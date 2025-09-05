import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import csv
from scipy.ndimage import zoom
import random
import argparse 
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import pandas as pd
import shutil
import time


#set the seed for splitting the data to training and validation
arr_ind = 0 #int(os.environ["SLURM_ARRAY_TASK_ID"])
seeds = [42, 33, 91, 9, 25]
seed = seeds[arr_ind]

#parser = argparse.ArgumentParser()
#parser.add_argument("--train_fraction", type=float, default=0.05, help="Train-set-out-of-original")
#args = parser.parse_args()

efm = False #set True when using AlphaEarth presentations
train_fraction = 1.0 #the fraction of data used (train + validation / original train set) #args.train_fraction 
run_id = "fraction_" + str(train_fraction)
print("Train fraction:", train_fraction)

only_testing = False #if True, uses existing checkpoints to do predictions
val_fraction = 0.2 #train to val split

num_workers = 4
n_epochs = 80
learning_rate = 1e-4
n_test_files = 2773
batch_size = 4

do_new_split = True #if True, uses the seed to make the splits
writeout = False #if True, writes out files in train/val splits

thresholded = False

####
#set correct paths
root_dir = "/m/cs/scratch/networks/silja/forests/btfm_project/"
ground_truth_dir = "/m/cs/scratch/networks/silja/forests/btfm_project/ground_truth_clipped/"

if efm == False:
    root_dir = root_dir + "tessera/"
    representation_folder = "representations"
else:
    root_dir = root_dir + "efm/"
    representation_folder = "data"

pred_path = root_dir + "predictions_clipped_rmse_git/seed"+ str(seed) + "/" #path for predictions
checkpoint_dir = os.path.join(root_dir, "checkpoints_clipped_rmse_git", run_id, str(seed)) #path for model checkpoints


####

n_in_channels = 128
if efm == True:
    n_in_channels = 64
    
agbm_threshold = None

if thresholded == True:
    pred_path = root_dir + "predictions_thresholded/"+ str(seed)+ "/"
    checkpoint_dir = os.path.join("checkpoints_thresholded", run_id, str(seed))
    agbm_threshold = 400

os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, f"best_patch_ckpt_{run_id}.pth")
finetuning_checkpoint_path = os.path.join(checkpoint_dir, f"best_patch_ckpt_{run_id}_finetuned.pth")


def dequantize(rep_path, scale_path):

    int8_array = np.load(rep_path)
    scale = np.load(scale_path)

    arr = int8_array.astype(np.float32) * scale[..., np.newaxis]

    return arr


def compute_target_stats(target_files):

    """ calculates mean and std of target_files """
    
    all_values = []
    
    for f in target_files:
        arr = np.load(f)
        valid = arr[~np.isnan(arr)]
        all_values.append(valid)
    all_values = np.concatenate(all_values)
    
    return np.mean(all_values), np.std(all_values)

def remove_module_prefix(state_dict):
    """Removes 'module.' prefix from state_dict keys if present."""
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[new_key] = v
    return new_state_dict
    
####################################
# 3. 数据集划分与 DataLoader
####################################

def get_splits(root_dir, split_csv_start, train_fraction):

        
    csv_train = split_csv_start +  "trains_"+ str(train_fraction) + ".csv"
    csv_val = split_csv_start +  "vals_"+ str(train_fraction) + ".csv"

    train_representations_folder = os.path.join(root_dir, representation_folder + "/representations_train_all")
    test_representations_folder = os.path.join(root_dir, representation_folder + "/representations_test")
    train_agb_folder = os.path.join(ground_truth_dir, "train_agbm_all")
    test_agb_folder = os.path.join(ground_truth_dir, "test_agbm")

    #form lists of trains and vals
    train_rep, train_target = [],[]
    val_rep, val_target = [],[]

    for csv, reps, targets in zip([csv_train,csv_val],[train_rep,val_rep],[train_target,val_target]):
        
        #chosen = pd.read_csv(csv,header=False)[0]
        df = pd.read_csv(csv, header=None)
        chosen = df.iloc[:, 0]
        
        for file in chosen:
            basename = file.replace(".npy", "") #these are all with npy
            #check that file in rep folder
            if not os.path.isfile(train_representations_folder + "/"+basename +".npy"):
                raise FileNotFoundError(train_representations_folder + "/"+basename +".npy")
            #check that file in agb_folder
            if not os.path.isfile(train_agb_folder + "/"+basename + ".npy"):
                raise FileNotFoundError(train_agb_folder + "/"+basename + ".npy")
            #append both to list
            reps.append(train_representations_folder + "/"+basename +".npy")
            targets.append(train_agb_folder + "/"+ basename + ".npy")


    #test set
    test_representations_folder = os.path.join(root_dir, representation_folder + "/representations_test")
    test_agb_folder = os.path.join(ground_truth_dir, "test_agbm")


    test_rep = sorted(glob.glob(os.path.join(test_representations_folder, "*.npy")))
    test_target = sorted(glob.glob(os.path.join(test_agb_folder, "*.npy")))

    print("test_representations_folder",test_representations_folder)
    print("test_agb_folder",test_agb_folder)


    print("n train files:", len(train_rep))
    print("n val files:", len(val_rep))
    print("n test files:", len(test_rep))

    print(f"# in test_rep: {len(test_rep)}")
    print(f"# in test_target: {len(test_target)}")

    test_rep_names = {os.path.basename(p) for p in test_rep}
    test_target_names = {os.path.basename(p) for p in test_target}
    print(f"# matching test_rep ∩ test_target: {len(test_rep_names & test_target_names)}")

    return train_rep, train_target, val_rep, val_target, test_rep, test_target 




def load_filtered_file_list(rep_files, agb_files, year_csv, valid_years=[2020, 2021]):
    
    df = pd.read_csv(year_csv)
    year_dict = {fname.replace(".tif", ""): year for fname, year in zip(df["new_fname"], df["year"])}

    filtered_rep = []
    filtered_agb = []

    for rep_path in rep_files:
        rep_name = os.path.basename(rep_path).replace(".npy", "")
        if year_dict.get(rep_name) in valid_years:
            filtered_rep.append(rep_path)
            filtered_agb.append(os.path.join(os.path.dirname(agb_files[0]), rep_name + ".npy"))

    return filtered_rep, filtered_agb



def get_file_lists(root_dir):
    
    train_rep_dir = os.path.join(root_dir, representation_folder, "representations_train_all")
    test_rep_dir = os.path.join(root_dir, representation_folder, "representations_test")
    train_agb_dir = os.path.join(ground_truth_dir, "train_agbm_all")
    test_agb_dir = os.path.join(ground_truth_dir, "test_agbm")

    train_rep_files = sorted(glob.glob(os.path.join(train_rep_dir, "*.npy")))
    train_agb_files = sorted(glob.glob(os.path.join(train_agb_dir, "*.npy")))
    test_rep_files = sorted(glob.glob(os.path.join(test_rep_dir, "*.npy")))
    test_agb_files = sorted(glob.glob(os.path.join(test_agb_dir, "*.npy")))

    # Check file alignment 
    def check_alignment(rep_files, agb_files, split_name):
        rep_set = {os.path.basename(f) for f in rep_files}
        agb_set = {os.path.basename(f) for f in agb_files}
        if rep_set != agb_set:
            missing = rep_set - agb_set
            if missing:
                print(f"⚠️  Warning: missing AGB files for {split_name}: {len(missing)}")
            print(f"Mismatch in {split_name}: {len(rep_set)} reps, {len(agb_set)} agbs")

    check_alignment(train_rep_files, train_agb_files, "train")
    check_alignment(test_rep_files, test_agb_files, "test")


    print(f"n train files: {len(train_rep_files)}")
    print(f"n test files: {len(test_rep_files)}")

    return train_rep_files, train_agb_files, test_rep_files, test_agb_files





def get_limited_labels(train_rep, train_target, train_fraction, seed):
    
    """
    Selects a random subset of data based on `train_fraction`, then splits into
    training and validation sets using `val_fraction` of that subset.
    
    Returns:
        train_rep_sub, val_rep_sub, train_target_sub, val_target_sub
    """
    assert 0 < train_fraction <= 1, "train_fraction must be between 0 and 1"
    #assert 0 <= val_fraction < 1, "val_fraction must be between 0 and 1"
    assert len(train_rep) == len(train_target), "Mismatch in number of rep and target files"

    # Create a map from filename to target path
    target_map = {os.path.basename(p): p for p in train_target}

    # Shuffle and select subset
    random.seed(seed)
    selected_rep = random.sample(train_rep, int(train_fraction * len(train_rep)))

    # Get corresponding targets
    selected_target = []
    for rep_path in selected_rep:
        fname = os.path.basename(rep_path)
        if fname not in target_map:
            raise ValueError(f"Target file for {fname} not found")
        selected_target.append(target_map[fname])

    # Split selected_rep/selected_target into train/val
    val_size = int(val_fraction * len(selected_rep))
    random.seed(seed + 1)  # Different seed to avoid overlap with selection
    val_indices = set(random.sample(range(len(selected_rep)), val_size))

    train_rep_sub, val_rep_sub = [], []
    train_target_sub, val_target_sub = [], []

    for i, (rep, tgt) in enumerate(zip(selected_rep, selected_target)):
        if i in val_indices:
            val_rep_sub.append(rep)
            val_target_sub.append(tgt)
        else:
            train_rep_sub.append(rep)
            train_target_sub.append(tgt)

    
    start = ""

    if only_testing == False and writeout==True:
        
        csvname = start + "limited_labels_fraction_trains_"+ str(train_fraction) + "_seed_" + str(seed) + ".csv"
        
        with open("splits/" + csvname, mode="w", newline="") as f:
            writer = csv.writer(f)
            for file_path in train_rep_sub:
                writer.writerow([os.path.basename(file_path)])
    
        csvname = start + "limited_labels_fraction_vals_"+ str(train_fraction) + "_seed_" + str(seed) + ".csv"
        
        with open("splits/" + csvname, mode="w", newline="") as f:
            writer = csv.writer(f)
            for file_path in val_rep_sub:
                writer.writerow([os.path.basename(file_path)])

    return train_rep_sub, train_target_sub, val_rep_sub, val_target_sub

"""
def get_limited_labels(train_rep, train_target, train_fraction, seed = 42):

    assert 0 < train_fraction <= 1, "train_fraction must be between 0 and 1"
    assert len(train_rep) == len(train_target), "Mismatch in number of rep and target files"

    # Create a map from filename to target path
    target_map = {os.path.basename(p): p for p in train_target}

    # Shuffle and select subset
    random.seed(seed)
    selected_rep = random.sample(train_rep, int(train_fraction * len(train_rep)))

    # Get corresponding targets
    selected_target = []
    for rep_path in selected_rep:
        fname = os.path.basename(rep_path)
        if fname not in target_map:
            raise ValueError(f"Target file for {fname} not found")
        selected_target.append(target_map[fname])

    #csvname = "limited_labels_fraction_"+ str(train_fraction) + ".csv"
    
    #with open("splits/" + csvname, mode="w", newline="") as f:
        #writer = csv.writer(f)
        #for file_path in selected_rep:
            #writer.writerow([os.path.basename(file_path)])

    selected_val = random.sample(selected_rep, int(val_fraction*len(selected_rep))

    return selected_rep, selected_target

 """   
####################################
# 1. 定义数据集类，并过滤全为 NaN 的 patch
####################################

class BiomasstersDataset(Dataset):
    
    def __init__(self, rep_files, target_files, target_mean, target_std):
        """
        :param rep_files: representation .npy 文件列表，形状 (100,100,128)
        :param target_files: target .npy 文件列表，形状 (100,100)，其中部分值可能为 NaN
        """
        # 过滤掉那些 target 全为 NaN 的 patch
        valid_rep_files = []
        valid_target_files = []
        for rep_file, target_file in zip(rep_files, target_files):
            target = np.load(target_file)
            if np.all(np.isnan(target)):
                print(f"Skipping {target_file} as all values are NaN.")
                continue
            valid_rep_files.append(rep_file)
            valid_target_files.append(target_file)
        
        self.rep_files = valid_rep_files
        self.target_files = valid_target_files
        self.target_mean = target_mean
        self.target_std = target_std
        
    def __len__(self):
        return len(self.rep_files)
    
    def __getitem__(self, idx):
        # 加载 representation 和 target
        
        #rep = np.load(self.rep_files[idx])
        rep_path = self.rep_files[idx]

        if "_train" in rep_path:
            scale_file = self.rep_files[idx].replace("representations_train", "scales_train")
        elif "_val" in self.rep_files[idx]:
            scale_file = self.rep_files[idx].replace("representations_validation", "scales_validation")
        elif "_test" in self.rep_files[idx]:
            scale_file = self.rep_files[idx].replace("representations_test", "scales_test")
        else:
            raise ValueError(f"Representation path doesn't contain 'train', 'val' or 'test': {rep_path}")

        if efm == False:
            rep = dequantize(self.rep_files[idx], scale_file)
        else:
            rep = np.load(self.rep_files[idx])

        target = np.load(self.target_files[idx])
        
        # representation: (100,100,128) -> 转为 (128,100,100)
        rep = torch.from_numpy(rep).float().permute(2, 0, 1)
        
        # target: (100,100) -> 转为 (1,100,100)
        target = torch.from_numpy(target).float().unsqueeze(0)
        # 对 target 进行归一化，nan 保持不变
        target = (target - self.target_mean) / self.target_std

        filename = os.path.basename(self.target_files[idx]).replace(".npy", "")
            
        return rep, target,filename

####################################
# 2. 定义 UNet 模型
####################################
class DoubleConv(nn.Module):
    """两次 3x3 卷积 + BatchNorm + ReLU"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.double_conv(x)





class UNet(nn.Module):
    def __init__(self, in_channels=n_in_channels, out_channels=1, features=[256, 512]):
        """
        :param in_channels: 输入通道数（这里为 128）
        :param out_channels: 输出通道数（回归任务设为 1）
        :param features: 编码器中每层的特征数
        """
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups   = nn.ModuleList()
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 构建编码器（Downsampling）
        curr_in_channels = in_channels
        for feature in features:
            self.downs.append(DoubleConv(curr_in_channels, feature))
            curr_in_channels = feature
        
        # Bottleneck 层
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        
        # 构建解码器（Upsampling）
        decoder_in_channels = features[-1] * 2
        for feature in reversed(features):
            # 使用转置卷积进行上采样
            self.ups.append(nn.ConvTranspose2d(decoder_in_channels, feature, kernel_size=2, stride=2))
            # 拼接后通道数为 2*feature，再经过双卷积降回 feature
            self.ups.append(DoubleConv(feature*2, feature))
            decoder_in_channels = feature
        
        # 最终的 1x1 卷积，将通道数映射为 1
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
    def forward(self, x):
        skip_connections = []
        # 编码器部分：保存每个阶段的特征用于跳跃连接
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
            
        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # 反转以便与解码器对应
        
        # 解码器部分：上采样、拼接、双卷积
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            # 如果尺寸不匹配，则进行插值调整
            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(x, size=skip_connection.shape[2:])
            x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](x)
            
        return self.final_conv(x)





####################################
# 4. 定义指标计算与 Masked Loss
####################################


def masked_mse_single(pred, target):
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return (pred * 0).sum()
    return torch.mean((pred[mask] - target[mask]) ** 2)


def chipwise_rmse_loss(pred, target, threshold=None):
    losses = []
    for b in range(pred.shape[0]):
        # Always mask out NaNs
        mask = ~torch.isnan(target[b])
        
        # Optionally mask out values >= threshold
        if threshold is not None:
            mask &= target[b] < threshold

        if mask.sum() == 0:
            losses.append((pred[b] * 0).sum())  # zero loss tied to graph
        else:
            mse = torch.mean((pred[b][mask] - target[b][mask]) ** 2)
            losses.append(torch.sqrt(mse))
    return torch.stack(losses).mean()


"""
def chipwise_rmse_loss(pred, target):

    #Computes RMSE per chip and averages across the batch.
    #Input shape: (B, 1, H, W)

    losses = []
    for b in range(pred.shape[0]):
        mse = masked_mse_single(pred[b], target[b])  # shape: (1, H, W)
        rmse = torch.sqrt(mse)
        losses.append(rmse)
    return torch.stack(losses).mean()
"""


def masked_mse_loss(pred, target):
    """
    仅对 target 中非 NaN 部分计算 MSE Loss；
    如果全部为 NaN，则返回一个与计算图关联的零损失。
    """
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        # 返回一个与 pred 计算图相关联的零损失
        return (pred * 0).sum()
    loss = torch.mean((pred[mask] - target[mask]) ** 2)
    return loss 
    #return torch.sqrt(loss)

    
def compute_metrics(pred, target, target_mean, target_std):
    """
    计算 MAE, RMSE, R2 指标，排除 target 中的 NaN 值
    先反归一化，再计算指标
    """
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return 0.0, 0.0, 0.0
    
    # 反归一化
    pred_denorm = pred * target_std + target_mean
    target_denorm = target * target_std + target_mean

    # 仅计算有效像素
    pred_valid = pred_denorm[mask]
    target_valid = target_denorm[mask]
    
    mae = torch.mean(torch.abs(pred_valid - target_valid))
    bias = torch.mean(pred_valid - target_valid)
    rmse = torch.sqrt(torch.mean((pred_valid - target_valid) ** 2))
    
    # 计算 R2：先 flatten 有效像素
    pred_flat = pred_valid.view(-1)
    target_flat = target_valid.view(-1)
    ss_res = torch.sum((target_flat - pred_flat) ** 2)
    ss_tot = torch.sum((target_flat - torch.mean(target_flat)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else torch.tensor(0.0)
    
    return mae.item(), bias.item(), rmse.item(), r2.item()

####################################
# 5. 训练、验证与测试过程（含checkpoint保存与加载）
####################################
    
def train_model(model, train_loader, val_loader, device, target_mean, target_std, epochs=50, lr=1e-3,checkpoint_path=None,threshold=None):
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    
    # 用于保存最佳验证loss
    best_val_loss = float('inf')


    for epoch in range(1, epochs+1):
        model.train()
        train_losses = []
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        
        #for batch_idx, (rep, target) in enumerate(progress_bar):
        for batch_idx, (rep, target, _) in enumerate(progress_bar):
            rep = rep.to(device)       # (B, 128, 64, 64)
            target = target.to(device) # (B, 1, 64, 64) 其中部分像素可能为 NaN
            
            optimizer.zero_grad()
            output = model(rep)        # 输出应为 (B, 1, 64, 64)
            loss = chipwise_rmse_loss(output,target,threshold) 
            #loss = masked_mse_loss(output, target)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            mae, bias, rmse, r2 = compute_metrics(output, target, target_mean, target_std)
            
            # 实时更新进度条显示指标
            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "MAE": f"{mae:.4f}",
                "RMSE": f"{rmse:.4f}",
                "R2": f"{r2:.4f}"
            })
        
        # 验证集评估
        if val_loader is not None:
    
            model.eval()
            val_loss = 0.0
            val_mae = 0.0
            val_rmse = 0.0
            val_r2 = 0.0
            with torch.no_grad():
                for rep, target, _ in val_loader:
                    rep = rep.to(device)
                    target = target.to(device)
                    output = model(rep)
                    loss = chipwise_rmse_loss(output,target,threshold)
                    #loss = masked_mse_loss(output, target)
                    val_loss += loss.item()
                    mae, bias,rmse, r2 = compute_metrics(output, target, target_mean, target_std)
                    val_mae += mae
                    val_rmse += rmse
                    val_r2 += r2
            
            num_val = len(val_loader)
            avg_val_loss = val_loss / num_val
            print(f"Epoch {epoch}/{epochs} - Train Loss: {np.mean(train_losses):.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | Val MAE: {val_mae/num_val:.4f} | "
                  f"Val RMSE: {val_rmse/num_val:.4f} | Val R2: {val_r2/num_val:.4f}")
            
            # 保存验证集loss最好的checkpoint
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss
                }
                torch.save(checkpoint, checkpoint_path)
                print(f"Saved new best checkpoint at epoch {epoch} with Val Loss: {avg_val_loss:.4f}")
            
        else:
            checkpoint = {'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': np.mean(train_losses),
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved final checkpoint to {checkpoint_path}")




def main():

    t1 = time.time()
    
    if do_new_split == True:
        print("✅ Doing a fresh train/val split from in-memory file list")
        train_rep, train_target, test_rep, test_target = get_file_lists(root_dir)
    
        #limited labels
        train_rep, train_target, val_rep, val_target = get_limited_labels(train_rep, train_target, train_fraction, seed)
    
    else:
        print("📄 Loading split from CSV files in 'splits/'")

        start = ""
    
        split_csv_start = "splits/" + start + "limited_labels_fraction_" #trains_"+ str(train_fraction) + ".csv"
        train_rep, train_target, val_rep, val_target, test_rep, test_target = get_splits(root_dir, split_csv_start, train_fraction)
        
    print("Using n labels in training:", len(train_rep), "nvals:", len(val_rep))

    #check that all pixels are valid
    trains = [train_rep, train_target]
    vals = [val_rep, val_target]
    tests = [test_rep, test_target]

    # Ensure no overlap between train/val and test filenames
    train_val_fnames = {os.path.basename(f) for f in train_rep + val_rep}
    test_fnames = {os.path.basename(f) for f in test_rep}
    overlap = train_val_fnames & test_fnames
    if overlap:
        raise ValueError(f"Train/val files found in test set: {overlap}")
    else:
        print("✅ No overlap between train/val and test sets.")

    #just another check
    full_train_dir = os.path.join(root_dir, representation_folder + "/representations_train_all")
    full_train_count = len([f for f in os.listdir(full_train_dir) if f.endswith(".npy")])
    
    used_total = len(train_rep) + len(val_rep)
    effective_fraction = used_total / full_train_count if full_train_count > 0 else 0
    actual_val_fraction = len(val_rep) / used_total if used_total > 0 else 0
    
    print(f"📊 Requested train_fraction: {train_fraction}")
    print(f"📊 Actual usable fraction (train+val): {effective_fraction:.3f}")
    print(f"📊 Requested val_fraction: {val_fraction}")
    print(f"📊 Actual val_fraction: {actual_val_fraction:.3f}")

    # Fail if mismatch is too large
    if abs(effective_fraction - train_fraction) > 0.05:
        raise ValueError("🚨 Effective train_fraction deviates significantly from requested value.")

    
    for files in [trains,  vals, tests]:
        rep_files, target_files = files[0], files[1]
        valid_pixel_counts = []
        for rep_file, target_file in zip(rep_files, target_files):
            target = np.load(target_file)
            if np.all(np.isnan(target)):
                print(f"Skipping {target_file} as all values are NaN.")
                continue
            valid_pixel_counts.append(np.sum(~np.isnan(target)))
        
        # Optional check:
        if len(set(valid_pixel_counts)) > 1:
            raise ValueError("Not all patches have the same number of valid pixels!")
    
    target_mean, target_std = compute_target_stats(train_target)
    
    if agbm_threshold is not None:
        agbm_threshold_normalized = (agbm_threshold - target_mean) / target_std
        if target_mean > agbm_threshold:
            raise Exception("SOMETHING WRONG with agbm threshold!!") 

    else:
        agbm_threshold_normalized = None
    
    
    
    # 创建数据集与 DataLoader
    train_dataset = BiomasstersDataset(train_rep, train_target, target_mean, target_std)
    val_dataset = BiomasstersDataset(val_rep, val_target, target_mean, target_std)
    test_dataset = BiomasstersDataset(test_rep, test_target, target_mean, target_std)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # 模型与设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=n_in_channels, out_channels=1)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_params} parameters.")
    
    if only_testing == False:
        
        train_model(model, train_loader, val_loader, device, target_mean, target_std, epochs=n_epochs, lr=learning_rate,checkpoint_path = checkpoint_path, threshold=agbm_threshold_normalized)
        """
        print("Finetuning on train+val before testing...")

        # Combine datasets
        combined_rep = train_rep + val_rep
        combined_target = train_target + val_target
        
        # Recreate dataset and loader
        finetune_dataset = BiomasstersDataset(combined_rep, combined_target, target_mean, target_std)
        finetune_loader = DataLoader(finetune_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        
        # Re-initialize model (optional, if you want a fresh fine-tuning pass)
        # model = UNet(in_channels=n_in_channels, out_channels=1).to(device)
        
        train_model(model, finetune_loader, val_loader=None, device=device, target_mean=target_mean, target_std=target_std, epochs=10, lr=learning_rate,checkpoint_path = finetuning_checkpoint_path)
        """
    # 加载最佳的checkpoint再进行测试

    #checkpoint = torch.load(checkpoint_path, map_location=device)
    
    t2 = time.time()
    print("training time:", (t2-t1)/60)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint["model_state_dict"]
        clean_state_dict = remove_module_prefix(state_dict)

        # Load into model
        model.load_state_dict(clean_state_dict)

        #model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        print(f"Loaded best checkpoint from epoch {checkpoint['epoch']} with Val Loss: {checkpoint['val_loss']:.4f}")
    else:
        print("No checkpoint found. Using current model for testing.")
    
    # 测试集评估
    model.eval()

    print(f"#Test samples: {len(test_loader.dataset)}")

    pred_dir = os.path.join(pred_path, run_id)
    if os.path.exists(pred_dir):
        print(f"🧹 Clearing prediction directory: {pred_dir}")
        shutil.rmtree(pred_dir)

    print(f"Writing to prediction directory: {pred_dir}")

    os.makedirs(pred_dir, exist_ok=True)
   
    with torch.no_grad():
        #for rep, target in test_loader:
        for i, (rep, target, filenames) in enumerate(test_loader):
            rep = rep.to(device)
            target = target.to(device)
            output = model(rep)

            preds = output * target_std + target_mean  # denormalize
            preds = torch.clamp(preds, min=0)
            preds_np = preds.cpu().numpy()  # shape (B, 1, H, W)
                
            for j in range(preds_np.shape[0]):
                
                base_name = filenames[j] 
                out_path = os.path.join(pred_dir, f"{base_name}.npy")
                np.save(out_path, preds_np[j, 0])  # save (H, W) array


if __name__ == "__main__":
    main()