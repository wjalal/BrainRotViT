"""
3D ResNet Training and Evaluation Pipeline
=================================================
This script uses 3D ResNet for brain age estimation instead of Vision Transformer.

The pipeline:
1. Loads 3D NIfTI medical images (.nii or .nii.gz)
2. Preprocesses brain volumes to (160, 192, 160)
3. Uses the same train/val split as original (80/20 split)
4. Trains a 3D ResNet model on the training set
5. Evaluates on the validation set
6. Reports Mean Absolute Error (MAE) for both training and validation sets

CHECKPOINT FEATURES:
- {model_name}_last_model_with_metadata.pkl: Saved after every epoch
- {model_name}_best_model_with_metadata.pkl: Saved only when validation MAE improves

USAGE:
------
python 3dresnet.py [model_name] [load_saved] [num_epochs]

Arguments:
    model_name  : Name for the model (default: 'resnet')
    load_saved  : Which checkpoint to load - 'none', 'last', or 'best' (default: 'none')
    num_epochs  : Number of epochs to train (default: 100)

Examples:
    python 3dresnet.py                          # Train from scratch for 100 epochs
    python 3dresnet.py my_model none 50         # Train new model for 50 epochs
    python 3dresnet.py my_model last 100        # Resume from last checkpoint
    python 3dresnet.py my_model best 100        # Resume from best checkpoint
"""

import os
import sys
import datetime
import warnings
import pickle
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
from scipy.ndimage import zoom, label, find_objects
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform, apply_orientation

warnings.filterwarnings("ignore")

# ============================================================================
# RANDOM SEED CONFIGURATION
# ============================================================================
def set_random_seed(seed=69420, enable_benchmark=True):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed value
        enable_benchmark: If True, enable cudnn.benchmark for faster training
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if enable_benchmark:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        print("⚡ cudnn.benchmark enabled for faster training")
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

UNIVERSAL_SEED = 69420
set_random_seed(UNIVERSAL_SEED, enable_benchmark=True)

# ============================================================================
# DATA PREPROCESSING FUNCTIONS
# ============================================================================
def white0(image, threshold=0):
    """Standardize voxels with value > threshold"""
    image = image.astype(np.float32)
    mask = (image > threshold).astype(int)
    image_h = image * mask
    
    non_zero_voxels = np.sum(mask)
    if non_zero_voxels > 0:
        mean = np.sum(image_h) / non_zero_voxels
        std_sum = np.sum((image_h - mean * mask) ** 2)
        std = np.sqrt(std_sum / non_zero_voxels)
        
        if std > 1e-8:
            normalized = mask * (image - mean) / (std + 1e-8)
            image = normalized + image * (1 - mask)
            image = np.clip(image, -10.0, 10.0)
            return image
    
    return np.zeros_like(image, dtype=np.float32)


def resample_nifti(img_data, target_shape=(160, 192, 160)):
    """Resample NIfTI volume to target shape"""
    current_shape = img_data.shape
    zoom_factors = [target_shape[i] / current_shape[i] for i in range(3)]
    resampled_data = zoom(img_data, zoom_factors, order=3)
    return resampled_data


def calculate_bounding_box_from_volume(volume, intensity_threshold=0.1):
    """Calculate bounding box from volume using connected components"""
    volume_normalized = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
    binary_mask = volume_normalized > intensity_threshold
    labeled_array, num_features = label(binary_mask)
    component_sizes = np.bincount(labeled_array.ravel())
    component_sizes[0] = 0
    largest_component = np.argmax(component_sizes)
    brain_mask = labeled_array == largest_component
    slices = find_objects(brain_mask.astype(int))[0]
    min_indices = [s.start for s in slices]
    max_indices = [s.stop - 1 for s in slices]
    return min_indices, max_indices


def crop_brain_volumes(brain_data):
    """Crop brain volume to bounding box"""
    min_indices, max_indices = calculate_bounding_box_from_volume(brain_data)
    cropped_brain = brain_data[min_indices[0]:max_indices[0] + 1,
                                min_indices[1]:max_indices[1] + 1,
                                min_indices[2]:max_indices[2] + 1]
    return cropped_brain


# ============================================================================
# 3D RESNET MODEL ARCHITECTURE
# ============================================================================
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet3D(nn.Module):
    """3D ResNet for brain age estimation"""
    def __init__(self, block, layers, num_classes=1,
                 channel_size=[64, 64, 128, 256, 512],
                 dropout=False):
        c = channel_size
        self.inplanes = c[0]
        super(ResNet3D, self).__init__()
        
        net = nn.Sequential()
        net.add_module('conv1', nn.Conv3d(1, c[0], kernel_size=7, stride=2, padding=3, bias=False))
        net.add_module('bn1', nn.BatchNorm3d(c[0]))
        net.add_module('relu', nn.ReLU(inplace=True))
        net.add_module('maxpool', nn.MaxPool3d(kernel_size=3, stride=2, padding=1))
        net.add_module('layer1', self._make_layer(block, c[1], layers[0]))
        net.add_module('layer2', self._make_layer(block, c[2], layers[1], stride=2))
        net.add_module('layer3', self._make_layer(block, c[3], layers[2], stride=2))
        net.add_module('layer4', self._make_layer(block, c[4], layers[3], stride=2))
        net.add_module('avgpool', nn.AdaptiveAvgPool3d((1, 1, 1)))
        
        if dropout:
            net.add_module('dropout', nn.Dropout(0.5))
        
        self.feature_extractor = net
        
        # Regression head
        self.fc1 = nn.Linear(c[4] * block.expansion, 1000)
        self.fc2 = nn.Linear(1000, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x shape: [B, D, H, W] -> add channel dim
        if x.dim() == 4:
            x = x.unsqueeze(1)  # [B, 1, D, H, W]
        
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


def resnet18(**kwargs):
    """ResNet-18 model"""
    model = ResNet3D(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """ResNet-34 model"""
    model = ResNet3D(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """ResNet-50 model"""
    model = ResNet3D(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


# ============================================================================
# DATASET CLASS
# ============================================================================
class BrainAgeDataset(Dataset):
    """Dataset class for loading brain MRI images"""
    def __init__(self, dataframe, target_shape=(160, 192, 160)):
        self.df = dataframe.reset_index(drop=True)
        self.target_shape = target_shape
        
        # Filter out non-existent files
        valid_indices = []
        for idx in range(len(self.df)):
            filepath = self.df.iloc[idx]['filepath']
            if os.path.exists(filepath):
                valid_indices.append(idx)
            else:
                print(f"Warning: File not found, skipping: {filepath}")
        
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)
        print(f"Dataset initialized with {len(self.df)} valid samples")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = row['filepath']
        age = row['Age']
        sex = row['Sex']
        
        try:
            nii_img = nib.load(filepath)
            
            # Reorient to RAS
            orig_ornt = io_orientation(nii_img.affine)
            ras_ornt = axcodes2ornt(("R", "A", "S"))
            ornt_trans = ornt_transform(orig_ornt, ras_ornt)
            
            data = nii_img.get_fdata()
            data = apply_orientation(data, ornt_trans)
            
            # Crop and resample
            data = crop_brain_volumes(data)
            data = resample_nifti(data, target_shape=self.target_shape)
            data = (data - data.min()) / (data.max() - data.min() + 0.000000001)


            # Normalize
            data = white0(data)
            
            # Convert to tensor [D, H, W]
            data = np.ascontiguousarray(data, dtype=np.float32)
            
            # Check for NaN or Inf
            if np.isnan(data).any() or np.isinf(data).any():
                print(f"Warning: NaN/Inf detected in {filepath}, replacing with zeros")
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            
            data = torch.from_numpy(data).type(torch.FloatTensor)
            
            # Encode sex (0 for M, 1 for F)
            sex_encoded = 0.0 if sex == 'M' else 1.0
            sex_tensor = torch.tensor(sex_encoded, dtype=torch.float32)
            
            # Age as float
            age_tensor = torch.tensor(age, dtype=torch.float32)
            
            return data, sex_tensor, age_tensor
            
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            dummy_data = torch.zeros(self.target_shape, dtype=torch.float32)
            return dummy_data, torch.tensor(0.0), torch.tensor(0.0)


# ============================================================================
# TRAINING UTILITIES
# ============================================================================
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve"""
    def __init__(self, patience=200, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_metric):
        score = val_metric

        if self.best_score is None:
            self.best_score = score
        elif score > self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}\n')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def metric(output, target):
    """Calculate MAE metric with NaN handling"""
    target_np = target.cpu().data.numpy()
    pred_np = output.cpu().data.numpy()
    
    if np.isnan(pred_np).any() or np.isinf(pred_np).any():
        print(f"Warning: NaN or Inf detected in predictions!")
        return float('inf')
    if np.isnan(target_np).any() or np.isinf(target_np).any():
        print(f"Warning: NaN or Inf detected in targets!")
        return float('inf')
    
    mae = mean_absolute_error(target_np, pred_np)
    return mae


def save_checkpoint(state, is_best, out_dir, model_name='resnet'):
    """Save model checkpoint"""
    os.makedirs(out_dir, exist_ok=True)
    
    last_model_path = os.path.join(out_dir, f'{model_name}_last_model_with_metadata.pkl')
    with open(last_model_path, 'wb') as f:
        pickle.dump(state, f)
    
    if is_best:
        best_model_path = os.path.join(out_dir, f'{model_name}_best_model_with_metadata.pkl')
        with open(best_model_path, 'wb') as f:
            pickle.dump(state, f)
        print("=======> This is the best model! It has been saved!\n")


def load_checkpoint(model, optimizer, out_dir, model_name='resnet', load_type='last'):
    """Load checkpoint and resume training"""
    if load_type == 'none':
        return 0, float('inf')
    
    best_model_path = os.path.join(out_dir, f'{model_name}_best_model_with_metadata.pkl')
    best_loss = float('inf')
    
    if os.path.exists(best_model_path):
        with open(best_model_path, 'rb') as f:
            best_checkpoint = pickle.load(f)
        best_loss = best_checkpoint.get('loss', float('inf'))
        print(f"Best model found with validation loss: {best_loss:.4f}")
    
    checkpoint_path = os.path.join(out_dir, f'{model_name}_{load_type}_model_with_metadata.pkl')
    
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}. Starting from scratch.")
        return 0, best_loss
    
    print(f"\nLoading checkpoint from {checkpoint_path}")
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    
    if 't_rng_st' in checkpoint:
        torch.set_rng_state(checkpoint['t_rng_st'])
    if 'n_rng_st' in checkpoint:
        np.random.set_state(checkpoint['n_rng_st'])
    if 'cuda_rng_st' in checkpoint and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng_st'])
    
    start_epoch = checkpoint['epoch'] + 1
    loaded_loss = checkpoint.get('loss', float('inf'))
    
    print(f"Checkpoint loaded successfully!")
    print(f"  Resuming from epoch: {start_epoch}")
    print(f"  Checkpoint val loss: {loaded_loss:.4f}")
    print(f"  Best val loss so far: {best_loss:.4f}")
    
    return start_epoch, best_loss


# ============================================================================
# TRAINING AND VALIDATION FUNCTIONS
# ============================================================================
def train_one_epoch(train_loader, model, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    losses = AverageMeter()
    MAE = AverageMeter()
    
    model.train()
    scaler = torch.amp.GradScaler('cuda')
    
    for i, (img, sex, target) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}")):
        img = img.to(device, non_blocking=True)
        sex = sex.to(device, non_blocking=True)
        target = target.unsqueeze(1).to(device, non_blocking=True)
        
        if torch.isnan(img).any() or torch.isinf(img).any():
            print(f"Warning: NaN/Inf in input at batch {i}, skipping...")
            continue
        
        with torch.amp.autocast('cuda'):
            output = model(img)
            loss = criterion(output, target)
        
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print(f"Warning: NaN/Inf loss at batch {i}, skipping...")
            continue
        
        mae = metric(output.detach(), target.detach())
        
        if mae == float('inf'):
            print(f"Warning: Invalid MAE at batch {i}, skipping...")
            continue
        
        losses.update(loss.item(), img.size(0))
        MAE.update(mae, img.size(0))
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    return losses.avg, MAE.avg


def validate(valid_loader, model, criterion, device):
    """Validate the model"""
    losses = AverageMeter()
    MAE = AverageMeter()
    
    model.eval()
    
    with torch.no_grad():
        for img, sex, target in tqdm(valid_loader, desc="Validation"):
            img = img.to(device, non_blocking=True)
            sex = sex.to(device, non_blocking=True)
            target = target.unsqueeze(1).to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                output = model(img)
                loss = criterion(output, target)
            
            mae = metric(output.detach(), target.detach())
            losses.update(loss.item(), img.size(0))
            MAE.update(mae, img.size(0))
    
    print(f'Valid: [steps {len(valid_loader)}], Loss {losses.avg:.4f}, MAE: {MAE.avg:.4f}')
    
    return losses.avg, MAE.avg


# ============================================================================
# DATASET LOADING
# ============================================================================
def load_datasets():
    """Load and prepare datasets"""
    print("=" * 80)
    print("LOADING DATASETS")
    print("=" * 80)
    
    # Load ADNI dataset
    df_adni = pd.read_csv("adni_storage/adni_brainrotnet_metadata.csv")
    df_adni['filepath'] = df_adni.apply(
        lambda row: f"adni_storage/ADNI_nii_gz_bias_corrected/I{row['ImageID'][4:]}_{row['SubjectID']}.stripped.N4.nii.gz",
        axis=1)
    df_adni = df_adni.loc[
        df_adni.groupby('SubjectID')['Age'].apply(lambda x: (x - x.median()).abs().idxmin())
    ].reset_index(drop=True)
    df_adni = df_adni.sort_values(by='Age', ascending=True).reset_index(drop=True).head(900)
    
    # Load IXI dataset
    df_ixi = pd.read_csv("ixi_storage/ixi_brainrotnet_metadata.csv")
    df_ixi['filepath'] = df_ixi.apply(
        lambda row: f"ixi_storage/IXI_bias_corrected/{row['ImageID']}.stripped.N4.nii.gz",
        axis=1
    )
    
    # Load ABIDE dataset
    df_abide = pd.read_csv("abide_storage/abide_brainrotnet_metadata.csv")
    df_abide['filepath'] = df_abide.apply(
        lambda row: f"abide_storage/ABIDEII_bias_corrected/{row['ImageID'][7:]}.stripped.N4.nii.gz",
        axis=1
    )
    df_abide = df_abide.sort_values(by='Age', ascending=False).reset_index(drop=True).head(750)
    
    # Load DLBS dataset
    df_dlbs = pd.read_csv("dlbs_storage/dlbs_brainrotnet_metadata.csv")
    df_dlbs['filepath'] = df_dlbs.apply(
        lambda row: f"dlbs_storage/DLBS_bias_corrected/{row['ImageID'][4:]}.stripped.N4.nii.gz",
        axis=1
    )
    
    # Load COBRE dataset
    df_cobre = pd.read_csv("cobre_storage/cobre_brainrotnet_metadata.csv")
    df_cobre['filepath'] = df_cobre.apply(
        lambda row: f"cobre_storage/COBRE_bias_corrected/{row['ImageID'][5:]}.stripped.N4.nii.gz",
        axis=1
    )
    
    # Load FCON1000 dataset
    df_fcon = pd.read_csv("fcon1000_storage/fcon1000_brainrotnet_metadata.csv")
    df_fcon['filepath'] = df_fcon.apply(
        lambda row: f"fcon1000_storage/fcon1000_bias_corrected/{row['ImageID'][8:]}.stripped.N4.nii.gz",
        axis=1
    )
    df_fcon = df_fcon.dropna()
    
    # Load CORR dataset
    df_corr = pd.read_csv("corr_storage/corr_brainrotnet_metadata.csv")
    df_corr['filepath'] = df_corr.apply(
        lambda row: f"corr_storage/CORR_bias_corrected/{row['ImageID'][5:]}.stripped.N4.nii.gz",
        axis=1
    )
    df_corr = df_corr.sort_values(by='Age', ascending=True).reset_index(drop=True)
    
    # Load OASIS1 dataset
    df_oas1 = pd.read_csv("oasis1_storage/oasis1_brainrotnet_metadata.csv")
    df_oas1['filepath'] = df_oas1.apply(
        lambda row: f"oasis1_storage/oasis_nii_gz_bias_corrected/{row['ImageID']}.stripped.N4.nii.gz",
        axis=1
    )
    df_oas1 = df_oas1.sort_values(by='Age', ascending=False).reset_index(drop=True).head(300)
    
    # Load CamCAN dataset
    df_camcan = pd.read_csv("camcan_storage/camcan_brainrotnet_metadata.csv")
    df_camcan['filepath'] = df_camcan.apply(
        lambda row: f"camcan_storage/CamCAN_nii_gz_bias_corrected/{row['ImageID']}.stripped.N4.nii.gz",
        axis=1
    )
    
    # Load NIMH dataset
    df_nimh = pd.read_csv("nimh_storage/nimh_mprage_brainrotnet_metadata.csv")
    df_nimh['filepath'] = df_nimh.apply(
        lambda row: f"nimh_storage/nimh_bias_corrected/{row['ImageID'][5:]}_ses-01_acq-MPRAGE_rec-SCIC_T1w.stripped.N4.nii.gz",
        axis=1
    )
    
    # Load BOLD dataset
    df_bold = pd.read_csv("bold_storage/bold_brainrotnet_metadata.csv")
    df_bold['filepath'] = df_bold.apply(
        lambda row: f"bold_storage/bold_bias_corrected/{row['ImageID'][5:]}_T1w.stripped.N4.nii.gz",
        axis=1
    )
    
    # Combine all datasets
    df = pd.concat([
        df_adni[['ImageID', 'Sex', 'Age', 'filepath']], 
        df_ixi[['ImageID', 'Sex', 'Age', 'filepath']], 
        df_abide[['ImageID', 'Sex', 'Age', 'filepath']],
        df_dlbs[['ImageID', 'Sex', 'Age', 'filepath']],
        df_cobre[['ImageID', 'Sex', 'Age', 'filepath']],
        df_fcon[['ImageID', 'Sex', 'Age', 'filepath']],
        df_corr[['ImageID', 'Sex', 'Age', 'filepath']], 
        df_oas1[['ImageID', 'Sex', 'Age', 'filepath']],
        df_camcan[['ImageID', 'Sex', 'Age', 'filepath']],
        df_nimh[['ImageID', 'Sex', 'Age', 'filepath']],
        df_bold[['ImageID', 'Sex', 'Age', 'filepath']]
    ], ignore_index=True)
    
    print(f"\nTotal samples: {len(df)}")
    print(f"Age range: {df['Age'].min():.1f} - {df['Age'].max():.1f}")
    print(f"Sex distribution:\n{df['Sex'].value_counts()}")
    
    # Split into train and validation (80/20)
    train_size = int(0.8 * len(df))
    indices = list(range(len(df)))
    np.random.seed(UNIVERSAL_SEED)
    np.random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_df = df.iloc[train_indices].reset_index(drop=True)
    val_df = df.iloc[val_indices].reset_index(drop=True)
    
    print(f"\nTrain size: {len(train_df)}, Val size: {len(val_df)}")
    
    return train_df, val_df


# ============================================================================
# MAIN FUNCTION
# ============================================================================
def main():
    """Main training and evaluation pipeline"""
    # Parse command-line arguments
    model_name = sys.argv[1] if len(sys.argv) > 1 else '3dresnet'
    load_saved = sys.argv[2] if len(sys.argv) > 2 else 'none'
    num_epochs_arg = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    
    if load_saved not in ['none', 'last', 'best']:
        print(f"Error: load_saved must be 'none', 'last', or 'best'. Got '{load_saved}'")
        sys.exit(1)
    
    # Configuration
    print("\n" + "=" * 80)
    print(" RESNET: 3D ResNet Brain Age Estimation Pipeline")
    print("=" * 80)
    
    # Hyperparameters (optimized for RTX 4090)
    BATCH_SIZE = 4
    NUM_EPOCHS = num_epochs_arg
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 5e-4
    NUM_WORKERS = 8
    OUTPUT_DIR = f'model_dumps/resnet/{model_name}/'
    
    print("\nConfiguration (optimized for RTX 4090 24GB):")
    print(f"  Model name:     {model_name}")
    print(f"  Architecture:   ResNet-34 (3D)")
    print(f"  Load saved:     {load_saved}")
    print(f"  Batch size:     {BATCH_SIZE}")
    print(f"  Learning rate:  {LEARNING_RATE}")
    print(f"  Weight decay:   {WEIGHT_DECAY}")
    print(f"  Num workers:    {NUM_WORKERS}")
    print(f"  Epochs:         {NUM_EPOCHS}")
    print(f"  Output dir:     {OUTPUT_DIR}")
    print(f"  Mixed precision: Enabled (FP16)")
    print(f"  cudnn.benchmark: Enabled")
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"  Device:         {device}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"  GPU Memory:     {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    train_df, val_df = load_datasets()
    
    print("\nCreating datasets...")
    train_dataset = BrainAgeDataset(train_df, target_shape=(160, 192, 160))
    val_dataset = BrainAgeDataset(val_df, target_shape=(160, 192, 160))
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        shuffle=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        shuffle=False,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    print("\n" + "=" * 80)
    print("BUILDING MODEL")
    print("=" * 80)
    
    # Create ResNet-34 model
    model = resnet34(num_classes=1, dropout=False)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total:      {total_params:,}")
    print(f"  Trainable:  {trainable_params:,}")
    
    # Define loss, optimizer, scheduler
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        amsgrad=True
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=5,
        factor=0.5
    )
    early_stopping = EarlyStopping(patience=200, verbose=True)
    
    # Load checkpoint if specified
    start_epoch, best_metric = load_checkpoint(
        model, optimizer, OUTPUT_DIR, model_name, load_saved
    )
    
    # Training loop
    print("\n" + "=" * 80)
    print("START TRAINING")
    print("=" * 80)
    
    train_history = []
    val_history = []
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\n{'=' * 80}")
        print(f"EPOCH {epoch + 1}/{NUM_EPOCHS}")
        print(f"{'=' * 80}")
        
        train_loss, train_mae = train_one_epoch(
            train_loader, model, criterion, optimizer, device, epoch
        )
        
        val_loss, val_mae = validate(val_loader, model, criterion, device)
        
        print(f"\n{'─' * 80}")
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} Summary:")
        print(f"  Training   → Loss: {train_loss:.4f}  |  MAE: {train_mae:.4f}")
        print(f"  Validation → Loss: {val_loss:.4f}  |  MAE: {val_mae:.4f}")
        print(f"{'─' * 80}")
        
        train_history.append({'epoch': epoch, 'loss': train_loss, 'mae': train_mae})
        val_history.append({'epoch': epoch, 'loss': val_loss, 'mae': val_mae})
        
        scheduler.step(val_mae)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {current_lr:.2e}")
        
        is_best = val_mae < best_metric
        if is_best:
            best_metric = val_mae
            print(f"✓ New best model! Validation MAE: {best_metric:.4f}")
        
        state = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'loss': val_loss,
            'best_metric': best_metric,
            'train_mae': train_mae,
            'val_mae': val_mae,
            't_rng_st': torch.get_rng_state(),
            'n_rng_st': np.random.get_state(),
            'cuda_rng_st': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        }
        save_checkpoint(state, is_best, OUTPUT_DIR, model_name=model_name)
        
        early_stopping(val_mae)
        if early_stopping.early_stop:
            print("\n=======> Early stopping triggered!")
            break
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Final evaluation
    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)
    
    best_model_path = os.path.join(OUTPUT_DIR, f'{model_name}_best_model_with_metadata.pkl')
    if os.path.exists(best_model_path):
        print(f"\nLoading best model from {best_model_path}")
        with open(best_model_path, 'rb') as f:
            checkpoint = pickle.load(f)
        model.load_state_dict(checkpoint['model_state'])
        print(f"Best model loaded with validation MAE: {checkpoint.get('val_mae', 'N/A'):.4f}")
    
    print("\nEvaluating on TRAINING set...")
    train_loss_final, train_mae_final = validate(train_loader, model, criterion, device)
    
    print("\nEvaluating on VALIDATION set...")
    val_loss_final, val_mae_final = validate(val_loader, model, criterion, device)
    
    # Results summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nFinal Training MAE:     {train_mae_final:.4f}")
    print(f"Final Validation MAE:   {val_mae_final:.4f}")
    print(f"\nBest Validation MAE:    {best_metric:.4f}")
    print(f"\nModel saved to:         {OUTPUT_DIR}")
    
    # Save training history
    history_df = pd.DataFrame({
        'epoch': [h['epoch'] for h in train_history],
        'train_loss': [h['loss'] for h in train_history],
        'train_mae': [h['mae'] for h in train_history],
        'val_loss': [h['loss'] for h in val_history],
        'val_mae': [h['mae'] for h in val_history]
    })
    history_path = os.path.join(OUTPUT_DIR, 'training_history.csv')
    history_df.to_csv(history_path, index=False)
    print(f"Training history saved: {history_path}")
    
    # Save final results
    results = {
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'train_mae': train_mae_final,
        'val_mae': val_mae_final,
        'best_val_mae': best_metric,
        'num_epochs': len(train_history),
        'hyperparameters': {
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY
        }
    }
    
    results_path = os.path.join(OUTPUT_DIR, 'results.txt')
    with open(results_path, 'w') as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    print(f"Results saved to:       {results_path}")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
