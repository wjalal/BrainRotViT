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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
import sys
import pickle
import datetime
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# ===========================
# 3D ViT Architecture
# ===========================

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class SelfAttention(nn.Module):
    def __init__(self, dim, heads=6, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
    
    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        attn = self.attend(dots)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, attn_layers, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(attn_layers):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, SelfAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, hidden_dim=mlp_dim, dropout=dropout))
            ]))
    
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT3D(nn.Module):
    def __init__(self, image_size, image_patch_size, depth, depth_patch_size, dim, channels=1,
                 dim_mlp=128, attn_layers=6, n_heads=8, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)
        
        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'
        assert depth % depth_patch_size == 0, \
            'Depth must be divisible by frame patch size'
        
        num_patches = (image_height // patch_height) * (image_width // patch_width) * (depth // depth_patch_size)
        patch_dim = channels * patch_height * patch_width * depth_patch_size
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (d pd) (h p1) (w p2) -> b (d h w) (p1 p2 pd c)', p1=patch_height, p2=patch_width,
                      pd=depth_patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        self.transformer = Transformer(dim, attn_layers, n_heads, dim_head, dim_mlp, dropout)
        
        self.to_latent = nn.Identity()
        
        # Regression head for age prediction
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1)
        )
    
    def forward(self, x):

        if x.ndim == 4:
            x = x.unsqueeze(1)  # Add channel dimension
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape
        
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)
        
        x = self.transformer(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        x = self.to_latent(x)
        return self.mlp_head(x)

# ===========================
# Training Utilities
# ===========================

class EarlyStopping:
    def __init__(self, patience=20, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
    
    def __call__(self, val_loss):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def save_checkpoint(state, is_best, output_dir, model_name='model'):
    """Save model checkpoint"""
    last_path = os.path.join(output_dir, f'{model_name}_last_model_with_metadata.pkl')
    with open(last_path, 'wb') as f:
        pickle.dump(state, f)
    
    if is_best:
        best_path = os.path.join(output_dir, f'{model_name}_best_model_with_metadata.pkl')
        with open(best_path, 'wb') as f:
            pickle.dump(state, f)

def load_checkpoint(model, optimizer, output_dir, model_name, load_saved):
    """Load model checkpoint"""
    start_epoch = 0
    best_metric = float('inf')
    
    if load_saved != 'none':
        checkpoint_name = f'{model_name}_{load_saved}_model_with_metadata.pkl'
        checkpoint_path = os.path.join(output_dir, checkpoint_name)
        
        if os.path.exists(checkpoint_path):
            print(f"\nLoading checkpoint: {checkpoint_path}")
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
            
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            start_epoch = checkpoint['epoch'] + 1
            best_metric = checkpoint.get('best_metric', float('inf'))
            
            if 't_rng_st' in checkpoint:
                torch.set_rng_state(checkpoint['t_rng_st'])
            if 'n_rng_st' in checkpoint:
                np.random.set_state(checkpoint['n_rng_st'])
            if 'cuda_rng_st' in checkpoint and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(checkpoint['cuda_rng_st'])
            
            print(f"Resumed from epoch {start_epoch}, best metric: {best_metric:.4f}")
        else:
            print(f"Checkpoint not found: {checkpoint_path}")
            print("Starting from scratch")
    
    return start_epoch, best_metric

from tqdm import tqdm
import torch

def train_one_epoch(train_loader, model, criterion, optimizer, device, epoch):
    """Train for one epoch with tqdm progress bar"""
    model.train()
    running_loss = 0.0
    running_mae = 0.0

    # Wrap the dataloader with tqdm
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch} [Train]", ncols=100)
    
    for i, (images, sex, ages) in progress_bar:
        images = images.to(device, non_blocking=True)
        ages = ages.to(device, non_blocking=True).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, ages)

        loss.backward()
        optimizer.step()

        mae = torch.abs(outputs - ages).mean().item()

        running_loss += loss.item()
        running_mae += mae

        # Update progress bar dynamically
        progress_bar.set_postfix({
            "Loss": f"{loss.item():.4f}",
            "MAE": f"{mae:.4f}"
        })

    epoch_loss = running_loss / len(train_loader)
    epoch_mae = running_mae / len(train_loader)

    return epoch_loss, epoch_mae


def validate(val_loader, model, criterion, device, epoch=None):
    """Validate the model with tqdm progress bar"""
    model.eval()
    running_loss = 0.0
    running_mae = 0.0

    desc = f"Epoch {epoch} [Val]" if epoch is not None else "[Val]"
    progress_bar = tqdm(val_loader, total=len(val_loader), desc=desc, ncols=100)

    with torch.no_grad():
        for images, sex, ages in progress_bar:
            images = images.to(device, non_blocking=True)
            ages = ages.to(device, non_blocking=True).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, ages)
            mae = torch.abs(outputs - ages).mean().item()

            running_loss += loss.item()
            running_mae += mae

            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "MAE": f"{mae:.4f}"
            })

    epoch_loss = running_loss / len(val_loader)
    epoch_mae = running_mae / len(val_loader)

    return epoch_loss, epoch_mae

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


# ===========================
# Main Pipeline
# ===========================

def main():
    """Main training and evaluation pipeline"""
    # Parse command-line arguments
    model_name = sys.argv[1] if len(sys.argv) > 1 else '3dvit'
    load_saved = sys.argv[2] if len(sys.argv) > 2 else 'none'
    num_epochs_arg = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    
    if load_saved not in ['none', 'last', 'best']:
        print(f"Error: load_saved must be 'none', 'last', or 'best'. Got '{load_saved}'")
        sys.exit(1)
    
    # Configuration
    print("\n" + "=" * 80)
    print(" 3D Vision Transformer: Brain Age Estimation Pipeline")
    print("=" * 80)
    
    # Hyperparameters (optimized for RTX 4090)
    BATCH_SIZE = 4  # Reduced for ViT memory requirements
    NUM_EPOCHS = num_epochs_arg
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 8
    OUTPUT_DIR = f'model_dumps/vit/{model_name}/'
    
    # ViT Architecture parameters
    IMAGE_SIZE = 192  # H and W
    DEPTH = 160       # D
    PATCH_SIZE = 16   # Spatial patch size
    DEPTH_PATCH_SIZE = 16  # Temporal/depth patch size
    DIM = 512         # Embedding dimension
    DIM_MLP = 2048    # MLP hidden dimension
    ATTN_LAYERS = 8   # Number of transformer layers
    N_HEADS = 8       # Number of attention heads
    DIM_HEAD = 64     # Dimension per head
    DROPOUT = 0.1
    EMB_DROPOUT = 0.1
    
    print("\nConfiguration (optimized for RTX 4090 24GB):")
    print(f"  Model name:       {model_name}")
    print(f"  Architecture:     3D Vision Transformer")
    print(f"  Load saved:       {load_saved}")
    print(f"  Batch size:       {BATCH_SIZE}")
    print(f"  Learning rate:    {LEARNING_RATE}")
    print(f"  Weight decay:     {WEIGHT_DECAY}")
    print(f"  Num workers:      {NUM_WORKERS}")
    print(f"  Epochs:           {NUM_EPOCHS}")
    print(f"  Output dir:       {OUTPUT_DIR}")
    print(f"\nViT Architecture:")
    print(f"  Image size:       {IMAGE_SIZE}x{IMAGE_SIZE}x{DEPTH}")
    print(f"  Patch size:       {PATCH_SIZE}x{PATCH_SIZE}x{DEPTH_PATCH_SIZE}")
    print(f"  Embedding dim:    {DIM}")
    print(f"  Transformer layers: {ATTN_LAYERS}")
    print(f"  Attention heads:  {N_HEADS}")
    print(f"  MLP hidden dim:   {DIM_MLP}")
    print(f"  Dropout:          {DROPOUT}")
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"  Device:           {device}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"  GPU Memory:       {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    
    train_df, val_df = load_datasets()
    
    print("\nCreating datasets...")
    train_dataset = BrainAgeDataset(train_df, target_shape=(DEPTH, IMAGE_SIZE, IMAGE_SIZE))
    val_dataset = BrainAgeDataset(val_df, target_shape=(DEPTH, IMAGE_SIZE, IMAGE_SIZE))
    
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
    
    # Create 3D ViT model
    model = ViT3D(
        image_size=IMAGE_SIZE,
        image_patch_size=PATCH_SIZE,
        depth=DEPTH,
        depth_patch_size=DEPTH_PATCH_SIZE,
        dim=DIM,
        channels=1,  # Single channel for MRI
        dim_mlp=DIM_MLP,
        attn_layers=ATTN_LAYERS,
        n_heads=N_HEADS,
        dim_head=DIM_HEAD,
        dropout=DROPOUT,
        emb_dropout=EMB_DROPOUT
    )
    
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
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2
    )
    early_stopping = EarlyStopping(patience=20, verbose=True)
    
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
        
        scheduler.step()
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
            'weight_decay': WEIGHT_DECAY,
            'image_size': IMAGE_SIZE,
            'depth': DEPTH,
            'patch_size': PATCH_SIZE,
            'depth_patch_size': DEPTH_PATCH_SIZE,
            'dim': DIM,
            'attn_layers': ATTN_LAYERS,
            'n_heads': N_HEADS
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
