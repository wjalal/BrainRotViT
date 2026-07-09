"""
Triamese Training and Evaluation Pipeline
==========================================
This script combines data loading, preprocessing, model training, and evaluation
for the MultiViewViT brain age estimation model.

The pipeline:
1. Loads 3D NIfTI medical images (.nii or .nii.gz)
2. Preprocesses and extracts three orthogonal 2D planes (axial, sagittal, coronal)
3. Uses the same train/val split as in 3dmap_grad_vit_cnn_main_mix_roi.py (80/20 split)
4. Trains the MultiViewViT model on the training set
5. Evaluates on the validation set
6. Reports Mean Absolute Error (MAE) for both training and validation sets

CHECKPOINT FEATURES:

- {model_name}_last_model_with_metadata.pkl: Saved after every epoch
- {model_name}_best_model_with_metadata.pkl: Saved only when validation MAE improves

USAGE:
------
python triamese.py [model_name] [load_saved] [num_epochs]

Arguments:
    model_name  : Name for the model (default: 'triamese')
    load_saved  : Which checkpoint to load - 'none', 'last', or 'best' (default: 'none')
    num_epochs  : Number of epochs to train (default: 100)

Examples:
    python triamese.py                          # Train from scratch for 100 epochs
    python triamese.py my_model none 50         # Train new model for 50 epochs
    python triamese.py my_model last 100        # Resume from last checkpoint for 100 epochs
    python triamese.py my_model best 100        # Resume from best checkpoint for 100 epochs
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
def set_random_seed(seed=69420):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

UNIVERSAL_SEED = 69420
set_random_seed(UNIVERSAL_SEED)

# ============================================================================
# DATA PREPROCESSING FUNCTIONS (from load_data.py)
# ============================================================================
def white0(image, threshold=0):
    """
    Standardize voxels with value > threshold
    
    Args:
        image: Input image array
        threshold: Threshold value for normalization
        
    Returns:
        Normalized image
    """
    image = image.astype(np.float32)
    mask = (image > threshold).astype(int)
    image_h = image * mask
    
    non_zero_voxels = np.sum(mask)
    if non_zero_voxels > 0:
        mean = np.sum(image_h) / non_zero_voxels
        std_sum = np.sum((image_h - mean * mask) ** 2)
        std = np.sqrt(std_sum / non_zero_voxels)
        
        if std > 1e-8:  # Add small epsilon to prevent division by zero
            normalized = mask * (image - mean) / (std + 1e-8)
            image = normalized + image * (1 - mask)
            # Clip extreme values to prevent instability
            image = np.clip(image, -10.0, 10.0)
            return image
    
    return np.zeros_like(image, dtype=np.float32)


def resample_nifti(img_data, target_shape=(160, 192, 160)):
    """
    Resample NIfTI volume to target shape
    
    Args:
        img_data: 3D numpy array
        target_shape: Target shape (D, H, W) for all three dimensions
        
    Returns:
        Resampled 3D array
    """
    current_shape = img_data.shape
    zoom_factors = [target_shape[i] / current_shape[i] for i in range(3)]
    resampled_data = zoom(img_data, zoom_factors, order=3)
    return resampled_data


def calculate_bounding_box_from_volume(volume, intensity_threshold=0.1):
    """
    Calculate bounding box from volume using connected components
    
    Args:
        volume: 3D numpy array
        intensity_threshold: Threshold for brain mask
        
    Returns:
        min_indices, max_indices: Bounding box coordinates
    """
    # Normalize the volume
    volume_normalized = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
    
    # Apply intensity threshold
    binary_mask = volume_normalized > intensity_threshold
    
    # Label connected components
    labeled_array, num_features = label(binary_mask)
    
    # Find the largest connected component
    component_sizes = np.bincount(labeled_array.ravel())
    component_sizes[0] = 0  # Exclude background
    largest_component = np.argmax(component_sizes)
    
    # Create a mask for the largest component
    brain_mask = labeled_array == largest_component
    
    # Find the bounding box of the largest component
    slices = find_objects(brain_mask.astype(int))[0]
    min_indices = [s.start for s in slices]
    max_indices = [s.stop - 1 for s in slices]
    
    return min_indices, max_indices


def crop_brain_volumes(brain_data):
    """
    Crop brain volume to bounding box
    
    Args:
        brain_data: 3D numpy array
        
    Returns:
        Cropped brain volume
    """
    min_indices, max_indices = calculate_bounding_box_from_volume(brain_data)
    cropped_brain = brain_data[min_indices[0]:max_indices[0] + 1,
                                min_indices[1]:max_indices[1] + 1,
                                min_indices[2]:max_indices[2] + 1]
    return cropped_brain


# ============================================================================
# MODEL ARCHITECTURE (from MultiViewVit.py and model/vit.py)
# ============================================================================
class PositionEmbs(nn.Module):
    """Positional Embeddings with dropout"""
    def __init__(self, num_patches, emb_dim, dropout_rate=0.1):
        super(PositionEmbs, self).__init__()
        # Use smaller initialization to prevent instability
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, emb_dim) * 0.02)
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def forward(self, x):
        out = x + self.pos_embedding
        if self.dropout:
            out = self.dropout(out)
        return out


class MlpBlock(nn.Module):
    """Transformer Feed-Forward Block"""
    def __init__(self, in_dim, mlp_dim, out_dim, dropout_rate=0.1):
        super(MlpBlock, self).__init__()
        self.fc1 = nn.Linear(in_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, out_dim)
        self.act = nn.GELU()
        if dropout_rate > 0.0:
            self.dropout1 = nn.Dropout(dropout_rate)
            self.dropout2 = nn.Dropout(dropout_rate)
        else:
            self.dropout1 = None
            self.dropout2 = None

    def forward(self, x):
        out = self.fc1(x)
        out = self.act(out)
        if self.dropout1:
            out = self.dropout1(out)
        out = self.fc2(out)
        if self.dropout2:
            out = self.dropout2(out)
        return out


class LinearGeneral(nn.Module):
    """General linear layer with configurable dimensions"""
    def __init__(self, in_dim=(768,), feat_dim=(12, 64)):
        super(LinearGeneral, self).__init__()
        # Use Xavier initialization for better stability
        self.weight = nn.Parameter(torch.randn(*in_dim, *feat_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(*feat_dim))

    def forward(self, x, dims):
        a = torch.tensordot(x, self.weight, dims=dims) + self.bias
        return a


class SelfAttention(nn.Module):
    """Multi-head Self Attention"""
    def __init__(self, in_dim, heads=8, dropout_rate=0.1):
        super(SelfAttention, self).__init__()
        self.heads = heads
        self.head_dim = in_dim // heads
        self.scale = self.head_dim ** 0.5

        self.query = LinearGeneral((in_dim,), (self.heads, self.head_dim))
        self.key = LinearGeneral((in_dim,), (self.heads, self.head_dim))
        self.value = LinearGeneral((in_dim,), (self.heads, self.head_dim))
        self.out = LinearGeneral((self.heads, self.head_dim), (in_dim,))

        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def forward(self, x, return_attention_weights=True):
        b, n, _ = x.shape

        q = self.query(x, dims=([2], [0]))
        k = self.key(x, dims=([2], [0]))
        v = self.value(x, dims=([2], [0]))

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_weights = torch.softmax(attn_weights, dim=-1)
        out = torch.matmul(attn_weights, v)
        out = out.permute(0, 2, 1, 3)

        out = self.out(out, dims=([2, 3], [0, 1]))

        if return_attention_weights:
            return out, attn_weights
        else:
            return out


class EncoderBlock(nn.Module):
    """Transformer Encoder Block with Pre-LayerNorm"""
    def __init__(self, in_dim, mlp_dim, num_heads, dropout_rate=0.1, attn_dropout_rate=0.1):
        super(EncoderBlock, self).__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = SelfAttention(in_dim, heads=num_heads, dropout_rate=attn_dropout_rate)
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None
        self.norm2 = nn.LayerNorm(in_dim)
        self.mlp = MlpBlock(in_dim, mlp_dim, in_dim, dropout_rate)

    def forward(self, x, return_attention_weights=True):
        residual = x
        out = self.norm1(x)
        out, attn_weights = self.attn(out, return_attention_weights=return_attention_weights)
        if self.dropout:
            out = self.dropout(out)
        out += residual
        residual = out

        out = self.norm2(out)
        out = self.mlp(out)
        out += residual
        
        if return_attention_weights:
            return out, attn_weights
        else:
            return out


class Encoder(nn.Module):
    """Transformer Encoder with multiple layers"""
    def __init__(self, num_patches, emb_dim, mlp_dim, num_layers=10, num_heads=12, 
                 dropout_rate=0.1, attn_dropout_rate=0.0):
        super(Encoder, self).__init__()
        # Positional embedding
        self.pos_embedding = PositionEmbs(num_patches, emb_dim, dropout_rate)

        # Encoder blocks
        in_dim = emb_dim
        self.encoder_layers = nn.ModuleList()
        for i in range(num_layers):
            layer = EncoderBlock(in_dim, mlp_dim, num_heads, dropout_rate, attn_dropout_rate)
            self.encoder_layers.append(layer)
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, x, return_attention_weights=True):
        out = self.pos_embedding(x)
        attention_weights = []

        for layer in self.encoder_layers:
            if return_attention_weights:
                out, attn_weights = layer(out, return_attention_weights=True)
                attention_weights.append(attn_weights)
            else:
                out = layer(out, return_attention_weights=False)

        out = self.norm(out)
        
        if return_attention_weights:
            return out, attention_weights
        else:
            return out


class VisionTransformer(nn.Module):
    """Vision Transformer - Original Implementation"""
    def __init__(self,
                 image_size=(109, 91),
                 patch_size=(7, 7),
                 emb_dim=768,
                 mlp_dim=3072,
                 num_heads=12,
                 num_layers=10,
                 num_classes=1,
                 attn_dropout_rate=0.0,
                 dropout_rate=0.1,
                 num_channals=91,
                 feat_dim=None):
        super(VisionTransformer, self).__init__()
        h, w = image_size

        # Embedding layer
        fh, fw = patch_size
        gh, gw = h // fh, w // fw
        num_patches = gh * gw
        self.embedding = nn.Conv2d(num_channals, emb_dim, kernel_size=(fh, fw), stride=(fh, fw))
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))

        # Transformer
        self.transformer = Encoder(
            num_patches=num_patches,
            emb_dim=emb_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            attn_dropout_rate=attn_dropout_rate)

        # No classifier here - output embeddings for MultiViewViT to process
        # self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, x, return_attention_weights=True):
        emb = self.embedding(x)  # (n, c, gh, gw)
        emb = emb.permute(0, 2, 3, 1)  # (n, gh, hw, c)
        b, h, w, c = emb.shape
        emb = emb.reshape(b, h * w, c)

        # Prepend class token
        cls_token = self.cls_token.repeat(b, 1, 1)
        emb = torch.cat([cls_token, emb], dim=1)

        # Transformer
        if return_attention_weights:
            feat, attention_weights = self.transformer(emb, return_attention_weights=True)
        else:
            feat = self.transformer(emb, return_attention_weights=False)

        # Output CLS token embeddings (not classified)
        cls_output = feat[:, 0]
        
        if return_attention_weights:
            return cls_output, attention_weights
        else:
            return cls_output


class MultiViewViT(nn.Module):
    """
    Multi-view Vision Transformer for brain age estimation
    Processes three orthogonal views: axial, sagittal, coronal
    """
    def __init__(self, image_sizes, patch_sizes, num_channals, vit_args, mlp_dims):
        """
        Args:
            image_sizes: List of sizes for each of the 3 views e.g. [(91, 109), (91, 91), (109, 91)]
            patch_sizes: List of patch sizes for each of the 3 views e.g. [(7, 7), (7, 7), (7, 7)]
            num_channals: List of number of channels for each view e.g. [91, 109, 91]
            vit_args: Dictionary containing ViT arguments (emb_dim, mlp_dim, num_heads, etc.)
            mlp_dims: List of dimensions for the MLP layers e.g. [3, 128, 256, 512, 1024, 512, 256, 128, 1]
        """
        super(MultiViewViT, self).__init__()

        # Creating 3 ViT models for each view
        self.vit_1 = VisionTransformer(
            image_size=image_sizes[0], 
            num_channals=num_channals[0], 
            patch_size=patch_sizes[0],
            emb_dim=vit_args['emb_dim'],
            mlp_dim=vit_args['mlp_dim'],
            num_heads=vit_args['num_heads'],
            num_layers=vit_args['num_layers'],
            num_classes=vit_args['num_classes'],
            dropout_rate=vit_args['dropout_rate'],
            attn_dropout_rate=vit_args['attn_dropout_rate']
        )
        self.vit_2 = VisionTransformer(
            image_size=image_sizes[1], 
            num_channals=num_channals[1], 
            patch_size=patch_sizes[1],
            emb_dim=vit_args['emb_dim'],
            mlp_dim=vit_args['mlp_dim'],
            num_heads=vit_args['num_heads'],
            num_layers=vit_args['num_layers'],
            num_classes=vit_args['num_classes'],
            dropout_rate=vit_args['dropout_rate'],
            attn_dropout_rate=vit_args['attn_dropout_rate']
        )
        self.vit_3 = VisionTransformer(
            image_size=image_sizes[2], 
            num_channals=num_channals[2], 
            patch_size=patch_sizes[2],
            emb_dim=vit_args['emb_dim'],
            mlp_dim=vit_args['mlp_dim'],
            num_heads=vit_args['num_heads'],
            num_layers=vit_args['num_layers'],
            num_classes=vit_args['num_classes'],
            dropout_rate=vit_args['dropout_rate'],
            attn_dropout_rate=vit_args['attn_dropout_rate']
        )

        # MLP for final prediction after concatenating outputs of three ViTs
        layers = []
        for i in range(len(mlp_dims) - 1):
            layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
            if i < len(mlp_dims) - 2:  # No activation after last layer
                layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x, return_attention_weights=True):
        """
        Forward pass through multi-view ViT
        
        Args:
            x: Input tensor [B, D1, D2, D3] - 3D volume
            return_attention_weights: Whether to return attention weights
            
        Returns:
            prediction: Age prediction
            (attn1, attn2, attn3): Attention weights from three views (if requested)
        """
        # Extract three orthogonal views
        x1 = x.permute(0, 3, 1, 2)  # Sagittal view
        x2 = x.permute(0, 2, 1, 3)  # Coronal view
        x3 = x                       # Axial view
        
        if return_attention_weights:
            out1, attn1 = self.vit_1(x1, return_attention_weights=True)
            out2, attn2 = self.vit_2(x2, return_attention_weights=True)
            out3, attn3 = self.vit_3(x3, return_attention_weights=True)
        else:
            out1 = self.vit_1(x1)
            out2 = self.vit_2(x2)
            out3 = self.vit_3(x3)

        # Concatenate the outputs
        combined_out = torch.cat([out1, out2, out3], dim=1)

        # Pass through MLP
        prediction = self.mlp(combined_out)

        if return_attention_weights:
            return prediction, (attn1, attn2, attn3)
        else:
            return prediction


# ============================================================================
# DATASET CLASS
# ============================================================================
class BrainAgeDataset(Dataset):
    """
    Dataset class for loading brain MRI images and extracting 3D volumes
    """
    def __init__(self, dataframe, target_shape=(160, 192, 160)):
        """
        Args:
            dataframe: DataFrame with columns ['filepath', 'Age', 'Sex']
            target_shape: Target shape (D, H, W) for resampling
        """
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
        
        # Load NIfTI file
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
            
            # Check for NaN or Inf in preprocessed data
            if np.isnan(data).any() or np.isinf(data).any():
                print(f"Warning: NaN/Inf detected in preprocessed data for {filepath}, replacing with zeros")
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
            # Return dummy data in case of error with correct shape
            dummy_data = torch.zeros(self.target_shape, dtype=torch.float32)
            return dummy_data, torch.tensor(0.0), torch.tensor(0.0)


# ============================================================================
# TRAINING UTILITIES (from Training.py)
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
    """Early stops the training if validation loss doesn't improve after a given patience."""
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


def weights_init(w):
    """Initialize weights for the model"""
    classname = w.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(w, 'weight'):
            nn.init.kaiming_normal_(w.weight, mode='fan_in', nonlinearity='leaky_relu')
        if hasattr(w, 'bias') and w.bias is not None:
            nn.init.constant_(w.bias, 0)
    if classname.find('Linear') != -1:
        if hasattr(w, 'weight'):
            torch.nn.init.xavier_normal_(w.weight)
        if hasattr(w, 'bias') and w.bias is not None:
            nn.init.constant_(w.bias, 0)
    if classname.find('BatchNorm') != -1:
        if hasattr(w, 'weight') and w.weight is not None:
            nn.init.constant_(w.weight, 1)
        if hasattr(w, 'bias') and w.bias is not None:
            nn.init.constant_(w.bias, 0)


def metric(output, target):
    """Calculate MAE metric with NaN handling"""
    target_np = target.cpu().data.numpy()
    pred_np = output.cpu().data.numpy()
    
    # Check for NaN or Inf values
    if np.isnan(pred_np).any() or np.isinf(pred_np).any():
        print(f"Warning: NaN or Inf detected in predictions!")
        return float('inf')
    if np.isnan(target_np).any() or np.isinf(target_np).any():
        print(f"Warning: NaN or Inf detected in targets!")
        return float('inf')
    
    mae = mean_absolute_error(target_np, pred_np)
    return mae


def save_checkpoint(state, is_best, out_dir, model_name='triamese'):
    """
    Save model checkpoint with last and best versions
    
    Args:
        state: Dictionary containing model state, optimizer state, etc.
        is_best: Whether this is the best model so far
        out_dir: Output directory
        model_name: Name of the model
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Save last model with metadata (using pickle for consistency with reference implementation)
    last_model_path = os.path.join(out_dir, f'{model_name}_last_model_with_metadata.pkl')
    with open(last_model_path, 'wb') as f:
        pickle.dump(state, f)
    
    # Save best model if this is the best one
    if is_best:
        best_model_path = os.path.join(out_dir, f'{model_name}_best_model_with_metadata.pkl')
        with open(best_model_path, 'wb') as f:
            pickle.dump(state, f)
        print("=======> This is the best model! It has been saved!\n")


def load_checkpoint(model, optimizer, out_dir, model_name='triamese', load_type='last'):
    """
    Load checkpoint and resume training
    
    Args:
        model: The model to load weights into
        optimizer: The optimizer to load state into
        out_dir: Output directory where checkpoints are saved
        model_name: Name of the model
        load_type: Which checkpoint to load - 'best', 'last', or 'none'
        
    Returns:
        start_epoch: Epoch to resume from
        best_loss: Best validation loss so far
    """
    if load_type == 'none':
        return 0, float('inf')
    
    # First, try to load the best model to get best_loss
    best_model_path = os.path.join(out_dir, f'{model_name}_best_model_with_metadata.pkl')
    best_loss = float('inf')
    
    if os.path.exists(best_model_path):
        with open(best_model_path, 'rb') as f:
            best_checkpoint = pickle.load(f)
        best_loss = best_checkpoint.get('loss', float('inf'))
        print(f"Best model found with validation loss: {best_loss:.4f}")
    
    # Now load the requested checkpoint (best or last)
    checkpoint_path = os.path.join(out_dir, f'{model_name}_{load_type}_model_with_metadata.pkl')
    
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}. Starting from scratch.")
        return 0, best_loss
    
    print(f"\nLoading checkpoint from {checkpoint_path}")
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    # Restore model and optimizer state
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    
    # Restore RNG states for reproducibility
    if 't_rng_st' in checkpoint:
        torch.set_rng_state(checkpoint['t_rng_st'])
    if 'n_rng_st' in checkpoint:
        np.random.set_state(checkpoint['n_rng_st'])
    if 'cuda_rng_st' in checkpoint and torch.cuda.is_available() and checkpoint['cuda_rng_st'] is not None:
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng_st'])
    
    # Retrieve metadata
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
def train_one_epoch(train_loader, model, criterion, optimizer, device, epoch, print_freq=10, 
                    accumulation_steps=4):
    """
    Train for one epoch with gradient accumulation
    
    Args:
        train_loader: Training data loader
        model: MultiViewViT model
        criterion: Loss function
        optimizer: Optimizer
        device: Device (CPU/GPU)
        epoch: Current epoch number
        print_freq: Print frequency
        accumulation_steps: Number of steps to accumulate gradients
        
    Returns:
        Average loss and MAE for the epoch
    """
    losses = AverageMeter()
    MAE = AverageMeter()
    
    model.train()
    
    # Use mixed precision training
    scaler = torch.amp.GradScaler('cuda')
    
    for i, (img, sex, target) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}")):
        # Move data to device
        img = img.to(device)
        sex = sex.to(device)
        target = target.unsqueeze(1).to(device)
        
        # Check for NaN in input data
        if torch.isnan(img).any() or torch.isinf(img).any():
            print(f"Warning: NaN/Inf in input data at batch {i}, skipping...")
            continue
        
        # Forward pass with mixed precision
        with torch.amp.autocast('cuda'):
            output, (attn1, attn2, attn3) = model(img, return_attention_weights=True)
            loss = criterion(output, target)
            loss = loss / accumulation_steps  # Normalize loss for accumulation
        
        # Check for NaN in loss
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print(f"Warning: NaN/Inf loss at batch {i}, skipping...")
            continue
        
        # Compute MAE
        mae = metric(output.detach(), target.detach())
        
        # Skip if MAE is invalid
        if mae == float('inf'):
            print(f"Warning: Invalid MAE at batch {i}, skipping...")
            continue
        
        # Update meters
        losses.update(loss.item() * accumulation_steps, img.size(0))
        MAE.update(mae, img.size(0))
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Update weights every accumulation_steps
        if (i + 1) % accumulation_steps == 0:
            # Unscale and clip gradients before stepping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Print progress
        # if i % print_freq == 0:
        #     print(f'Epoch: [{epoch}]  [step {i}/{len(train_loader)}]\t'
        #           f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
        #           f'MAE {MAE.val:.3f} ({MAE.avg:.3f})')
    
    # Final update if there are remaining gradients
    if (i + 1) % accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    return losses.avg, MAE.avg


def validate(valid_loader, model, criterion, device):
    """
    Validate the model with mixed precision
    
    Args:
        valid_loader: Validation data loader
        model: MultiViewViT model
        criterion: Loss function
        device: Device (CPU/GPU)
        
    Returns:
        Average loss and MAE for validation set
    """
    losses = AverageMeter()
    MAE = AverageMeter()
    
    model.eval()
    
    with torch.no_grad():
        for img, sex, target in tqdm(valid_loader, desc="Validation"):
            img = img.to(device)
            sex = sex.to(device)
            target = target.unsqueeze(1).to(device)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                output, (attn1, attn2, attn3) = model(img, return_attention_weights=True)
                loss = criterion(output, target)
            
            # Compute MAE
            mae = metric(output.detach(), target.detach())
            
            # Update meters
            losses.update(loss.item(), img.size(0))
            MAE.update(mae, img.size(0))
    
    print(f'Valid: [steps {len(valid_loader)}], Loss {losses.avg:.4f}, MAE: {MAE.avg:.4f}')
    
    return losses.avg, MAE.avg


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================
def load_datasets():
    """
    Load and prepare datasets using the same logic as 3dmap_grad_vit_cnn_main_mix_roi.py
    
    Returns:
        train_df, val_df: Training and validation dataframes
    """
    print("=" * 80)
    print("LOADING DATASETS")
    print("=" * 80)
    
    # Load ADNI dataset
    csv_path = "adni_storage/adni_brainrotnet_metadata.csv"
    df_adni = pd.read_csv(csv_path)
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
    df_abide = df_abide.sort_values(by='Age', ascending=False).reset_index(drop=True)
    df_abide = df_abide.head(750)
    
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
    df_oas1 = df_oas1.sort_values(by='Age', ascending=False)
    df_oas1 = df_oas1.reset_index(drop=True).head(300)
    
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
    
    # Split into train and validation (80/20 split with same seed as original)
    train_size = int(0.8 * len(df))
    val_size = len(df) - train_size
    
    # Create indices using same logic as original
    indices = list(range(len(df)))
    np.random.seed(UNIVERSAL_SEED)
    np.random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_df = df.iloc[train_indices].reset_index(drop=True)
    val_df = df.iloc[val_indices].reset_index(drop=True)
    
    print(f"\nTrain size: {len(train_df)}, Val size: {len(val_df)}")
    
    return train_df, val_df


def main():
    """
    Main training and evaluation pipeline
    
    Command-line arguments:
        sys.argv[1]: model_name (optional, default='triamese')
        sys.argv[2]: load_saved (optional, 'none', 'last', or 'best', default='none')
        sys.argv[3]: num_epochs (optional, default=100)
    
    Examples:
        python triamese.py                          # Train from scratch for 100 epochs
        python triamese.py triamese none 50         # Train from scratch for 50 epochs
        python triamese.py triamese last 100        # Resume from last checkpoint
        python triamese.py triamese best 100        # Resume from best checkpoint
    """
    # ========================================================================
    # PARSE COMMAND-LINE ARGUMENTS
    # ========================================================================
    model_name = sys.argv[1] if len(sys.argv) > 1 else 'triamese'
    load_saved = sys.argv[2] if len(sys.argv) > 2 else 'none'  # 'none', 'last', or 'best'
    num_epochs_arg = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    
    # Validate load_saved argument
    if load_saved not in ['none', 'last', 'best']:
        print(f"Error: load_saved must be 'none', 'last', or 'best'. Got '{load_saved}'")
        print("Usage: python triamese.py [model_name] [none|last|best] [num_epochs]")
        sys.exit(1)
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    print("\n" + "=" * 80)
    print("TRIAMESE: Multi-View ViT Brain Age Estimation Pipeline")
    print("=" * 80)
    
    # Hyperparameters
    BATCH_SIZE = 2  # Reduced from 8 to save memory
    NUM_EPOCHS = num_epochs_arg
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    NUM_WORKERS = 2  # Reduced from 4 to save memory
    OUTPUT_DIR = f'model_dumps/triamese/{model_name}/'
    
    print("\nConfiguration:")
    print(f"  Model name:     {model_name}")
    print(f"  Load saved:     {load_saved}")
    print(f"  Batch size:     {BATCH_SIZE} (with gradient accumulation x4 = effective batch size 8)")
    print(f"  Learning rate:  {LEARNING_RATE}")
    print(f"  Weight decay:   {WEIGHT_DECAY}")
    print(f"  Epochs:         {NUM_EPOCHS}")
    print(f"  Output dir:     {OUTPUT_DIR}")
    
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"  Device:         {device}")
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"  GPU Memory:     {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # ========================================================================
    # LOAD DATA
    # ========================================================================
    train_df, val_df = load_datasets()
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = BrainAgeDataset(train_df, target_shape=(160, 192, 160))
    val_dataset = BrainAgeDataset(val_df, target_shape=(160, 192, 160))
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        shuffle=False
    )
    
    print("\n" + "=" * 80)
    print("BUILDING MODEL")
    print("=" * 80)
    
    # With target_shape=(160, 192, 160) as (D, H, W):
    # x1 = x.permute(0, 3, 1, 2) → [B, W, D, H] = [B, 160, 160, 192]
    # x2 = x.permute(0, 2, 1, 3) → [B, H, D, W] = [B, 192, 160, 160]
    # x3 = x → [B, D, H, W] = [B, 160, 192, 160]
    
    model = MultiViewViT(
        image_sizes=[(160, 192), (160, 160), (192, 160)],  # (H, W) for each view
        patch_sizes=[(8, 8), (8, 8), (8, 8)],  # Larger patches = fewer patches = less memory
        num_channals=[160, 192, 160],  # Number of "slices" in each view
        vit_args={
            'emb_dim': 768,  # Reduced from 768 to save memory
            'mlp_dim': 3072,  # Reduced from 3072 to save memory
            'num_heads': 12,  # Reduced from 12 to save memory
            'num_layers': 12,  # Reduced from 12 to save memory
            'num_classes': 1,
            'dropout_rate': 0.1, 
            'attn_dropout_rate': 0.0
        },
        mlp_dims=[768 * 3, 128, 256, 512, 1024, 512, 256, 128, 1]  # MLP layers
    )
    
    # Initialize weights
    model.apply(weights_init)
    
    # Move to device (use DataParallel if multiple GPUs available)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total:      {total_params:,}")
    print(f"  Trainable:  {trainable_params:,}")
    
    # ========================================================================
    # DEFINE LOSS, OPTIMIZER, SCHEDULER
    # ========================================================================
    criterion = nn.L1Loss().to(device)  # MAE loss
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
    
    # ========================================================================
    # LOAD CHECKPOINT (if specified)
    # ========================================================================
    start_epoch, best_metric = load_checkpoint(
        model, optimizer, OUTPUT_DIR, model_name, load_saved
    )
    
    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    print("\n" + "=" * 80)
    print("START TRAINING")
    print("=" * 80)
    
    train_history = []
    val_history = []
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\n{'=' * 80}")
        print(f"EPOCH {epoch + 1}/{NUM_EPOCHS}")
        print(f"{'=' * 80}")
        
        # Train for one epoch
        train_loss, train_mae = train_one_epoch(
            train_loader, model, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_mae = validate(val_loader, model, criterion, device)
        
        # Report metrics for this epoch
        print(f"\n{'─' * 80}")
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} Summary:")
        print(f"  Training   → Loss: {train_loss:.4f}  |  MAE: {train_mae:.4f}")
        print(f"  Validation → Loss: {val_loss:.4f}  |  MAE: {val_mae:.4f}")
        print(f"{'─' * 80}")
        
        # Store history
        train_history.append({'epoch': epoch, 'loss': train_loss, 'mae': train_mae})
        val_history.append({'epoch': epoch, 'loss': val_loss, 'mae': val_mae})
        
        # Learning rate scheduling
        scheduler.step(val_mae)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {current_lr:.2e}")
        
        # Check if best model
        is_best = val_mae < best_metric
        if is_best:
            best_metric = val_mae
            print(f"✓ New best model! Validation MAE: {best_metric:.4f}")
        
        # Save checkpoint with RNG states for reproducibility
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
        
        # Early stopping
        early_stopping(val_mae)
        if early_stopping.early_stop:
            print("\n=======> Early stopping triggered!")
            break
        
        # Clear CUDA cache after each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # ========================================================================
    # FINAL EVALUATION
    # ========================================================================
    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)
    
    # Load best model
    best_model_path = os.path.join(OUTPUT_DIR, f'{model_name}_best_model_with_metadata.pkl')
    if os.path.exists(best_model_path):
        print(f"\nLoading best model from {best_model_path}")
        with open(best_model_path, 'rb') as f:
            checkpoint = pickle.load(f)
        model.load_state_dict(checkpoint['model_state'])
        print(f"Best model loaded with validation MAE: {checkpoint.get('val_mae', 'N/A'):.4f}")
    else:
        print(f"\nNo best model found at {best_model_path}. Using current model state.")
    
    # Evaluate on training set
    print("\nEvaluating on TRAINING set...")
    train_loss_final, train_mae_final = validate(train_loader, model, criterion, device)
    
    # Evaluate on validation set
    print("\nEvaluating on VALIDATION set...")
    val_loss_final, val_mae_final = validate(val_loader, model, criterion, device)
    
    # ========================================================================
    # RESULTS SUMMARY
    # ========================================================================
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
