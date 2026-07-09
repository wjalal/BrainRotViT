import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import nibabel as nib
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform, apply_orientation
from scipy.ndimage import label, find_objects
from scipy.ndimage import zoom
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os

# Import your model
from GlobalLocalTransformer import GlobalLocalBrainAge  # Assuming your model code is in model.py

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

class BrainAgeDataset(Dataset):
    def __init__(self, dataframe, slice_type='sagittal', transform=None):
        """
        Args:
            dataframe: pandas DataFrame with columns ['ImageID', 'Sex', 'Age', 'filepath']
            slice_type: 'sagittal', 'axial', or 'coronal'
            transform: optional transform to be applied on a sample
        """
        self.df = dataframe.reset_index(drop=True)
        self.slice_type = slice_type
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def calculate_bounding_box_from_volume(self, volume, intensity_threshold=0.1):
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
    
    def crop_brain_volumes(self, brain_data):
        # Calculate bounding box from the brain volume
        min_indices, max_indices = self.calculate_bounding_box_from_volume(brain_data)
        # Crop the volume
        cropped_brain = brain_data[min_indices[0]:max_indices[0] + 1,
                                   min_indices[1]:max_indices[1] + 1,
                                   min_indices[2]:max_indices[2] + 1]
        return cropped_brain
    
    def resize_volume(self, volume, target_shape=(130, 170, 120)):
        """Resize volume to target shape"""
        zoom_factors = [target_shape[i] / volume.shape[i] for i in range(3)]
        resized_volume = zoom(volume, zoom_factors, order=1)
        return resized_volume
    
    def extract_slice(self, volume, slice_type):
        """Extract central slice based on type"""
        if slice_type == 'sagittal':
            # Extract central sagittal slice (along x-axis)
            central_idx = volume.shape[0] // 2
            slice_2d = volume[central_idx, :, :]  # Shape: (170, 120)
        elif slice_type == 'axial':
            # Extract central axial slice (along z-axis)
            central_idx = volume.shape[2] // 2
            slice_2d = volume[:, :, central_idx]  # Shape: (130, 170)
        elif slice_type == 'coronal':
            # Extract central coronal slice (along y-axis)
            central_idx = volume.shape[1] // 2
            slice_2d = volume[:, central_idx, :]  # Shape: (130, 120)
        else:
            raise ValueError(f"Unknown slice_type: {slice_type}")
        
        return slice_2d
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = row['filepath']
        age = row['Age']
        sex = 1.0 if row['Sex'] == 'M' else 0.0
        
        # Load and process MRI
        nii_img = nib.load(filepath)
        
        # Get current orientation and reorient to RAS
        orig_ornt = io_orientation(nii_img.affine)
        ras_ornt = axcodes2ornt(("R", "A", "S"))
        ornt_trans = ornt_transform(orig_ornt, ras_ornt)
        data = nii_img.get_fdata()
        data = apply_orientation(data, ornt_trans)
        
        # Crop brain volumes
        data = self.crop_brain_volumes(data)
        
        # Resize to target shape (130, 170, 120)
        data = self.resize_volume(data, target_shape=(130, 170, 120))
        
        # Normalize
        data = (data - np.mean(data)) / (np.std(data) + 1e-8)
        # data = (data - data.min()) / (data.max() - data.min() + 0.000000001)
        
        # Extract central slice
        slice_2d = self.extract_slice(data, self.slice_type)
        
        # Add channel dimension and convert to tensor
        # For model input, we need (C, H, W) - using 1 channel (grayscale)
        slice_tensor = torch.from_numpy(slice_2d).float().unsqueeze(0)
        
        # Repeat channel to match model expectation (model expects 5 channels in example)
        # We'll use 1 channel for single modality MRI
        
        age_tensor = torch.tensor(age, dtype=torch.float32)
        sex_tensor = torch.tensor(sex, dtype=torch.float32)
        
        return {
            'image': slice_tensor,
            'age': age_tensor,
            'sex': sex_tensor,
            'image_id': row['ImageID']
        }


class BrainAgeTrainer:
    def __init__(self, model, device, learning_rate=0.0001):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=25, gamma=0.5)
        self.criterion = nn.L1Loss()  # MAE loss
        
    def compute_loss(self, predictions_list, target):
        """
        predictions_list: list of predictions [global, local1, local2, ...]
        target: ground truth age
        """
        total_loss = 0.0
        for pred in predictions_list:
            loss = self.criterion(pred.squeeze(), target)
            total_loss += loss
        return total_loss / len(predictions_list)
    
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
        for batch in pbar:
            images = batch['image'].to(self.device)
            ages = batch['age'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions_list = self.model(images)
            
            # Compute loss
            loss = self.compute_loss(predictions_list, ages)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader):
        self.model.eval()
        total_mae = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                images = batch['image'].to(self.device)
                ages = batch['age'].to(self.device)
                
                # Forward pass
                predictions_list = self.model(images)
                
                # Average all predictions (global + local patches)
                avg_pred = torch.mean(torch.stack([p.squeeze() for p in predictions_list]), dim=0)
                
                # Calculate MAE
                mae = torch.abs(avg_pred - ages).mean()
                total_mae += mae.item()
                
                all_predictions.extend(avg_pred.cpu().numpy())
                all_targets.extend(ages.cpu().numpy())
        
        avg_mae = total_mae / len(val_loader)
        return avg_mae, np.array(all_predictions), np.array(all_targets)
    
    def train(self, train_loader, val_loader, num_epochs=80, save_path='best_model.pth'):
        best_mae = float('inf')
        
        for epoch in range(num_epochs):
            # Training
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_mae, _, _ = self.validate(val_loader)
            
            # Step scheduler
            self.scheduler.step()
            
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val MAE: {val_mae:.4f}, LR: {self.scheduler.get_last_lr()[0]:.6f}")
            
            # Save best model
            if val_mae < best_mae:
                best_mae = val_mae
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_mae': val_mae,
                }, save_path)
                print(f"  --> Saved best model with MAE: {val_mae:.4f}")
        
        return best_mae


def main(df, batch_size=4, num_workers=2):
    """
    Main training function
    
    Args:
        df: DataFrame with columns ['ImageID', 'Sex', 'Age', 'filepath']
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
    """
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Split dataset 80:20
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
    test_df = df.iloc[val_indices].reset_index(drop=True)
    
    print(f"\nTrain size: {len(train_df)}, Val size: {len(test_df)}")
    
    
    # Train three models for sagittal, axial, and coronal slices
    slice_types = ['sagittal', 'axial', 'coronal']
    trained_models = {}
    test_predictions = {}
    
    for slice_type in slice_types:
        print(f"\n{'='*60}")
        print(f"Training model for {slice_type.upper()} slices")
        print(f"{'='*60}\n")
        
        # Create datasets
        train_dataset = BrainAgeDataset(train_df, slice_type=slice_type)
        test_dataset = BrainAgeDataset(test_df, slice_type=slice_type)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                 shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                                shuffle=False, num_workers=num_workers)
        
        # Initialize model (1 channel input for single modality MRI)
        model = GlobalLocalBrainAge(
            inplace=1,  # 1 channel for grayscale MRI
            patch_size=64,
            step=32,
            nblock=6,
            drop_rate=0.5,
            backbone='vgg8'
        )
        
        # Initialize trainer
        trainer = BrainAgeTrainer(model, device, learning_rate=0.0001)
        
        # Train model
        save_path = f'best_model_{slice_type}.pth'
        best_mae = trainer.train(train_loader, test_loader, num_epochs=80, save_path=save_path)
        
        # Load best model for testing
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        trained_models[slice_type] = model
        
        # Test predictions
        print(f"\nTesting {slice_type} model...")
        test_mae, predictions, targets = trainer.validate(test_loader)
        test_predictions[slice_type] = predictions
        print(f"{slice_type.upper()} Test MAE: {test_mae:.4f}")
    
    # Calculate ensemble prediction (average of 3 models)
    print(f"\n{'='*60}")
    print("ENSEMBLE RESULTS (Averaging 3 models)")
    print(f"{'='*60}\n")
    
    ensemble_predictions = np.mean([
        test_predictions['sagittal'],
        test_predictions['axial'],
        test_predictions['coronal']
    ], axis=0)
    
    # Get ground truth from test set
    test_ages = test_df['Age'].values
    
    # Calculate final MAE
    final_mae = np.mean(np.abs(ensemble_predictions - test_ages))
    print(f"Final Ensemble MAE: {final_mae:.4f}")
    
    # Save results
    results_df = pd.DataFrame({
        'ImageID': test_df['ImageID'].values,
        'True_Age': test_ages,
        'Predicted_Age': ensemble_predictions,
        'Sagittal_Pred': test_predictions['sagittal'],
        'Axial_Pred': test_predictions['axial'],
        'Coronal_Pred': test_predictions['coronal'],
        'Absolute_Error': np.abs(ensemble_predictions - test_ages)
    })
    results_df.to_csv('test_predictions.csv', index=False)
    print("\nResults saved to 'test_predictions.csv'")
    
    return trained_models, results_df, final_mae


# Example usage:
# Assuming you have your dataframe 'df' loaded
# trained_models, results_df, final_mae = main(df, batch_size=4, num_workers=2)




# Load the CSV file into a pandas DataFrame
csv_path = "adni_storage/adni_brainrotnet_metadata.csv"
df_adni = pd.read_csv(csv_path)
# df = df.sample(n=1000, random_state=69420)
# Add a new column 'filepath' with the constructed file paths
df_adni['filepath'] = df_adni.apply(
    lambda row: f"adni_storage/ADNI_nii_gz_bias_corrected/I{row['ImageID'][4:]}_{row['SubjectID']}.stripped.N4.nii.gz",
    axis=1)
df_adni = df_adni.loc[
    df_adni.groupby('SubjectID')['Age'].apply(lambda x: (x - x.median()).abs().idxmin())
].reset_index(drop=True)
df_adni = df_adni.sort_values(by='Age', ascending=True).reset_index(drop=True).head(900)

# Load independent dataset metadata
metadata_path = "ixi_storage/ixi_brainrotnet_metadata.csv"
df_ixi = pd.read_csv(metadata_path)
# Update filepaths for the independent dataset
df_ixi['filepath'] = df_ixi.apply(
    lambda row: f"ixi_storage/IXI_bias_corrected/{row['ImageID']}.stripped.N4.nii.gz",
    axis=1
)

metadata_path = "abide_storage/abide_brainrotnet_metadata.csv"
df_abide = pd.read_csv(metadata_path)
# Update filepaths for the independent dataset
df_abide['filepath'] = df_abide.apply(
    lambda row: f"abide_storage/ABIDEII_bias_corrected/{row['ImageID'][7:]}.stripped.N4.nii.gz",
    axis=1
)
df_abide = df_abide.sort_values(by='Age', ascending=False).reset_index(drop=True)
df_abide = df_abide.head(750)
# df_abide=df_abide.sample(n=200)

metadata_path = "dlbs_storage/dlbs_brainrotnet_metadata.csv"
df_dlbs = pd.read_csv(metadata_path)
# Update filepaths for the independent dataset
df_dlbs['filepath'] = df_dlbs.apply(
    lambda row: f"dlbs_storage/DLBS_bias_corrected/{row['ImageID'][4:]}.stripped.N4.nii.gz",
    axis=1
)

metadata_path = "cobre_storage/cobre_brainrotnet_metadata.csv"
df_cobre = pd.read_csv(metadata_path)
# Update filepaths for the independent dataset
df_cobre['filepath'] = df_cobre.apply(
    lambda row: f"cobre_storage/COBRE_bias_corrected/{row['ImageID'][5:]}.stripped.N4.nii.gz",
    axis=1
)

metadata_path = "fcon1000_storage/fcon1000_brainrotnet_metadata.csv"
df_fcon = pd.read_csv(metadata_path)
# Update filepaths for the independent dataset
df_fcon['filepath'] = df_fcon.apply(
    lambda row: f"fcon1000_storage/fcon1000_bias_corrected/{row['ImageID'][8:]}.stripped.N4.nii.gz",
    axis=1
)
df_fcon = df_fcon.dropna()
# df_fcon = df_fcon.sort_values(by='Age', ascending=False).reset_index(drop=True).head(300)
# df_fcon = df_fcon.sample(n=300)

metadata_path = "sald_storage/sald_brainrotnet_metadata.csv"
df_sald = pd.read_csv(metadata_path)
# Update filepaths for the independent dataset
df_sald['filepath'] = df_sald.apply(
    lambda row: f"sald_storage/SALD_bias_corrected/sub-{row['ImageID'][4:]}.stripped.N4.nii.gz",
    axis=1
)
# df_sald = df_sald.sort_values(by='Age', ascending=False).reset_index(drop=True).head(300)
# df_sald = df_sald.sample(n=300)

metadata_path = "corr_storage/corr_brainrotnet_metadata.csv"
df_corr = pd.read_csv(metadata_path)
# Update filepaths for the independent dataset
df_corr['filepath'] = df_corr.apply(
    lambda row: f"corr_storage/CORR_bias_corrected/{row['ImageID'][5:]}.stripped.N4.nii.gz",
    axis=1
)
df_corr = df_corr.sort_values(by='Age', ascending=True).reset_index(drop=True)
# df_corr = df_corr.head(300)
# df_corr = df_corr.sample(n=200)


metadata_path = "oasis1_storage/oasis1_brainrotnet_metadata.csv"
df_oas1 = pd.read_csv(metadata_path)
# Update filepaths for the independent dataset
df_oas1['filepath'] = df_oas1.apply(
    lambda row: f"oasis1_storage/oasis_nii_gz_bias_corrected/{row['ImageID']}.stripped.N4.nii.gz",
    axis=1
)
df_oas1 = df_oas1.sort_values(by='Age', ascending=False)
df_oas1 = df_oas1.reset_index(drop=True).head(300)

metadata_path = "camcan_storage/camcan_brainrotnet_metadata.csv"
df_camcan = pd.read_csv(metadata_path)
# Update filepaths for the independent dataset
df_camcan['filepath'] = df_camcan.apply(
    lambda row: f"camcan_storage/CamCAN_nii_gz_bias_corrected/{row['ImageID']}.stripped.N4.nii.gz",
    axis=1
)

metadata_path = "nimh_storage/nimh_mprage_brainrotnet_metadata.csv"
df_nimh = pd.read_csv(metadata_path)
# Update filepaths for the independent dataset
df_nimh['filepath'] = df_nimh.apply(
    lambda row: f"nimh_storage/nimh_bias_corrected/{row['ImageID'][5:]}_ses-01_acq-MPRAGE_rec-SCIC_T1w.stripped.N4.nii.gz",
    axis=1
)

metadata_path = "bold_storage/bold_brainrotnet_metadata.csv"
df_bold = pd.read_csv(metadata_path)
# Update filepaths for the independent dataset
df_bold['filepath'] = df_bold.apply(
    lambda row: f"bold_storage/bold_bias_corrected/{row['ImageID'][5:]}_T1w.stripped.N4.nii.gz",
    axis=1
)

df = pd.concat ([
                 df_adni[['ImageID', 'Sex', 'Age', 'filepath']], 
                 df_ixi[['ImageID', 'Sex', 'Age', 'filepath']], 
                 df_abide[['ImageID', 'Sex', 'Age', 'filepath']],
                 df_dlbs[['ImageID', 'Sex', 'Age', 'filepath']],
                 df_cobre[['ImageID', 'Sex', 'Age', 'filepath']],
                 df_fcon[['ImageID', 'Sex', 'Age', 'filepath']],
                #  df_sald[['ImageID', 'Sex', 'Age', 'filepath']],
                 df_corr[['ImageID', 'Sex', 'Age', 'filepath']], 
                 df_oas1[['ImageID', 'Sex', 'Age', 'filepath']],
                 df_camcan[['ImageID', 'Sex', 'Age', 'filepath']],
                 df_nimh[['ImageID', 'Sex', 'Age', 'filepath']],
                 df_bold[['ImageID', 'Sex', 'Age', 'filepath']]
                 ], ignore_index=True)



# Run training with your dataframe
trained_models, results_df, final_mae = main(
    df=df,  # Your dataframe
    batch_size=4,
    num_workers=2
)