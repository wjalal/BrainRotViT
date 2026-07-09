import os
import sys
import pickle
import importlib
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform, apply_orientation
from torchvision import transforms
from transformers import ViTModel, ViTFeatureExtractor
from peft import LoraConfig, get_peft_model
from scipy.ndimage import zoom, label, find_objects
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt


def set_random_seed(seed=69420):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


universal_seed = 69420
set_random_seed(universal_seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Load metadata for every dataset (same sources/filters as the cached pipeline
# in vit_train_feature_cnn_main_mix_roi.py)
# ---------------------------------------------------------------------------
csv_path = "adni_storage/adni_brainrotnet_metadata.csv"
df_adni = pd.read_csv(csv_path)
df_adni['filepath'] = df_adni.apply(
    lambda row: f"adni_storage/ADNI_nii_gz_bias_corrected/I{row['ImageID'][4:]}_{row['SubjectID']}.stripped.N4.nii.gz",
    axis=1
)
# Keep a single scan per subject: the one closest to that subject's median age
df_adni = df_adni.loc[
    df_adni.groupby('SubjectID')['Age'].apply(lambda x: (x - x.median()).abs().idxmin())
].reset_index(drop=True)
df_adni = df_adni.sort_values(by='Age', ascending=True).reset_index(drop=True)
df_adni = df_adni.head(900)

metadata_path = "ixi_storage/ixi_brainrotnet_metadata.csv"
df_ixi = pd.read_csv(metadata_path)
df_ixi['filepath'] = df_ixi.apply(
    lambda row: f"ixi_storage/IXI_bias_corrected/{row['ImageID']}.stripped.N4.nii.gz",
    axis=1
)

metadata_path = "abide_storage/abide_brainrotnet_metadata.csv"
df_abide = pd.read_csv(metadata_path)
df_abide['filepath'] = df_abide.apply(
    lambda row: f"abide_storage/ABIDEII_bias_corrected/{row['ImageID'][7:]}.stripped.N4.nii.gz",
    axis=1
)
df_abide = df_abide.sort_values(by='Age', ascending=False).reset_index(drop=True)
df_abide = df_abide.head(750)

metadata_path = "dlbs_storage/dlbs_brainrotnet_metadata.csv"
df_dlbs = pd.read_csv(metadata_path)
df_dlbs['filepath'] = df_dlbs.apply(
    lambda row: f"dlbs_storage/DLBS_bias_corrected/{row['ImageID'][4:]}.stripped.N4.nii.gz",
    axis=1
)

metadata_path = "cobre_storage/cobre_brainrotnet_metadata.csv"
df_cobre = pd.read_csv(metadata_path)
df_cobre['filepath'] = df_cobre.apply(
    lambda row: f"cobre_storage/COBRE_bias_corrected/{row['ImageID'][5:]}.stripped.N4.nii.gz",
    axis=1
)

metadata_path = "fcon1000_storage/fcon1000_brainrotnet_metadata.csv"
df_fcon = pd.read_csv(metadata_path)
df_fcon['filepath'] = df_fcon.apply(
    lambda row: f"fcon1000_storage/fcon1000_bias_corrected/{row['ImageID'][8:]}.stripped.N4.nii.gz",
    axis=1
)
df_fcon = df_fcon.dropna()

metadata_path = "corr_storage/corr_brainrotnet_metadata.csv"
df_corr = pd.read_csv(metadata_path)
df_corr['filepath'] = df_corr.apply(
    lambda row: f"corr_storage/CORR_bias_corrected/{row['ImageID'][5:]}.stripped.N4.nii.gz",
    axis=1
)
df_corr = df_corr.sort_values(by='Age', ascending=True).reset_index(drop=True)

metadata_path = "oasis1_storage/oasis1_brainrotnet_metadata.csv"
df_oas1 = pd.read_csv(metadata_path)
df_oas1['filepath'] = df_oas1.apply(
    lambda row: f"oasis1_storage/oasis_nii_gz_bias_corrected/{row['ImageID']}.stripped.N4.nii.gz",
    axis=1
)
df_oas1 = df_oas1.sort_values(by='Age', ascending=False)
df_oas1 = df_oas1.reset_index(drop=True).head(300)

metadata_path = "camcan_storage/camcan_brainrotnet_metadata.csv"
df_camcan = pd.read_csv(metadata_path)
df_camcan['filepath'] = df_camcan.apply(
    lambda row: f"camcan_storage/CamCAN_nii_gz_bias_corrected/{row['ImageID']}.stripped.N4.nii.gz",
    axis=1
)

metadata_path = "nimh_storage/nimh_mprage_brainrotnet_metadata.csv"
df_nimh = pd.read_csv(metadata_path)
df_nimh['filepath'] = df_nimh.apply(
    lambda row: f"nimh_storage/nimh_bias_corrected/{row['ImageID'][5:]}_ses-01_acq-MPRAGE_rec-SCIC_T1w.stripped.N4.nii.gz",
    axis=1
)

metadata_path = "bold_storage/bold_brainrotnet_metadata.csv"
df_bold = pd.read_csv(metadata_path)
df_bold['filepath'] = df_bold.apply(
    lambda row: f"bold_storage/bold_bias_corrected/{row['ImageID'][5:]}_T1w.stripped.N4.nii.gz",
    axis=1
)

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
    df_bold[['ImageID', 'Sex', 'Age', 'filepath']],
], ignore_index=True)

# Drop rows whose NIfTI file is missing on disk
missing = ~df['filepath'].apply(os.path.exists)
if missing.any():
    print(f"Skipping {missing.sum()} rows with missing files.")
df = df[~missing].reset_index(drop=True)
print(df)

sex_encoded = df['Sex'].apply(lambda x: 0 if x == 'M' else 1).tolist()
age_list = df['Age'].tolist()
filepath_list = df['filepath'].tolist()

num_slices = 160  # every resampled slice is used and kept in the backward graph

# ---------------------------------------------------------------------------
# Preprocessing helpers: volume -> normalized [num_slices, 3, 224, 224] tensor.
# Nothing is written to disk -- no cached slices, no cached ViT features.
# ---------------------------------------------------------------------------
def resample_nifti(img_data, target_slices=160):
    current_slices = img_data.shape[0]
    zoom_factor = target_slices / current_slices
    return zoom(img_data, (zoom_factor, 1, 1), order=3)


def calculate_bounding_box_from_volume(volume, intensity_threshold=0.1):
    volume_normalized = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
    binary_mask = volume_normalized > intensity_threshold
    labeled_array, _ = label(binary_mask)
    component_sizes = np.bincount(labeled_array.ravel())
    component_sizes[0] = 0  # exclude background
    largest_component = np.argmax(component_sizes)
    brain_mask = labeled_array == largest_component
    slices = find_objects(brain_mask.astype(int))[0]
    min_indices = [s.start for s in slices]
    max_indices = [s.stop - 1 for s in slices]
    return min_indices, max_indices


def crop_brain_volumes(brain_data):
    min_indices, max_indices = calculate_bounding_box_from_volume(brain_data)
    return brain_data[min_indices[0]:max_indices[0] + 1,
                       min_indices[1]:max_indices[1] + 1,
                       min_indices[2]:max_indices[2] + 1]


feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
])


def load_volume_slices(filepath, target_slices=num_slices):
    nii_img = nib.load(filepath)
    orig_ornt = io_orientation(nii_img.affine)
    ras_ornt = axcodes2ornt(("R", "A", "S"))
    ornt_trans = ornt_transform(orig_ornt, ras_ornt)
    data = nii_img.get_fdata()
    data = apply_orientation(data, ornt_trans)
    data = crop_brain_volumes(data)
    data = resample_nifti(data, target_slices=target_slices)

    slices = []
    for slice_idx in range(data.shape[0]):
        slice_data = data[slice_idx, :, :]
        slice_min, slice_max = np.min(slice_data), np.max(slice_data)
        denom = slice_max - slice_min
        # Uniform slices (e.g. all-zero background after resampling) have
        # max == min; leave them as zeros instead of dividing by zero -> NaN.
        if denom > 0:
            slice_data = (slice_data - slice_min) / denom
        else:
            slice_data = np.zeros_like(slice_data)
        slices.append(transform(slice_data))
    return torch.stack(slices)  # [target_slices, 3, 224, 224]


class VolumeDataset(Dataset):
    """Lazily loads and preprocesses whole volumes on every access -- no
    slices or ViT features are ever cached to disk."""

    def __init__(self, filepath_list, sex_list, age_list, target_slices=num_slices):
        self.filepath_list = filepath_list
        self.sex_list = sex_list
        self.age_list = age_list
        self.target_slices = target_slices

    def __len__(self):
        return len(self.age_list)

    def __getitem__(self, idx):
        slices = load_volume_slices(self.filepath_list[idx], self.target_slices)
        sex = torch.tensor(self.sex_list[idx], dtype=torch.float32)
        age = torch.tensor(self.age_list[idx], dtype=torch.float32)
        return slices, sex, age


dataset = VolumeDataset(filepath_list, sex_encoded, age_list, target_slices=num_slices)

batch_size = 1
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
generator = torch.Generator().manual_seed(universal_seed)
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)
train_indices = train_dataset.indices
val_indices = val_dataset.indices

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

train_outputs = {}
val_outputs = {}

# ---------------------------------------------------------------------------
# End-to-end model: a LoRA-adapted ViT backbone feeds its per-slice embeddings
# straight into the CNN -- there's no frozen feature-extraction stage, so
# every one of the `num_slices` ViT forward passes stays in the graph and
# gets a gradient during backprop.
# ---------------------------------------------------------------------------
module_name = sys.argv[1]
class_name = "AgePredictionCNN"
module = importlib.import_module(module_name)
AgePredictionCNN = getattr(module, class_name)
print(f"Successfully imported {class_name} from {module_name}.")

vit_backbone = ViTModel.from_pretrained("google/vit-base-patch16-224", add_pooling_layer=False)

# Gradient checkpointing: recompute each encoder layer's activations during the
# backward pass instead of storing them. With every one of the num_slices slices
# living in the batch dimension, this is what keeps peak memory tractable
# (~26 GB -> ~4.6 GB at num_slices=160). enable_input_require_grads() is needed
# so the checkpointed graph reaches the (frozen) input embeddings.
vit_backbone.gradient_checkpointing_enable()
vit_backbone.enable_input_require_grads()

# DoRA (weight-decomposed LoRA) on every linear projection in the transformer:
# the attention Q/K/V projections and both MLP projections all share the "dense"
# module name, so this covers the whole backbone with a small number of
# trainable parameters.
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    target_modules=["query", "key", "value", "dense"],
    use_dora=True,
)
vit_backbone = get_peft_model(vit_backbone, lora_config)
vit_backbone.print_trainable_parameters()


class ViTCNNEndToEnd(nn.Module):
    def __init__(self, vit, cnn):
        super().__init__()
        self.vit = vit
        self.cnn = cnn

    def forward(self, slices, sex):
        b, n, c, h, w = slices.shape
        slices = slices.view(b * n, c, h, w)
        embeddings = self.vit(pixel_values=slices).last_hidden_state.mean(dim=1)
        embeddings = embeddings.view(b, n, -1).unsqueeze(1)  # [b, 1, n, hidden]
        return self.cnn(embeddings, sex)


cnn_head = AgePredictionCNN((1, num_slices, vit_backbone.config.hidden_size))
model = ViTCNNEndToEnd(vit_backbone, cnn_head).to(device)

criterion = nn.MSELoss()
eval_crit = nn.L1Loss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0005)
best_loss = np.inf
start_epoch = 0

filename = f"{sys.argv[1]}_e2e"

load_saved = sys.argv[2]  # "none", "last", "best"
if load_saved != "none":
    with open(f"model_dumps/mix/{filename}_best_model_with_metadata.pkl", "rb") as f:
        checkpoint = pickle.load(f)
    best_loss = checkpoint["loss"]

    with open(f"model_dumps/mix/{filename}_{load_saved}_model_with_metadata.pkl", "rb") as f:
        checkpoint = pickle.load(f)

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    torch.set_rng_state(checkpoint["t_rng_st"])
    np.random.set_state(checkpoint["n_rng_st"])
    if torch.cuda.is_available() and checkpoint["cuda_rng_st"] is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_st"])

    start_epoch = checkpoint["epoch"] + 1
    print(f"Loaded model from epoch {start_epoch} with validation loss {checkpoint['loss']:.4f}, best loss {best_loss:.4f}")

epochs = int(sys.argv[3])
csv_file = f"model_dumps/mix/{filename}.csv"

if os.path.exists(csv_file):
    epoch_data = pd.read_csv(csv_file).to_dict(orient="records")
    print(f"Loaded existing epoch data from {csv_file}.")
else:
    epoch_data = []
    print("No existing epoch data found. Starting fresh.")


def update_loss_plot(epoch_data, filename):
    df_loss = pd.DataFrame(epoch_data)
    df_loss.to_csv(f"model_dumps/mix/{filename}.csv", index=False)

    plt.figure(figsize=(8, 6))
    plt.plot(df_loss['epoch'], df_loss['train_loss'] ** 0.5, label="Train RMSE", marker="o")
    plt.plot(df_loss['epoch'], df_loss['val_loss'], label="Validation MAE", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs. Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"model_dumps/mix/{filename}.png")
    plt.close()


for epoch in range(start_epoch, epochs):
    model.train()
    train_loss = 0.0

    for idx, (slices, sex, age) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]")):
        slices, sex, age = slices.to(device), sex.to(device), age.to(device)

        optimizer.zero_grad()
        outputs = model(slices, sex)
        for i in range(outputs.size(0)):
            train_outputs[train_indices[idx * batch_size + i]] = outputs[i].item()

        loss = criterion(outputs.squeeze(), age)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for idx, (slices, sex, age) in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [val]")):
            slices, sex, age = slices.to(device), sex.to(device), age.to(device)
            outputs = model(slices, sex)
            loss = eval_crit(outputs.squeeze(), age)
            val_loss += loss.item()

            for i in range(outputs.size(0)):
                val_outputs[val_indices[idx * batch_size + i]] = outputs[i].item()

    val_loss /= len(val_loader)
    print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss:.4f}")

    print("Saving last model...")
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "loss": val_loss,
        "t_rng_st": torch.get_rng_state(),
        "n_rng_st": np.random.get_state(),
        "cuda_rng_st": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    with open(f"model_dumps/mix/{filename}_last_model_with_metadata.pkl", "wb") as f:
        pickle.dump(checkpoint, f)

    if val_loss < best_loss:
        best_loss = val_loss
        print(f"Validation loss improved to {best_loss:.4f}. Saving model...")
        with open(f"model_dumps/mix/{filename}_best_model_with_metadata.pkl", "wb") as f:
            pickle.dump(checkpoint, f)

    epoch_data.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
    update_loss_plot(epoch_data, filename)

    max_index = max(train_outputs.keys())
    df_trn = pd.DataFrame(index=range(max_index + 1), columns=["Predicted_Age"])
    for index, value in train_outputs.items():
        df_trn.loc[index, "Predicted_Age"] = value
    df2 = df.copy()
    df2['Predicted_Age'] = df_trn['Predicted_Age']
    df2.loc[train_outputs.keys()].to_csv(f"model_dumps/mix/{filename}_predicted_ages_train.csv")

    max_index = max(val_outputs.keys())
    df_pred = pd.DataFrame(index=range(max_index + 1), columns=["Predicted_Age"])
    for index, value in val_outputs.items():
        df_pred.loc[index, "Predicted_Age"] = value
    df1 = df.copy()
    df1['Predicted_Age'] = df_pred['Predicted_Age']
    test_df = df1.loc[val_outputs.keys()]
    test_df.to_csv(f"model_dumps/mix/{filename}_predicted_ages_val.csv")

    unique_groups = test_df['ImageID'].str[:3].unique()
    group_to_color = {group: i for i, group in enumerate(unique_groups)}
    cmap = plt.get_cmap('tab10')
    colors = [cmap(group_to_color[group]) for group in test_df['ImageID'].str[:3]]

    plt.figure(figsize=(8, 6))
    plt.scatter(test_df['Age'], test_df['Predicted_Age'], color=colors, label='Predicted vs Actual')
    handles = [plt.Line2D([0], [0], marker='o', color=cmap(i), linestyle='', markersize=10)
               for i, group in enumerate(unique_groups)]
    plt.legend(handles, unique_groups, title="ImageID Groups")
    plt.xlabel('Age')
    plt.ylabel('Predicted Age')
    plt.title('Age vs Predicted Age')
    plt.grid(True)
    plt.savefig(f"model_dumps/mix/{filename}_age_vs_predicted.png")
    plt.close()
