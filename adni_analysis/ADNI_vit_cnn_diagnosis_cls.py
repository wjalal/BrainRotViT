"""
ADNI_vit_cnn_diagnosis_cls.py

Diagnosis classifier (CN / MCI / AD) built on top of the EXISTING, frozen,
age-trained ViT encoder.

The ViT was already fine-tuned on age and its per-slice embeddings are cached in
    ../adni_storage/ADNI_features/train_adni_e{vit_train_epochs}_s{slice_count}/
by ADNI_vit_train_feature_cnn_main_mix_roi.py. This script does NOT touch the
ViT at all -- it simply reloads those frozen features and trains a new CNN head
to predict cognitive status instead of age.

Everything else is kept identical to the age pipeline:
  * the same row-4190-to-top reordering of df_adni,
  * the same subject-level train/val partition (no visit-level leakage),
  * the same feature directory, ROI crop, and CNN trunk.
Only the head (3-way softmax), the loss (class-weighted cross-entropy) and the
reported metrics (accuracy / balanced accuracy / macro-F1 / confusion matrix)
differ from the age-regression model.

Usage:
    python ADNI_vit_cnn_diagnosis_cls.py <epochs> [recover]
        <epochs>   number of training epochs (e.g. 100)
        recover    optional; resume from the last saved checkpoint
"""
import os
import sys
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             confusion_matrix, classification_report, roc_auc_score)

from cnn_mx_bigdo_ch_sw_res_cls import DiagnosisCNN


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

# Must match the age-feature-extraction pipeline exactly so the cached feature
# files and the subject split line up.
vit_train_epochs = 5
slice_count = 32
roi = 160
FEATURE_DIR = f"../adni_storage/ADNI_features/train_adni_e{vit_train_epochs}_s{slice_count}"

CLASS_ORDER = ["CN", "MCI", "AD"]
label_to_idx = {c: i for i, c in enumerate(CLASS_ORDER)}
num_classes = len(CLASS_ORDER)

OUT_PREFIX = "vit_cnn_diagnosis_cls"
os.makedirs("model_dumps/mix", exist_ok=True)
os.makedirs("model_dumps/mix/plots", exist_ok=True)

# ---------------------------------------------------------------------------
# Metadata (identical prep + row-4190-to-top reorder as the age pipeline)
# ---------------------------------------------------------------------------
csv_path = "../adni_storage/adni_brainrotnet_metadata.csv"
df_adni = pd.read_csv(csv_path)
df_adni['filepath'] = df_adni.apply(
    lambda row: f"../adni_storage/ADNI_nii_gz_bias_corrected/I{row['ImageID'][4:]}_{row['SubjectID']}.stripped.N4.nii.gz",
    axis=1
)
row_to_top = df_adni.iloc[[4190]]
rest = df_adni.drop(df_adni.index[4190])
df_adni = pd.concat([row_to_top, rest], ignore_index=True)

df = df_adni[['ImageID', 'SubjectID', 'Sex', 'Age', 'Group', 'filepath']].copy().reset_index(drop=True)

sex_encoded = df['Sex'].apply(lambda x: 0 if x == 'M' else 1).tolist()
age_list = df['Age'].tolist()
group_list = df['Group'].tolist()
class_list = [label_to_idx[g] for g in group_list]
print("Class distribution (all rows):",
      {c: int((np.array(class_list) == i).sum()) for c, i in label_to_idx.items()})

# ---------------------------------------------------------------------------
# Subject-level train/val partition (identical to the age pipeline).
# ---------------------------------------------------------------------------
subject_ids = df_adni['SubjectID'].to_numpy()
unique_subjects = np.array(sorted(df_adni['SubjectID'].unique()))
_subj_rng = np.random.RandomState(universal_seed)
_subj_perm = _subj_rng.permutation(len(unique_subjects))
_n_train_subj = int(0.8 * len(unique_subjects))
train_subjects = set(unique_subjects[_subj_perm[:_n_train_subj]].tolist())
val_subjects = set(unique_subjects[_subj_perm[_n_train_subj:]].tolist())
subject_train_indices = [i for i, s in enumerate(subject_ids) if s in train_subjects]
subject_val_indices = [i for i, s in enumerate(subject_ids) if s in val_subjects]
assert set(train_subjects).isdisjoint(val_subjects), "Subject leak between splits!"
print(f"Subject-level split: {len(train_subjects)} train / {len(val_subjects)} val subjects "
      f"({len(subject_train_indices)} / {len(subject_val_indices)} samples)")

# ---------------------------------------------------------------------------
# Load the FROZEN, cached ViT features (no ViT forward pass here).
# features_list[k] corresponds to df row k (same order as df_adni).
# ---------------------------------------------------------------------------
features_list = []
missing = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading frozen ViT features"):
    image_title = f"{row['ImageID'][4:]}_{row['SubjectID']}"
    feature_file_path = f"{FEATURE_DIR}/{image_title}_features.npy"
    if not os.path.exists(feature_file_path):
        missing.append(feature_file_path)
        features_list.append(None)
        continue
    features = np.load(feature_file_path)
    features = features[len(features) // 2 - roi // 2: len(features) // 2 + roi // 2]
    features_list.append(features)

if missing:
    raise FileNotFoundError(
        f"{len(missing)} feature files missing (first: {missing[0]}). "
        f"Run ADNI_vit_train_feature_cnn_main_mix_roi.py first to cache them.")

feat_shape = features_list[0].shape
print("Feature shape per sample:", feat_shape)
assert len(features_list) == len(df), "feature/df misalignment"


# ---------------------------------------------------------------------------
# Dataset: (features, sex, class label)
# ---------------------------------------------------------------------------
class ADNIClsDataset(Dataset):
    def __init__(self, features, sex, labels):
        self.features = features
        self.sex = sex
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.sex[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


dataset = ADNIClsDataset(features_list, sex_encoded, class_list)
train_dataset = torch.utils.data.Subset(dataset, subject_train_indices)
val_dataset = torch.utils.data.Subset(dataset, subject_val_indices)
train_indices = subject_train_indices
val_indices = subject_val_indices

batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Class weights (inverse frequency on the TRAIN split) to counter CN dominance.
train_labels = np.array([class_list[i] for i in train_indices])
class_counts = np.array([(train_labels == i).sum() for i in range(num_classes)])
class_weights = class_counts.sum() / (num_classes * np.maximum(class_counts, 1))
class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)
print("Train class counts:", dict(zip(CLASS_ORDER, class_counts.tolist())))
print("Class weights:", dict(zip(CLASS_ORDER, np.round(class_weights, 3).tolist())))

# ---------------------------------------------------------------------------
# Model / optim
# ---------------------------------------------------------------------------
model = DiagnosisCNN((1, feat_shape[0], feat_shape[1]), num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_t)
optimizer = optim.Adam(model.parameters(), lr=0.0005)

epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 100
best_metric = -np.inf          # track best balanced accuracy
start_epoch = 0

last_ckpt_path = f"model_dumps/mix/{OUT_PREFIX}_last_model_with_metadata.pkl"
best_ckpt_path = f"model_dumps/mix/{OUT_PREFIX}_best_model_with_metadata.pkl"

if len(sys.argv) > 2 and sys.argv[2] == "recover" and os.path.exists(last_ckpt_path):
    with open(last_ckpt_path, "rb") as f:
        ckpt = pickle.load(f)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    torch.set_rng_state(ckpt["t_rng_st"])
    np.random.set_state(ckpt["n_rng_st"])
    if torch.cuda.is_available() and ckpt["cuda_rng_st"] is not None:
        torch.cuda.set_rng_state_all(ckpt["cuda_rng_st"])
    start_epoch = ckpt["epoch"] + 1
    if os.path.exists(best_ckpt_path):
        with open(best_ckpt_path, "rb") as f:
            best_metric = pickle.load(f).get("metric", -np.inf)
    print(f"Recovered from epoch {start_epoch} (best bal-acc {best_metric:.4f}).")

csv_file = f"model_dumps/mix/{OUT_PREFIX}.csv"
epoch_data = pd.read_csv(csv_file).to_dict(orient="records") if os.path.exists(csv_file) else []


def evaluate(loader, indices):
    """Return (avg_loss, y_true, y_pred, probs, sample_indices)."""
    model.eval()
    loss_sum = 0.0
    y_true, y_pred, probs, samp_idx = [], [], [], []
    with torch.no_grad():
        for bidx, (features, sex, label) in enumerate(loader):
            features = features.unsqueeze(1).to(device)
            sex = sex.to(device)
            label = label.to(device)
            logits = model(features, sex)
            loss_sum += criterion(logits, label).item()
            p = torch.softmax(logits, dim=1)
            pred = p.argmax(dim=1)
            y_true.extend(label.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())
            probs.extend(p.cpu().tolist())
            for i in range(label.size(0)):
                samp_idx.append(indices[bidx * batch_size + i])
    return loss_sum / max(len(loader), 1), np.array(y_true), np.array(y_pred), np.array(probs), samp_idx


def write_predictions(y_true, y_pred, probs, samp_idx, out_csv):
    rec = df.iloc[samp_idx].copy()
    rec["TrueLabel"] = [CLASS_ORDER[t] for t in y_true]
    rec["PredLabel"] = [CLASS_ORDER[p] for p in y_pred]
    for i, c in enumerate(CLASS_ORDER):
        rec[f"prob_{c}"] = probs[:, i]
    rec.to_csv(out_csv, index=False)


def save_confusion(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(num_classes)); ax.set_xticklabels(CLASS_ORDER)
    ax.set_yticks(range(num_classes)); ax.set_yticklabels(CLASS_ORDER)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Validation confusion matrix")
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def update_curve():
    d = pd.DataFrame(epoch_data)
    d.to_csv(csv_file, index=False)
    plt.figure(figsize=(8, 6))
    plt.plot(d['epoch'], d['train_loss'], label="Train loss", marker="o")
    plt.plot(d['epoch'], d['val_loss'], label="Val loss", marker="o")
    plt.plot(d['epoch'], d['val_bal_acc'], label="Val balanced acc", marker="s")
    plt.plot(d['epoch'], d['val_macro_f1'], label="Val macro-F1", marker="^")
    plt.xlabel("Epoch"); plt.ylabel("Loss / Metric"); plt.legend(); plt.grid(True)
    plt.title("Diagnosis classifier training")
    plt.savefig(f"model_dumps/mix/{OUT_PREFIX}.png")
    plt.close()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
for epoch in range(start_epoch, epochs):
    model.train()
    train_loss = 0.0
    for features, sex, label in train_loader:
        features = features.unsqueeze(1).to(device)
        sex = sex.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        logits = model(features, sex)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= max(len(train_loader), 1)

    val_loss, yt, yp, probs, samp_idx = evaluate(val_loader, val_indices)
    acc = accuracy_score(yt, yp)
    bal_acc = balanced_accuracy_score(yt, yp)
    macro_f1 = f1_score(yt, yp, average="macro")
    print(f"Epoch {epoch+1}/{epochs}  train_loss {train_loss:.4f}  val_loss {val_loss:.4f}  "
          f"acc {acc:.4f}  bal_acc {bal_acc:.4f}  macro_f1 {macro_f1:.4f}")

    ckpt = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "loss": val_loss,
        "metric": bal_acc,
        "t_rng_st": torch.get_rng_state(),
        "n_rng_st": np.random.get_state(),
        "cuda_rng_st": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    with open(last_ckpt_path, "wb") as f:
        pickle.dump(ckpt, f)

    if bal_acc > best_metric:
        best_metric = bal_acc
        with open(best_ckpt_path, "wb") as f:
            pickle.dump(ckpt, f)
        print(f"  -> new best balanced accuracy {best_metric:.4f}; saved.")
        write_predictions(yt, yp, probs, samp_idx, f"model_dumps/mix/{OUT_PREFIX}_predicted_val.csv")
        save_confusion(yt, yp, f"model_dumps/mix/plots/{OUT_PREFIX}_confusion.png")

    epoch_data.append({
        "epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss,
        "val_acc": acc, "val_bal_acc": bal_acc, "val_macro_f1": macro_f1,
    })
    update_curve()

# ---------------------------------------------------------------------------
# Final report on the BEST model
# ---------------------------------------------------------------------------
print("\nReloading best model for final report...")
if not os.path.exists(best_ckpt_path):
    print("No best checkpoint found (ran 0 epochs or interrupted before first save); skipping report.")
    sys.exit(0)
with open(best_ckpt_path, "rb") as f:
    best = pickle.load(f)
model.load_state_dict(best["model_state"])

val_loss, yt, yp, probs, samp_idx = evaluate(val_loader, val_indices)
print(f"\n=== BEST MODEL (epoch {best['epoch']+1}) validation report ===")
print(f"accuracy      {accuracy_score(yt, yp):.4f}")
print(f"balanced acc  {balanced_accuracy_score(yt, yp):.4f}")
print(f"macro F1      {f1_score(yt, yp, average='macro'):.4f}")
try:
    auc = roc_auc_score(yt, probs, multi_class="ovr", labels=range(num_classes))
    print(f"macro AUC-OVR {auc:.4f}")
except Exception as e:
    print(f"AUC unavailable: {e}")
print("\nConfusion matrix (rows=true, cols=pred) order", CLASS_ORDER)
print(confusion_matrix(yt, yp, labels=range(num_classes)))
print("\n" + classification_report(yt, yp, labels=range(num_classes), target_names=CLASS_ORDER, digits=3))

write_predictions(yt, yp, probs, samp_idx, f"model_dumps/mix/{OUT_PREFIX}_predicted_val.csv")
save_confusion(yt, yp, f"model_dumps/mix/plots/{OUT_PREFIX}_confusion.png")
print(f"Saved: model_dumps/mix/{OUT_PREFIX}_predicted_val.csv "
      f"and plots/{OUT_PREFIX}_confusion.png")
