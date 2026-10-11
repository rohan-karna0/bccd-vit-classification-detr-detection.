"""Notebook cells: problem, setup, data, and the from-scratch ViT."""

CELLS = []


def md(text):
    CELLS.append(("md", text.strip() + "\n"))


def code(text):
    CELLS.append(("code", text.strip() + "\n"))


md(r'''
# BCCD blood-cell classification and detection

This is **one project** with **two tasks** on the same microscopy images.

1. **Classification.** Look at one cropped cell and name it RBC, WBC, or Platelets.
2. **Detection.** Look at a full image and return a box, a class, and a score for each cell.

The classifier does not search the image. The detector does. Both are trained in this notebook, then a final cell opens a window where you can upload your own image.
''')

md(r'''
## Section 1: Problem

A blood-smear image contains many red blood cells, fewer white blood cells, and small platelets. The labels in this dataset are **RBC**, **WBC**, and **Platelets**.

The data is the public BCCD collection: about 364 images, Pascal VOC boxes, downloaded from Kaggle as `konstantinazov/bccd-dataset`.

### What is being compared

Classification uses the **same crops and the same train / val / test split** for two models:

| Model | Starting weights | Input size |
| --- | --- | --- |
| Custom ViT | Random. Trained only on BCCD crops. | 128 x 128 |
| torchvision `vit_b_16` | ImageNet-1K (`ViT_B_16_Weights.IMAGENET1K_V1`). The last layer is replaced, then the network is fine-tuned on BCCD. | 224 x 224 |

The custom ViT is **not** ViT-Base. It uses patch size 16, embedding size 192, 6 attention heads, and 6 encoder blocks.

Detection uses one model: Hugging Face `facebook/detr-resnet-50`, pretrained on COCO. The classification head is replaced for the three BCCD classes, then the model is fine-tuned on full images. That is separate from the ViT comparison.
''')

md(r'''
## Section 2: Setup

Keys are read from `.env` in the project root. Copy `.env.example` to `.env` before you run the download cell.

- `KAGGLE_API_TOKEN` is required. Create it at https://www.kaggle.com/settings/api
- `HF_TOKEN` is optional. The DETR checkpoint is public.
''')

code(r'''
import os
import sys
from pathlib import Path

FLAGS = {}


def mark(step):
    FLAGS[step] = True


def require(*steps):
    missing = [step for step in steps if not FLAGS.get(step)]
    if missing:
        raise RuntimeError("Run these cells first: " + ", ".join(missing))


def find_project_root():
    cwd = Path.cwd().resolve()
    if (cwd / "src" / "bccd").exists():
        return cwd
    if cwd.name == "notebooks" and (cwd.parent / "src" / "bccd").exists():
        return cwd.parent
    raise RuntimeError("Open the notebook with the project root as the working directory.")


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bccd.credentials import huggingface_token, load_env

ENV_PATH = load_env(PROJECT_ROOT)
HF_TOKEN = huggingface_token()
print("Project root:", PROJECT_ROOT)
print(".env file:", ENV_PATH, "(found)" if ENV_PATH.exists() else "(missing — copy .env.example to .env)")
print("HF_TOKEN:", "set" if HF_TOKEN else "not set")
mark("env_detected")
''')

code(r'''
require("env_detected")

CFG = {
    "seed": 42,
    "kaggle_dataset_slug": "konstantinazov/bccd-dataset",
    "min_crop_side": 16,
    "cls_image_size": 128,
    "cls_patch_size": 16,
    "cls_embed_dim": 192,
    "cls_depth": 6,
    "cls_num_heads": 6,
    "cls_mlp_ratio": 4.0,
    "cls_dropout": 0.1,
    "cls_batch_size": 32,
    "cls_epochs": 20,
    "cls_lr": 3e-4,
    "cls_weight_decay": 0.05,
    "cls_patience": 5,
    "cls_mean": (0.5, 0.5, 0.5),
    "cls_std": (0.5, 0.5, 0.5),
    "pretrained_epochs": 10,
    "pretrained_batch_size": 16,
    "pretrained_lr": 1e-4,
    "detr_model_name": "facebook/detr-resnet-50",
    "detr_epochs": 10,
    "detr_batch_size": 2,
    "detr_lr": 1e-4,
    "detr_lr_backbone": 1e-5,
    "detr_weight_decay": 1e-4,
    "detr_grad_clip": 0.1,
    "detr_patience": 4,
    "detr_confidence_threshold": 0.5,
    "detr_shortest_edge": 800,
    "detr_longest_edge": 1333,
    "num_workers": 2,
}

if CFG["cls_embed_dim"] % CFG["cls_num_heads"] != 0:
    raise ValueError("cls_embed_dim must be divisible by cls_num_heads.")
if CFG["cls_image_size"] % CFG["cls_patch_size"] != 0:
    raise ValueError("cls_image_size must be divisible by cls_patch_size.")
print("From-scratch ViT: 128px, patch 16, embed 192, depth 6, heads 6, epochs", CFG["cls_epochs"])
print("Pretrained comparison: torchvision vit_b_16, ImageNet-1K, epochs", CFG["pretrained_epochs"])
print("Detector: facebook/detr-resnet-50, epochs", CFG["detr_epochs"])
mark("config_ready")
''')

code(r'''
require("config_ready")
import random
import numpy as np
import pandas as pd
import torch
from PIL import Image

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(CFG["seed"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("Device:", DEVICE)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("No CUDA GPU was found. Training will be slow on CPU.")
mark("env_ready")
''')

md(r'''
## Section 3: Import the dataset

The download uses `KAGGLE_API_TOKEN` from `.env`. If the archive is already in `data/downloads/`, this cell reuses it.
''')

code(r'''
require("env_ready")
import json
from bccd.credentials import apply_kaggle_credentials
from bccd.voc import discover_files, format_tree

KAGGLE_DATASET_SLUG = CFG["kaggle_dataset_slug"]
DOWNLOAD_DIR = PROJECT_ROOT / "data" / "downloads" / KAGGLE_DATASET_SLUG.split("/")[-1]
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)


def looks_like_bccd(root):
    saw_image = False
    saw_xml = False
    image_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    for path in Path(root).rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        saw_image = saw_image or suffix in image_ext
        saw_xml = saw_xml or suffix == ".xml"
        if saw_image and saw_xml:
            return True
    return False


if looks_like_bccd(DOWNLOAD_DIR):
    DATASET_ROOT = DOWNLOAD_DIR
    print("Using the dataset already extracted at", DATASET_ROOT)
else:
    apply_kaggle_credentials()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    print("Downloading", KAGGLE_DATASET_SLUG)
    api.dataset_download_files(KAGGLE_DATASET_SLUG, path=str(DOWNLOAD_DIR), unzip=True, quiet=False)
    if not looks_like_bccd(DOWNLOAD_DIR):
        raise RuntimeError("Download finished, but no image and XML pair was found.")
    DATASET_ROOT = DOWNLOAD_DIR

discovered = discover_files(DATASET_ROOT)
print(format_tree(DATASET_ROOT))
print(f"Images: {len(discovered['images'])} | XML files: {len(discovered['xml'])}")
mark("data_ready")
''')

md(r'''
## Section 4: Inspect annotations

Each XML file lists cells as `xmin, ymin, xmax, ymax` on the original image. Class names are mapped to RBC, WBC, and Platelets.
''')

code(r'''
require("data_ready")
from bccd.voc import build_records, class_to_index

RECORDS, DATA_SUMMARY = build_records(DATASET_ROOT)
if not RECORDS:
    raise RuntimeError("No image could be paired with an XML file.")
print(pd.Series({
    "images": DATA_SUMMARY["n_images"],
    "paired": DATA_SUMMARY["n_paired"],
    "objects": DATA_SUMMARY["n_objects"],
    "empty_images": DATA_SUMMARY["n_empty_images"],
}).to_string())
print("\nInstances per class:")
print(pd.Series(DATA_SUMMARY["class_counts"], name="count").sort_values(ascending=False).to_string())
mark("explored")
''')

code(r'''
require("explored")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

CLASS_COLORS = {"RBC": "#d62728", "WBC": "#1f77b4", "Platelets": "#2ca02c"}


def draw_xyxy(ax, image, objects, title):
    ax.imshow(image)
    for obj in objects:
        x1, y1, x2, y2 = obj["bbox_xyxy"]
        color = CLASS_COLORS.get(obj["name"], "#ff7f0e")
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=1.4))
        ax.text(x1, max(y1 - 2, 0), obj["name"], color="white", fontsize=8,
                bbox={"facecolor": color, "alpha": 0.75, "pad": 0.2})
    ax.set_title(title)
    ax.axis("off")


samples = [record for record in RECORDS if record["objects"]][:4]
figure, axes = plt.subplots(1, len(samples), figsize=(4 * len(samples), 4))
if len(samples) == 1:
    axes = [axes]
for axis, record in zip(axes, samples):
    with Image.open(record["image_path"]) as image:
        draw_xyxy(axis, image.convert("RGB"), record["objects"], record["stem"])
figure.tight_layout()
plt.show()
mark("visualized")
''')

md(r'''
## Section 5: Split images and build crops

Images are split first. Crops from one image stay in one split. The official `ImageSets/Main` lists are used when they cover the archive. Otherwise the split is 70% train, 15% validation, 15% test.

Crops shorter than 16 pixels on either side are dropped. The rest are saved at 128 x 128 for the from-scratch ViT. The pretrained ViT-B/16 resizes those same files to 224 x 224 during training.
''')

code(r'''
require("visualized")
from bccd.splits import assign_splits, split_counts

PATHS = {
    "classification": PROJECT_ROOT / "data" / "classification",
    "detection": PROJECT_ROOT / "data" / "detection",
    "checkpoints": PROJECT_ROOT / "models" / "checkpoints",
    "outputs_cls": PROJECT_ROOT / "outputs" / "classification",
    "outputs_det": PROJECT_ROOT / "outputs" / "detection",
    "plots": PROJECT_ROOT / "outputs" / "plots",
}
for path in PATHS.values():
    path.mkdir(parents=True, exist_ok=True)

SPLIT_RECORDS, SPLIT_DESCRIPTION = assign_splits(RECORDS, DATASET_ROOT, seed=CFG["seed"])
CLASS_TO_IDX = class_to_index(SPLIT_RECORDS)
IDX_TO_CLASS = {index: name for name, index in CLASS_TO_IDX.items()}
print("Split rule:", SPLIT_DESCRIPTION)
print("Images per split:", split_counts(SPLIT_RECORDS))
print("Class index:", CLASS_TO_IDX)

DETECTION_RECORDS = []
for record in SPLIT_RECORDS:
    labels, boxes, names = [], [], []
    for obj in record["objects"]:
        if obj["name"] not in CLASS_TO_IDX:
            continue
        labels.append(CLASS_TO_IDX[obj["name"]])
        boxes.append([float(value) for value in obj["bbox_xyxy"]])
        names.append(obj["name"])
    DETECTION_RECORDS.append({
        "split": record["split"],
        "stem": record["stem"],
        "image_path": record["image_path"],
        "width": record["width"],
        "height": record["height"],
        "labels": labels,
        "boxes_xyxy": boxes,
        "class_names": names,
    })
(PATHS["detection"] / "records.json").write_text(json.dumps({
    "class_to_idx": CLASS_TO_IDX,
    "split_rule": SPLIT_DESCRIPTION,
    "records": DETECTION_RECORDS,
}, indent=2), encoding="utf-8")
print("Detection records:", len(DETECTION_RECORDS))
mark("splits_ready")
''')

code(r'''
require("splits_ready")
from bccd.crops import export_classification_crops

CROP_ROWS, CROP_SUMMARY = export_classification_crops(
    SPLIT_RECORDS,
    PATHS["classification"],
    CLASS_TO_IDX,
    min_side=CFG["min_crop_side"],
    image_size=CFG["cls_image_size"],
    stamp={
        "min_side": CFG["min_crop_side"],
        "image_size": CFG["cls_image_size"],
        "n_source_images": len(SPLIT_RECORDS),
        "seed": CFG["seed"],
        "split_rule": SPLIT_DESCRIPTION,
    },
)
crop_frame = pd.DataFrame(CROP_ROWS)
print("Crops:", CROP_SUMMARY.get("n_crops"), "| reused:", CROP_SUMMARY.get("reused"))
print(pd.crosstab(crop_frame["split"], crop_frame["class_name"]))
TRAIN_ROWS = [row for row in CROP_ROWS if row["split"] == "train"]
VAL_ROWS = [row for row in CROP_ROWS if row["split"] == "val"]
TEST_ROWS = [row for row in CROP_ROWS if row["split"] == "test"]
if not TRAIN_ROWS or not VAL_ROWS or not TEST_ROWS:
    raise RuntimeError("Train, validation, and test crops must all be non-empty.")
mark("crops_ready")
''')

code(r'''
require("crops_ready")
counts = pd.DataFrame(CROP_ROWS).groupby(["split", "class_name"]).size().unstack(fill_value=0)
figure, axis = plt.subplots(figsize=(7, 4))
counts.plot(kind="bar", ax=axis, color=[CLASS_COLORS.get(name, "#999999") for name in counts.columns])
axis.set_ylabel("Crops")
axis.set_title("Crops by split")
figure.tight_layout()
figure.savefig(PATHS["plots"] / "class_distribution.png", dpi=120)
plt.show()
mark("crops_preview_ready")
''')

md(r'''
## Section 6: From-scratch Vision Transformer

This is the model trained from random weights.

- Input: 128 x 128 RGB
- Patch: 16 x 16, so 8 x 8 = 64 patch tokens, plus one class token
- Embedding size 192, 6 heads, 6 encoder blocks
- Output: one logit for RBC, WBC, and Platelets

The next section trains it. The ImageNet ViT-B/16 is a different network and is trained later.
''')

code(r'''
require("crops_preview_ready")
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(f"Image size {img_size} is not divisible by patch size {patch_size}.")
        self.grid = img_size // patch_size
        self.num_patches = self.grid * self.grid
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, length, dim = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        attn = (query @ key.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = self.dropout(attn.softmax(dim=-1))
        out = (attn @ value).transpose(1, 2).reshape(batch, length, dim)
        return self.proj(out)


class MLPBlock(nn.Module):
    def __init__(self, embed_dim, mlp_ratio, dropout):
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLPBlock(embed_dim, mlp_ratio, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ClassificationHead(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, cls_token):
        return self.fc(self.norm(cls_token))


class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim, depth, num_heads, mlp_ratio, dropout, num_classes):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)
        ])
        self.head = ClassificationHead(embed_dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        batch = x.shape[0]
        tokens = self.patch_embed(x)
        cls = self.cls_token.expand(batch, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.blocks(self.dropout(tokens + self.pos_embed))
        return self.head(tokens[:, 0])


def build_vit(num_classes):
    return VisionTransformer(
        img_size=CFG["cls_image_size"],
        patch_size=CFG["cls_patch_size"],
        embed_dim=CFG["cls_embed_dim"],
        depth=CFG["cls_depth"],
        num_heads=CFG["cls_num_heads"],
        mlp_ratio=CFG["cls_mlp_ratio"],
        dropout=CFG["cls_dropout"],
        num_classes=num_classes,
    )


probe = build_vit(len(CLASS_TO_IDX))
with torch.no_grad():
    logits = probe(torch.randn(2, 3, CFG["cls_image_size"], CFG["cls_image_size"]))
print(f"From-scratch ViT parameters: {sum(p.numel() for p in probe.parameters()):,}")
print("Forward check:", tuple(logits.shape))
mark("vit_defined")
''')

md(r'''
## Section 7: Train the from-scratch ViT

Each epoch prints train loss, train accuracy, validation loss, and validation accuracy. The checkpoint with the highest validation accuracy is kept. The test set is not used here.
''')

code(r'''
require("vit_defined")
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class CellCropDataset(Dataset):
    def __init__(self, rows, transform):
        self.rows = list(rows)
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["crop_path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(row["label"])


def scratch_transform(train):
    ops = [transforms.Resize((CFG["cls_image_size"], CFG["cls_image_size"]))]
    if train:
        ops.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        ])
    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(CFG["cls_mean"], CFG["cls_std"]),
    ])
    return transforms.Compose(ops)


def make_loader(rows, train, batch_size):
    return DataLoader(
        CellCropDataset(rows, scratch_transform(train)),
        batch_size=batch_size,
        shuffle=train,
        num_workers=CFG["num_workers"],
        pin_memory=DEVICE.type == "cuda",
    )


def class_weights(rows):
    labels = [row["label"] for row in rows]
    counts = np.bincount(labels, minlength=len(CLASS_TO_IDX)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def run_epoch(model, loader, criterion, optimizer):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item()) * labels.size(0)
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total_seen += int(labels.size(0))
    return total_loss / total_seen, total_correct / total_seen


print(f"Crops: train {len(TRAIN_ROWS)}, val {len(VAL_ROWS)}, test {len(TEST_ROWS)}")
mark("cls_loop_ready")
''')

code(r'''
require("cls_loop_ready")
import time
from bccd.metrics_io import save_metrics

VIT_CHECKPOINT = PATHS["checkpoints"] / "vit_bccd_best.pt"
VIT_MODEL = build_vit(len(CLASS_TO_IDX)).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights(TRAIN_ROWS))
optimizer = torch.optim.AdamW(VIT_MODEL.parameters(), lr=CFG["cls_lr"], weight_decay=CFG["cls_weight_decay"])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG["cls_epochs"])
train_loader = make_loader(TRAIN_ROWS, True, CFG["cls_batch_size"])
val_loader = make_loader(VAL_ROWS, False, CFG["cls_batch_size"])

VIT_HISTORY = []
best_acc = -1.0
best_epoch = -1
waits = CFG["cls_patience"]
started = time.time()
for epoch in range(CFG["cls_epochs"]):
    train_loss, train_acc = run_epoch(VIT_MODEL, train_loader, criterion, optimizer)
    val_loss, val_acc = run_epoch(VIT_MODEL, val_loader, criterion, None)
    scheduler.step()
    VIT_HISTORY.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
    })
    print(
        f"epoch {epoch + 1}/{CFG['cls_epochs']}  "
        f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
        f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
    )
    if val_acc > best_acc:
        best_acc = val_acc
        best_epoch = epoch + 1
        waits = CFG["cls_patience"]
        torch.save({
            "model_state": VIT_MODEL.state_dict(),
            "class_to_idx": CLASS_TO_IDX,
            "model_name": "vit_from_scratch",
            "image_size": CFG["cls_image_size"],
            "epoch": best_epoch,
            "val_accuracy": best_acc,
        }, VIT_CHECKPOINT)
    else:
        waits -= 1
        if waits <= 0:
            print(f"Stopped early. Best validation accuracy {best_acc:.4f} at epoch {best_epoch}.")
            break

pd.DataFrame(VIT_HISTORY).to_csv(PATHS["outputs_cls"] / "vit_history.csv", index=False)
save_metrics(PATHS["outputs_cls"] / "vit_metrics.json", {
    "model": "ViT from scratch",
    "best_epoch": best_epoch,
    "best_val_accuracy": best_acc,
    "minutes": (time.time() - started) / 60,
})
print(f"Best validation accuracy {best_acc:.4f} at epoch {best_epoch}")
print("Checkpoint:", VIT_CHECKPOINT)
mark("vit_trained")
''')

code(r'''
require("vit_trained")
history = pd.read_csv(PATHS["outputs_cls"] / "vit_history.csv")
figure, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(history["epoch"], history["train_loss"], label="train")
axes[0].plot(history["epoch"], history["val_loss"], label="validation")
axes[0].set_title("From-scratch ViT loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[1].plot(history["epoch"], history["train_accuracy"], label="train")
axes[1].plot(history["epoch"], history["val_accuracy"], label="validation")
axes[1].set_title("From-scratch ViT accuracy")
axes[1].set_xlabel("Epoch")
axes[1].legend()
figure.tight_layout()
figure.savefig(PATHS["plots"] / "vit_training_curves.png", dpi=120)
plt.show()
mark("vit_curves_ready")
''')
