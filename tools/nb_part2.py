"""Notebook cells for sections 7 through 14."""

CELLS = []


def md(text):
    CELLS.append(("md", text.strip() + "\n"))


def code(text):
    CELLS.append(("code", text.strip() + "\n"))


md(r'''
## Section 7: Implement the Vision Transformer in PyTorch

A Vision Transformer cuts the image into a grid of patches, linearly projects each patch to a vector, and runs a Transformer encoder over that sequence. A learned class token is prepended. Its final state is the input to the classification head.

For a 128 by 128 RGB crop and 16 by 16 patches:

- the patch grid is 8 by 8, so there are 64 patch tokens
- each token has `cls_embed_dim` channels (192 in the full configuration)
- with the class token, the sequence length is 65
- each attention head produces a 65 by 65 attention matrix
- the head maps the class token to one logit per cell class

The full configuration uses embedding size 192, 6 heads, and 6 encoder blocks. That is small enough to train from scratch on a few thousand crops. It is not ImageNet ViT-Base. An optional cell later fine-tunes torchvision's pretrained ViT-B/16, which is a different model: those weights were trained by someone else on ImageNet, and only the head is replaced.

The implementation uses a strided convolution for the patch projection, so there is no Python loop over patches.
''')

code(r'''
require("crops_preview_ready")
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(f"Image size {img_size} is not divisible by patch size {patch_size}.")
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid = img_size // patch_size
        self.num_patches = self.grid * self.grid
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, 3, H, W) -> (B, D, grid, grid) -> (B, N, D)
        x = self.proj(x)
        if x.shape[-2:] != (self.grid, self.grid):
            raise AssertionError(f"Patch grid {tuple(x.shape[-2:])} != {(self.grid, self.grid)}")
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim {embed_dim} is not divisible by num_heads {num_heads}.")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, N, D). q, k, v: (B, heads, N, head_dim). attn: (B, heads, N, N).
        batch, length, dim = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        scale = self.head_dim ** -0.5
        attn = (query @ key.transpose(-2, -1)) * scale
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
        # Pre-norm residual block. The skip connection is the residual.
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
mark("vit_blocks_ready")
''')

code(r'''
require("vit_blocks_ready")


class ClassificationHead(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, cls_token):
        # cls_token: (B, D) -> logits: (B, num_classes)
        return self.fc(self.norm(cls_token))


class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim, depth, num_heads, mlp_ratio, dropout, num_classes):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans=3, embed_dim=embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)
        ])
        self.head = ClassificationHead(embed_dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        batch, channels, height, width = x.shape
        if channels != 3 or height != self.img_size or width != self.img_size:
            raise AssertionError(f"Expected (B, 3, {self.img_size}, {self.img_size}), got {tuple(x.shape)}")
        tokens = self.patch_embed(x)
        if tokens.shape != (batch, self.num_patches, self.cls_token.shape[-1]):
            raise AssertionError(f"Unexpected patch tokens: {tuple(tokens.shape)}")
        cls = self.cls_token.expand(batch, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.dropout(tokens + self.pos_embed)
        tokens = self.blocks(tokens)
        logits = self.head(tokens[:, 0])
        if logits.shape != (batch, self.num_classes):
            raise AssertionError(f"Unexpected logits: {tuple(logits.shape)}")
        return logits


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


_probe = build_vit(len(CLASS_TO_IDX))
_probe.eval()
with torch.no_grad():
    _logits = _probe(torch.randn(2, 3, CFG["cls_image_size"], CFG["cls_image_size"]))
if _logits.shape != (2, len(CLASS_TO_IDX)):
    raise AssertionError(_logits.shape)
print(
    f"ViT forward pass OK. Image (2, 3, {CFG['cls_image_size']}, {CFG['cls_image_size']}) "
    f"-> tokens {_probe.num_patches + 1} -> logits {tuple(_logits.shape)}"
)
del _probe, _logits
mark("vit_defined")
''')

md(r'''
## Section 8: Train the ViT classifier

The best checkpoint is the one with the highest validation accuracy. The test set is not used for that choice, for the learning rate, or for early stopping. Class-weighted cross-entropy counters the large RBC count. AdamW and a cosine schedule are the optimizer. Early stopping watches validation accuracy.

Set `CFG["run_pretrained_vit"] = True` to fine-tune ImageNet ViT-B/16 afterward. Set `CFG["run_cnn_baseline"] = True` to train the small CNN used in the comparison. Both are off by default so the GPU run starts with the from-scratch ViT.
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


def make_cls_transform(train):
    ops = [transforms.Resize((CFG["cls_image_size"], CFG["cls_image_size"]))]
    if train:
        ops.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
        ])
    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(CFG["cls_mean"], CFG["cls_std"]),
    ])
    return transforms.Compose(ops)


def make_loader(rows, train, batch_size):
    dataset = CellCropDataset(rows, make_cls_transform(train))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=CFG["num_workers"],
        pin_memory=DEVICE.type == "cuda",
    )


TRAIN_ROWS = [row for row in CROP_ROWS if row["split"] == "train"]
VAL_ROWS = [row for row in CROP_ROWS if row["split"] == "val"]
TEST_ROWS = [row for row in CROP_ROWS if row["split"] == "test"]
if not TRAIN_ROWS or not VAL_ROWS or not TEST_ROWS:
    raise RuntimeError("Train, val, and test crops must all be non-empty before training.")
print(f"Crop loaders: train {len(TRAIN_ROWS)}, val {len(VAL_ROWS)}, test {len(TEST_ROWS)}")
mark("cls_loaders_ready")
''')

code(r'''
require("cls_loaders_ready")
import time
from tqdm.auto import tqdm


def run_epoch(model, loader, criterion, optimizer, max_batches):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for step, (images, labels) in enumerate(tqdm(loader, leave=False)):
            if max_batches is not None and step >= max_batches:
                break
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
    if total_seen == 0:
        raise RuntimeError("The loader produced no batches.")
    return total_loss / total_seen, total_correct / total_seen


def fit_classifier(model, train_rows, val_rows, epochs, lr, weight_decay, patience, checkpoint_path, max_train_batches, max_val_batches):
    labels = [row["label"] for row in train_rows]
    counts = np.bincount(labels, minlength=len(CLASS_TO_IDX)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    train_loader = make_loader(train_rows, train=True, batch_size=CFG["cls_batch_size"])
    val_loader = make_loader(val_rows, train=False, batch_size=CFG["cls_batch_size"])
    history = []
    best_acc = -1.0
    best_epoch = -1
    waits = patience
    started = time.time()
    model.to(DEVICE)
    for epoch in range(epochs):
        epoch_started = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, max_train_batches)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, max_val_batches)
        scheduler.step()
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        })
        improved = val_acc > best_acc
        if improved:
            best_acc = val_acc
            best_epoch = epoch
            waits = patience
            torch.save({
                "model_state": model.state_dict(),
                "class_to_idx": CLASS_TO_IDX,
                "config": CFG,
                "epoch": epoch,
                "val_accuracy": val_acc,
                "val_loss": val_loss,
            }, checkpoint_path)
        else:
            waits -= 1
        elapsed = time.time() - epoch_started
        remaining = elapsed * (epochs - epoch - 1)
        print(
            f"Epoch {epoch + 1}/{epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f} best={best_acc:.4f} epoch_min={elapsed / 60:.2f} "
            f"eta_min={remaining / 60:.1f}"
        )
        if waits <= 0:
            print(f"Early stopping at epoch {epoch + 1}. Best validation accuracy was {best_acc:.4f} at epoch {best_epoch + 1}.")
            break
    return history, best_acc, best_epoch, time.time() - started
mark("cls_loop_ready")
''')

code(r'''
require("cls_loop_ready")
from bccd.metrics_io import save_metrics

VIT_CHECKPOINT = PATHS["checkpoints"] / "vit_bccd_best.pt"
VIT_MODEL = build_vit(len(CLASS_TO_IDX))
print(f"ViT parameters: {sum(p.numel() for p in VIT_MODEL.parameters()):,}")
VIT_HISTORY, VIT_BEST_ACC, VIT_BEST_EPOCH, VIT_SECONDS = fit_classifier(
    VIT_MODEL,
    TRAIN_ROWS,
    VAL_ROWS,
    epochs=CFG["cls_epochs"],
    lr=CFG["cls_lr"],
    weight_decay=CFG["cls_weight_decay"],
    patience=CFG["cls_patience"],
    checkpoint_path=VIT_CHECKPOINT,
    max_train_batches=CFG["max_train_batches"],
    max_val_batches=CFG["max_val_batches"],
)
if not VIT_CHECKPOINT.exists():
    raise RuntimeError("ViT training finished without writing a checkpoint.")
history_frame = pd.DataFrame(VIT_HISTORY)
history_frame.to_csv(PATHS["outputs_cls"] / "vit_history.csv", index=False)
config_text = (
    f"from scratch, image {CFG['cls_image_size']}, patch {CFG['cls_patch_size']}, "
    f"embed {CFG['cls_embed_dim']}, depth {CFG['cls_depth']}, heads {CFG['cls_num_heads']}, "
    f"epochs {CFG['cls_epochs']}, lr {CFG['cls_lr']}, batch {CFG['cls_batch_size']}, "
    f"smoke_test={CFG['smoke_test']}"
)
save_metrics(PATHS["outputs_cls"] / "vit_metrics.json", {
    "model": "ViT from scratch",
    "smoke_test": CFG["smoke_test"],
    "training_configuration": config_text,
    "validation_metrics": f"loss and accuracy tracked; best val accuracy={VIT_BEST_ACC:.4f} at epoch {VIT_BEST_EPOCH + 1}",
    "test_metrics": "not yet run",
    "training_duration": f"{VIT_SECONDS / 60:.2f} min on {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
    "inference_latency": "not yet run",
    "limitations": "Smoke-test metrics are not full-training results." if CFG["smoke_test"] else "BCCD crops are imbalanced and come from only a few hundred fields. The checkpoint was selected on validation accuracy.",
    "best_val_accuracy": VIT_BEST_ACC,
    "best_epoch": VIT_BEST_EPOCH,
    "history": VIT_HISTORY,
})
print(f"Saved best ViT checkpoint to {VIT_CHECKPOINT}")
mark("vit_trained")
''')

code(r'''
require("vit_trained")
history = pd.read_csv(PATHS["outputs_cls"] / "vit_history.csv")
figure, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(history["epoch"], history["train_loss"], label="train")
axes[0].plot(history["epoch"], history["val_loss"], label="validation")
axes[0].set_title("ViT loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[1].plot(history["epoch"], history["train_accuracy"], label="train")
axes[1].plot(history["epoch"], history["val_accuracy"], label="validation")
axes[1].set_title("ViT accuracy")
axes[1].set_xlabel("Epoch")
axes[1].legend()
figure.tight_layout()
figure.savefig(PATHS["plots"] / "vit_training_curves.png", dpi=120)
plt.show()
mark("vit_curves_ready")
''')

code(r'''
require("vit_curves_ready")
import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

def load_torch(path):
    try:
        return torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=DEVICE)

checkpoint = load_torch(VIT_CHECKPOINT)
VIT_MODEL = build_vit(len(checkpoint["class_to_idx"]))
VIT_MODEL.load_state_dict(checkpoint["model_state"])
VIT_MODEL.to(DEVICE).eval()
test_loader = make_loader(TEST_ROWS if not CFG["smoke_test"] else TEST_ROWS[: CFG["cls_batch_size"]], train=False, batch_size=CFG["cls_batch_size"])
y_true, y_pred, images_kept = [], [], []
with torch.no_grad():
    for images, labels in test_loader:
        logits = VIT_MODEL(images.to(DEVICE))
        preds = logits.argmax(dim=1).cpu()
        y_true.extend(labels.tolist())
        y_pred.extend(preds.tolist())
        images_kept.append(images.cpu())
y_true = np.asarray(y_true)
y_pred = np.asarray(y_pred)
names = [IDX_TO_CLASS[i] for i in range(len(IDX_TO_CLASS))]
test_accuracy = accuracy_score(y_true, y_pred)
test_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
test_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
test_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
print(classification_report(y_true, y_pred, target_names=names, zero_division=0))
matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(names))))
figure, axis = plt.subplots(figsize=(5, 4))
image = axis.imshow(matrix, cmap="Blues")
axis.set_xticks(range(len(names)), names, rotation=45, ha="right")
axis.set_yticks(range(len(names)), names)
axis.set_xlabel("Predicted")
axis.set_ylabel("Ground truth")
axis.set_title("ViT test confusion matrix")
for row in range(matrix.shape[0]):
    for col in range(matrix.shape[1]):
        axis.text(col, row, int(matrix[row, col]), ha="center", va="center")
figure.colorbar(image, ax=axis)
figure.tight_layout()
figure.savefig(PATHS["plots"] / "vit_confusion_matrix.png", dpi=120)
plt.show()

# Latency on the held-out crops, batch size 1, after one warmup.
latency_loader = make_loader(TEST_ROWS[:20], train=False, batch_size=1)
warmup = True
times = []
with torch.no_grad():
    for images, _labels in latency_loader:
        images = images.to(DEVICE)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        VIT_MODEL(images)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        if warmup:
            warmup = False
            continue
        times.append(time.perf_counter() - started)
latency_ms = float(np.mean(times) * 1000) if times else float("nan")
device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
print(f"ViT latency: {latency_ms:.2f} ms per crop on {device_name}, batch size 1, n={len(times)}")

from bccd.metrics_io import load_metrics, save_metrics
metrics = load_metrics(PATHS["outputs_cls"] / "vit_metrics.json") or {}
scope = "SMOKE TEST, not a full test evaluation. " if CFG["smoke_test"] else ""
metrics.update({
    "test_metrics": (
        f"{scope}accuracy={test_accuracy:.4f}, macro_precision={test_precision:.4f}, "
        f"macro_recall={test_recall:.4f}, macro_f1={test_f1:.4f}"
    ),
    "inference_latency": f"{latency_ms:.2f} ms per crop on {device_name}, batch size 1",
    "test_accuracy": test_accuracy,
    "test_precision_macro": test_precision,
    "test_recall_macro": test_recall,
    "test_f1_macro": test_f1,
})
save_metrics(PATHS["outputs_cls"] / "vit_metrics.json", metrics)

gallery = torch.cat(images_kept, dim=0)
correct = [i for i in range(len(y_true)) if y_true[i] == y_pred[i]][:4]
wrong = [i for i in range(len(y_true)) if y_true[i] != y_pred[i]][:4]
mean = torch.tensor(CFG["cls_mean"]).view(3, 1, 1)
std = torch.tensor(CFG["cls_std"]).view(3, 1, 1)

def show_group(indexes, title):
    if not indexes:
        print(title + ": none in this run.")
        return
    figure, axes = plt.subplots(1, len(indexes), figsize=(3 * len(indexes), 3))
    if len(indexes) == 1:
        axes = [axes]
    for axis, index in zip(axes, indexes):
        view = (gallery[index] * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        axis.imshow(view)
        axis.set_title(f"true {IDX_TO_CLASS[int(y_true[index])]}\npred {IDX_TO_CLASS[int(y_pred[index])]}")
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    plt.show()

show_group(correct, "Correct ViT predictions")
show_group(wrong, "Misclassified ViT predictions")
mark("vit_evaluated")
''')

md(r'''
### Pretrained ViT versus this implementation

The classes above are the architecture. Training them from random initialization teaches the patch embedding and the attention layers using only BCCD crops.

`torchvision.models.vit_b_16` is a larger encoder, with a 224 pixel input, that already learned visual features on ImageNet. Fine-tuning it means loading those weights and replacing the final linear layer. That comparison is useful, and it is not a substitute for being able to explain the implementation. Leave `CFG["run_pretrained_vit"]` false unless you want to spend the extra GPU time and the ImageNet download.
''')

code(r'''
require("vit_evaluated")
from bccd.metrics_io import save_metrics

if not CFG["run_pretrained_vit"]:
    save_metrics(PATHS["outputs_cls"] / "pretrained_vit_metrics.json", {
        "model": "torchvision ViT-B/16 fine-tune",
        "training_configuration": "disabled because CFG['run_pretrained_vit'] is False",
        "validation_metrics": "not trained",
        "test_metrics": "not trained",
        "training_duration": "not trained",
        "inference_latency": "not trained",
        "limitations": "Enable CFG['run_pretrained_vit'] to fine-tune ImageNet weights. This is separate from the from-scratch ViT.",
    })
    print("Skipped pretrained ViT fine-tuning.")
else:
    from torchvision.models import ViT_B_16_Weights, vit_b_16
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    pretrained = vit_b_16(weights=weights)
    pretrained.heads.head = nn.Linear(pretrained.heads.head.in_features, len(CLASS_TO_IDX))
    # This loader uses the 224 px ImageNet preprocessing that belongs to these weights.
    preprocess = weights.transforms()

    class PretrainedCropDataset(Dataset):
        def __init__(self, rows):
            self.rows = list(rows)
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, index):
            row = self.rows[index]
            with Image.open(row["crop_path"]) as image:
                tensor = preprocess(image.convert("RGB"))
            return tensor, int(row["label"])

    original_size = CFG["cls_image_size"]
    original_batch = CFG["cls_batch_size"]
    CFG["cls_batch_size"] = min(original_batch, 16)
    # fit_classifier calls make_loader, which uses make_cls_transform. For this
    # comparison, temporarily swap the crop dataset transform by wrapping rows
    # through a local loader inside a dedicated loop copy is avoided: we call
    # fit_classifier only for the scratch model. Here we reuse run_epoch.
    def loader_for(rows, train):
        return DataLoader(PretrainedCropDataset(rows), batch_size=CFG["cls_batch_size"], shuffle=train, num_workers=CFG["num_workers"])
    labels = [row["label"] for row in TRAIN_ROWS]
    counts = np.bincount(labels, minlength=len(CLASS_TO_IDX)).astype(np.float32)
    class_weights = torch.tensor(counts.sum() / np.maximum(counts, 1.0), dtype=torch.float32, device=DEVICE)
    class_weights = class_weights / class_weights.mean()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(pretrained.parameters(), lr=1e-4, weight_decay=CFG["cls_weight_decay"])
    pretrained.to(DEVICE)
    best_acc = -1.0
    best_state = None
    started = time.time()
    epochs = 1 if CFG["smoke_test"] else min(CFG["cls_epochs"], 10)
    for epoch in range(epochs):
        train_loss, train_acc = run_epoch(pretrained, loader_for(TRAIN_ROWS, True), criterion, optimizer, CFG["max_train_batches"])
        val_loss, val_acc = run_epoch(pretrained, loader_for(VAL_ROWS, False), criterion, None, CFG["max_val_batches"])
        print(f"Pretrained ViT epoch {epoch + 1}/{epochs} val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in pretrained.state_dict().items()}
    pretrained.load_state_dict(best_state)
    torch.save({"model_state": best_state, "class_to_idx": CLASS_TO_IDX, "source": "torchvision vit_b_16 IMAGENET1K_V1"}, PATHS["checkpoints"] / "vit_b16_finetune.pt")
    save_metrics(PATHS["outputs_cls"] / "pretrained_vit_metrics.json", {
        "model": "torchvision ViT-B/16 fine-tune",
        "smoke_test": CFG["smoke_test"],
        "training_configuration": f"ImageNet init, replaced head, lr 1e-4, epochs {epochs}, smoke_test={CFG['smoke_test']}",
        "validation_metrics": f"best val accuracy={best_acc:.4f}",
        "test_metrics": "not computed in this optional cell; use the from-scratch test cell as the primary result",
        "training_duration": f"{(time.time() - started) / 60:.2f} min",
        "inference_latency": "not measured",
        "limitations": "Fine-tuning uses external ImageNet features. It is not the architecture implemented above.",
    })
    CFG["cls_image_size"] = original_size
    CFG["cls_batch_size"] = original_batch
    print("Saved pretrained ViT fine-tune checkpoint.")
mark("pretrained_vit_ready")
''')

code(r'''
require("pretrained_vit_ready")


class SmallCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.head(x)


if not CFG["run_cnn_baseline"]:
    save_metrics(PATHS["outputs_cls"] / "cnn_metrics.json", {
        "model": "Small CNN baseline",
        "training_configuration": "disabled because CFG['run_cnn_baseline'] is False",
        "validation_metrics": "not trained",
        "test_metrics": "not trained",
        "training_duration": "not trained",
        "inference_latency": "not trained",
        "limitations": "Enable CFG['run_cnn_baseline'] to train this CNN on the same crops and splits as the ViT.",
    })
    print("Skipped CNN baseline. Set CFG['run_cnn_baseline'] = True to train it.")
else:
    cnn = SmallCNN(len(CLASS_TO_IDX))
    cnn_seconds_started = time.time()
    history, best_acc, best_epoch, seconds = fit_classifier(
        cnn, TRAIN_ROWS, VAL_ROWS,
        epochs=CFG["cls_epochs"], lr=1e-3, weight_decay=1e-4, patience=CFG["cls_patience"],
        checkpoint_path=PATHS["checkpoints"] / "cnn_bccd_best.pt",
        max_train_batches=CFG["max_train_batches"], max_val_batches=CFG["max_val_batches"],
    )
    save_metrics(PATHS["outputs_cls"] / "cnn_metrics.json", {
        "model": "Small CNN baseline",
        "smoke_test": CFG["smoke_test"],
        "training_configuration": f"3 conv blocks, AdamW lr 1e-3, same crops and early stopping, smoke_test={CFG['smoke_test']}",
        "validation_metrics": f"best val accuracy={best_acc:.4f} at epoch {best_epoch + 1}",
        "test_metrics": "not yet run" if not CFG["smoke_test"] else "smoke test did not run a separate CNN test pass",
        "training_duration": f"{seconds / 60:.2f} min",
        "inference_latency": "not measured",
        "limitations": "A small CNN has a locality bias that often helps on tiny medical sets. Compare validation accuracy with the ViT before drawing a conclusion.",
    })
    print(f"CNN baseline best validation accuracy: {best_acc:.4f}")
mark("cnn_ready")
''')

md(r'''
## Section 9: Transformer object detection with DETR

DETR treats detection as set prediction. A CNN backbone, here ResNet-50, builds a feature map. A transformer encoder mixes that map. A decoder attends from a fixed set of object queries to the encoder memory. Each query is decoded by two heads: a linear classifier, including a no-object class, and a small MLP that outputs one box as normalized `cx, cy, w, h`.

Training needs a one-to-one match between predictions and ground-truth boxes, because the queries are not tied to anchors. Hungarian matching solves that assignment. Matched queries take cross-entropy on the class and an L1 plus generalized IoU loss on the box. Unmatched queries are trained as no-object. BCCD labels use ids 0, 1, and 2. The extra logit is the no-object class inside the model, not a fourth dataset class.

The training cell fine-tunes `facebook/detr-resnet-50`. Those weights were trained on COCO. The classification layer is reinitialized for the BCCD classes (`ignore_mismatched_sizes=True`). The backbone and the transformer start from the COCO checkpoint. That is pretrained initialization, followed by training on BCCD. If the checkpoint cannot be downloaded and `CFG["allow_detr_from_scratch"]` is true, the same DETR architecture is built from `DetrConfig` and the metrics are labeled as random initialization. The notebook does not swap in a CNN detector.

The cell below is a readable Hungarian matcher and a numerical check. The training loss itself is the criterion inside `DetrForObjectDetection`, which runs the same matching when `labels` are passed.
''')

code(r'''
require("cnn_ready")
import importlib.util
import inspect
import subprocess

for module_name, pip_name in (("transformers", "transformers"), ("torchmetrics", "torchmetrics"), ("scipy", "scipy")):
    if importlib.util.find_spec(module_name) is None:
        print(f"Installing {pip_name}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])

from scipy.optimize import linear_sum_assignment


def box_cxcywh_to_xyxy(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack((cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h), dim=-1)


def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)


def generalized_box_iou(boxes1, boxes2):
    # boxes are xyxy. Returns an (N, M) GIoU matrix.
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    iou = inter / union.clamp(min=1e-6)
    enclose_lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    enclose_rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    enclose = (enclose_rb - enclose_lt).clamp(min=0)
    enclose_area = enclose[:, :, 0] * enclose[:, :, 1]
    return iou - (enclose_area - union) / enclose_area.clamp(min=1e-6)


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=1.0, cost_bbox=5.0, cost_giou=2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, pred_logits, pred_boxes, targets):
        # pred_logits: (B, queries, classes + 1), pred_boxes: (B, queries, 4) cxcywh
        out_prob = pred_logits.softmax(-1)
        matches = []
        for batch_index in range(pred_logits.shape[0]):
            target_ids = targets[batch_index]["class_labels"]
            target_boxes = targets[batch_index]["boxes"]
            if target_ids.numel() == 0:
                empty = torch.zeros(0, dtype=torch.int64)
                matches.append((empty, empty))
                continue
            cost_class = -out_prob[batch_index][:, target_ids]
            cost_bbox = torch.cdist(pred_boxes[batch_index], target_boxes, p=1)
            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(pred_boxes[batch_index]),
                box_cxcywh_to_xyxy(target_boxes),
            )
            cost = self.cost_class * cost_class + self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
            row, col = linear_sum_assignment(cost.detach().cpu().numpy())
            matches.append((torch.as_tensor(row, dtype=torch.int64), torch.as_tensor(col, dtype=torch.int64)))
        return matches


_matcher = HungarianMatcher()
_logits = torch.tensor([[[5.0, 0.0, 0.0, 0.0], [0.0, 5.0, 0.0, 0.0], [0.0, 0.0, 0.0, 5.0]]])
_boxes = torch.tensor([[[0.25, 0.25, 0.2, 0.2], [0.75, 0.75, 0.2, 0.2], [0.5, 0.5, 0.1, 0.1]]])
_targets = [{
    "class_labels": torch.tensor([0, 1]),
    "boxes": torch.tensor([[0.25, 0.25, 0.2, 0.2], [0.75, 0.75, 0.2, 0.2]]),
}]
_src, _tgt = _matcher(_logits, _boxes, _targets)[0]
print("Hungarian assignment, queries -> targets:", list(zip(_src.tolist(), _tgt.tolist())))
if set(_src.tolist()) != {0, 1}:
    raise AssertionError("The matcher did not select the two queries that explain the two targets.")
mark("matcher_ready")
''')

code(r'''
require("matcher_ready")
from transformers import DetrForObjectDetection, DetrImageProcessor

try:
    from transformers import DetrConfig
except ImportError as exc:
    raise RuntimeError("transformers is installed but DetrConfig could not be imported.") from exc

print("DetrForObjectDetection.forward signature:")
print(inspect.signature(DetrForObjectDetection.forward))
ID2LABEL = {index: name for name, index in CLASS_TO_IDX.items()}
LABEL2ID = {name: index for index, name in ID2LABEL.items()}
DETR_SIZE = {"shortest_edge": CFG["detr_shortest_edge"], "longest_edge": CFG["detr_longest_edge"]}
try:
    DETR_PROCESSOR = DetrImageProcessor.from_pretrained(CFG["detr_model_name"], size=DETR_SIZE)
except TypeError:
    DETR_PROCESSOR = DetrImageProcessor.from_pretrained(CFG["detr_model_name"])
    DETR_PROCESSOR.size = DETR_SIZE

DETR_INIT_MODE = "coco_pretrained_classification_head_reinitialized"
try:
    DETR_MODEL = DetrForObjectDetection.from_pretrained(
        CFG["detr_model_name"],
        num_labels=len(CLASS_TO_IDX),
        ignore_mismatched_sizes=True,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
except Exception as exc:
    if not CFG["allow_detr_from_scratch"]:
        raise RuntimeError(
            "Could not load the COCO-pretrained DETR checkpoint. "
            "Check the network connection, or set CFG['allow_detr_from_scratch'] = True to train the "
            "same DETR architecture from random weights. A CNN detector will not be substituted.\n"
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc
    config = DetrConfig(num_labels=len(CLASS_TO_IDX), id2label=ID2LABEL, label2id=LABEL2ID)
    DETR_MODEL = DetrForObjectDetection(config)
    DETR_INIT_MODE = "random_detr_config_not_coco_pretrained"
    print("Training DETR from random initialization. Do not describe this run as COCO fine-tuning.")

head = getattr(DETR_MODEL, "class_labels_classifier", None)
if head is None:
    raise RuntimeError("This transformers build has no class_labels_classifier. Inspect the model before training.")
if head.out_features != len(CLASS_TO_IDX) + 1:
    raise RuntimeError(
        f"Expected {len(CLASS_TO_IDX) + 1} class logits (dataset classes plus no-object), got {head.out_features}."
    )
print("DETR init:", DETR_INIT_MODE)
print("No-object logit index:", len(CLASS_TO_IDX))
print("Processor size:", DETR_PROCESSOR.size)
mark("detr_built")
''')

code(r'''
require("detr_built")


class BCCDDetectionDataset(Dataset):
    def __init__(self, records, augment):
        self.records = list(records)
        self.augment = augment
        self.color = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(record["image_path"]) as image:
            image = image.convert("RGB")
        boxes = np.asarray(record["boxes_xyxy"], dtype=np.float32).reshape(-1, 4)
        labels = np.asarray(record["labels"], dtype=np.int64)
        if self.augment and random.random() < 0.5:
            image = transforms.functional.hflip(image)
            boxes = hflip_xyxy(boxes, image.width)
        if self.augment:
            image = self.color(image)
        width, height = image.size
        boxes_cxcywh = xyxy_to_cxcywh_norm(boxes, width, height)
        target = {
            "class_labels": torch.tensor(labels, dtype=torch.long),
            "boxes": torch.tensor(boxes_cxcywh, dtype=torch.float32),
            "orig_size": torch.tensor([height, width], dtype=torch.int64),
            "image_path": record["image_path"],
        }
        return image, target


def collate_detections(batch):
    images, targets = zip(*batch)
    encoding = DETR_PROCESSOR(images=list(images), return_tensors="pt")
    labels = [{"class_labels": target["class_labels"], "boxes": target["boxes"]} for target in targets]
    return {
        "pixel_values": encoding["pixel_values"],
        "pixel_mask": encoding.get("pixel_mask"),
        "labels": labels,
        "orig_sizes": torch.stack([target["orig_size"] for target in targets]),
        "paths": [target["image_path"] for target in targets],
    }


def detection_loader(split, train):
    rows = [record for record in DETECTION_RECORDS if record["split"] == split]
    if not rows:
        raise RuntimeError(f"No detection records in the {split} split.")
    return DataLoader(
        BCCDDetectionDataset(rows, augment=train),
        batch_size=CFG["detr_batch_size"],
        shuffle=train,
        num_workers=CFG["num_workers"],
        collate_fn=collate_detections,
        pin_memory=DEVICE.type == "cuda",
    )

print("Detection images:", {split: sum(r["split"] == split for r in DETECTION_RECORDS) for split in ("train", "val", "test")})
mark("detr_loaders_ready")
''')

code(r'''
require("detr_loaders_ready")

def detr_forward(model, batch):
    labels = [{key: value.to(DEVICE) for key, value in target.items()} for target in batch["labels"]]
    kwargs = {"pixel_values": batch["pixel_values"].to(DEVICE), "labels": labels}
    if batch["pixel_mask"] is not None:
        kwargs["pixel_mask"] = batch["pixel_mask"].to(DEVICE)
    return model(**kwargs)


def detr_epoch(model, loader, optimizer, max_batches):
    training = optimizer is not None
    model.train(training)
    totals = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for step, batch in enumerate(tqdm(loader, leave=False)):
            if max_batches is not None and step >= max_batches:
                break
            if training:
                optimizer.zero_grad(set_to_none=True)
            outputs = detr_forward(model, batch)
            loss = outputs.loss
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["detr_grad_clip"])
                optimizer.step()
            loss_terms = {}
            if getattr(outputs, "loss_dict", None) is not None:
                loss_terms = {key: float(value.detach().cpu()) for key, value in outputs.loss_dict.items()}
            totals.append((float(loss.detach().cpu()), loss_terms))
            if step == 0:
                print("DETR loss terms on the first batch:", loss_terms or {"loss": totals[-1][0]})
    if not totals:
        raise RuntimeError("DETR loader produced no batches.")
    return float(np.mean([item[0] for item in totals]))


backbone_params = [p for name, p in DETR_MODEL.named_parameters() if "backbone" in name and p.requires_grad]
other_params = [p for name, p in DETR_MODEL.named_parameters() if "backbone" not in name and p.requires_grad]
param_groups = []
if other_params:
    param_groups.append({"params": other_params, "lr": CFG["detr_lr"]})
if backbone_params:
    param_groups.append({"params": backbone_params, "lr": CFG["detr_lr_backbone"]})
if not param_groups:
    raise RuntimeError("DETR has no trainable parameters.")
DETR_OPTIMIZER = torch.optim.AdamW(param_groups, lr=CFG["detr_lr"], weight_decay=CFG["detr_weight_decay"])
DETR_SCHEDULER = torch.optim.lr_scheduler.CosineAnnealingLR(DETR_OPTIMIZER, T_max=max(CFG["detr_epochs"], 1))
DETR_MODEL.to(DEVICE)

DETR_CHECKPOINT = PATHS["checkpoints"] / "detr_bccd_best.pt"
DETR_EXPORT = PATHS["checkpoints"] / "detr_bccd_best"
start_epoch = 0
best_val = float("inf")
if CFG["resume_detr"] and DETR_CHECKPOINT.exists():
    saved = load_torch(DETR_CHECKPOINT)
    DETR_MODEL.load_state_dict(saved["model_state"])
    DETR_OPTIMIZER.load_state_dict(saved["optimizer_state"])
    start_epoch = int(saved["epoch"]) + 1
    best_val = float(saved.get("best_val_loss", saved.get("val_loss", float("inf"))))
    print(f"Resumed DETR from epoch {saved['epoch'] + 1}. Next epoch is {start_epoch + 1}.")

train_loader = detection_loader("train", train=True)
val_loader = detection_loader("val", train=False)
detr_history = []
best_epoch = -1
waits = CFG["detr_patience"]
train_started = time.time()
if start_epoch >= CFG["detr_epochs"]:
    print("The saved DETR checkpoint already completed the configured epochs. Training is skipped.")
else:
    for epoch in range(start_epoch, CFG["detr_epochs"]):
        epoch_started = time.time()
        train_loss = detr_epoch(DETR_MODEL, train_loader, DETR_OPTIMIZER, CFG["max_train_batches"])
        val_loss = detr_epoch(DETR_MODEL, val_loader, None, CFG["max_val_batches"])
        DETR_SCHEDULER.step()
        detr_history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(
            f"DETR epoch {epoch + 1}/{CFG['detr_epochs']} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} minutes={((time.time() - epoch_started) / 60):.2f}"
        )
        payload = {
            "model_state": DETR_MODEL.state_dict(),
            "optimizer_state": DETR_OPTIMIZER.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "best_val_loss": min(best_val, val_loss),
            "class_to_idx": CLASS_TO_IDX,
            "init_mode": DETR_INIT_MODE,
            "config": CFG,
        }
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            waits = CFG["detr_patience"]
            torch.save(payload, DETR_CHECKPOINT)
            DETR_MODEL.save_pretrained(DETR_EXPORT)
            DETR_PROCESSOR.save_pretrained(DETR_EXPORT)
            (DETR_EXPORT / "bccd_meta.json").write_text(json.dumps({
                "class_to_idx": CLASS_TO_IDX,
                "init_mode": DETR_INIT_MODE,
                "smoke_test": CFG["smoke_test"],
            }, indent=2), encoding="utf-8")
        else:
            waits -= 1
            if waits <= 0:
                print("Early stopping DETR on validation loss.")
                break
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

if not DETR_CHECKPOINT.exists():
    raise RuntimeError("DETR training did not write a checkpoint.")
pd.DataFrame(detr_history).to_csv(PATHS["outputs_det"] / "detr_history.csv", index=False)
save_metrics(PATHS["outputs_det"] / "detr_metrics.json", {
    "model": "DETR ResNet-50",
    "smoke_test": CFG["smoke_test"],
    "training_configuration": (
        f"{DETR_INIT_MODE}, epochs {CFG['detr_epochs']}, batch {CFG['detr_batch_size']}, "
        f"lr {CFG['detr_lr']}, backbone lr {CFG['detr_lr_backbone']}, "
        f"size {CFG['detr_shortest_edge']}/{CFG['detr_longest_edge']}, smoke_test={CFG['smoke_test']}"
    ),
    "validation_metrics": f"best val loss={best_val:.4f} at epoch {best_epoch + 1}" if best_epoch >= 0 else f"resumed, stored best val loss={best_val:.4f}",
    "test_metrics": "not yet run",
    "training_duration": f"{(time.time() - train_started) / 60:.2f} min on {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
    "inference_latency": "not yet run",
    "limitations": (
        "Smoke-test metrics are not full-training results. "
        if CFG["smoke_test"] else
        "COCO pretraining is not medical pretraining. BCCD has few fields and many touching red cells, so box metrics move slowly."
    ),
    "init_mode": DETR_INIT_MODE,
})
print("DETR checkpoint:", DETR_CHECKPOINT)
mark("detr_trained")
''')

md(r'''
## Section 10: Evaluate object detection

TorchMetrics `MeanAveragePrecision` expects absolute `xyxy` boxes. Predictions come from the DETR post-processor, which converts normalized `cxcywh` back to pixel corners on the original image. Ground truth uses the same original `xyxy` corners stored in the manifest.

`map_50` is mAP at IoU 0.5. `map` is mAP averaged over IoU thresholds from 0.5 to 0.95. Precision and recall below use confidence 0.5 and a greedy IoU match of 0.5. Latency is batch size 1, with a warmup prediction excluded, and CUDA synchronization when a GPU is present.
''')

code(r'''
require("detr_trained")
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def pairwise_iou(boxes_a, boxes_b):
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    a = np.asarray(boxes_a, dtype=np.float32)
    b = np.asarray(boxes_b, dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, a_min=0, a_max=None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-6, None)


def postprocess_batch(outputs, orig_sizes, threshold):
    sizes = orig_sizes.cpu()
    try:
        return DETR_PROCESSOR.post_process_object_detection(outputs, target_sizes=sizes, threshold=threshold)
    except TypeError:
        raw = DETR_PROCESSOR.post_process_object_detection(outputs, target_sizes=sizes)
        kept = []
        for result in raw:
            mask = result["scores"] >= threshold
            kept.append({key: value[mask] for key, value in result.items()})
        return kept


def collect_predictions():
    rows = [record for record in DETECTION_RECORDS if record["split"] == "test"]
    if CFG["smoke_test"]:
        rows = rows[:2]
    loader = DataLoader(BCCDDetectionDataset(rows, augment=False), batch_size=1, shuffle=False, collate_fn=collate_detections)
    predictions, targets, paths = [], [], []
    DETR_MODEL.eval()
    with torch.no_grad():
        for batch in loader:
            kwargs = {"pixel_values": batch["pixel_values"].to(DEVICE)}
            if batch["pixel_mask"] is not None:
                kwargs["pixel_mask"] = batch["pixel_mask"].to(DEVICE)
            outputs = DETR_MODEL(**kwargs)
            processed = postprocess_batch(outputs, batch["orig_sizes"], threshold=0.0)
            for result, target, path in zip(processed, batch["labels"], batch["paths"]):
                predictions.append({
                    "boxes": result["boxes"].detach().cpu(),
                    "scores": result["scores"].detach().cpu(),
                    "labels": result["labels"].detach().cpu(),
                })
                gt_record = next(item for item in rows if item["image_path"] == path)
                gt_boxes = torch.tensor(gt_record["boxes_xyxy"], dtype=torch.float32).reshape(-1, 4)
                gt_labels = torch.tensor(gt_record["labels"], dtype=torch.int64)
                targets.append({"boxes": gt_boxes, "labels": gt_labels})
                paths.append(path)
    return predictions, targets, paths


best_detr = load_torch(DETR_CHECKPOINT)
DETR_MODEL.load_state_dict(best_detr["model_state"])
DETR_MODEL.to(DEVICE).eval()
PREDICTIONS, DET_TARGETS, DET_PATHS = collect_predictions()
map_metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
map_metric.update(PREDICTIONS, DET_TARGETS)
map_scores = map_metric.compute()


def as_float(value):
    if hasattr(value, "numel") and value.numel() == 1:
        return float(value.detach().cpu())
    return value


map_50 = as_float(map_scores["map_50"])
map_5095 = as_float(map_scores["map"])
per_class = {}
if "map_per_class" in map_scores and "classes" in map_scores:
    for class_id, score in zip(map_scores["classes"].tolist(), map_scores["map_per_class"].tolist()):
        per_class[ID2LABEL[int(class_id)]] = None if score != score else float(score)
print(f"mAP@0.5 = {map_50:.4f}")
print(f"mAP@0.5:0.95 = {map_5095:.4f}")
print("Per-class AP:", per_class)

tp = fp = fn = 0
per_class_counts = {name: {"tp": 0, "fp": 0, "fn": 0} for name in ID2LABEL.values()}
for prediction, target in zip(PREDICTIONS, DET_TARGETS):
    scores = prediction["scores"].numpy()
    labels = prediction["labels"].numpy()
    boxes = prediction["boxes"].numpy()
    gt_boxes = target["boxes"].numpy()
    gt_labels = target["labels"].numpy()
    order = np.argsort(-scores)
    matched = set()
    for index in order:
        if scores[index] < CFG["detr_confidence_threshold"]:
            continue
        name = ID2LABEL[int(labels[index])]
        ious = pairwise_iou(boxes[index:index + 1], gt_boxes).reshape(-1) if len(gt_boxes) else np.zeros(0)
        best_j = None
        best_iou = 0.5
        for j, iou in enumerate(ious):
            if j in matched or int(gt_labels[j]) != int(labels[index]):
                continue
            if iou >= best_iou:
                best_iou = float(iou)
                best_j = j
        if best_j is None:
            fp += 1
            per_class_counts[name]["fp"] += 1
        else:
            tp += 1
            matched.add(best_j)
            per_class_counts[name]["tp"] += 1
    for j, label in enumerate(gt_labels):
        if j not in matched:
            fn += 1
            per_class_counts[ID2LABEL[int(label)]]["fn"] += 1

precision = tp / (tp + fp) if (tp + fp) else 0.0
recall = tp / (tp + fn) if (tp + fn) else 0.0
print(f"Precision@{CFG['detr_confidence_threshold']:.2f}, IoU 0.5: {precision:.4f}")
print(f"Recall@{CFG['detr_confidence_threshold']:.2f}, IoU 0.5: {recall:.4f}")
print("Per-class TP/FP/FN:", per_class_counts)

latency_rows = [record for record in DETECTION_RECORDS if record["split"] == "test"][:20]
latency_loader = DataLoader(BCCDDetectionDataset(latency_rows, augment=False), batch_size=1, shuffle=False, collate_fn=collate_detections)
detr_times = []
warmup = True
DETR_MODEL.eval()
with torch.no_grad():
    for batch in latency_loader:
        kwargs = {"pixel_values": batch["pixel_values"].to(DEVICE)}
        if batch["pixel_mask"] is not None:
            kwargs["pixel_mask"] = batch["pixel_mask"].to(DEVICE)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        outputs = DETR_MODEL(**kwargs)
        postprocess_batch(outputs, batch["orig_sizes"], CFG["detr_confidence_threshold"])
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        if warmup:
            warmup = False
            continue
        detr_times.append(time.perf_counter() - started)
detr_latency_ms = float(np.mean(detr_times) * 1000) if detr_times else float("nan")
device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
print(
    f"DETR latency: {detr_latency_ms:.2f} ms per image on {device_name}, "
    f"batch size 1, shortest edge {CFG['detr_shortest_edge']}, n={len(detr_times)}"
)

from bccd.metrics_io import load_metrics
detr_metrics = load_metrics(PATHS["outputs_det"] / "detr_metrics.json") or {}
scope = "SMOKE TEST subset, not the full test set. " if CFG["smoke_test"] else ""
detr_metrics.update({
    "test_metrics": (
        f"{scope}mAP@0.5={map_50:.4f}, mAP@0.5:0.95={map_5095:.4f}, "
        f"precision={precision:.4f}, recall={recall:.4f} at confidence {CFG['detr_confidence_threshold']}"
    ),
    "inference_latency": (
        f"{detr_latency_ms:.2f} ms per image on {device_name}, batch size 1, "
        f"shortest edge {CFG['detr_shortest_edge']}"
    ),
    "map_50": map_50,
    "map_50_95": map_5095,
    "per_class_ap": per_class,
    "precision_at_threshold": precision,
    "recall_at_threshold": recall,
})
save_metrics(PATHS["outputs_det"] / "detr_metrics.json", detr_metrics)
mark("detr_evaluated")
''')

code(r'''
require("detr_evaluated")

def classify_image(prediction, target):
    ious = pairwise_iou(prediction["boxes"].numpy(), target["boxes"].numpy())
    scores = prediction["scores"].numpy()
    has_tp = False
    has_fp = False
    matched = set()
    for index in np.argsort(-scores):
        if scores[index] < CFG["detr_confidence_threshold"]:
            continue
        candidates = []
        if ious.shape[1]:
            candidates = [
                (ious[index, j], j) for j in range(ious.shape[1])
                if int(prediction["labels"][index]) == int(target["labels"][j])
            ]
        if candidates and max(candidates)[0] >= 0.5:
            has_tp = True
            matched.add(max(candidates)[1])
        else:
            has_fp = True
    has_fn = len(matched) < len(target["labels"])
    gt = target["boxes"].numpy()
    crowded = False
    if len(gt) >= 2:
        gt_iou = pairwise_iou(gt, gt)
        np.fill_diagonal(gt_iou, 0)
        crowded = bool(gt_iou.max() >= 0.15)
    return has_tp, has_fp, has_fn, crowded


picked = {"correct": None, "false_positive": None, "missed": None, "overlap": None}
for prediction, target, path in zip(PREDICTIONS, DET_TARGETS, DET_PATHS):
    has_tp, has_fp, has_fn, crowded = classify_image(prediction, target)
    if has_tp and picked["correct"] is None:
        picked["correct"] = (prediction, target, path)
    if has_fp and picked["false_positive"] is None:
        picked["false_positive"] = (prediction, target, path)
    if has_fn and picked["missed"] is None:
        picked["missed"] = (prediction, target, path)
    if crowded and picked["overlap"] is None:
        picked["overlap"] = (prediction, target, path)

shown = [(name, item) for name, item in picked.items() if item is not None]
if not shown:
    print("No test predictions were available to draw.")
else:
    figure, axes = plt.subplots(len(shown), 1, figsize=(8, 4 * len(shown)))
    if len(shown) == 1:
        axes = [axes]
    for axis, (name, (prediction, target, path)) in zip(axes, shown):
        with Image.open(path) as image:
            axis.imshow(image.convert("RGB"))
        for box, label in zip(target["boxes"].numpy(), target["labels"].numpy()):
            x1, y1, x2, y2 = box
            axis.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="#2ca02c", linewidth=1.2))
        for box, score, label in zip(prediction["boxes"].numpy(), prediction["scores"].numpy(), prediction["labels"].numpy()):
            if score < CFG["detr_confidence_threshold"]:
                continue
            x1, y1, x2, y2 = box
            axis.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="#d62728", linewidth=1.2))
            axis.text(x1, y1, f"{ID2LABEL[int(label)]} {score:.2f}", color="white", fontsize=7,
                      bbox={"facecolor": "#d62728", "alpha": 0.7, "pad": 0.1})
        axis.set_title(f"{name}: green = ground truth, red = prediction")
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(PATHS["plots"] / "detr_qualitative.png", dpi=120)
    plt.show()
mark("detr_visualized")
''')

md(r'''
## Section 11: Inference demo

`predict_cell` accepts a file path, a PIL image, or an array, and returns a class plus a confidence score. `predict_detections` accepts a full microscopy image and returns boxes in absolute `xyxy` pixels, class names, and scores. `draw_detections` paints those boxes. Both functions work without the upload widget.
''')

code(r'''
require("detr_visualized")
from PIL import ImageDraw


def _as_pil(image):
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            return opened.convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    return image.convert("RGB")


def predict_cell(image, model=None, class_to_idx=None, device=None, image_size=None):
    model = VIT_MODEL if model is None else model
    class_to_idx = CLASS_TO_IDX if class_to_idx is None else class_to_idx
    device = DEVICE if device is None else device
    image_size = CFG["cls_image_size"] if image_size is None else image_size
    index_to_class = {index: name for name, index in class_to_idx.items()}
    tensor = make_cls_transform(train=False)(_as_pil(image)).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=-1)[0]
    score, index = torch.max(probabilities, dim=0)
    return {
        "class_name": index_to_class[int(index)],
        "confidence": float(score),
        "probabilities": {index_to_class[i]: float(probabilities[i]) for i in range(len(probabilities))},
    }


def predict_detections(image, model=None, processor=None, device=None, threshold=None, id2label=None):
    model = DETR_MODEL if model is None else model
    processor = DETR_PROCESSOR if processor is None else processor
    device = DEVICE if device is None else device
    threshold = CFG["detr_confidence_threshold"] if threshold is None else threshold
    id2label = ID2LABEL if id2label is None else id2label
    pil_image = _as_pil(image)
    width, height = pil_image.size
    encoding = processor(images=pil_image, return_tensors="pt")
    kwargs = {"pixel_values": encoding["pixel_values"].to(device)}
    if encoding.get("pixel_mask") is not None:
        kwargs["pixel_mask"] = encoding["pixel_mask"].to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(**kwargs)
    try:
        result = processor.post_process_object_detection(
            outputs, target_sizes=torch.tensor([[height, width]]), threshold=threshold
        )[0]
    except TypeError:
        result = processor.post_process_object_detection(outputs, target_sizes=torch.tensor([[height, width]]))[0]
        mask = result["scores"] >= threshold
        result = {key: value[mask] for key, value in result.items()}
    detections = []
    for box, score, label in zip(result["boxes"].cpu(), result["scores"].cpu(), result["labels"].cpu()):
        detections.append({
            "box_xyxy": [float(v) for v in box.tolist()],
            "class_name": id2label[int(label)],
            "confidence": float(score),
        })
    return {"detections": detections, "image_size": [width, height]}


def draw_detections(image, detections):
    canvas = _as_pil(image)
    pen = ImageDraw.Draw(canvas)
    for det in detections:
        x1, y1, x2, y2 = det["box_xyxy"]
        pen.rectangle([x1, y1, x2, y2], outline=(214, 39, 40), width=2)
        pen.text((x1, max(0, y1 - 12)), f"{det['class_name']} {det['confidence']:.2f}", fill=(214, 39, 40))
    return canvas


demo_crop = TEST_ROWS[0]["crop_path"]
demo_image = next(record["image_path"] for record in DETECTION_RECORDS if record["split"] == "test")
print("Classifier demo:", predict_cell(demo_crop))
demo_result = predict_detections(demo_image)
print(f"Detector demo: {len(demo_result['detections'])} boxes on {Path(demo_image).name}")
plt.figure(figsize=(7, 5))
plt.imshow(draw_detections(demo_image, demo_result["detections"]))
plt.axis("off")
plt.title("DETR predictions on one test image")
plt.show()
mark("inference_ready")
''')

code(r'''
require("inference_ready")


def launch_upload_demo():
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except Exception:
        print("ipywidgets is not available. Call predict_cell(path) or predict_detections(path) directly.")
        return

    def _uploaded_image(new_value):
        if not new_value:
            return None
        if isinstance(new_value, dict):
            name, payload = next(iter(new_value.items()))
            content = payload["content"]
        else:
            item = new_value[0]
            if isinstance(item, dict):
                name, content = item.get("name", "upload"), item["content"]
            else:
                name, content = item.name, bytes(item.content)
        return Image.open(io.BytesIO(content)).convert("RGB"), name

    def on_classify(change):
        loaded = _uploaded_image(change["new"])
        if loaded is None:
            return
        image, name = loaded
        print(name, predict_cell(image))

    def on_detect(change):
        loaded = _uploaded_image(change["new"])
        if loaded is None:
            return
        image, name = loaded
        result = predict_detections(image)
        print(name, result["detections"][:10])
        plt.figure(figsize=(7, 5))
        plt.imshow(draw_detections(image, result["detections"]))
        plt.axis("off")
        plt.show()

    classify_upload = widgets.FileUpload(accept="image/*", multiple=False)
    detect_upload = widgets.FileUpload(accept="image/*", multiple=False)
    classify_upload.observe(on_classify, names="value")
    detect_upload.observe(on_detect, names="value")
    display(widgets.VBox([
        widgets.HTML("<b>Classify one cell crop</b>"),
        classify_upload,
        widgets.HTML("<b>Detect cells in a full field</b>"),
        detect_upload,
    ]))


import io
launch_upload_demo()
mark("upload_demo_ready")
''')

md(r'''
## Section 12: Save and load models

The training cells already wrote the best checkpoints. This section loads those files into new modules and runs one prediction with each, which checks that the saved weights, the class map, and the processor config are sufficient without the training objects.
''')

code(r'''
require("upload_demo_ready")
if not VIT_CHECKPOINT.exists() or not DETR_EXPORT.exists():
    raise RuntimeError("Best checkpoints are missing. Run the ViT and DETR training cells first.")

vit_payload = load_torch(VIT_CHECKPOINT)
reloaded_vit = build_vit(len(vit_payload["class_to_idx"]))
reloaded_vit.load_state_dict(vit_payload["model_state"])
reloaded_vit.to(DEVICE).eval()
reloaded_class_to_idx = vit_payload["class_to_idx"]
vit_check = predict_cell(TEST_ROWS[0]["crop_path"], model=reloaded_vit, class_to_idx=reloaded_class_to_idx)
print("Reloaded ViT prediction:", vit_check)

reloaded_detr = DetrForObjectDetection.from_pretrained(DETR_EXPORT)
reloaded_processor = DetrImageProcessor.from_pretrained(DETR_EXPORT)
meta = json.loads((DETR_EXPORT / "bccd_meta.json").read_text(encoding="utf-8"))
reloaded_id2label = {int(index): name for name, index in meta["class_to_idx"].items()}
reloaded_detr.to(DEVICE).eval()
detr_check = predict_detections(
    demo_image,
    model=reloaded_detr,
    processor=reloaded_processor,
    id2label=reloaded_id2label,
)
print(f"Reloaded DETR boxes: {len(detr_check['detections'])}. Init mode recorded as {meta['init_mode']}.")
if "class_name" not in vit_check or "detections" not in detr_check:
    raise RuntimeError("Reloaded models did not return the expected inference fields.")
mark("reload_ready")
''')

md(r'''
## Section 13: Final comparison and report

The table is built from metric JSON files. If a training cell has not been executed, that row says `not yet run`. Nothing in this section invents an accuracy or an mAP.

A small CNN is a strong baseline on BCCD because a few thousand cell crops do not give a transformer much data to learn spatial structure from scratch. Convolution has a locality bias and needs fewer examples to find edges and texture. Attention can relate distant patches, which is useful when context matters, and it is easier to overfit when the training set is a few hundred microscopy fields. Pretrained initialization changes that trade-off: ImageNet or COCO features supply the missing data, at the cost of a domain gap between natural photographs and stained cells.

What this notebook implements is the data split, the from-scratch ViT, the DETR fine-tune, and the evaluation. What it has trained depends entirely on the cells you have run.
''')

code(r'''
require("reload_ready")
from bccd.metrics_io import metrics_row_or_missing

rows = [
    metrics_row_or_missing(PATHS["outputs_cls"] / "vit_metrics.json", "ViT from scratch"),
    metrics_row_or_missing(PATHS["outputs_cls"] / "pretrained_vit_metrics.json", "torchvision ViT-B/16 fine-tune"),
    metrics_row_or_missing(PATHS["outputs_cls"] / "cnn_metrics.json", "Small CNN baseline"),
    metrics_row_or_missing(PATHS["outputs_det"] / "detr_metrics.json", "DETR ResNet-50"),
]
report = pd.DataFrame(rows)
pd.set_option("display.max_colwidth", 200)
try:
    from IPython.display import display
    display(report)
except Exception:
    print(report.to_string(index=False))
report_path = PATHS["reports"] / "summary.md"
lines = [
    "# BCCD experiment summary",
    "",
    "Numbers below were read from metric files written by the training and evaluation cells.",
    "A value of `not yet run` means that cell has not been executed in this project.",
    "",
    "| Model | Training configuration | Validation metrics | Test metrics | Training duration | Inference latency | Limitations |",
    "| --- | --- | --- | --- | --- | --- | --- |",
]
for row in rows:
    cells = [str(row[column]).replace("|", "/") for column in [
        "model", "training_configuration", "validation_metrics", "test_metrics",
        "training_duration", "inference_latency", "limitations",
    ]]
    lines.append("| " + " | ".join(cells) + " |")
lines.extend([
    "",
    "The ViT classifier and the DETR detector were trained separately.",
    "The classifier does not localize cells. DETR does not take pre-cropped cells as its training input.",
    "",
    "Attention models can overfit this dataset because there are only a few hundred source fields.",
    "A CNN baseline is available by setting CFG['run_cnn_baseline'] = True.",
    "Possible improvements: stain normalization, more fields, class-balanced sampling, and longer DETR fine-tuning with a lower learning rate.",
])
report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Wrote {report_path}")
mark("report_ready")
''')

md(r'''
## Section 14: Interview preparation

### How does a Vision Transformer differ from a CNN?

A CNN applies local filters and builds a hierarchy of receptive fields. A ViT splits the image into patches, embeds them as tokens, and mixes every token with every other token through self-attention. The CNN's locality is built in. The ViT has to learn locality, which is one reason it wants more data or a pretrained initialization.

### How does patch embedding work?

The image is divided into non-overlapping patches, here 16 by 16. Each patch is a vector of `16 * 16 * 3` pixel values. A linear layer, implemented as a convolution with stride 16, maps that vector to the model dimension. The result is a sequence of patch tokens.

### How are query, key, and value generated?

Each token is multiplied by three learned matrices, often implemented as one linear layer that outputs three blocks. Splitting that output produces Q, K, and V. In multi-head attention those vectors are reshaped into one Q, K, and V per head.

### What does self-attention compute?

For every token, it takes a weighted sum of the value vectors. The weights are a softmax over the scaled dot products of that token's query with every token's key. The attention matrix for one head has shape `(sequence length, sequence length)`.

### Why are positional embeddings needed?

Self-attention itself does not know which patch was on the left or the right. The order of the sequence is not a position. Adding a learned vector for each slot gives the model a way to tell patch `(0, 0)` from patch `(7, 7)`.

### Why is a ViT classifier insufficient for object detection?

The classifier emits one label for one crop. A microscopy field contains many cells, and the task is to say where each one is. The classifier has no box head, no set of object queries, and no matching loss between predictions and ground-truth boxes. Cropping first also assumes the boxes are already known.

### How does DETR predict object locations?

Each of a fixed number of object queries is decoded into one class distribution and one box. The box head is an MLP that regresses the box center, width, and height, normalized to the image. Queries that do not match a real object are assigned the no-object class.

### What is Hungarian matching?

It is the assignment algorithm that finds a one-to-one pairing between predicted queries and ground-truth objects with the lowest total cost. The cost mixes classification cost, L1 box cost, and generalized IoU cost. DETR uses it so two queries are not trained on the same cell, and so extra queries can be trained as background.

### What are IoU and mAP?

Intersection over union is the overlap of two boxes divided by the area of their union. Average precision summarizes the precision-recall curve for one class at a chosen IoU threshold. mAP averages that score across classes. mAP@0.5 uses IoU 0.5. mAP@0.5:0.95 also averages across stricter thresholds from 0.5 to 0.95 in steps of 0.05.

### Why can transformers overfit on small datasets?

Attention can memorize individual training images because every token can depend on every other token, and the model has many parameters relative to a few hundred blood-smear fields. Crops from the same field are highly correlated. Regularization, early stopping, augmentation, and pretrained weights are the practical controls. A validation split that does not share source images with training is what tells you the overfitting is happening.

### How can very high-resolution pathology images be processed?

Whole-slide images are too large for one forward pass. The usual approach is to read the slide at a chosen magnification, cut it into tiles, run the model on each tile, and stitch the tile predictions back into slide coordinates. Tissue detection skips empty glass. For a transformer, tiles can also be sampled at more than one magnification when the question needs both cellular detail and tissue context.

### How would this project be adapted to histopathology images?

Keep the two-task split. Use a classifier only when the input is already a tile or a nucleus crop, and use a detector or a segmentation model when the output must be a location. Replace BCCD's three classes with the pathology labels, expect stain variation, and split by patient or slide rather than by tile so the test set is not another view of the same tissue. Pretraining on a pathology tile dataset is more relevant than ImageNet or COCO when it is available. The ViT and DETR code paths stay the same; the annotations, the split unit, and the magnification change.
''')
