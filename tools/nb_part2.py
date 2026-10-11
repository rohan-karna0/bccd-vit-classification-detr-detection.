"""Notebook cells: evaluation, pretrained ViT, DETR, and the upload window."""

CELLS = []


def md(text):
    CELLS.append(("md", text.strip() + "\n"))


def code(text):
    CELLS.append(("code", text.strip() + "\n"))


md(r'''
## Section 8: Test the from-scratch ViT

This cell loads the best validation checkpoint and scores the test crops.
''')

code(r'''
require("vit_curves_ready")
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

def load_torch(path):
    try:
        return torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=DEVICE)


def evaluate_classifier(model, rows, batch_size):
    loader = make_loader(rows, train=False, batch_size=batch_size)
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            preds = model(images.to(DEVICE)).argmax(dim=1).cpu()
            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    names = [IDX_TO_CLASS[i] for i in range(len(IDX_TO_CLASS))]
    print(classification_report(y_true, y_pred, target_names=names, zero_division=0))
    print(f"test accuracy {accuracy_score(y_true, y_pred):.4f}")
    print(f"test macro F1 {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(names))))
    figure, axis = plt.subplots(figsize=(5, 4))
    axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(names)), names, rotation=45, ha="right")
    axis.set_yticks(range(len(names)), names)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Ground truth")
    axis.set_title("From-scratch ViT, test set")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            axis.text(col, row, int(matrix[row, col]), ha="center", va="center")
    figure.tight_layout()
    figure.savefig(PATHS["plots"] / "vit_confusion_matrix.png", dpi=120)
    plt.show()
    return {
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


checkpoint = load_torch(VIT_CHECKPOINT)
VIT_MODEL = build_vit(len(checkpoint["class_to_idx"]))
VIT_MODEL.load_state_dict(checkpoint["model_state"])
VIT_MODEL.to(DEVICE)
SCRATCH_TEST = evaluate_classifier(VIT_MODEL, TEST_ROWS, CFG["cls_batch_size"])
from bccd.metrics_io import load_metrics, save_metrics
metrics = load_metrics(PATHS["outputs_cls"] / "vit_metrics.json") or {}
metrics.update(SCRATCH_TEST)
save_metrics(PATHS["outputs_cls"] / "vit_metrics.json", metrics)
mark("vit_evaluated")
''')

md(r'''
## Section 9: Fine-tune pretrained ViT-B/16

This is the other side of the comparison.

- Architecture: `torchvision.models.vit_b_16`
- Weights: `ViT_B_16_Weights.IMAGENET1K_V1` (ImageNet, 1,000 classes)
- Change: the final linear layer is replaced with 3 outputs (RBC, WBC, Platelets)
- Input: 224 x 224 with the ImageNet mean and standard deviation
- Data: the same crop files and the same split as the from-scratch ViT

Each epoch prints train loss, train accuracy, validation loss, and validation accuracy.
''')

code(r'''
require("vit_evaluated")
import time
from torchvision.models import ViT_B_16_Weights, vit_b_16

PRETRAINED_WEIGHTS = ViT_B_16_Weights.IMAGENET1K_V1
PRETRAINED_TRANSFORM = PRETRAINED_WEIGHTS.transforms()
PRETRAINED_MODEL = vit_b_16(weights=PRETRAINED_WEIGHTS)
PRETRAINED_MODEL.heads.head = nn.Linear(PRETRAINED_MODEL.heads.head.in_features, len(CLASS_TO_IDX))
PRETRAINED_MODEL.to(DEVICE)
print("Pretrained model: torchvision vit_b_16, ImageNet-1K")
print(f"Parameters: {sum(p.numel() for p in PRETRAINED_MODEL.parameters()):,}")


class PretrainedCropDataset(Dataset):
    def __init__(self, rows):
        self.rows = list(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["crop_path"]) as image:
            tensor = PRETRAINED_TRANSFORM(image.convert("RGB"))
        return tensor, int(row["label"])


def pretrained_loader(rows, train):
    return DataLoader(
        PretrainedCropDataset(rows),
        batch_size=CFG["pretrained_batch_size"],
        shuffle=train,
        num_workers=CFG["num_workers"],
    )


criterion = nn.CrossEntropyLoss(weight=class_weights(TRAIN_ROWS))
optimizer = torch.optim.AdamW(PRETRAINED_MODEL.parameters(), lr=CFG["pretrained_lr"], weight_decay=CFG["cls_weight_decay"])
train_loader = pretrained_loader(TRAIN_ROWS, True)
val_loader = pretrained_loader(VAL_ROWS, False)
PRETRAINED_HISTORY = []
best_acc = -1.0
best_epoch = -1
best_state = None
started = time.time()
for epoch in range(CFG["pretrained_epochs"]):
    train_loss, train_acc = run_epoch(PRETRAINED_MODEL, train_loader, criterion, optimizer)
    val_loss, val_acc = run_epoch(PRETRAINED_MODEL, val_loader, criterion, None)
    PRETRAINED_HISTORY.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
    })
    print(
        f"epoch {epoch + 1}/{CFG['pretrained_epochs']}  "
        f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
        f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
    )
    if val_acc > best_acc:
        best_acc = val_acc
        best_epoch = epoch + 1
        best_state = {key: value.detach().cpu().clone() for key, value in PRETRAINED_MODEL.state_dict().items()}

PRETRAINED_CHECKPOINT = PATHS["checkpoints"] / "vit_b16_finetune.pt"
torch.save({
    "model_state": best_state,
    "class_to_idx": CLASS_TO_IDX,
    "model_name": "torchvision_vit_b_16_imagenet1k",
    "epoch": best_epoch,
    "val_accuracy": best_acc,
}, PRETRAINED_CHECKPOINT)
PRETRAINED_MODEL.load_state_dict(best_state)
pd.DataFrame(PRETRAINED_HISTORY).to_csv(PATHS["outputs_cls"] / "pretrained_vit_history.csv", index=False)
save_metrics(PATHS["outputs_cls"] / "pretrained_vit_metrics.json", {
    "model": "torchvision ViT-B/16 fine-tuned from ImageNet-1K",
    "best_epoch": best_epoch,
    "best_val_accuracy": best_acc,
    "minutes": (time.time() - started) / 60,
})
print(f"Best validation accuracy {best_acc:.4f} at epoch {best_epoch}")
print("Checkpoint:", PRETRAINED_CHECKPOINT)
mark("pretrained_trained")
''')

code(r'''
require("pretrained_trained")
history = pd.read_csv(PATHS["outputs_cls"] / "pretrained_vit_history.csv")
figure, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(history["epoch"], history["train_loss"], label="train")
axes[0].plot(history["epoch"], history["val_loss"], label="validation")
axes[0].set_title("ImageNet ViT-B/16 loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[1].plot(history["epoch"], history["train_accuracy"], label="train")
axes[1].plot(history["epoch"], history["val_accuracy"], label="validation")
axes[1].set_title("ImageNet ViT-B/16 accuracy")
axes[1].set_xlabel("Epoch")
axes[1].legend()
figure.tight_layout()
figure.savefig(PATHS["plots"] / "pretrained_vit_curves.png", dpi=120)
plt.show()

test_loader = pretrained_loader(TEST_ROWS, False)
y_true, y_pred = [], []
PRETRAINED_MODEL.eval()
with torch.no_grad():
    for images, labels in test_loader:
        preds = PRETRAINED_MODEL(images.to(DEVICE)).argmax(dim=1).cpu()
        y_true.extend(labels.tolist())
        y_pred.extend(preds.tolist())
y_true = np.asarray(y_true)
y_pred = np.asarray(y_pred)
names = [IDX_TO_CLASS[i] for i in range(len(IDX_TO_CLASS))]
print(classification_report(y_true, y_pred, target_names=names, zero_division=0))
PRETRAINED_TEST = {
    "test_accuracy": float(accuracy_score(y_true, y_pred)),
    "test_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
}
print(f"test accuracy {PRETRAINED_TEST['test_accuracy']:.4f}")
print(f"test macro F1 {PRETRAINED_TEST['test_macro_f1']:.4f}")
metrics = load_metrics(PATHS["outputs_cls"] / "pretrained_vit_metrics.json") or {}
metrics.update(PRETRAINED_TEST)
save_metrics(PATHS["outputs_cls"] / "pretrained_vit_metrics.json", metrics)
mark("pretrained_evaluated")
''')

md(r'''
## Section 10: Compare the two classifiers

Both rows use the same BCCD crops. The only difference is the network and where its weights started.
''')

code(r'''
require("pretrained_evaluated")
scratch = load_metrics(PATHS["outputs_cls"] / "vit_metrics.json")
pretrained = load_metrics(PATHS["outputs_cls"] / "pretrained_vit_metrics.json")
comparison = pd.DataFrame([
    {
        "model": "Custom ViT, trained from scratch on BCCD",
        "start": "random weights",
        "input": "128 x 128",
        "best_val_accuracy": scratch["best_val_accuracy"],
        "test_accuracy": scratch["test_accuracy"],
        "test_macro_f1": scratch["test_macro_f1"],
    },
    {
        "model": "torchvision ViT-B/16",
        "start": "ImageNet-1K, then fine-tuned on BCCD",
        "input": "224 x 224",
        "best_val_accuracy": pretrained["best_val_accuracy"],
        "test_accuracy": pretrained["test_accuracy"],
        "test_macro_f1": pretrained["test_macro_f1"],
    },
])
print(comparison.to_string(index=False))
comparison.to_csv(PATHS["outputs_cls"] / "classifier_comparison.csv", index=False)
mark("comparison_ready")
''')

md(r'''
## Section 11: DETR detection model

`facebook/detr-resnet-50` is a detector pretrained on COCO. A ResNet-50 builds the image features. A transformer then predicts a fixed set of boxes.

The COCO classification layer is replaced with 3 BCCD classes plus one no-object class. The rest of the checkpoint is kept, then fine-tuned on the full microscopy images.
''')

code(r'''
require("comparison_ready")
from transformers import DetrForObjectDetection, DetrImageProcessor
from bccd.boxes import hflip_xyxy, xyxy_to_cxcywh_norm

ID2LABEL = {index: name for name, index in CLASS_TO_IDX.items()}
LABEL2ID = {name: index for index, name in ID2LABEL.items()}
token = HF_TOKEN
try:
    DETR_PROCESSOR = DetrImageProcessor.from_pretrained(
        CFG["detr_model_name"],
        token=token,
        size={"shortest_edge": CFG["detr_shortest_edge"], "longest_edge": CFG["detr_longest_edge"]},
    )
except TypeError:
    DETR_PROCESSOR = DetrImageProcessor.from_pretrained(CFG["detr_model_name"], token=token)
    DETR_PROCESSOR.size = {"shortest_edge": CFG["detr_shortest_edge"], "longest_edge": CFG["detr_longest_edge"]}

try:
    DETR_MODEL = DetrForObjectDetection.from_pretrained(
        CFG["detr_model_name"],
        num_labels=len(CLASS_TO_IDX),
        ignore_mismatched_sizes=True,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        token=token,
    )
except Exception as exc:
    raise RuntimeError(
        "Could not download facebook/detr-resnet-50. If Hugging Face asked you to log in, "
        "put a token in HF_TOKEN in the .env file and re-run this cell.\n"
        f"Original error: {type(exc).__name__}: {exc}"
    ) from exc
DETR_MODEL.to(DEVICE)
print("Loaded facebook/detr-resnet-50. Classification head was reset for", list(CLASS_TO_IDX))
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
        if self.augment and random.random() < 0.5 and len(boxes):
            image = transforms.functional.hflip(image)
            boxes = hflip_xyxy(boxes, image.width)
        if self.augment:
            image = self.color(image)
        boxes_cxcywh = xyxy_to_cxcywh_norm(boxes, image.width, image.height)
        target = {
            "class_labels": torch.tensor(labels, dtype=torch.long),
            "boxes": torch.tensor(boxes_cxcywh, dtype=torch.float32),
            "orig_size": torch.tensor([image.height, image.width], dtype=torch.int64),
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
    return DataLoader(
        BCCDDetectionDataset(rows, augment=train),
        batch_size=CFG["detr_batch_size"],
        shuffle=train,
        num_workers=CFG["num_workers"],
        collate_fn=collate_detections,
        pin_memory=DEVICE.type == "cuda",
    )


print("Detection images:", {split: sum(record["split"] == split for record in DETECTION_RECORDS) for split in ("train", "val", "test")})
mark("detr_loaders_ready")
''')

md(r'''
## Section 12: Train DETR

Each epoch prints the training loss and the validation loss. The checkpoint with the lowest validation loss is kept. Lower loss is better. Detection accuracy is measured after training as mAP.
''')

code(r'''
require("detr_loaders_ready")
from tqdm.auto import tqdm


def detr_epoch(model, loader, optimizer):
    training = optimizer is not None
    model.train(training)
    losses = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in tqdm(loader, leave=False):
            labels = [{key: value.to(DEVICE) for key, value in target.items()} for target in batch["labels"]]
            kwargs = {"pixel_values": batch["pixel_values"].to(DEVICE), "labels": labels}
            if batch["pixel_mask"] is not None:
                kwargs["pixel_mask"] = batch["pixel_mask"].to(DEVICE)
            if training:
                optimizer.zero_grad(set_to_none=True)
            loss = model(**kwargs).loss
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["detr_grad_clip"])
                optimizer.step()
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


backbone_params = [p for name, p in DETR_MODEL.named_parameters() if "backbone" in name and p.requires_grad]
other_params = [p for name, p in DETR_MODEL.named_parameters() if "backbone" not in name and p.requires_grad]
DETR_OPTIMIZER = torch.optim.AdamW([
    {"params": other_params, "lr": CFG["detr_lr"]},
    {"params": backbone_params, "lr": CFG["detr_lr_backbone"]},
], weight_decay=CFG["detr_weight_decay"])

DETR_CHECKPOINT = PATHS["checkpoints"] / "detr_bccd_best.pt"
DETR_EXPORT = PATHS["checkpoints"] / "detr_bccd_best"
train_loader = detection_loader("train", True)
val_loader = detection_loader("val", False)
DETR_HISTORY = []
best_val = float("inf")
best_epoch = -1
waits = CFG["detr_patience"]
started = time.time()
for epoch in range(CFG["detr_epochs"]):
    train_loss = detr_epoch(DETR_MODEL, train_loader, DETR_OPTIMIZER)
    val_loss = detr_epoch(DETR_MODEL, val_loader, None)
    DETR_HISTORY.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
    print(f"epoch {epoch + 1}/{CFG['detr_epochs']}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
    if val_loss < best_val:
        best_val = val_loss
        best_epoch = epoch + 1
        waits = CFG["detr_patience"]
        torch.save({
            "model_state": DETR_MODEL.state_dict(),
            "class_to_idx": CLASS_TO_IDX,
            "epoch": best_epoch,
            "val_loss": best_val,
        }, DETR_CHECKPOINT)
        DETR_MODEL.save_pretrained(DETR_EXPORT)
        DETR_PROCESSOR.save_pretrained(DETR_EXPORT)
    else:
        waits -= 1
        if waits <= 0:
            print(f"Stopped early. Best validation loss {best_val:.4f} at epoch {best_epoch}.")
            break

pd.DataFrame(DETR_HISTORY).to_csv(PATHS["outputs_det"] / "detr_history.csv", index=False)
save_metrics(PATHS["outputs_det"] / "detr_metrics.json", {
    "model": "facebook/detr-resnet-50 fine-tuned on BCCD",
    "best_epoch": best_epoch,
    "best_val_loss": best_val,
    "minutes": (time.time() - started) / 60,
})
print(f"Best validation loss {best_val:.4f} at epoch {best_epoch}")
print("Checkpoint:", DETR_CHECKPOINT)
mark("detr_trained")
''')

code(r'''
require("detr_trained")
history = pd.read_csv(PATHS["outputs_det"] / "detr_history.csv")
figure, axis = plt.subplots(figsize=(6, 4))
axis.plot(history["epoch"], history["train_loss"], label="train")
axis.plot(history["epoch"], history["val_loss"], label="validation")
axis.set_title("DETR loss")
axis.set_xlabel("Epoch")
axis.legend()
figure.tight_layout()
figure.savefig(PATHS["plots"] / "detr_loss.png", dpi=120)
plt.show()
mark("detr_curves_ready")
''')

md(r'''
## Section 13: Evaluate DETR

mAP@0.5 and mAP@0.5:0.95 are computed on the test images. Precision and recall use a confidence threshold of 0.5 and an IoU match of 0.5.
''')

code(r'''
require("detr_curves_ready")
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


saved = load_torch(DETR_CHECKPOINT)
DETR_MODEL.load_state_dict(saved["model_state"])
DETR_MODEL.to(DEVICE).eval()
rows = [record for record in DETECTION_RECORDS if record["split"] == "test"]
loader = DataLoader(BCCDDetectionDataset(rows, augment=False), batch_size=1, shuffle=False, collate_fn=collate_detections)
predictions, targets = [], []
with torch.no_grad():
    for batch in loader:
        kwargs = {"pixel_values": batch["pixel_values"].to(DEVICE)}
        if batch["pixel_mask"] is not None:
            kwargs["pixel_mask"] = batch["pixel_mask"].to(DEVICE)
        outputs = DETR_MODEL(**kwargs)
        processed = postprocess_batch(outputs, batch["orig_sizes"], threshold=0.0)
        for result, path in zip(processed, batch["paths"]):
            predictions.append({
                "boxes": result["boxes"].detach().cpu(),
                "scores": result["scores"].detach().cpu(),
                "labels": result["labels"].detach().cpu(),
            })
            gt = next(item for item in rows if item["image_path"] == path)
            targets.append({
                "boxes": torch.tensor(gt["boxes_xyxy"], dtype=torch.float32).reshape(-1, 4),
                "labels": torch.tensor(gt["labels"], dtype=torch.int64),
            })

metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
metric.update(predictions, targets)
scores = metric.compute()
map_50 = float(scores["map_50"])
map_all = float(scores["map"])
print(f"mAP@0.5 = {map_50:.4f}")
print(f"mAP@0.5:0.95 = {map_all:.4f}")

tp = fp = fn = 0
threshold = CFG["detr_confidence_threshold"]
for prediction, target in zip(predictions, targets):
    order = np.argsort(-prediction["scores"].numpy())
    matched = set()
    gt_boxes = target["boxes"].numpy()
    gt_labels = target["labels"].numpy()
    boxes = prediction["boxes"].numpy()
    labels = prediction["labels"].numpy()
    confidences = prediction["scores"].numpy()
    for index in order:
        if confidences[index] < threshold:
            continue
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
        else:
            tp += 1
            matched.add(best_j)
    fn += len(gt_labels) - len(matched)
precision = tp / (tp + fp) if (tp + fp) else 0.0
recall = tp / (tp + fn) if (tp + fn) else 0.0
print(f"precision @ {threshold:.2f} = {precision:.4f}")
print(f"recall @ {threshold:.2f} = {recall:.4f}")
metrics = load_metrics(PATHS["outputs_det"] / "detr_metrics.json") or {}
metrics.update({"map_50": map_50, "map_50_95": map_all, "precision": precision, "recall": recall})
save_metrics(PATHS["outputs_det"] / "detr_metrics.json", metrics)
mark("detr_evaluated")
''')

md(r'''
## Section 14: Upload a sample image

Run this cell after training. A window opens with two buttons.

- **Classify a cell** expects one cropped cell. It prints the class from the from-scratch ViT and from the ImageNet ViT-B/16.
- **Detect cells** expects a full smear. It draws the DETR boxes.

Close the window when you are done. The kernel stays busy until the window closes.
''')

code(r'''
require("detr_evaluated")
import tkinter as tk
from tkinter import filedialog
from PIL import ImageDraw, ImageTk


def _load_scratch():
    checkpoint = load_torch(VIT_CHECKPOINT)
    model = build_vit(len(checkpoint["class_to_idx"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE).eval()
    return model


def _load_pretrained():
    checkpoint = load_torch(PRETRAINED_CHECKPOINT)
    model = vit_b_16(weights=None)
    model.heads.head = nn.Linear(model.heads.head.in_features, len(checkpoint["class_to_idx"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE).eval()
    return model


def _load_detr():
    checkpoint = load_torch(DETR_CHECKPOINT)
    model = DetrForObjectDetection.from_pretrained(DETR_EXPORT)
    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE).eval()
    return model


SCRATCH_DEMO = _load_scratch()
PRETRAINED_DEMO = _load_pretrained()
DETR_DEMO = _load_detr()


def predict_cell(image):
    rgb = image.convert("RGB")
    scratch_tensor = scratch_transform(False)(rgb).unsqueeze(0).to(DEVICE)
    pretrained_tensor = PRETRAINED_TRANSFORM(rgb).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        scratch_prob = torch.softmax(SCRATCH_DEMO(scratch_tensor), dim=1)[0]
        pretrained_prob = torch.softmax(PRETRAINED_DEMO(pretrained_tensor), dim=1)[0]
    scratch_index = int(scratch_prob.argmax())
    pretrained_index = int(pretrained_prob.argmax())
    return (
        f"From-scratch ViT: {IDX_TO_CLASS[scratch_index]} ({float(scratch_prob[scratch_index]):.2f})\n"
        f"ImageNet ViT-B/16: {IDX_TO_CLASS[pretrained_index]} ({float(pretrained_prob[pretrained_index]):.2f})"
    )


def predict_detections(image):
    rgb = image.convert("RGB")
    encoding = DETR_PROCESSOR(images=rgb, return_tensors="pt")
    kwargs = {"pixel_values": encoding["pixel_values"].to(DEVICE)}
    if encoding.get("pixel_mask") is not None:
        kwargs["pixel_mask"] = encoding["pixel_mask"].to(DEVICE)
    with torch.no_grad():
        outputs = DETR_DEMO(**kwargs)
    target_sizes = torch.tensor([[rgb.height, rgb.width]])
    try:
        result = DETR_PROCESSOR.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=CFG["detr_confidence_threshold"]
        )[0]
    except TypeError:
        result = DETR_PROCESSOR.post_process_object_detection(outputs, target_sizes=target_sizes)[0]
        mask = result["scores"] >= CFG["detr_confidence_threshold"]
        result = {key: value[mask] for key, value in result.items()}
    drawn = rgb.copy()
    pen = ImageDraw.Draw(drawn)
    lines = []
    for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
        x1, y1, x2, y2 = [float(value) for value in box.tolist()]
        name = IDX_TO_CLASS[int(label)]
        pen.rectangle((x1, y1, x2, y2), outline=CLASS_COLORS.get(name, "#ff7f0e"), width=2)
        lines.append(f"{name} {float(score):.2f}")
    summary = "\n".join(lines) if lines else "No cells above the confidence threshold."
    return drawn, summary


def open_demo():
    window = tk.Tk()
    window.title("BCCD sample")
    status = tk.StringVar(value="Choose an image.")
    panel = tk.Label(window)
    panel.pack(padx=8, pady=8)
    tk.Label(window, textvariable=status, justify="left").pack(padx=8, pady=4)
    photo = {"image": None}

    def show(image, text):
        preview = image.copy()
        preview.thumbnail((640, 480))
        photo["image"] = ImageTk.PhotoImage(preview)
        panel.configure(image=photo["image"])
        status.set(text)

    def on_classify():
        path = filedialog.askopenfilename(title="Cell crop", filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if not path:
            return
        image = Image.open(path)
        show(image.convert("RGB"), predict_cell(image))

    def on_detect():
        path = filedialog.askopenfilename(title="Full smear", filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if not path:
            return
        image = Image.open(path)
        drawn, text = predict_detections(image)
        show(drawn, text)

    tk.Button(window, text="Classify a cell", command=on_classify).pack(side="left", padx=8, pady=8)
    tk.Button(window, text="Detect cells", command=on_detect).pack(side="left", padx=8, pady=8)
    window.mainloop()


try:
    open_demo()
except tk.TclError as exc:
    message = (
        "The upload window could not open. This machine needs a display and the Tk bindings "
        f"(python3-tk). Original error: {exc}"
    )
    print(message)
    raise RuntimeError(message) from exc
mark("demo_ready")
''')
