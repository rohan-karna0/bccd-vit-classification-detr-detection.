"""Notebook cells for sections 1 through 6."""

CELLS = []


def md(text):
    CELLS.append(("md", text.strip() + "\n"))


def code(text):
    CELLS.append(("code", text.strip() + "\n"))


md(r'''
# BCCD Blood Cell Classification and Detection

This notebook builds two separate models on the BCCD (Blood Cell Count and Detection) dataset:

1. A Vision Transformer that classifies a crop of one cell as RBC, WBC, or Platelets.
2. A DETR detector that finds cells in a full microscopy image and returns boxes, classes, and confidence scores.

The classifier does not detect objects. The detector is not a classifier of pre-cropped cells. Each model has its own data, training loop, checkpoint, and evaluation.

Run this notebook on a GPU. The training cells are complete, and this workspace does not contain precomputed metrics. Every number in the final table is written by those cells when you execute them. A smoke-test switch is available in the configuration cell; leave it off for the real run.

The dataset is the public BCCD collection used as a small hematology example. The same split of the work, classification of a region versus localization in a whole slide, is the kind of distinction that matters in digital pathology.
''')

md(r'''
## Section 1: Project introduction

### Problem

A peripheral blood smear is a microscopy image that contains many red blood cells (RBC), fewer white blood cells (WBC), and small platelets. Counting and identifying those cells supports hematology review. The same two questions show up in pathology image analysis:

- What is this region?
- Where are the objects, and what class is each one?

BCCD is small, about 364 images at 640 by 480 pixels, with Pascal VOC boxes. It is large enough to implement both pipelines honestly and small enough to fine-tune on a single GPU. It is not a whole-slide pathology dataset. Section 14 discusses how the same design extends to histopathology.

### Classification versus object detection

| | Classification | Object detection |
| --- | --- | --- |
| Input | One cropped cell | The full microscopy field |
| Output | One class and a confidence score | A set of boxes, classes, and scores |
| Supervision | The class of that crop | Every ground-truth box in the image |
| Model in this notebook | Vision Transformer encoder | DETR encoder-decoder |

A correct class on a crop does not say where the cell was. A detector has to propose locations as well as names.

### End-to-end architecture

```mermaid
flowchart TD
  kaggle[Kaggle BCCD VOC] --> discover[Discover images and XML]
  discover --> split[Split by source image]
  split --> crops[Cell crops for ViT]
  split --> frames[Full images plus boxes for DETR]
  crops --> vit[ViT classifier]
  frames --> detr[DETR detector]
```

### Tools, outputs, and hardware

Libraries used when a cell needs them: PyTorch, torchvision, NumPy, pandas, Matplotlib, Pillow, scikit-learn, PyYAML, tqdm, the Kaggle API, Hugging Face Transformers, TorchMetrics, SciPy, and optional ipywidgets.

The notebook writes:

- crop images and a manifest under `data/classification/`
- detection manifests under `data/detection/`
- checkpoints under `models/checkpoints/`
- plots and metric JSON under `outputs/`
- a text summary under `reports/` after you run evaluation

Hardware: a CUDA GPU with at least 8 GB of memory is the intended setup for DETR fine-tuning at the default image scale. The from-scratch ViT is small and can share that GPU. CPU execution is supported for a smoke test only. Set `SMOKE_TEST = True` in the configuration cell for that short run, and do not treat those metrics as the full experiment.
''')

md(r'''
## Section 2: Environment setup

The next cells detect Kaggle, Google Colab, or a local Jupyter process, install only missing packages, and do not upgrade an existing PyTorch install. They then print the Python version, the PyTorch version, and the GPU, and they set random seeds.
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
        raise RuntimeError(
            "These earlier steps have not finished: "
            + ", ".join(missing)
            + ". Re-run the notebook from the top. If a step failed, fix that error before continuing."
        )


def detect_environment():
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.environ.get("KAGGLE_URL_BASE"):
        return "kaggle"
    try:
        import google.colab  # noqa: F401
        return "colab"
    except ImportError:
        return "local"


def find_project_root():
    cwd = Path.cwd().resolve()
    if (cwd / "src" / "bccd").exists():
        return cwd
    if cwd.name == "notebooks" and (cwd.parent / "src" / "bccd").exists():
        return cwd.parent
    if detect_environment() == "kaggle" and Path("/kaggle/working").exists():
        return Path("/kaggle/working")
    return cwd


ENV_NAME = detect_environment()
PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
print(f"Environment: {ENV_NAME}")
print(f"Project root: {PROJECT_ROOT}")
mark("env_detected")
''')

code(r'''
require("env_detected")

# One configuration object for both models.
# SMOKE_TEST = False is the GPU experiment. Set it to True only for a tiny wiring check.
SMOKE_TEST = False

CFG = {
    "seed": 42,
    "smoke_test": SMOKE_TEST,
    # Verified against the Kaggle data page for konstantinazov/bccd-dataset:
    # BCCD/JPEGImages, BCCD/Annotations (Pascal VOC XML), BCCD/ImageSets/Main.
    # Another public VOC mirror is "quangnguynvnnn/bccd-dataset-pascal-voc".
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
    "num_workers": 0 if os.name == "nt" else 2,
    "run_cnn_baseline": False,
    "run_pretrained_vit": False,
    "allow_detr_from_scratch": False,
    "resume_detr": True,
    "max_train_batches": None,
    "max_val_batches": None,
}

if CFG["smoke_test"]:
    CFG.update({
        "cls_epochs": 1,
        "cls_batch_size": 8,
        "cls_depth": 2,
        "cls_embed_dim": 64,
        "cls_num_heads": 4,
        "cls_patience": 1,
        "detr_epochs": 1,
        "detr_batch_size": 1,
        "detr_patience": 1,
        "detr_shortest_edge": 320,
        "detr_longest_edge": 480,
        "max_train_batches": 2,
        "max_val_batches": 1,
    })
    print("SMOKE_TEST is on. Metrics from this run are wiring checks, not full-training results.")
else:
    print("Full training configuration. Run the training cells on a CUDA GPU.")

if CFG["cls_embed_dim"] % CFG["cls_num_heads"] != 0:
    raise ValueError("cls_embed_dim must be divisible by cls_num_heads.")
if CFG["cls_image_size"] % CFG["cls_patch_size"] != 0:
    raise ValueError("cls_image_size must be divisible by cls_patch_size.")
mark("config_ready")
''')

code(r'''
require("config_ready")
import importlib.util
import subprocess

REQUIRED = [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("PIL", "Pillow"),
    ("sklearn", "scikit-learn"),
    ("yaml", "pyyaml"),
    ("tqdm", "tqdm"),
]
missing = [pip_name for module, pip_name in REQUIRED if importlib.util.find_spec(module) is None]
if missing:
    print("Installing missing packages:", ", ".join(missing))
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
else:
    print("Core packages are already installed. An existing PyTorch install was left unchanged.")
mark("packages_ready")
''')

code(r'''
require("packages_ready")
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(CFG["seed"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cpu":
    CFG["detr_batch_size"] = 1
    CFG["cls_batch_size"] = min(CFG["cls_batch_size"], 16)
    print("CUDA is not available. Batch sizes were reduced so a CPU smoke test can run.")
    print("Train the full configuration on a GPU before reporting results.")

print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("Device:", DEVICE)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU memory (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
mark("env_ready")
''')

md(r'''
## Section 3: Kaggle authentication and dataset download

The verified dataset slug is `konstantinazov/bccd-dataset`. Its page shows the upstream BCCD archive: `BCCD/JPEGImages`, `BCCD/Annotations` in Pascal VOC XML, and `BCCD/ImageSets/Main`, with the classes RBC, WBC, and Platelets. Change `CFG["kaggle_dataset_slug"]` if you mount a different mirror. A second public VOC copy is `quangnguynvnnn/bccd-dataset-pascal-voc` (train, valid, and test folders). Confirm the slug on the dataset URL before you rely on it:

`https://www.kaggle.com/datasets/<owner>/<dataset>`

### Create a token

1. Open [Kaggle account settings](https://www.kaggle.com/settings/account) and sign in.
2. In the API section, create a new token. Kaggle downloads `kaggle.json`, or shows a single API token.
3. Do not paste the token into this notebook.

Use one of these methods:

- Local Jupyter: put `kaggle.json` at `~/.kaggle/kaggle.json`, or set `KAGGLE_USERNAME` and `KAGGLE_KEY`, or set `KAGGLE_API_TOKEN`.
- Colab: add those same names as Colab secrets, or upload `kaggle.json` into `~/.kaggle/`.
- Kaggle notebooks: add this dataset with Add Data so it is already under `/kaggle/input/`. A secret named `KAGGLE_API_TOKEN` is used only if the files are not mounted.

The download cell stops if authentication fails.
''')

code(r'''
require("env_ready")
import json
import shutil
import subprocess

KAGGLE_DATASET_SLUG = CFG["kaggle_dataset_slug"]
DOWNLOAD_DIR = PROJECT_ROOT / "data" / "downloads" / KAGGLE_DATASET_SLUG.split("/")[-1]
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

AUTH_HELP = """
Kaggle authentication or download failed. No credentials are stored in this notebook.

1. Open https://www.kaggle.com/settings/account and create an API token.
2. Local Jupyter or Colab: save kaggle.json to ~/.kaggle/kaggle.json
   or export KAGGLE_USERNAME and KAGGLE_KEY, or export KAGGLE_API_TOKEN.
3. Colab: you can store KAGGLE_API_TOKEN, or KAGGLE_USERNAME and KAGGLE_KEY, as secrets.
4. On Kaggle, attach the dataset so it appears under /kaggle/input/.
5. Check CFG["kaggle_dataset_slug"]. The verified default is konstantinazov/bccd-dataset.
""".strip()


def _looks_like_bccd(root):
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


def find_kaggle_mount(slug):
    root = Path("/kaggle/input")
    if not root.exists():
        return None
    direct = root / slug.split("/")[-1]
    if direct.exists() and _looks_like_bccd(direct):
        return direct
    if not root.is_dir():
        return None
    for child in sorted(root.iterdir()):
        if child.is_dir() and _looks_like_bccd(child):
            return child
    return None


def _write_kaggle_json(username, key):
    destination = Path.home() / ".kaggle"
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "kaggle.json"
    path.write_text(json.dumps({"username": username, "key": key}), encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass


def load_secret(name):
    if os.environ.get(name):
        return os.environ[name]
    if ENV_NAME == "colab":
        try:
            from google.colab import userdata
            value = userdata.get(name)
            if value:
                return value
        except Exception:
            return None
    if ENV_NAME == "kaggle":
        try:
            from kaggle_secrets import UserSecretsClient
            return UserSecretsClient().get_secret(name)
        except Exception:
            return None
    return None


def configure_kaggle_credentials():
    token = load_secret("KAGGLE_API_TOKEN")
    username = load_secret("KAGGLE_USERNAME")
    key = load_secret("KAGGLE_KEY")
    if token:
        os.environ["KAGGLE_API_TOKEN"] = token
    if username and key:
        os.environ["KAGGLE_USERNAME"] = username
        os.environ["KAGGLE_KEY"] = key
        _write_kaggle_json(username, key)
    json_path = Path.home() / ".kaggle" / "kaggle.json"
    if json_path.exists() or os.environ.get("KAGGLE_API_TOKEN") or (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
        return
    raise RuntimeError(AUTH_HELP)


def ensure_kaggle_package():
    if importlib.util.find_spec("kaggle") is not None:
        return
    print("Installing the Kaggle package.")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])


mounted = find_kaggle_mount(KAGGLE_DATASET_SLUG) if ENV_NAME == "kaggle" else None
if mounted is not None:
    DATASET_ROOT = mounted
    print(f"Using the dataset already mounted at {DATASET_ROOT}")
elif _looks_like_bccd(DOWNLOAD_DIR):
    DATASET_ROOT = DOWNLOAD_DIR
    print(f"Using the dataset already extracted at {DATASET_ROOT}")
else:
    try:
        configure_kaggle_credentials()
        ensure_kaggle_package()
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print(f"Downloading {KAGGLE_DATASET_SLUG} to {DOWNLOAD_DIR}")
        api.dataset_download_files(KAGGLE_DATASET_SLUG, path=str(DOWNLOAD_DIR), unzip=True, quiet=False)
    except Exception as exc:
        raise RuntimeError(AUTH_HELP + f"\n\nOriginal error: {type(exc).__name__}: {exc}") from exc
    if not _looks_like_bccd(DOWNLOAD_DIR):
        raise RuntimeError(
            f"The download finished, but no image/XML pair was found under {DOWNLOAD_DIR}. "
            "Confirm the dataset slug and that the archive contains Pascal VOC annotations."
        )
    DATASET_ROOT = DOWNLOAD_DIR

from bccd.voc import discover_files, format_tree

discovered = discover_files(DATASET_ROOT)
print(format_tree(DATASET_ROOT))
print(
    f"Images: {len(discovered['images'])} | XML: {len(discovered['xml'])} | "
    f"JSON: {len(discovered['json'])} | CSV: {len(discovered['csv'])} | YAML: {len(discovered['yaml'])}"
)
if not discovered["images"] or not discovered["xml"]:
    raise RuntimeError("Dataset verification failed: images and Pascal VOC XML are both required.")
mark("data_ready")
''')

md(r'''
## Section 4: Dataset exploration

The next cell walks the extracted directory, pairs each image with an XML file by filename, and reads Pascal VOC objects. It does not assume that the images live in a folder named `JPEGImages`, although that is the layout of the verified archive. Sample figures use the ground-truth boxes from those XML files.

A VOC object looks like this:

```xml
<object>
  <name>WBC</name>
  <bndbox>
    <xmin>260</xmin>
    <ymin>177</ymin>
    <xmax>491</xmax>
    <ymax>376</ymax>
  </bndbox>
</object>
```

`xmin, ymin, xmax, ymax` are absolute pixel corners on the original image, with the origin at the top-left. Class names are normalized so `Platelet` and `Platelets` land on one label. Unknown names are kept and reported rather than dropped silently.
''')

code(r'''
require("data_ready")
from collections import Counter
from bccd.voc import build_records, class_to_index

RECORDS, DATA_SUMMARY = build_records(DATASET_ROOT)
if not RECORDS:
    raise RuntimeError("No image could be paired with a VOC XML file. Check the extracted archive.")
if DATA_SUMMARY["parse_errors"]:
    print("XML files skipped:")
    for message in DATA_SUMMARY["parse_errors"][:10]:
        print(" -", message)

print(pd.Series({key: DATA_SUMMARY[key] for key in [
    "n_images", "n_xml", "n_paired", "n_unpaired_images", "n_unpaired_xml",
    "n_objects", "n_empty_images", "n_invalid_boxes", "annotation_format",
]}).to_string())
print("\nInstances per class:")
class_table = pd.Series(DATA_SUMMARY["class_counts"], name="instances").sort_values(ascending=False)
print(class_table.to_string())
expected = {"RBC", "WBC", "Platelets"}
found = set(DATA_SUMMARY["class_counts"])
if not expected & found:
    raise RuntimeError(f"None of RBC, WBC, or Platelets were found. Classes present: {sorted(found)}")
if expected - found:
    print("Warning: some expected classes are absent:", sorted(expected - found))
print("YAML files seen:", DATA_SUMMARY["yaml_files"] or "none")
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
if not samples:
    raise RuntimeError("Every paired image has zero valid boxes, so there is nothing to draw.")
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
## Section 5: Create the data directories

Raw images stay in the download location. `data/raw` points at that folder and is not a second copy of the archive. Crops and JSON manifests are small derived files. Checkpoints, plots, and reports get their own directories.
''')

code(r'''
require("visualized")

PATHS = {
    "raw": DATASET_ROOT,
    "processed": PROJECT_ROOT / "data" / "processed",
    "classification": PROJECT_ROOT / "data" / "classification",
    "detection": PROJECT_ROOT / "data" / "detection",
    "checkpoints": PROJECT_ROOT / "models" / "checkpoints",
    "outputs_cls": PROJECT_ROOT / "outputs" / "classification",
    "outputs_det": PROJECT_ROOT / "outputs" / "detection",
    "plots": PROJECT_ROOT / "outputs" / "plots",
    "reports": PROJECT_ROOT / "reports",
    "src_dataset": PROJECT_ROOT / "src" / "dataset",
    "src_models": PROJECT_ROOT / "src" / "models",
    "src_training": PROJECT_ROOT / "src" / "training",
    "src_evaluation": PROJECT_ROOT / "src" / "evaluation",
    "notebooks": PROJECT_ROOT / "notebooks",
}
for path in PATHS.values():
    Path(path).mkdir(parents=True, exist_ok=True)
if not PATHS["raw"].exists():
    raise RuntimeError(f"Raw dataset path does not exist: {PATHS['raw']}")
print("Project directories are ready.")
for name, path in PATHS.items():
    print(f"  {name}: {path}")
mark("dirs_ready")
''')

md(r'''
## Section 6: Data preprocessing and splitting

Classification crops are cut from the same boxes the detector uses, but the two tasks cannot share one representation.

- The detector sees the whole field. Its target is a variable-length set of boxes. Empty images are valid and simply have no objects. Boxes stay in the original pixel geometry, then become normalized `cx, cy, w, h` for DETR. Horizontal flip rewrites the x corners. Color jitter does not move boxes. The DETR processor resizes the image while preserving aspect ratio, so those normalized coordinates stay aligned. A separate `scale_xyxy` helper is checked below for the case where a resize does stretch the axes.
- The classifier sees one cell. The crop's label is the box class. Crops are created only after the source image has a split, so cells from one field cannot land in both train and test. A crop is discarded when its shorter side is below 16 pixels. Remaining crops are resized to 128 by 128, which is the from-scratch ViT resolution.

The official `ImageSets/Main` lists are used when they cover the archive. Folder names such as `train` and `valid` are used when the archive is already split. Otherwise the notebook splits source images 70/15/15 with seed 42.
''')

code(r'''
require("dirs_ready")
from bccd.boxes import cxcywh_norm_to_xyxy, hflip_xyxy, scale_xyxy, xyxy_to_cxcywh_norm
from bccd.splits import assign_splits, split_counts

SPLIT_RECORDS, SPLIT_DESCRIPTION = assign_splits(RECORDS, DATASET_ROOT, seed=CFG["seed"])
CLASS_TO_IDX = class_to_index(SPLIT_RECORDS)
IDX_TO_CLASS = {index: name for name, index in CLASS_TO_IDX.items()}
print("Split rule:", SPLIT_DESCRIPTION)
print("Images per split:", split_counts(SPLIT_RECORDS))
print("Class to index:", CLASS_TO_IDX)

# Box-convention checks. These do not use dataset images.
sample = np.array([[10, 20, 40, 60]], dtype=np.float32)
flipped = hflip_xyxy(sample, width=100)
assert np.allclose(flipped, [[60, 20, 90, 60]]), flipped
scaled = scale_xyxy(sample, scale_x=0.5, scale_y=2.0)
assert np.allclose(scaled, [[5, 40, 20, 120]]), scaled
restored = cxcywh_norm_to_xyxy(xyxy_to_cxcywh_norm(sample, 100, 80), 100, 80)
assert np.allclose(restored, sample), restored

DETECTION_RECORDS = []
for record in SPLIT_RECORDS:
    labels, boxes, names = [], [], []
    for obj in record["objects"]:
        if obj["name"] not in CLASS_TO_IDX:
            continue
        labels.append(CLASS_TO_IDX[obj["name"]])
        boxes.append([float(v) for v in obj["bbox_xyxy"]])
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

detection_path = PATHS["detection"] / "records.json"
detection_path.write_text(json.dumps({
    "class_to_idx": CLASS_TO_IDX,
    "split_rule": SPLIT_DESCRIPTION,
    "coordinate_format": "absolute xyxy pixels, origin top-left, VOC corners",
    "records": DETECTION_RECORDS,
}, indent=2), encoding="utf-8")
(PATHS["processed"] / "class_to_idx.json").write_text(json.dumps(CLASS_TO_IDX, indent=2), encoding="utf-8")
print(f"Wrote {len(DETECTION_RECORDS)} detection records to {detection_path}")
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
if not CROP_ROWS:
    raise RuntimeError("No classification crops were created. Check the box sizes and class names.")

crop_frame = pd.DataFrame(CROP_ROWS)
overlap = {}
for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
    shared = set(crop_frame.loc[crop_frame["split"] == left, "source_stem"]) & set(
        crop_frame.loc[crop_frame["split"] == right, "source_stem"]
    )
    overlap[f"{left}-{right}"] = len(shared)
if any(overlap.values()):
    raise RuntimeError(f"Source images leaked across crop splits: {overlap}")

print("Crop export:", CROP_SUMMARY)
print("Crops per split and class:")
print(pd.crosstab(crop_frame["split"], crop_frame["class_name"]))
print(
    f"Rejected boxes with a side shorter than {CFG['min_crop_side']} px "
    f"or with invalid coordinates: {CROP_SUMMARY.get('rejected_small', 'see stamp')}"
)
mark("crops_ready")
''')

code(r'''
require("crops_ready")
counts = pd.DataFrame(CROP_ROWS).groupby(["split", "class_name"]).size().unstack(fill_value=0)
figure, axis = plt.subplots(figsize=(7, 4))
counts.plot(kind="bar", ax=axis, color=[CLASS_COLORS.get(name, "#999999") for name in counts.columns])
axis.set_ylabel("Crops")
axis.set_title("Classification crops by split")
axis.legend(title="Class")
figure.tight_layout()
figure.savefig(PATHS["plots"] / "class_distribution.png", dpi=120)
plt.show()

preview = pd.DataFrame(CROP_ROWS).groupby("class_name", group_keys=False).head(2)
figure, axes = plt.subplots(1, len(preview), figsize=(3 * len(preview), 3))
if len(preview) == 1:
    axes = [axes]
for axis, (_, row) in zip(axes, preview.iterrows()):
    with Image.open(row["crop_path"]) as image:
        axis.imshow(image)
    axis.set_title(row["class_name"])
    axis.axis("off")
figure.suptitle(f"Example crops resized to {CFG['cls_image_size']} px")
figure.tight_layout()
figure.savefig(PATHS["plots"] / "example_crops.png", dpi=120)
plt.show()
mark("crops_preview_ready")
''')
