# BCCD ViT classification and DETR detection

Two models are trained on the BCCD blood-cell dataset. They are separate.

- A Vision Transformer classifies one cropped cell as RBC, WBC, or Platelets.
- DETR detects cells in a full microscopy image and returns boxes, classes, and scores.

The notebook is [notebooks/BCCD_ViT_Classification_DETR_Detection.ipynb](notebooks/BCCD_ViT_Classification_DETR_Detection.ipynb). Run it on a CUDA GPU. This repository does not include trained weights or metrics. The comparison table stays `not yet run` until you execute the training and evaluation cells.

## Dataset

The default Kaggle slug is `konstantinazov/bccd-dataset`. That archive is the public BCCD collection in Pascal VOC format:

- `BCCD/JPEGImages`
- `BCCD/Annotations`
- `BCCD/ImageSets/Main`

Confirm the slug on the dataset page before you change it. An optional VOC mirror with `train`, `valid`, and `test` folders is `quangnguynvnnn/bccd-dataset-pascal-voc`. Set `CFG["kaggle_dataset_slug"]` in the configuration cell if you use it.

## Setup

1. Open [Kaggle account settings](https://www.kaggle.com/settings/account) and create an API token.
2. Save `kaggle.json` to `~/.kaggle/kaggle.json`, or set `KAGGLE_USERNAME` and `KAGGLE_KEY`, or set `KAGGLE_API_TOKEN`. Do not paste the token into the notebook.
3. On Kaggle itself, attach the dataset with Add Data so it is mounted under `/kaggle/input/`.
4. Install dependencies with the GPU build of PyTorch that matches your CUDA version. The notebook installs a package only when it is missing, and it does not upgrade an existing PyTorch install.

```bash
pip install -r requirements.txt
```

Install PyTorch from the official index if `pip install torch` would not select a CUDA build.

## Run

Open the project root in Jupyter or VS Code / Cursor so `src/` is importable, then run the notebook from the top.

`SMOKE_TEST` defaults to `False`. That is the full GPU experiment. Set `SMOKE_TEST = True` in the configuration cell only for a one-epoch wiring check, and do not report those numbers as the final result.

Optional switches in the same cell:

- `run_pretrained_vit`: fine-tune torchvision ViT-B/16 from ImageNet weights
- `run_cnn_baseline`: train the small CNN on the same crops
- `allow_detr_from_scratch`: build DETR from `DetrConfig` if the COCO checkpoint cannot be downloaded
- `resume_detr`: continue from `models/checkpoints/detr_bccd_best.pt` when that file exists

DETR fine-tuning expects a GPU with at least 8 GB of memory at the default 800-pixel scale. If it runs out of memory, set `detr_batch_size` to 1 and run the training cell again.

## Outputs

- `data/classification/`: cell crops and `crops_manifest.csv`
- `data/detection/records.json`: boxes for DETR
- `models/checkpoints/`: best ViT and DETR checkpoints
- `outputs/classification/vit_metrics.json` and `outputs/detection/detr_metrics.json`
- `reports/summary.md`: written by the report cell from those files

Raw images are not copied. `data/raw` points at the extracted download.
