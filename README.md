# Classification and Detection of Blood Cells
### A comparison of Vision Transformers, then a detector that finds every cell

One blood-smear dataset. Two questions.

**What is this cell?** Crop it, then let two Vision Transformers compete: one trained from scratch, one fine-tuned from ImageNet.

**Where are the cells?** Hand the full image to DETR and get a box, a class, and a score for each RBC, WBC, and platelet.

The notebook is `notebooks/BCCD_ViT_Classification_DETR_Detection.ipynb`.

| Task | Model | Starts from | Input |
| --- | --- | --- | --- |
| Classify one cell | Custom ViT | Random weights | 128 × 128 crop |
| Classify the same cell | torchvision ViT-B/16 | ImageNet-1K | 224 × 224 crop |
| Find every cell | `facebook/detr-resnet-50` | COCO | Full smear |

The custom ViT is smaller than ViT-Base: patch size 16, embedding size 192, 6 heads, 6 encoder blocks. Both classifiers use the same crops and the same train / val / test split. Detection is a separate fine-tune on the full images.

Data is the public BCCD collection (`konstantinazov/bccd-dataset`): about 364 microscopy images with Pascal VOC boxes. Labels are **RBC**, **WBC**, and **Platelets**.

---

## Results

Numbers below are from the latest notebook run.

### Classifier comparison

The pretrained model wins, and it peaks earlier. The from-scratch model is already very strong, and it trains in about a third of the time.

| Model | Best epoch | Val accuracy | Test accuracy | Test macro F1 | Train time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Custom ViT, from scratch | 12 | 99.47% | 99.47% | 97.82% | 1.5 min |
| ViT-B/16, ImageNet-1K then BCCD | 3 | 99.91% | 99.89% | 99.52% | 4.0 min |

Saved table: `outputs/classification/classifier_comparison.csv`

### From-scratch ViT

![From-scratch ViT training curves](outputs/plots/vit_training_curves.png)

Loss falls and accuracy climbs past 99% on both train and validation. Best validation accuracy is **99.47%** at epoch 12.

![From-scratch ViT confusion matrix](outputs/plots/vit_confusion_matrix.png)

On the test set the model is almost perfect on red blood cells (805 / 805). The remaining mistakes are a handful of WBC and platelet swaps: 2 WBC called platelets, 1 platelet called RBC, 2 platelets called WBC.

Test accuracy **99.47%**. Test macro F1 **97.82%**.

### ImageNet ViT-B/16

![ImageNet ViT-B/16 training curves](outputs/plots/pretrained_vit_curves.png)

Validation accuracy is already near the ceiling from epoch 1. Best validation accuracy is **99.91%** at epoch 3. Test accuracy **99.89%**. Test macro F1 **99.52%**.

### DETR detection

![DETR training and validation loss](outputs/plots/detr_loss.png)

`facebook/detr-resnet-50` is fine-tuned with a new 3-class head. Best validation loss is **0.816** at epoch 8 (about 6.7 minutes).

| Metric | Score |
| --- | ---: |
| mAP @ 0.50 | 0.842 |
| mAP @ 0.50:0.95 | 0.520 |
| Precision | 0.526 |
| Recall | 0.960 |

Recall is high: most true cells are found. Precision is lower: the model still proposes extra boxes. Metrics file: `outputs/detection/detr_metrics.json`.

### How the crops are split

![Crops by split and class](outputs/plots/class_distribution.png)

Red blood cells dominate every split. White cells and platelets are the minority classes, which is why macro F1 matters more than accuracy alone.

---

## Where the outputs live

| What | Path |
| --- | --- |
| From-scratch metrics | `outputs/classification/vit_metrics.json` |
| From-scratch history | `outputs/classification/vit_history.csv` |
| Pretrained metrics | `outputs/classification/pretrained_vit_metrics.json` |
| Pretrained history | `outputs/classification/pretrained_vit_history.csv` |
| Side-by-side comparison | `outputs/classification/classifier_comparison.csv` |
| DETR metrics | `outputs/detection/detr_metrics.json` |
| DETR history | `outputs/detection/detr_history.csv` |
| Plots | `outputs/plots/` |
| Checkpoints | `models/checkpoints/` |

These folders are produced by the notebook. They are not committed.

---

## Setup

```bash
cd bccd-vit-classification-detr-detection.
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put a Kaggle API token in `.env` at the project root. `facebook/detr-resnet-50` is public, so a Hugging Face token is optional. Add `HF_TOKEN` only if a download asks you to log in.

Open the project root in Cursor, select the `.venv` interpreter (**BCCD (Python 3.13)**), and run the notebook from the top. If a new package was installed after the kernel started, restart the kernel before continuing. Training cells print epoch, loss, and accuracy. The last cell opens a window to upload a sample image.
