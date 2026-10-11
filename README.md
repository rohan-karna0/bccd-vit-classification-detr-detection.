# BCCD classification and detection

One project, two tasks, on the BCCD blood-cell images.

- A custom Vision Transformer, trained from scratch, classifies one cell as RBC, WBC, or Platelets.
- torchvision ViT-B/16, started from ImageNet-1K weights, is fine-tuned on the same crops so the two classifiers can be compared.
- DETR (`facebook/detr-resnet-50`, started from COCO weights) finds cells in a full image.

The notebook is `notebooks/BCCD_ViT_Classification_DETR_Detection.ipynb`.

## Setup

```bash
cd bccd-vit-classification-detr-detection.
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Put your Kaggle API token in `.env`. A Hugging Face token is optional. `facebook/detr-resnet-50` is public. Add `HF_TOKEN` only if the download asks you to log in.

Open the project root in Cursor, select the `.venv` interpreter, and run the notebook from the top. Each training cell prints the epoch, loss, and accuracy. The last cell opens a window to upload a sample image.
