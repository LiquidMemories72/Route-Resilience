# Road Extraction & Road Graph Generation

DeepGlobe road extraction project using:

* SegFormer MiT-B2 encoder
* FPN decoder (Segmentation Models PyTorch)
* BCE + clDice loss
* Graph extraction and topology repair pipeline

---

# Project Structure

```text
ROAD RESEARCH/

├── train/
│   ├── *_sat.jpg
│   └── *_mask.png
│
├── valid/
│   └── *_sat.jpg
│
├── test/
│
├── graph_module/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── skeleton.py
│   ├── graph_builder.py
│   ├── graph_healing.py
│   ├── visualization.py
│   └── run_pipeline.py
│
├── train.py
├── evaluate.py
├── evaluate_full.py
├── export_predictions.py
├── soft_cldice_loss.py
│
├── best_model.pth
│
├── predictions/
├── pred_masks/
├── graph_test/
└── graph_smoke_output/
```

---

# Environment Setup

Create virtual environment:

```bash
python -m venv venv
```

Activate:

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Training

Train model:

```bash
python train.py
```

Custom settings:

```bash
python train.py ^
    --encoder mit_b2 ^
    --img_size 512 ^
    --batch_size 4 ^
    --epochs 30 ^
    --lr 1e-4
```

Resume training from checkpoint:

```bash
python train.py --resume best_model.pth
```

---

# Model Architecture

```text
RGB Image
      ↓
SegFormer Encoder (MiT-B2)
      ↓
FPN Decoder
      ↓
1-channel Road Mask
```

Loss:

```text
Loss =
0.5 × BCE
+
0.5 × clDice
```

Optimizer:

```text
AdamW
```

Scheduler:

```text
ReduceLROnPlateau
```

---

# Evaluation

Evaluate current validation split:

```bash
python evaluate.py
```

Outputs:

```text
IoU
Dice
F1
Precision
Recall
Accuracy
```

---

# Full Dataset Evaluation

Evaluate on all train images:

```bash
python evaluate_full.py
```

Example output:

```text
IoU
Dice
Precision
Recall
Accuracy
```

---

# Export Prediction Masks

Generate binary masks from trained model:

```bash
python export_predictions.py
```

Outputs:

```text
pred_masks/

0001_sat_pred.png
0002_sat_pred.png
...
```

These masks are required by the graph pipeline.

---

# Graph Pipeline

The graph module converts segmentation masks into road networks.

Pipeline:

```text
Road Mask
      ↓
Morphological Cleanup
      ↓
Skeletonization
      ↓
Graph Extraction
      ↓
Graph Healing
      ↓
Visualization
```

---

# Run Graph Extraction

Single mask:

```bash
python graph_module\run_pipeline.py ^
    --mask pred_masks\0001_sat_pred.png ^
    --output-dir graph_test
```

Mask + satellite overlay:

```bash
python graph_module\run_pipeline.py ^
    --mask pred_masks\0001_sat_pred.png ^
    --satellite valid\0001_sat.jpg ^
    --output-dir graph_test
```

---

# Graph Parameters

Adjust graph healing:

```bash
python graph_module\run_pipeline.py ^
    --mask pred_masks\0001_sat_pred.png ^
    --output-dir graph_test ^
    --endpoint-distance 15 ^
    --heading-angle 45 ^
    --junction-merge-distance 3
```

---

# Graph Outputs

Generated inside:

```text
graph_test/
```

Files:

```text
cleaned_mask.png

skeleton.png
skeleton_visualization.png

nodes.png

graph_on_mask.png
graph_on_satellite.png

nodes.json
edges.json

graph_summary.json
```

---


# Git Commands

Initialize repository:

```bash
git init
```

Check status:

```bash
git status
```

Add files:

```bash
git add .
```

Commit:

```bash
git commit -m "message"
```

Push:

```bash
git push origin main
```

Remove cached files after editing .gitignore:

```bash
git rm -r --cached .
git add .
git commit -m "Fix gitignore"
```

---

# Current Results

Best observed metrics:

```text
Train IoU ≈ 0.58
Validation IoU ≈ 0.55

Full Dataset IoU ≈ 0.61
Dice ≈ 0.76
Recall ≈ 0.88 with specific binarisation 
Precision ≈ 0.70
```

---


```
```
