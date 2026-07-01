# End-to-End Pipeline Documentation

This document serves as the comprehensive, single-source-of-truth guide for the entire **Road Extraction and Graph Generation** pipeline. It details the role of every script, the data flow between modules, and the step-by-step process to run the system from raw satellite imagery to a fully vectorized and topologically repaired road network.

---

## 1. Project Overview & Directory Structure

The system is split into two primary phases:
1. **Machine Learning Pipeline**: Translates raw RGB satellite imagery into pixel-wise binary road masks and probability maps.
2. **Graph Extraction & Repair Pipeline**: Ingests these raster masks, skeletonizes them, repairs topological breaks using A* routing, and exports them as vector graphs.

### Key Files and Directories:
- `train/`, `valid/`: Contains `*_sat.jpg` (RGB images) and `*_mask.png` (Ground truth binary masks).
- `pred_masks/`: Output directory where the ML model saves its predictions.
- `graph_test/`: Output directory for the final vectorized graphs and diagnostic images.
- **Root Scripts**: Handle training, evaluation, and exporting predictions (`train.py`, `evaluate.py`, `export_predictions.py`, `soft_cldice_loss.py`).
- `graph_module/`: A standalone package for converting masks to graphs, containing core routing and extraction logic.

---

## 2. Machine Learning Pipeline

The ML pipeline is built on PyTorch and `segmentation_models_pytorch`, featuring a **SegFormer MiT-B2 Encoder** and an **FPN (Feature Pyramid Network) Decoder**. 

### 2.1 `train.py`
The primary training script. 
- **Data Loading**: Loads images, resizes to 512x512, normalizes them, and splits them 80-20 into train/validation sets (seed=67).
- **Augmentation**: Uses `albumentations` for heavy data augmentation (flips, color jitter, rotations) to prevent overfitting.
- **Optimization**: Uses `AdamW` (lr=1e-4) paired with a `ReduceLROnPlateau` scheduler.
- **Checkpoints**: Saves the best model weights to `best_model.pth`.

### 2.2 `soft_cldice_loss.py`
Defines the custom loss function critical for road extraction.
- Standard losses like BCE focus purely on pixel-wise overlap. This script implements a hybrid loss: `0.5 × BCE + 0.5 × soft clDice`.
- **clDice (Centerline Dice)** focuses on the topological structure of the prediction by performing differentiable morphological skeletonization (`soft_erode`, `soft_dilate`, `soft_skel`), heavily penalizing broken roads and disconnected segments.

### 2.3 `evaluate.py` & `evaluate_full.py`
Evaluation scripts that load `best_model.pth` and calculate metrics against the ground truth.
- **Metrics Tracked**: Intersection over Union (IoU), F1-Score (Dice), Precision, Recall, and Pixel Accuracy.
- Generates visual comparison grids overlaid on the original imagery.

### 2.4 `export_predictions.py`
The bridge between the ML phase and the Graph phase.
- Runs inference over all images in the `valid/` directory.
- Exports the raw continuous probability logits (after a sigmoid activation) into binary `[0, 255]` `.png` masks inside the `pred_masks/` folder.

---

## 3. Graph Extraction & Repair Pipeline (`graph_module/`)

The graph module converts the raster `pred_masks/` into mathematical graphs (`nodes.json`, `edges.json`), applying sophisticated topological repairs in the process.

### 3.1 Driver Scripts
- **`run_pipeline.py`**: The main entry point for processing a single mask. It orchestrates cleanup, skeletonization, repair, graph building, and simplification.
- **`run_random_samples.py`**: A batch utility that runs the pipeline over multiple random samples in a directory, useful for evaluating pipeline robustness.

### 3.2 Core Extraction Logic
- **`preprocess.py`**: Cleans up the raw ML mask by applying morphological closing to fill small holes and removing tiny disconnected pixel islands.
- **`skeleton.py`**: Uses `skimage.morphology.skeletonize` to reduce the binary mask to a 1-pixel-wide centerline network.
- **`endpoint_detection.py`**: Identifies dead-ends (pixels with 1 neighbor) and uses PCA to estimate the local tangent vector (direction) of the road at that point.
- **`branch_detection.py`**: Identifies junctions/intersections (pixels with >2 neighbors) and segments the skeleton into discrete continuous branches.

### 3.3 Topology Repair Logic
Because the ML model may occasionally fail under tree cover or shadows, the skeleton is often broken.
- **`topology_repair.py`**: The orchestrator for bridging gaps. It runs in three progressive stages (`Stage 1: Small (20px)`, `Stage 2: Medium (50px)`, `Stage 3: Large (80px)`). 
- **`candidate_pairs.py`**: For each endpoint, it searches for nearby endpoints or branches to connect to, generating "candidate pairs".
- **`scoring.py`**: Scores candidates based on Euclidean distance, tangent alignment (do the endpoints face each other?), and model probability.
- **`astar.py`**: Instead of connecting endpoints with straight lines, this runs a Bidirectional A* Search over the ML model's raw *probability map*. This forces the repaired connection to follow the most likely actual path of the road.
- **`validation.py`**: Ensures the generated A* path maintains a minimum average probability so the algorithm doesn't hallucinate roads across empty fields.

### 3.4 Graph Conversion & Output
- **`graph_builder.py`**: Traverses the repaired pixel skeleton and converts it into a `NetworkX` MultiGraph. Intersections become nodes; continuous pixel paths become edges containing spatial geometry and length properties.
- **`graph_healing.py`**: Simplifies the mathematical graph by pruning short dead-end spurs, removing redundant degree-2 nodes, and deduplicating overlapping edges.
- **`visualization.py` & `debug_viz.py`**: Renders output PNGs (nodes overlaid on satellite images, skeleton visualizations) and logs the internal decisions made during the A* repair phase.

---

## 4. End-to-End Walkthrough

To run the entire pipeline from scratch, execute the following commands in order:

1. **Train the ML Model**
   ```bash
   # Trains the model using Mit-B2 and BCE+clDice loss over 30 epochs
   python train.py --epochs 30 --batch_size 4
   ```

2. **Evaluate Performance**
   ```bash
   # Prints IoU, Dice, Recall, and saves visualization grids
   python evaluate_full.py
   ```

3. **Export Mask Predictions**
   ```bash
   # Generates binary masks in the pred_masks/ directory
   python export_predictions.py
   ```

4. **Extract and Repair Graph**
   ```bash
   # Converts a mask into a JSON vector graph, running A* topology repair
   python graph_module\run_pipeline.py ^
       --mask pred_masks\0001_sat_pred.png ^
       --satellite valid\0001_sat.jpg ^
       --output-dir graph_test
   ```

**Final Output**: Check the `graph_test/` directory for `nodes.json`, `edges.json`, and visual overlays showing the fully repaired road network!
