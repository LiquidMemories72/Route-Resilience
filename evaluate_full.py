import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm
import random
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Reuse dataset and utilities from evaluate.py
from evaluate import RoadDataset, get_val_transform, compute_metrics, save_grid

def evaluate_full(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load all training images and masks (no split)
    train_dir = os.path.join(args.data_dir, 'train')
    all_images = sorted([os.path.join(train_dir, f)
                         for f in os.listdir(train_dir) if f.endswith('_sat.jpg')])
    all_masks = sorted([os.path.join(train_dir, f)
                        for f in os.listdir(train_dir) if f.endswith('_mask.png')])

    assert len(all_images) == len(all_masks), "Number of images and masks must match"
    print(f"Total training samples: {len(all_images)}")

    full_ds = RoadDataset(all_images, all_masks, transform=get_val_transform(args.img_size))
    full_loader = DataLoader(full_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    # Load model
    model = smp.FPN(
        encoder_name=args.encoder,
        encoder_weights=None,   # weights loaded from checkpoint
        in_channels=3,
        classes=1,
    ).to(device)

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    # Aggregate metrics over the full training set
    agg = dict(iou=0, dice=0, f1=0, precision=0, recall=0, accuracy=0)
    n_batches = 0

    with torch.no_grad():
        for images, masks in tqdm(full_loader, desc='Evaluating full training set'):
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            m = compute_metrics(logits, masks, threshold=args.threshold)
            for k in agg:
                agg[k] += m[k]
            n_batches += 1

    # Print averaged metrics
    print("\n" + "=" * 45)
    print(f"{'Metric':<15} {'Value':>10}")
    print("=" * 45)
    for k, v in agg.items():
        print(f"  {k:<13} {v / n_batches:>10.4f}")
    print("=" * 45)

    # Visualise random samples from the training set
    indices = random.sample(range(len(full_ds)), min(args.num_samples, len(full_ds)))
    samples = []
    with torch.no_grad():
        for idx in indices:
            img_t, gt_t = full_ds[idx]
            logits = model(img_t.unsqueeze(0).to(device))
            pred = (torch.sigmoid(logits) > args.threshold).float().cpu().squeeze().numpy()
            samples.append((img_t, gt_t, pred))

    save_path = os.path.join(args.output_dir, 'full_training_evaluation_grid.png')
    save_grid(samples, save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate road segmentation model on the full training set')
    parser.add_argument('--checkpoint',   type=str,   default='best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir',     type=str,   default='.',
                        help='Dataset root directory')
    parser.add_argument('--encoder',      type=str,   default='mit_b2',
                        help='Encoder used during training')
    parser.add_argument('--img_size',     type=int,   default=512)
    parser.add_argument('--batch_size',   type=int,   default=8)
    parser.add_argument('--num_samples',  type=int,   default=8,
                        help='Number of random samples to visualise')
    parser.add_argument('--seed',         type=int,   default=67,
                        help='Random seed (kept for reproducibility)')
    parser.add_argument('--threshold',    type=float, default=0.5,
                        help='Binarisation threshold (0-1). Lower = more road pixels detected.')
    parser.add_argument('--output_dir',   type=str,   default='predictions',
                        help='Directory to save visualisation grids')
    args = parser.parse_args()
    # Ensure reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    evaluate_full(args)
