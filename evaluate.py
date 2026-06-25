import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
import random
from tqdm import tqdm

# ─── Reuse the same Dataset & transform from train.py ──────────────────────────

class RoadDataset(Dataset):
    def __init__(self, image_files, mask_files, transform=None):
        self.image_files = image_files
        self.mask_files  = mask_files
        self.transform   = transform
        assert len(image_files) == len(mask_files)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = cv2.cvtColor(cv2.imread(self.image_files[idx]), cv2.COLOR_BGR2RGB)
        mask  = cv2.cvtColor(cv2.imread(self.mask_files[idx]),  cv2.COLOR_BGR2RGB)
        binary_mask = np.all(mask == [255, 255, 255], axis=-1).astype(np.float32)

        if self.transform:
            out = self.transform(image=image, mask=binary_mask)
            image, binary_mask = out['image'], out['mask']

        return image, binary_mask.unsqueeze(0)


def get_val_transform(img_size=512):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ─── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(logits, targets, threshold=0.5):
    preds = (torch.sigmoid(logits) > threshold).float()

    TP = (preds * targets).sum()
    FP = (preds * (1 - targets)).sum()
    FN = ((1 - preds) * targets).sum()
    TN = ((1 - preds) * (1 - targets)).sum()

    precision = (TP / (TP + FP + 1e-6)).item()
    recall    = (TP / (TP + FN + 1e-6)).item()
    dice      = (2 * TP / (2 * TP + FP + FN + 1e-6)).item()   # same as F1
    iou       = (TP / (TP + FP + FN + 1e-6)).item()
    accuracy  = ((TP + TN) / (TP + TN + FP + FN + 1e-6)).item()

    return dict(iou=iou, dice=dice, f1=dice, precision=precision,
                recall=recall, accuracy=accuracy)


# ─── Visualisation ─────────────────────────────────────────────────────────────

MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])


def denorm(tensor):
    img = tensor.permute(1, 2, 0).numpy()
    return np.clip(STD * img + MEAN, 0, 1)


def save_grid(samples, save_path):
    """
    samples: list of (image_tensor, gt_mask_tensor, pred_mask_np)
    """
    n = len(samples)
    fig = plt.figure(figsize=(18, 4.5 * n))
    gs  = gridspec.GridSpec(n, 4, figure=fig,
                            hspace=0.05, wspace=0.05)

    col_titles = ['RGB Image', 'Ground Truth', 'Prediction', 'Overlay']

    for row, (img_t, gt_t, pred) in enumerate(samples):
        img  = denorm(img_t)
        gt   = gt_t.squeeze().numpy()

        overlay = img.copy()
        overlay[pred == 1] = overlay[pred == 1] * 0.45 + np.array([1.0, 0.15, 0.15]) * 0.55

        panels = [img, gt, pred, overlay]
        cmaps  = [None, 'gray', 'gray', None]

        for col, (panel, cmap, title) in enumerate(zip(panels, cmaps, col_titles)):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(panel, cmap=cmap, vmin=0, vmax=1 if cmap else None)
            if row == 0:
                ax.set_title(title, fontsize=12, fontweight='bold', pad=6)
            ax.axis('off')

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualisation → {save_path}")


# ─── Main ──────────────────────────────────────────────────────────────────────

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ── Build validation set (same split as training) ──
    train_dir  = os.path.join(args.data_dir, 'train')
    all_images = sorted([os.path.join(train_dir, f)
                         for f in os.listdir(train_dir) if f.endswith('_sat.jpg')])
    all_masks  = sorted([os.path.join(train_dir, f)
                         for f in os.listdir(train_dir) if f.endswith('_mask.png')])

    _, val_images, _, val_masks = train_test_split(
        all_images, all_masks,
        test_size=0.2, random_state=args.seed, shuffle=True
    )

    val_ds     = RoadDataset(val_images, val_masks, transform=get_val_transform(args.img_size))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    print(f"Validation samples: {len(val_ds)}")

    # ── Load model ──
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

    # ── Run full-dataset evaluation ──
    agg = dict(iou=0, dice=0, f1=0, precision=0, recall=0, accuracy=0)
    n_batches = 0

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc='Evaluating'):
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            m = compute_metrics(logits, masks, threshold=args.threshold)
            for k in agg:
                agg[k] += m[k]
            n_batches += 1

    print("\n" + "=" * 45)
    print(f"{'Metric':<15} {'Value':>10}")
    print("=" * 45)
    for k, v in agg.items():
        print(f"  {k:<13} {v / n_batches:>10.4f}")
    print("=" * 45)

    # ── Visualise random samples ──
    indices = random.sample(range(len(val_ds)), min(args.num_samples, len(val_ds)))
    samples = []
    with torch.no_grad():
        for idx in indices:
            img_t, gt_t = val_ds[idx]
            logits = model(img_t.unsqueeze(0).to(device))
            pred   = (torch.sigmoid(logits) > args.threshold).float().cpu().squeeze().numpy()
            samples.append((img_t, gt_t, pred))

    save_path = os.path.join(args.output_dir, 'evaluation_grid.png')
    save_grid(samples, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate road segmentation model')
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
                        help='Must match random_state used in training split')
    parser.add_argument('--threshold',    type=float, default=0.5,
                        help='Binarisation threshold (0-1). Lower = more road pixels detected.')
    parser.add_argument('--output_dir',   type=str,   default='predictions',
                        help='Directory to save visualisation grids')
    args = parser.parse_args()
    evaluate(args)
