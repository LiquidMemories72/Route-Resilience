import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm
import argparse
import random
import matplotlib.pyplot as plt
from soft_cldice_loss import BCE_clDice_Loss
from sklearn.model_selection import train_test_split

class RoadDataset(Dataset):
    def __init__(self, image_files, mask_files, transform=None):
        self.image_files = image_files
        self.mask_files = mask_files
        self.transform = transform
        
        assert len(self.image_files) == len(self.mask_files), "Mismatch in number of images and masks"
        
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        mask_path = self.mask_files[idx]
        
        # Read image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Read mask
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        
        # Convert RGB mask to binary mask
        # Road is (255, 255, 255). We can just check one channel or all
        binary_mask = np.all(mask == [255, 255, 255], axis=-1).astype(np.float32)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=binary_mask)
            image = augmented['image']
            binary_mask = augmented['mask']
            
        # Add channel dimension to mask [1, H, W]
        binary_mask = binary_mask.unsqueeze(0)
            
        return image, binary_mask

def get_transforms(img_size=512):
    train_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    val_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    return train_transform, val_transform

def compute_iou(preds, targets, threshold=0.5):
    preds = (torch.sigmoid(preds) > threshold).float()
    intersection = (preds * targets).sum((1, 2, 3))
    union = (preds + targets).sum((1, 2, 3)) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()

def save_validation_grid(model, valid_dataset, device, epoch, num_samples=4, save_dir='predictions'):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    # Randomly select samples
    indices = random.sample(range(len(valid_dataset)), min(num_samples, len(valid_dataset)))
    
    fig, axes = plt.subplots(len(indices), 4, figsize=(16, 4 * len(indices)))
    if len(indices) == 1:
        axes = [axes]
        
    with torch.no_grad():
        for row, idx in enumerate(indices):
            image_tensor, mask_tensor = valid_dataset[idx]
            
            img_batch = image_tensor.unsqueeze(0).to(device)
            logits = model(img_batch)
            pred_mask = (torch.sigmoid(logits) > 0.5).float().cpu().squeeze()
            
            # Denormalize image
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_vis = image_tensor.permute(1, 2, 0).numpy()
            image_vis = std * image_vis + mean
            image_vis = np.clip(image_vis, 0, 1)
            
            gt_mask_vis = mask_tensor.squeeze().numpy()
            pred_mask_vis = pred_mask.numpy()
            
            # Overlay (Prediction on Image in Red)
            overlay = image_vis.copy()
            overlay[pred_mask_vis == 1] = overlay[pred_mask_vis == 1] * 0.5 + np.array([1.0, 0.0, 0.0]) * 0.5
            
            axes[row][0].imshow(image_vis)
            axes[row][0].set_title("RGB Image")
            axes[row][0].axis('off')
            
            axes[row][1].imshow(gt_mask_vis, cmap='gray', vmin=0, vmax=1)
            axes[row][1].set_title("Ground Truth Mask")
            axes[row][1].axis('off')
            
            axes[row][2].imshow(pred_mask_vis, cmap='gray', vmin=0, vmax=1)
            axes[row][2].set_title("Predicted Mask")
            axes[row][2].axis('off')
            
            axes[row][3].imshow(overlay)
            axes[row][3].set_title("Overlay")
            axes[row][3].axis('off')
            
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'epoch_{epoch+1}.png'))
    plt.close()

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Transforms
    train_transform, val_transform = get_transforms(img_size=args.img_size)
    
    # Datasets
    train_dir = os.path.join(args.data_dir, 'train')
    all_images = sorted([os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('_sat.jpg')])
    all_masks = sorted([os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('_mask.png')])
    
    # Shuffled 80-20 split (random_state=42 for reproducibility)
    train_images, val_images, train_masks, val_masks = train_test_split(
        all_images, all_masks, test_size=0.2, random_state=67, shuffle=True
    )
    
    train_dataset = RoadDataset(train_images, train_masks, transform=train_transform)
    valid_dataset = RoadDataset(val_images, val_masks, transform=val_transform)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(valid_dataset)}")
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Model: Segformer encoder (mit_b2) + FPN decoder
    model = smp.FPN(
        encoder_name=args.encoder,        # e.g., 'mit_b2'
        encoder_weights="imagenet",     # use pre-trained weights
        in_channels=3,
        classes=1,                      # binary segmentation
    )
    model = model.to(device)
    
    # Loss and Optimizer
    criterion = BCE_clDice_Loss(alpha=0.5) # Combines BCE and clDice
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    best_iou = 0.0
    start_epoch = 0
    
    # Resume from checkpoint
    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        # Support both full checkpoints and bare state_dicts
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_iou = checkpoint['best_iou']
        else:
            # Bare weights saved with torch.save(model.state_dict(), ...)
            model.load_state_dict(checkpoint)
        print(f"Resumed. Starting from epoch {start_epoch + 1}, best IoU so far: {best_iou:.4f}")
    
    # Training Loop
    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        train_loss = 0.0
        train_iou = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{start_epoch + args.epochs} [Train]")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            iou = compute_iou(logits, masks)
            train_iou += iou
            
            pbar.set_postfix({'loss': loss.item(), 'iou': iou})
            
        train_loss /= len(train_loader)
        train_iou /= len(train_loader)
        
        # Validation
        model.eval()
        valid_loss = 0.0
        valid_iou = 0.0
        
        with torch.no_grad():
            pbar = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{start_epoch + args.epochs} [Valid]")
            for images, masks in pbar:
                images = images.to(device)
                masks = masks.to(device)
                
                logits = model(images)
                loss = criterion(logits, masks)
                
                valid_loss += loss.item()
                iou = compute_iou(logits, masks)
                valid_iou += iou
                
                pbar.set_postfix({'loss': loss.item(), 'iou': iou})
                
        valid_loss /= len(valid_loader)
        valid_iou /= len(valid_loader)
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"Train Loss: {train_loss:.4f} | Train IoU: {train_iou:.4f}")
        print(f"Valid Loss: {valid_loss:.4f} | Valid IoU: {valid_iou:.4f}")
        
        scheduler.step(valid_iou)
        
        if valid_iou > best_iou:
            best_iou = valid_iou
            print(f"--> New best valid IoU: {best_iou:.4f}. Saving model...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_iou': best_iou,
            }, 'best_model.pth')
            
        # Save validation grid
        save_validation_grid(model, valid_dataset, device, epoch)
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='.', help='Dataset directory')
    parser.add_argument('--encoder', type=str, default='mit_b2', help='Segformer encoder version (mit_b0 - mit_b5)')
    parser.add_argument('--img_size', type=int, default=512, help='Image resolution')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from (e.g. best_model.pth)')
    args = parser.parse_args()
    
    train(args)
