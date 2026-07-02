import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
import sys

# Add current directory to path so we can import graph_module
sys.path.append(os.path.dirname(__file__))
from graph_module.endpoint_detection import get_endpoints

def main():
    base_dir = r"d:\S files\Samples"
    
    if not os.path.exists(base_dir):
        print(f"Base directory does not exist: {base_dir}")
        return

    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        if os.path.isdir(folder_path):
            mask_path = os.path.join(folder_path, "cleaned_mask.png")
            if not os.path.exists(mask_path):
                print(f"Skipping {folder_name}: No cleaned_mask.png found")
                continue
            
            # Read mask
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            
            # Binarize and skeletonize (1-pixel wide for endpoint detection)
            binary_mask = mask > 0
            skeleton = skeletonize(binary_mask)
            
            # Get endpoints
            endpoints = get_endpoints(skeleton)
            
            # Dilate skeleton for visualization
            skeleton_viz = (skeleton * 255).astype(np.uint8)
            kernel = np.ones((3, 3), np.uint8)
            skeleton_viz = cv2.dilate(skeleton_viz, kernel, iterations=1)
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.set_aspect('equal')
            ax.imshow(skeleton_viz, cmap='gray')
            
            # Plot endpoints (y, x coordinates)
            if endpoints:
                y_coords, x_coords = zip(*endpoints)
                ax.scatter(x_coords, y_coords, color='red', s=100, label='Endpoints')
            
            plt.axis('off')
            plt.title(f"Endpoints for {folder_name}")
            
            # Save
            out_path = os.path.join(folder_path, "endpoints_visualization.png")
            plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)
            print(f"Saved endpoint visualization for {folder_name} at {out_path}")

if __name__ == "__main__":
    main()
