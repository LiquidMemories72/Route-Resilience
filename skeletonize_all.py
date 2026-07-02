import os
import cv2
import numpy as np
from skimage.morphology import skeletonize
from PIL import Image

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
                print(f"Failed to read {mask_path}")
                continue
            
            # Binarize and skeletonize
            binary_mask = mask > 0
            skeleton = skeletonize(binary_mask)
            
            # Convert back to uint8
            skeleton_img = (skeleton * 255).astype(np.uint8)
            
            # Dilate to make it thicker
            kernel = np.ones((3, 3), np.uint8)
            skeleton_img = cv2.dilate(skeleton_img, kernel, iterations=1)
            
            # Save
            out_path = os.path.join(folder_path, "skeleton.png")
            cv2.imwrite(out_path, skeleton_img)
            print(f"Saved skeleton for {folder_name} at {out_path}")

if __name__ == "__main__":
    main()
