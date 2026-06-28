import numpy as np
import cv2
import matplotlib.pyplot as plt
from graph_module import TopologyRepair

def create_synthetic_data():
    # Create a 100x100 synthetic image
    size = 100
    
    # Probability map (0 to 1)
    prob_map = np.zeros((size, size), dtype=np.float32)
    binary_mask = np.zeros((size, size), dtype=np.uint8)
    skeleton = np.zeros((size, size), dtype=np.uint8)
    
    # Draw a vertical road from y=10 to y=90 at x=50, with a gap between y=45 and y=55
    # Segment 1
    prob_map[10:45, 48:53] = 0.9
    binary_mask[10:45, 48:53] = 1
    skeleton[10:45, 50] = 1
    
    # Segment 2
    prob_map[55:90, 48:53] = 0.9
    binary_mask[55:90, 48:53] = 1
    skeleton[55:90, 50] = 1
    
    # But in the gap (45 to 55), let's say probability is slightly lower (e.g. 0.6) 
    # due to tree occlusion, so it was thresholded out of the binary mask
    prob_map[45:55, 48:53] = 0.6
    
    return prob_map, binary_mask, skeleton

def main():
    print("Generating synthetic data with a gap...")
    prob_map, binary_mask, skeleton = create_synthetic_data()
    
    print("Running TopologyRepair...")
    repair = TopologyRepair(
        probability_map=prob_map,
        binary_mask=binary_mask,
        skeleton=skeleton,
        w_prob=1.0,           
        w_dist=1.0,
        w_dir=1.0,
        w_curve=1.0
    )
    
    repaired_skeleton = repair.run()
    
    # Check if gap is bridged (at y=50, x=50)
    bridged = repaired_skeleton[50, 50] == 1
    print(f"Gap bridged successfully? {'Yes' if bridged else 'No'}")
    
    # Visualise
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Road Probability")
    plt.imshow(prob_map, cmap='viridis')
    plt.colorbar()
    
    plt.subplot(1, 3, 2)
    plt.title("Original Skeleton (Broken)")
    plt.imshow(skeleton, cmap='gray')
    
    plt.subplot(1, 3, 3)
    plt.title("Repaired Skeleton")
    plt.imshow(repaired_skeleton, cmap='gray')
    
    plt.savefig("synthetic_repair_test.png")
    print("Saved visualisation to synthetic_repair_test.png")

if __name__ == "__main__":
    main()
