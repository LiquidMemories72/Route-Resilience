import numpy as np
import cv2
from scipy import ndimage

def get_branch_points(skeleton):
    """
    Detect branch points (junctions) in a 1-pixel wide skeleton image.
    A branch point is a skeleton pixel that has > 2 neighbors in its 8-neighborhood.
    
    Args:
        skeleton: 2D numpy array (binary, 0 or >0)
        
    Returns:
        List of tuples (y, x) representing branch point coordinates.
    """
    skel = (skeleton > 0).astype(np.uint8)
    
    # Kernel to count neighbors including center
    kernel = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]], dtype=np.uint8)
    
    neighbor_count = cv2.filter2D(skel, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    
    # A branch point has > 3 in the neighbor count (itself + >2 neighbors)
    # AND it must be part of the skeleton
    branch_mask = (neighbor_count > 3) & (skel == 1)
    
    y_coords, x_coords = np.where(branch_mask)
    branch_points = list(zip(y_coords, x_coords))
    
    return branch_points, branch_mask

def extract_branches(skeleton, branch_mask):
    """
    Extract individual branches by removing branch points from the skeleton.
    
    Args:
        skeleton: 2D numpy array (binary, 0 or >0)
        branch_mask: 2D boolean or binary array indicating branch points
        
    Returns:
        labeled_branches: 2D numpy array where each branch has a unique label
        num_branches: integer count of branches
    """
    skel = (skeleton > 0).astype(np.uint8)
    
    # Remove branch points
    branches_only = skel.copy()
    branches_only[branch_mask > 0] = 0
    
    # Label the remaining connected components (branches)
    # 8-connectivity
    structure = np.ones((3, 3), dtype=np.int32)
    labeled_branches, num_branches = ndimage.label(branches_only, structure=structure)
    
    return labeled_branches, num_branches
