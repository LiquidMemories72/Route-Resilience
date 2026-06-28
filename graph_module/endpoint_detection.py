import numpy as np
from scipy import ndimage
import cv2

def get_endpoints(skeleton):
    """
    Detect endpoints in a 1-pixel wide skeleton image.
    An endpoint is a skeleton pixel that has exactly 1 neighbor in its 8-neighborhood.
    
    Args:
        skeleton: 2D numpy array (binary, 0 or >0)
        
    Returns:
        List of tuples (y, x) representing endpoint coordinates.
    """
    # Ensure binary
    skel = (skeleton > 0).astype(np.uint8)
    
    # Kernel to count neighbors including center
    kernel = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]], dtype=np.uint8)
    
    # Convolve to get neighbor count for each pixel
    neighbor_count = cv2.filter2D(skel, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    
    # An endpoint has exactly 2 in the neighbor count (itself + 1 neighbor)
    # AND it must be part of the skeleton
    endpoints_mask = (neighbor_count == 2) & (skel == 1)
    
    y_coords, x_coords = np.where(endpoints_mask)
    endpoints = list(zip(y_coords, x_coords))
    
    return endpoints

def get_connected_components(skeleton):
    """
    Get connected components of the skeleton.
    
    Args:
        skeleton: 2D numpy array (binary, 0 or >0)
        
    Returns:
        labeled_array, num_features
    """
    skel = (skeleton > 0).astype(np.uint8)
    # 8-connectivity
    structure = np.ones((3, 3), dtype=np.int32)
    labeled_array, num_features = ndimage.label(skel, structure=structure)
    
    return labeled_array, num_features

def group_endpoints_by_component(endpoints, labeled_array):
    """
    Groups endpoints by their connected component label.
    
    Args:
        endpoints: List of (y, x) tuples
        labeled_array: 2D numpy array of labels
        
    Returns:
        Dictionary mapping component label -> list of endpoints
    """
    grouped = {}
    for y, x in endpoints:
        label = labeled_array[y, x]
        if label not in grouped:
            grouped[label] = []
        grouped[label].append((y, x))
    return grouped

def estimate_endpoint_tangent(endpoint, skeleton, branch_mask, length=10):
    """
    Estimate the tangent direction at an endpoint by walking backward along the skeleton.
    
    Args:
        endpoint: (y, x) tuple
        skeleton: 2D numpy array (binary)
        branch_mask: 2D numpy array (binary) indicating branch points
        length: Maximum number of pixels to walk backward
        
    Returns:
        (dy, dx) normalized vector pointing OUT from the endpoint.
    """
    y, x = endpoint
    path = [(y, x)]
    
    skel = (skeleton > 0).astype(np.uint8)
    h, w = skel.shape
    
    # 8-connectivity directions
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]
                  
    curr = (y, x)
    prev = None
    
    for _ in range(length):
        if branch_mask[curr[0], curr[1]] > 0 and curr != endpoint:
            break # Stop at branch point
            
        next_pixel = None
        for dy, dx in directions:
            ny, nx = curr[0] + dy, curr[1] + dx
            if 0 <= ny < h and 0 <= nx < w:
                if skel[ny, nx] == 1:
                    n_pt = (ny, nx)
                    if n_pt != prev:
                        next_pixel = n_pt
                        break
                        
        if next_pixel is None:
            break # End of line
            
        path.append(next_pixel)
        prev = curr
        curr = next_pixel
        
    if len(path) < 2:
        return (0.0, 0.0)
        
    # Vector from the last point found pointing towards the endpoint
    # This represents the outward tangent
    start_pt = path[-1]
    end_pt = path[0]
    
    dy = end_pt[0] - start_pt[0]
    dx = end_pt[1] - start_pt[1]
    
    norm = np.hypot(dy, dx)
    if norm > 0:
        return (dy / norm, dx / norm)
    return (0.0, 0.0)
