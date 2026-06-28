import numpy as np

def validate_path(path, prob_map, max_length=50, min_avg_prob=0.3, max_low_conf_ratio=0.5, prob_threshold=0.5):
    """
    Validate a reconstructed path based on defined criteria.
    Rejects only if multiple criteria fail, or if a critical threshold is breached.
    
    Args:
        path: List of (y, x) tuples
        prob_map: 2D numpy array of road probabilities
        max_length: maximum allowed path length in pixels
        min_avg_prob: minimum average road probability along the path
        max_low_conf_ratio: maximum fraction of path pixels below prob_threshold
        prob_threshold: threshold for considering a pixel 'low confidence'
        
    Returns:
        is_valid: boolean
        reason: string (for debugging/logging)
    """
    if path is None or len(path) == 0:
        return False, "Empty path"
        
    if len(path) > max_length:
        return False, f"Path too long ({len(path)} > {max_length})"
        
    probs = [prob_map[y, x] for y, x in path]
    avg_prob = np.mean(probs)
    low_conf_count = sum(1 for p in probs if p < prob_threshold)
    low_conf_ratio = low_conf_count / len(path)
    
    failures = []
    
    if avg_prob < min_avg_prob:
        failures.append(f"Low avg prob ({avg_prob:.2f} < {min_avg_prob})")
        
    if low_conf_ratio > max_low_conf_ratio:
        failures.append(f"Too many low conf ({low_conf_ratio:.2f} > {max_low_conf_ratio})")
        
    # Curvature check
    if len(path) > 3:
        for i in range(1, len(path) - 1):
            p1 = np.array(path[i-1])
            p2 = np.array(path[i])
            p3 = np.array(path[i+1])
            
            v1 = p2 - p1
            v2 = p3 - p2
            
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 > 0 and norm2 > 0:
                cos_sim = np.dot(v1, v2) / (norm1 * norm2)
                # If cos_sim < -0.5 (more than 120 degree turn in 1 pixel step)
                if cos_sim < -0.5:
                    failures.append("Excessive curvature")
                    break
                    
    # Reject if 2 or more non-critical failures
    if len(failures) >= 2:
        return False, " | ".join(failures)
        
    # Reject if avg_prob is EXTREMELY low (critical failure)
    if avg_prob < (min_avg_prob / 2.0):
        return False, f"Critical low avg prob ({avg_prob:.2f})"
        
    if len(failures) == 1:
        return True, f"Passed with warning: {failures[0]}"
        
    return True, "Valid"
