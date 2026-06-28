import numpy as np

def validate_path(path, prob_map, max_length=50, min_avg_prob=0.3, max_low_conf_ratio=0.5, prob_threshold=0.5):
    """
    Validate a reconstructed path using multiple metrics.
    Rejects only if multiple criteria fail, or if a critical threshold is breached.
    """
    if path is None or len(path) == 0:
        return False, "Empty path"
        
    if len(path) > max_length:
        return False, f"Path too long ({len(path)} > {max_length})"
        
    probs = [prob_map[y, x] for y, x in path]
    avg_prob = np.mean(probs)
    median_prob = np.median(probs)
    
    low_conf_count = sum(1 for p in probs if p < prob_threshold)
    low_conf_ratio = low_conf_count / len(path)
    
    euclidean_dist = np.hypot(path[-1][0] - path[0][0], path[-1][1] - path[0][1])
    # Path efficiency = length / euclidean_dist. Closer to 1.0 is better.
    efficiency = len(path) / (euclidean_dist + 1e-5)
    
    failures = []
    
    if avg_prob < min_avg_prob:
        failures.append(f"Low avg prob ({avg_prob:.2f} < {min_avg_prob})")
        
    if median_prob < (min_avg_prob * 0.8): # slightly looser threshold for median
        failures.append(f"Low median prob ({median_prob:.2f})")
        
    if low_conf_ratio > max_low_conf_ratio:
        failures.append(f"Too many low conf ({low_conf_ratio:.2f} > {max_low_conf_ratio})")
        
    if efficiency > 2.0: # Path is more than twice as long as straight line
        failures.append(f"Low efficiency ({efficiency:.2f})")
        
    # Curvature check
    excessive_curves = 0
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
                    excessive_curves += 1
                    
    if excessive_curves > 1:
        failures.append(f"Excessive curvature ({excessive_curves} sharp turns)")
                    
    # Reject if 2 or more non-critical failures
    if len(failures) >= 2:
        return False, " | ".join(failures)
        
    # Reject if avg_prob is EXTREMELY low (critical failure)
    if avg_prob < (min_avg_prob / 2.0):
        return False, f"Critical low avg prob ({avg_prob:.2f})"
        
    if len(failures) == 1:
        return True, f"Passed with warning: {failures[0]}"
        
    return True, "Valid"
