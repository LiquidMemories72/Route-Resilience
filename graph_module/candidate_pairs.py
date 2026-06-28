"""Module for generating candidate connection pairs between endpoints and branches.

The original heuristic based on average road probability has been removed.
Candidates are now ranked using distance, endpoint tangent alignment,
orientation agreement, and connected component geometry.
"""

import numpy as np

def score_candidate(p1, p2, t1, t2, search_radius):
    """Score a candidate connection between two endpoints.

    Parameters
    ----------
    p1, p2 : tuple
        (y, x) coordinates of the two endpoints.
    t1, t2 : tuple or None
        Tangent vectors at the endpoints. ``t2`` may be ``None`` for branch connections.
    search_radius : float
        Maximum allowable distance for a candidate.

    Returns
    -------
    score : float
        Confidence score (higher is better).
    dist : float
        Euclidean distance between the points.
    reason : str
        Reason string indicating validity.
    """
    dist = np.hypot(p1[0] - p2[0], p1[1] - p2[1])
    if dist > search_radius or dist == 0:
        return -1.0, 0.0, "distance_exceeded"
        
    line_dir = ((p2[0] - p1[0]) / dist, (p2[1] - p1[1]) / dist)
    
    # How well does t1 point towards p2?
    t1_align = t1[0] * line_dir[0] + t1[1] * line_dir[1]
    
    # Distance score (1.0 for dist=0, 0.0 for dist=search_radius)
    dist_score = 1.0 - (dist / search_radius)
    
    score = dist_score + max(0, t1_align) * 0.5
    
    if t2 is not None:
        # How well does t2 point towards p1?
        t2_align = t2[0] * (-line_dir[0]) + t2[1] * (-line_dir[1])
        # How well are t1 and t2 aligned to each other? (should be opposite)
        ep_align = t1[0] * (-t2[0]) + t1[1] * (-t2[1])
        score += max(0, t2_align) * 0.5 + max(0, ep_align) * 0.5
    else:
        # If no t2 (e.g. branch), give a boost to keep scales comparable
        score += 1.0
        
    return score, dist, "valid"

def generate_candidate_pairs(endpoints, tangents, labeled_components, search_radius, min_score=0.1):
    candidates = []
    n = len(endpoints)
    
    for i in range(n):
        for j in range(i + 1, n):
            p1 = endpoints[i]
            p2 = endpoints[j]
            
            l1 = labeled_components[p1]
            l2 = labeled_components[p2]
            
            if l1 == l2:
                continue
                
            score, dist, reason = score_candidate(p1, p2, tangents[i], tangents[j], search_radius)
            if score >= min_score and reason == "valid":
                candidates.append({
                    'src': p1, 'dst': p2, 
                    'src_tangent': tangents[i], 'dst_tangent': tangents[j],
                    'type': 'ep-ep', 'dist': dist, 'score': score
                })
    return candidates

def generate_endpoint_to_branch_candidates(endpoints, tangents, labeled_components, labeled_branches, search_radius, min_score=0.1):
    candidates = []
    branch_coords = {}
    y_coords, x_coords = np.where(labeled_branches > 0)
    for y, x in zip(y_coords, x_coords):
        label = labeled_branches[y, x]
        if label not in branch_coords:
            branch_coords[label] = []
        branch_coords[label].append((y, x))
        
    for label in branch_coords:
        branch_coords[label] = np.array(branch_coords[label])
        
    for i, ep in enumerate(endpoints):
        ep_label = labeled_components[ep]
        t1 = tangents[i]
        
        for branch_label, coords in branch_coords.items():
            first_pt = tuple(coords[0])
            if labeled_components[first_pt] == ep_label:
                continue
                
            # Find the best point on the branch to connect to
            diffs = coords - np.array(ep)
            dists = np.hypot(diffs[:, 0], diffs[:, 1])
            min_dist_idx = np.argmin(dists)
            min_dist = dists[min_dist_idx]
            
            if min_dist <= search_radius:
                best_pt = tuple(coords[min_dist_idx])
                score, dist, reason = score_candidate(ep, best_pt, t1, None, search_radius)
                
                if score >= min_score and reason == "valid":
                    candidates.append({
                        'src': ep, 'dst_branch_label': branch_label, 'dst_branch_coords': coords,
                        'src_tangent': t1, 'dst_tangent': None,
                        'type': 'ep-branch', 'min_dist': min_dist, 'score': score, 'target_pt': best_pt
                    })
    return candidates
