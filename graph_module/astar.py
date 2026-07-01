import numpy as np
import heapq

def run_bidirectional_astar(src, target, target_pt, cost_func, img_shape, margin, traversable_mask=None, max_explored_nodes=800):
    """
    Run Bidirectional A* in an adaptive elliptical search window.
    
    Args:
        src: (y, x) starting point
        target: (y, x) end point OR list of (y, x) representing a branch
        target_pt: (y, x) the specific point we are trying to reach (for line-of-sight/ellipse foci)
        cost_func: CostFunction object
        img_shape: (H, W) of the full image
        margin: adaptive margin for the elliptical search region
        traversable_mask: boolean array (H, W) where True means allowed to traverse
        max_explored_nodes: hard cap on total nodes expanded before giving up
        
    Returns:
        path: list of (y, x) coordinates from src to dst, or None if no path found
        explored_count: number of nodes explored (for debugging)
        final_cost: total cost of the path
    """
    if isinstance(target, tuple):
        targets = set([target])
    else:
        targets = set([tuple(pt) for pt in target])
        
    y1, x1 = src
    y2, x2 = target_pt
    focal_dist = np.hypot(y2 - y1, x2 - x1)
    
    fwd_dist = {src: 0.0}
    bwd_dist = {t: 0.0 for t in targets}
    
    fwd_prev = {src: None}
    bwd_prev = {t: None for t in targets}
    
    fwd_prev_dir = {src: None}
    
    fwd_pq = []
    bwd_pq = []
    
    heapq.heappush(fwd_pq, (focal_dist, 0.0, src[0], src[1]))
    for t in targets:
        h = np.hypot(src[0] - t[0], src[1] - t[1])
        heapq.heappush(bwd_pq, (h, 0.0, t[0], t[1]))
        
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]
                  
    mu = float('inf')
    best_meeting_point = None
    explored_count = 0
    
    while fwd_pq and bwd_pq:
        if explored_count >= max_explored_nodes:
            break
            
        fwd_min_f = fwd_pq[0][0]
        bwd_min_f = bwd_pq[0][0]
        
        if fwd_min_f + bwd_min_f >= mu:
            break
            
        if fwd_pq and (not bwd_pq or fwd_pq[0][0] <= bwd_pq[0][0]):
            _, d, y, x = heapq.heappop(fwd_pq)
            u = (y, x)
            is_fwd = True
        else:
            _, d, y, x = heapq.heappop(bwd_pq)
            u = (y, x)
            is_fwd = False
            
        dist_map = fwd_dist if is_fwd else bwd_dist
        if d > dist_map.get(u, float('inf')):
            continue
            
        explored_count += 1
        
        if is_fwd:
            p_dir = fwd_prev_dir[u]
        else:
            p_dir = None
            
        for dy, dx in directions:
            vy, vx = y + dy, x + dx
            v = (vy, vx)
            
            if not (0 <= vy < img_shape[0] and 0 <= vx < img_shape[1]):
                continue
                
            if traversable_mask is not None and not traversable_mask[vy, vx]:
                continue
                
            d_src_v = np.hypot(vy - y1, vx - x1)
            d_v_tgt = np.hypot(vy - y2, vx - x2)
            if d_src_v + d_v_tgt > focal_dist + margin:
                continue
                
            if is_fwd:
                edge_cost, new_dir = cost_func.get_cost(u, v, p_dir)
                new_dist = d + edge_cost
                if new_dist < fwd_dist.get(v, float('inf')):
                    fwd_dist[v] = new_dist
                    fwd_prev[v] = u
                    fwd_prev_dir[v] = new_dir
                    
                    h = np.hypot(vy - y2, vx - x2)
                    heapq.heappush(fwd_pq, (new_dist + h, new_dist, vy, vx))
                    
                    if v in bwd_dist:
                        path_len = fwd_dist[v] + bwd_dist[v]
                        if path_len < mu:
                            mu = path_len
                            best_meeting_point = v
            else:
                edge_cost, _ = cost_func.get_cost(v, u, None) 
                new_dist = d + edge_cost
                
                if new_dist < bwd_dist.get(v, float('inf')):
                    bwd_dist[v] = new_dist
                    bwd_prev[v] = u
                    
                    h = np.hypot(vy - y1, vx - x1)
                    heapq.heappush(bwd_pq, (new_dist + h, new_dist, vy, vx))
                    
                    if v in fwd_dist:
                        path_len = fwd_dist[v] + bwd_dist[v]
                        if path_len < mu:
                            mu = path_len
                            best_meeting_point = v
                            
    if best_meeting_point is None:
        return None, explored_count, -1.0
        
    path = []
    curr = best_meeting_point
    while curr is not None:
        path.append(curr)
        curr = fwd_prev[curr]
    path.reverse()
    
    curr = bwd_prev[best_meeting_point]
    while curr is not None:
        path.append(curr)
        curr = bwd_prev[curr]
        
    return path, explored_count, mu
