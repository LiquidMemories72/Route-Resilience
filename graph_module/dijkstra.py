import numpy as np
import heapq

def run_dijkstra_local(src, target, target_pt, cost_func, img_shape, margin=15):
    """
    Run Dijkstra's algorithm in an adaptive search window (tube around the straight line).
    
    Args:
        src: (y, x) starting point
        target: (y, x) end point OR list of (y, x) representing a branch
        target_pt: (y, x) the specific point we are trying to reach (for line-of-sight)
        cost_func: CostFunction object
        img_shape: (H, W) of the full image
        margin: maximum perpendicular distance allowed from the straight line
        
    Returns:
        path: list of (y, x) coordinates from src to dst, or None if no path found
    """
    if isinstance(target, tuple):
        targets = set([target])
        bbox_targets = [target]
    else:
        # Target is a branch (array of coordinates)
        targets = set([tuple(pt) for pt in target])
        bbox_targets = target
        
    # Bounding box for quick filtering
    ys = [src[0]] + [pt[0] for pt in bbox_targets]
    xs = [src[1]] + [pt[1] for pt in bbox_targets]
    
    min_y = max(0, min(ys) - margin)
    max_y = min(img_shape[0] - 1, max(ys) + margin)
    min_x = max(0, min(xs) - margin)
    max_x = min(img_shape[1] - 1, max(xs) + margin)
    
    # Line equation parameters for the "tube" constraint
    y1, x1 = src
    y2, x2 = target_pt
    line_dist = np.hypot(y2 - y1, x2 - x1)
    
    dist_map = {}
    prev = {}
    prev_dir = {}
    
    pq = []
    
    start_pos = src
    dist_map[start_pos] = 0.0
    prev[start_pos] = None
    prev_dir[start_pos] = None
    
    heapq.heappush(pq, (0.0, start_pos[0], start_pos[1]))
    
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]
                  
    target_reached = None
    
    while pq:
        d, y, x = heapq.heappop(pq)
        u = (y, x)
        
        if d > dist_map.get(u, float('inf')):
            continue
            
        if u in targets:
            target_reached = u
            break
            
        p_dir = prev_dir[u]
        
        for dy, dx in directions:
            vy, vx = y + dy, x + dx
            
            # 1. Check bounds
            if not (min_y <= vy <= max_y and min_x <= vx <= max_x):
                continue
                
            # 2. Check tube constraint (perpendicular distance to line src->target_pt)
            if line_dist > 0:
                perp_dist = abs((x2 - x1) * (y1 - vy) - (y2 - y1) * (x1 - vx)) / line_dist
                if perp_dist > margin:
                    continue
                
            v = (vy, vx)
            
            edge_cost, new_dir = cost_func.get_cost(u, v, p_dir)
            new_dist = d + edge_cost
            
            if new_dist < dist_map.get(v, float('inf')):
                dist_map[v] = new_dist
                prev[v] = u
                prev_dir[v] = new_dir
                heapq.heappush(pq, (new_dist, vy, vx))
                
    if target_reached is None:
        return None
        
    path = []
    curr = target_reached
    while curr is not None:
        path.append(curr)
        curr = prev[curr]
        
    path.reverse()
    return path
