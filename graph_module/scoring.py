import numpy as np

class CostFunction:
    def __init__(self, probability_map, w_prob=1.0, w_dist=1.0, w_dir=1.0, w_curve=1.0, 
                 w_target_align=1.0, prob_threshold=0.5, w_low_conf=2.0, target_pt=None,
                 w_ep_tangent=1.0, src_tangent=None, dst_tangent=None, src_pt=None):
        """
        Args:
            probability_map: 2D numpy array of probabilities [0, 1]
            w_prob: weight for inverse probability
            w_dist: weight for euclidean distance
            w_dir: weight for direction change
            w_curve: weight for curvature (sharp turns)
            w_target_align: weight to reward moving towards target_pt
            prob_threshold: threshold below which a pixel is considered low confidence
            w_low_conf: additional penalty for moving into low confidence regions
            target_pt: (y, x) the final destination
            w_ep_tangent: weight for endpoint tangent consistency
            src_tangent: (dy, dx) tangent vector at source
            dst_tangent: (dy, dx) tangent vector at destination
            src_pt: (y, x) source point
        """
        self.prob_map = probability_map
        self.w_prob = w_prob
        self.w_dist = w_dist
        self.w_dir = w_dir
        self.w_curve = w_curve
        self.w_target_align = w_target_align
        self.prob_threshold = prob_threshold
        self.w_low_conf = w_low_conf
        self.target_pt = target_pt
        self.w_ep_tangent = w_ep_tangent
        self.src_tangent = src_tangent
        self.dst_tangent = dst_tangent
        self.src_pt = src_pt
        
    def get_cost(self, u, v, prev_dir=None):
        # 1. Distance cost
        dy = v[0] - u[0]
        dx = v[1] - u[1]
        dist = np.hypot(dy, dx)
        cost_dist = self.w_dist * dist
        
        if dist > 0:
            new_dir = (dy / dist, dx / dist)
        else:
            new_dir = (0, 0)
            
        # 2. Probability cost
        p = self.prob_map[v[0], v[1]]
        cost_prob = self.w_prob * (1.0 - p)
        
        if p < self.prob_threshold:
            cost_prob += self.w_low_conf
            
        # 3. Direction and Curvature cost
        cost_dir = 0.0
        cost_curve = 0.0
        
        if prev_dir is not None and dist > 0:
            cos_sim = prev_dir[0] * new_dir[0] + prev_dir[1] * new_dir[1]
            dir_penalty = 1.0 - cos_sim
            cost_dir = self.w_dir * dir_penalty
            
            if cos_sim < 0:
                cost_curve = self.w_curve * abs(cos_sim)
                
        # 4. Target alignment (Reward moving towards the target)
        cost_align = 0.0
        if self.target_pt is not None and dist > 0:
            ty = self.target_pt[0] - u[0]
            tx = self.target_pt[1] - u[1]
            tdist = np.hypot(ty, tx)
            if tdist > 0:
                t_dir = (ty / tdist, tx / tdist)
                t_sim = new_dir[0] * t_dir[0] + new_dir[1] * t_dir[1]
                # t_sim is [-1, 1]. We want to penalize moving away, reward moving towards
                cost_align = self.w_target_align * (1.0 - t_sim)
                
        # 5. Endpoint Tangent Consistency
        cost_ep_tangent = 0.0
        if dist > 0:
            if self.src_pt is not None and self.src_tangent is not None:
                d_src = np.hypot(u[0] - self.src_pt[0], u[1] - self.src_pt[1])
                if d_src < 20: 
                    sim_src = new_dir[0] * self.src_tangent[0] + new_dir[1] * self.src_tangent[1]
                    weight = max(0, 1.0 - d_src / 20.0)
                    cost_ep_tangent += self.w_ep_tangent * weight * (1.0 - sim_src)
                    
            if self.target_pt is not None and self.dst_tangent is not None:
                d_dst = np.hypot(u[0] - self.target_pt[0], u[1] - self.target_pt[1])
                if d_dst < 20:
                    # new_dir should be aligned with -dst_tangent (entering target)
                    sim_dst = new_dir[0] * (-self.dst_tangent[0]) + new_dir[1] * (-self.dst_tangent[1])
                    weight = max(0, 1.0 - d_dst / 20.0)
                    cost_ep_tangent += self.w_ep_tangent * weight * (1.0 - sim_dst)
                
        total_cost = cost_dist + cost_prob + cost_dir + cost_curve + cost_align + cost_ep_tangent
        return total_cost, new_dir
