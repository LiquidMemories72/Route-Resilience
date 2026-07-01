import numpy as np
import cv2
from skimage.morphology import skeletonize

try:
    from .endpoint_detection import get_endpoints, get_connected_components, group_endpoints_by_component, estimate_endpoint_tangent, consolidate_topology_endpoints
    from .branch_detection import get_branch_points, extract_branches
    from .candidate_pairs import generate_candidate_pairs, generate_endpoint_to_branch_candidates
    from .scoring import CostFunction
    from .astar import run_bidirectional_astar
    from .validation import validate_path
    from .debug_viz import DebugLogger
except ImportError:
    from endpoint_detection import get_endpoints, get_connected_components, group_endpoints_by_component, estimate_endpoint_tangent, consolidate_topology_endpoints
    from branch_detection import get_branch_points, extract_branches
    from candidate_pairs import generate_candidate_pairs, generate_endpoint_to_branch_candidates
    from scoring import CostFunction
    from astar import run_bidirectional_astar
    from validation import validate_path
    from debug_viz import DebugLogger

class TopologyRepair:
    def __init__(self, probability_map, binary_mask, skeleton, 
                 w_prob=2.0, w_dist=1.0, w_dir=2.0, w_curve=1.5, w_target_align=2.0, w_ep_tangent=1.5,
                 prob_threshold=0.5, w_low_conf=5.0,
                 min_avg_prob=0.3, max_low_conf_ratio=0.5,
                 debug_dir="graph_debug"):
        
        self.prob_map = probability_map
        self.binary_mask = binary_mask
        self.skeleton = skeleton.copy()
        
        self.w_prob = w_prob
        self.w_dist = w_dist
        self.w_dir = w_dir
        self.w_curve = w_curve
        self.w_target_align = w_target_align
        self.w_ep_tangent = w_ep_tangent
        self.prob_threshold = prob_threshold
        self.w_low_conf = w_low_conf
        
        self.min_avg_prob = min_avg_prob
        self.max_low_conf_ratio = max_low_conf_ratio
        
        self.logger = DebugLogger(output_dir=debug_dir)
        
        self.stages = [
            {'name': 'Stage 1 (Small)', 'max_dist': 20, 'min_score': 0.1, 'margin': 5, 'max_path_length': 30},
            {'name': 'Stage 2 (Medium)', 'max_dist': 50, 'min_score': 0.3, 'margin': 10, 'max_path_length': 70},
            {'name': 'Stage 3 (Large)', 'max_dist': 80, 'min_score': 0.5, 'margin': 15, 'max_path_length': 100}
        ]

    def _run_stage(self, repaired_skeleton, stage_config):
        stage_name = stage_config['name']
        search_radius = stage_config['max_dist']
        min_score = stage_config['min_score']
        margin = stage_config['margin']
        max_path_length = stage_config['max_path_length']
        
        endpoints = get_endpoints(repaired_skeleton)
        labeled_comp, num_comp = get_connected_components(repaired_skeleton)
        
        branch_points, branch_mask = get_branch_points(repaired_skeleton)
        labeled_branches, num_branches = extract_branches(repaired_skeleton, branch_mask)
        
        tangents = [estimate_endpoint_tangent(ep, repaired_skeleton, branch_mask) for ep in endpoints]
        
        original_endpoints = endpoints.copy()
        
        # Topology-aware consolidation
        endpoints, tangents = consolidate_topology_endpoints(
            endpoints, tangents, repaired_skeleton, branch_mask, 
            length_threshold=15, tangent_dot_threshold=0.5
        )
        
        searches_saved_estimate = len(original_endpoints) - len(endpoints)
        self.logger.log_endpoints(original_endpoints, endpoints, searches_saved_estimate)
        
        ep_ep_candidates = generate_candidate_pairs(
            endpoints, tangents, labeled_comp, search_radius, min_score
        )
        
        ep_branch_candidates = generate_endpoint_to_branch_candidates(
            endpoints, tangents, labeled_comp, labeled_branches, search_radius, min_score
        )
        
        all_candidates = ep_ep_candidates + ep_branch_candidates
        all_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        paths_merged = 0
        
        # Traversable mask: prob > threshold OR close to skeleton
        skeleton_uint8 = (repaired_skeleton > 0).astype(np.uint8)
        dist_to_skeleton = cv2.distanceTransform(1 - skeleton_uint8, cv2.DIST_L2, 5)
        traversable_mask = (self.prob_map > self.prob_threshold) | (dist_to_skeleton <= margin)
        
        for cand in all_candidates:
            src = cand['src']
            src_tangent = cand.get('src_tangent')
            
            if cand['type'] == 'ep-ep':
                target = cand['dst']
                target_pt = target
                dst_tangent = cand.get('dst_tangent')
                src_label = labeled_comp[src[0], src[1]]
                dst_label = labeled_comp[target_pt[0], target_pt[1]]
                if src_label != 0 and src_label == dst_label:
                    self.logger.log_candidate(stage_name, src, target_pt,
                                              cand.get('dist', 0), cand['score'],
                                              None, 0, 0.0, False, "Already connected after earlier merge")
                    continue
            else:
                target = cand['dst_branch_coords']
                target_pt = cand['target_pt']
                dst_tangent = None
                
            cost_func = CostFunction(
                probability_map=self.prob_map,
                w_prob=self.w_prob, w_dist=self.w_dist, w_dir=self.w_dir, 
                w_curve=self.w_curve, w_target_align=self.w_target_align,
                prob_threshold=self.prob_threshold, w_low_conf=self.w_low_conf,
                target_pt=target_pt,
                w_ep_tangent=self.w_ep_tangent,
                src_tangent=src_tangent, dst_tangent=dst_tangent, src_pt=src
            )
            
            path, explored_count, final_cost = run_bidirectional_astar(
                src=src,
                target=target,
                target_pt=target_pt,
                cost_func=cost_func,
                img_shape=self.prob_map.shape,
                margin=margin,
                traversable_mask=traversable_mask
            )
            
            is_valid, reason = validate_path(
                path, self.prob_map, 
                max_length=max_path_length,
                min_avg_prob=self.min_avg_prob,
                max_low_conf_ratio=self.max_low_conf_ratio,
                prob_threshold=self.prob_threshold
            )
            
            dist_val = cand.get('dist', cand.get('min_dist'))
            score_val = cand['score']
            self.logger.log_candidate(stage_name, src, target_pt, dist_val, score_val, path, explored_count, final_cost, is_valid, reason)
            
            if is_valid and path:
                for y, x in path:
                    repaired_skeleton[y, x] = 1
                repaired_skeleton = skeletonize(repaired_skeleton > 0).astype(np.uint8)
                labeled_comp, _ = get_connected_components(repaired_skeleton)
                
                # Update traversable mask around the new path
                for y, x in path:
                    y_min = max(0, y - margin)
                    y_max = min(self.prob_map.shape[0], y + margin + 1)
                    x_min = max(0, x - margin)
                    x_max = min(self.prob_map.shape[1], x + margin + 1)
                    traversable_mask[y_min:y_max, x_min:x_max] = True
                    
                paths_merged += 1
            
        print(f"[TopologyRepair] {stage_name}: Merged {paths_merged} paths out of {len(all_candidates)} candidates.")
        return repaired_skeleton

    def run(self):
        """
        Run the topology repair pipeline through all hierarchical stages.
        """
        repaired_skeleton = self.skeleton.copy()
        
        for stage_config in self.stages:
            repaired_skeleton = self._run_stage(repaired_skeleton, stage_config)
            
        self.logger.save_log()
        self.logger.visualize(self.prob_map)
                
        return repaired_skeleton
