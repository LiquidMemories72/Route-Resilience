import matplotlib.pyplot as plt
import numpy as np
import os
import json

class DebugLogger:
    def __init__(self, output_dir="graph_debug"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.candidates = []
        self.original_endpoints = set()
        self.consolidated_endpoints = set()
        self.searches_saved = 0
        
    def log_endpoints(self, original_eps, consolidated_eps, searches_saved_estimate):
        self.original_endpoints.update(original_eps)
        self.consolidated_endpoints.update(consolidated_eps)
        self.searches_saved += searches_saved_estimate
        
    def log_candidate(self, stage, src, dst, dist, score, path, explored_nodes, final_cost, is_valid, reason):
        self.candidates.append({
            'stage': stage,
            'src': (int(src[0]), int(src[1])),
            'dst': (int(dst[0]), int(dst[1])) if dst is not None else None,
            'dist': float(dist),
            'score': float(score),
            'path_length': len(path) if path else 0,
            'explored_nodes': int(explored_nodes),
            'final_cost': float(final_cost),
            'is_valid': is_valid,
            'reason': reason,
            'path': path
        })
        
    def save_log(self, filename="repair_log.json"):
        # Save JSON without full paths to save space
        summary = []
        for c in self.candidates:
            summary.append({
                'stage': c['stage'],
                'src': c['src'],
                'dst': c['dst'],
                'dist': c['dist'],
                'score': c['score'],
                'path_length': c['path_length'],
                'explored_nodes': c['explored_nodes'],
                'final_cost': c['final_cost'],
                'is_valid': c['is_valid'],
                'reason': c['reason']
            })
            
        full_log = {
            'candidates': summary,
            'optimization_stats': {
                'original_endpoints': len(self.original_endpoints),
                'consolidated_endpoints': len(self.consolidated_endpoints),
                'pruned_endpoints': len(self.original_endpoints) - len(self.consolidated_endpoints),
                'estimated_searches_saved': self.searches_saved
            }
        }
            
        with open(os.path.join(self.output_dir, filename), 'w') as f:
            json.dump(full_log, f, indent=2)
            
    def visualize(self, prob_map, filename="repair_debug_viz.png"):
        plt.figure(figsize=(12, 12))
        plt.imshow(prob_map, cmap='gray', alpha=0.5)
        plt.title(f"Topology Repair Debug\nSaved ~{self.searches_saved} redundant A* searches")
        
        for c in self.candidates:
            path = c['path']
            if not path:
                continue
                
            path_np = np.array(path)
            
            if c['is_valid']:
                plt.plot(path_np[:, 1], path_np[:, 0], 'g-', linewidth=2, alpha=0.8)
            else:
                plt.plot(path_np[:, 1], path_np[:, 0], 'r--', linewidth=1, alpha=0.5)
                
                mid_idx = len(path_np) // 2
                plt.text(path_np[mid_idx, 1], path_np[mid_idx, 0], c['reason'], 
                         color='red', fontsize=6, alpha=0.7)
                         
        # Plot endpoints
        # Original but pruned (red cross)
        pruned = self.original_endpoints - self.consolidated_endpoints
        if pruned:
            pruned_y, pruned_x = zip(*pruned)
            plt.plot(pruned_x, pruned_y, 'rx', markersize=4, label='Pruned Endpoints')
            
        # Consolidated / Representative (blue circle)
        if self.consolidated_endpoints:
            cons_y, cons_x = zip(*self.consolidated_endpoints)
            plt.plot(cons_x, cons_y, 'bo', markersize=4, label='Representative Endpoints')
            
        plt.legend(loc='upper right', fontsize=8)
        plt.axis('off')
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
