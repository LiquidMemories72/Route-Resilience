import matplotlib.pyplot as plt
import numpy as np
import os
import json

class DebugLogger:
    def __init__(self, output_dir="graph_debug"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.candidates = []
        
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
            
        with open(os.path.join(self.output_dir, filename), 'w') as f:
            json.dump(summary, f, indent=2)
            
    def visualize(self, prob_map, filename="repair_debug_viz.png"):
        plt.figure(figsize=(12, 12))
        plt.imshow(prob_map, cmap='gray', alpha=0.5)
        plt.title("Topology Repair Debug Visualization")
        
        for c in self.candidates:
            path = c['path']
            if not path:
                continue
                
            path_np = np.array(path)
            
            if c['is_valid']:
                # Plot accepted in green
                plt.plot(path_np[:, 1], path_np[:, 0], 'g-', linewidth=2, alpha=0.8)
                # Plot endpoints
                plt.plot(c['src'][1], c['src'][0], 'go', markersize=4)
                plt.plot(c['dst'][1], c['dst'][0], 'go', markersize=4)
            else:
                # Plot rejected in red
                plt.plot(path_np[:, 1], path_np[:, 0], 'r--', linewidth=1, alpha=0.5)
                # Plot endpoints
                plt.plot(c['src'][1], c['src'][0], 'ro', markersize=3)
                plt.plot(c['dst'][1], c['dst'][0], 'ro', markersize=3)
                
                # Add text for reason near the middle of the path
                mid_idx = len(path_np) // 2
                plt.text(path_np[mid_idx, 1], path_np[mid_idx, 0], c['reason'], 
                         color='red', fontsize=6, alpha=0.7)
                         
        plt.axis('off')
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
