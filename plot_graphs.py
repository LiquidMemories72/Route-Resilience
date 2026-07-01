import os
import json
import matplotlib.pyplot as plt

def plot_graph(nodes_file, edges_file, output_file):
    if not os.path.exists(nodes_file) or not os.path.exists(edges_file):
        print(f"Skipping {os.path.basename(os.path.dirname(nodes_file))}: Missing nodes.json or edges.json")
        return

    with open(nodes_file, 'r') as f:
        nodes = json.load(f)

    with open(edges_file, 'r') as f:
        edges = json.load(f)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    
    # Invert y-axis assuming row=y, col=x to match typical image coordinates (row 0 at top)
    ax.invert_yaxis() 

    # Plot edges based on geometry
    for edge in edges:
        geom = edge.get('geometry', [])
        if geom:
            # geometry is a list of [row, col]
            cols = [pt[1] for pt in geom]
            rows = [pt[0] for pt in geom]
            ax.plot(cols, rows, color='blue', linewidth=1.5, alpha=0.8)

    # Plot nodes
    node_cols = [n['col'] for n in nodes]
    node_rows = [n['row'] for n in nodes]
    
    # Differentiate endpoints and junctions if 'type' is present
    end_cols = [n['col'] for n in nodes if n.get('type') == 'endpoint']
    end_rows = [n['row'] for n in nodes if n.get('type') == 'endpoint']
    junc_cols = [n['col'] for n in nodes if n.get('type') == 'junction']
    junc_rows = [n['row'] for n in nodes if n.get('type') == 'junction']

    if junc_cols:
        ax.scatter(junc_cols, junc_rows, color='red', s=20, zorder=5)
    if end_cols:
        ax.scatter(end_cols, end_rows, color='green', s=20, zorder=5)
    
    # Fallback if no type is found
    if not junc_cols and not end_cols:
        ax.scatter(node_cols, node_rows, color='red', s=15, zorder=5)

    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0,0)
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"Saved {output_file}")

import sys

if __name__ == "__main__":
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        base_dir = r"d:\S files\Samples"
    
    if not os.path.exists(base_dir):
        print(f"Base directory does not exist: {base_dir}")
        exit(1)

    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        if os.path.isdir(folder_path):
            nodes_file = os.path.join(folder_path, "nodes.json")
            edges_file = os.path.join(folder_path, "edges.json")
            output_file = os.path.join(folder_path, "graph_visualization.png")
            
            plot_graph(nodes_file, edges_file, output_file)
