"""Command-line and Python API for road-mask graph extraction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Union
import cv2
import networkx as nx
import numpy as np

try:
    from .graph_builder import extract_graph, graph_to_records
    from .graph_healing import heal_graph
    from .preprocess import cleanup_mask, load_binary_mask, save_binary_mask
    from .skeleton import make_skeleton, save_skeleton
    from .visualization import save_graph_overlay, save_node_visualization, save_skeleton_visualization
except ImportError:
    from graph_builder import extract_graph, graph_to_records
    from graph_healing import heal_graph
    from preprocess import cleanup_mask, load_binary_mask, save_binary_mask
    from skeleton import make_skeleton, save_skeleton
    from visualization import save_graph_overlay, save_node_visualization, save_skeleton_visualization

MaskInput = Union[str, Path, np.ndarray]


def _load_satellite(path: Union[str, Path]) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read satellite image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _write_records(graph: nx.MultiGraph, output_dir: Path) -> None:
    nodes, edges = graph_to_records(graph)
    (output_dir / "nodes.json").write_text(json.dumps(nodes, indent=2), encoding="utf-8")
    (output_dir / "edges.json").write_text(json.dumps(edges, indent=2), encoding="utf-8")
    summary = {
        "nodes": graph.number_of_nodes(), "edges": graph.number_of_edges(),
        "connected_components": nx.number_connected_components(graph),
        "total_length_pixels": float(sum(d.get("length", 0.0) for _, _, d in graph.edges(data=True))),
        "healed_edges": sum(bool(d.get("healed", False)) for _, _, d in graph.edges(data=True)),
        "coordinate_order": "row_col",
    }
    (output_dir / "graph_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def run_pipeline(
    mask: MaskInput,
    output_dir: Union[str, Path],
    satellite_image: Optional[Union[str, Path, np.ndarray]] = None,
    *,
    mask_threshold: int = 127,
    closing_kernel_size: int = 5,
    closing_iterations: int = 1,
    min_component_area: int = 50,
    endpoint_distance: float = 15.0,
    max_heading_angle: Optional[float] = 45.0,
    junction_merge_distance: float = 3.0,
    dangling_length: float = 5.0,
    allow_same_component_bridges: bool = False,
) -> nx.MultiGraph:
    """Run the full pipeline and return the healed graph; distances are pixels."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    binary = load_binary_mask(mask, mask_threshold) if isinstance(mask, (str, Path)) else np.asarray(mask)
    cleaned = cleanup_mask(binary, closing_kernel_size, closing_iterations, min_component_area, mask_threshold)
    skeleton = make_skeleton(cleaned)
    graph = heal_graph(
        extract_graph(skeleton), endpoint_distance, max_heading_angle,
        junction_merge_distance, dangling_length, allow_same_component_bridges,
    )
    save_binary_mask(cleaned, output / "cleaned_mask.png")
    save_skeleton(skeleton, output / "skeleton.png")
    save_skeleton_visualization(skeleton, output / "skeleton_visualization.png")
    save_node_visualization(skeleton, graph, output / "nodes.png")
    save_graph_overlay(cleaned, graph, output / "graph_on_mask.png", "Healed road graph on mask")
    if satellite_image is not None:
        satellite = _load_satellite(satellite_image) if isinstance(satellite_image, (str, Path)) else np.asarray(satellite_image)
        if satellite.shape[:2] != cleaned.shape:
            satellite = cv2.resize(
                satellite,
                (cleaned.shape[1], cleaned.shape[0])
            )
        save_graph_overlay(satellite, graph, output / "graph_on_satellite.png", "Healed road graph on satellite image")
    _write_records(graph, output)
    return graph


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract and rule-heal a road graph from a binary mask.")
    parser.add_argument("--mask", required=True)
    parser.add_argument("--output-dir", default="graph_output")
    parser.add_argument("--satellite")
    parser.add_argument("--mask-threshold", type=int, default=127)
    parser.add_argument("--closing-kernel", type=int, default=5)
    parser.add_argument("--closing-iterations", type=int, default=1)
    parser.add_argument("--min-component-area", type=int, default=50)
    parser.add_argument("--endpoint-distance", type=float, default=15.0)
    parser.add_argument("--heading-angle", type=float, default=45.0, help="Negative disables heading checks")
    parser.add_argument("--junction-merge-distance", type=float, default=3.0)
    parser.add_argument("--dangling-length", type=float, default=5.0)
    parser.add_argument("--allow-same-component-bridges", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    graph = run_pipeline(
        args.mask, args.output_dir, args.satellite,
        mask_threshold=args.mask_threshold, closing_kernel_size=args.closing_kernel,
        closing_iterations=args.closing_iterations, min_component_area=args.min_component_area,
        endpoint_distance=args.endpoint_distance,
        max_heading_angle=None if args.heading_angle < 0 else args.heading_angle,
        junction_merge_distance=args.junction_merge_distance, dangling_length=args.dangling_length,
        allow_same_component_bridges=args.allow_same_component_bridges,
    )
    print(f"Saved {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
