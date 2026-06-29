"""Command-line and Python API for road-mask graph extraction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Union
import cv2
import shutil
import networkx as nx
import numpy as np
from skimage.morphology import skeletonize as skimage_skeletonize

try:
    from .graph_builder import GraphSimplificationConfig, extract_graph, graph_to_records, simplify_graph
    from .graph_healing import heal_graph
    from .preprocess import cleanup_mask, load_binary_mask, save_binary_mask
    from .skeleton import make_skeleton, save_skeleton
    from .visualization import save_graph_overlay, save_node_visualization, save_skeleton_visualization
    from .topology_repair import TopologyRepair
except ImportError:
    from graph_builder import GraphSimplificationConfig, extract_graph, graph_to_records, simplify_graph
    from graph_healing import heal_graph
    from preprocess import cleanup_mask, load_binary_mask, save_binary_mask
    from skeleton import make_skeleton, save_skeleton
    from visualization import save_graph_overlay, save_node_visualization, save_skeleton_visualization
    from topology_repair import TopologyRepair

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
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "connected_components": nx.number_connected_components(graph),
        "total_length_pixels": float(sum(d.get("length", 0.0) for _, _, d in graph.edges(data=True))),
        "healed_edges": sum(bool(d.get("healed", False)) for _, _, d in graph.edges(data=True)),
        "coordinate_order": "row_col",
        "simplification_diagnostics": graph.graph.get("simplification_diagnostics", {}),
    }
    (output_dir / "graph_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def run_pipeline(
    mask: MaskInput,
    output_dir: Union[str, Path],
    satellite_image: Optional[Union[str, Path, np.ndarray]] = None,
    prob_map: Optional[Union[str, Path, np.ndarray]] = None,
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
    simplify: bool = True,
    contract_degree2: bool = True,
    collapse_short_edges: bool = True,
    deduplicate_paths: bool = True,
    remove_tiny_cycles: bool = True,
    short_edge_threshold: float = 7.0,
    tiny_cycle_perimeter: float = 18.0,
    tiny_cycle_radius: float = 4.0,
    max_artifact_edge_length: float = 8.0,
) -> nx.MultiGraph:
    """Run the full pipeline and return the healed, simplified graph."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    binary = load_binary_mask(mask, mask_threshold) if isinstance(mask, (str, Path)) else np.asarray(mask)
    cleaned = cleanup_mask(binary, closing_kernel_size, closing_iterations, min_component_area, mask_threshold)
    skeleton = make_skeleton(cleaned)

    if prob_map is not None:
        # Visualization of probability map is handled by TopologyRepair

        # Copy original satellite image to the output folder if provided
        if satellite_image is not None:
            src_sat = Path(satellite_image) if isinstance(satellite_image, (str, Path)) else None
            if src_sat and src_sat.exists():
                shutil.copy2(src_sat, output / src_sat.name)

        # Copy debug visualizations to a debug subdirectory within the output folder
        debug_src = Path("graph_debug")
        if debug_src.is_dir():
            dest_debug = output / "debug"
            if dest_debug.exists():
                shutil.rmtree(dest_debug)
            shutil.copytree(debug_src, dest_debug)
            
        probability = np.load(prob_map) if isinstance(prob_map, (str, Path)) else np.asarray(prob_map)
        repair = TopologyRepair(probability, cleaned, skeleton)
        skeleton = repair.run()
        # Post-repair cleanup: dilate slightly and re-skeletonize to merge nearby
        # junction pixel clusters that the repair may have introduced.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        skeleton_u8 = (skeleton > 0).astype(np.uint8)
        dilated = cv2.dilate(skeleton_u8, kernel, iterations=1)
        skeleton = skimage_skeletonize(dilated > 0).astype(np.uint8)

    graph = heal_graph(
        extract_graph(skeleton), endpoint_distance, max_heading_angle,
        junction_merge_distance, dangling_length, allow_same_component_bridges,
    )
    if simplify:
        graph = simplify_graph(
            graph,
            GraphSimplificationConfig(
                contract_degree2=contract_degree2,
                collapse_short_edges=collapse_short_edges,
                deduplicate_paths=deduplicate_paths,
                remove_tiny_cycles=remove_tiny_cycles,
                short_edge_threshold=short_edge_threshold,
                tiny_cycle_perimeter=tiny_cycle_perimeter,
                tiny_cycle_radius=tiny_cycle_radius,
                max_artifact_edge_length=max_artifact_edge_length,
            ),
        )

    save_binary_mask(cleaned, output / "cleaned_mask.png")
    save_skeleton(skeleton, output / "skeleton.png")
    save_skeleton_visualization(skeleton, output / "skeleton_visualization.png")
    save_node_visualization(skeleton, graph, output / "nodes.png")
    save_graph_overlay(cleaned, graph, output / "graph_on_mask.png", "Healed road graph on mask")
    if satellite_image is not None:
        satellite = _load_satellite(satellite_image) if isinstance(satellite_image, (str, Path)) else np.asarray(satellite_image)
        if satellite.shape[:2] != cleaned.shape:
            satellite = cv2.resize(satellite, (cleaned.shape[1], cleaned.shape[0]))
        save_graph_overlay(satellite, graph, output / "graph_on_satellite.png", "Healed road graph on satellite image")
    _write_records(graph, output)
    return graph


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract and rule-heal a road graph from a binary mask.")
    parser.add_argument("--mask", required=True)
    parser.add_argument("--prob-map", default=None, help="Path to .npy probability map for topology repair")
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
    parser.add_argument("--no-simplify", action="store_true")
    parser.add_argument("--no-contract-degree2", action="store_true")
    parser.add_argument("--no-collapse-short-edges", action="store_true")
    parser.add_argument("--no-deduplicate-paths", action="store_true")
    parser.add_argument("--no-remove-tiny-cycles", action="store_true")
    parser.add_argument("--short-edge-threshold", type=float, default=4.0)
    parser.add_argument("--tiny-cycle-perimeter", type=float, default=18.0)
    parser.add_argument("--tiny-cycle-radius", type=float, default=4.0)
    parser.add_argument("--max-artifact-edge-length", type=float, default=8.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    graph = run_pipeline(
        args.mask, args.output_dir, args.satellite, prob_map=args.prob_map,
        mask_threshold=args.mask_threshold, closing_kernel_size=args.closing_kernel,
        closing_iterations=args.closing_iterations, min_component_area=args.min_component_area,
        endpoint_distance=args.endpoint_distance,
        max_heading_angle=None if args.heading_angle < 0 else args.heading_angle,
        junction_merge_distance=args.junction_merge_distance, dangling_length=args.dangling_length,
        allow_same_component_bridges=args.allow_same_component_bridges,
        simplify=not args.no_simplify,
        contract_degree2=not args.no_contract_degree2,
        collapse_short_edges=not args.no_collapse_short_edges,
        deduplicate_paths=not args.no_deduplicate_paths,
        remove_tiny_cycles=not args.no_remove_tiny_cycles,
        short_edge_threshold=args.short_edge_threshold,
        tiny_cycle_perimeter=args.tiny_cycle_perimeter,
        tiny_cycle_radius=args.tiny_cycle_radius,
        max_artifact_edge_length=args.max_artifact_edge_length,
    )
    print(f"Saved {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()

