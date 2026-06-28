"""Compress a pixel skeleton into a geometric NetworkX graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import cv2
import networkx as nx
import numpy as np

Pixel = Tuple[int, int]  # (row, column)
OFFSETS = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
)


def _neighbors(pixel: Pixel, skeleton: np.ndarray) -> List[Pixel]:
    row, col = pixel
    height, width = skeleton.shape
    return [
        (row + dr, col + dc)
        for dr, dc in OFFSETS
        if 0 <= row + dr < height
        and 0 <= col + dc < width
        and skeleton[row + dr, col + dc]
    ]


def _segment_key(a: Pixel, b: Pixel) -> Tuple[Pixel, Pixel]:
    return (a, b) if a <= b else (b, a)


def _path_length(points: Sequence[Sequence[float]]) -> float:
    if len(points) < 2:
        return 0.0
    coordinates = np.asarray(points, dtype=float)
    return float(np.linalg.norm(np.diff(coordinates, axis=0), axis=1).sum())


def _coord(graph: nx.MultiGraph, node: int) -> np.ndarray:
    return np.asarray(graph.nodes[node]["coord"], dtype=float)


def _edge_geometry(graph: nx.MultiGraph, u: int, v: int, data: dict) -> List[Tuple[float, float]]:
    geometry = [tuple(map(float, point)) for point in data.get("geometry", [])]
    if len(geometry) < 2:
        geometry = [tuple(_coord(graph, u)), tuple(_coord(graph, v))]
    return geometry


def _oriented_geometry(
    graph: nx.MultiGraph, edge_u: int, edge_v: int, data: dict, start_node: int
) -> List[Tuple[float, float]]:
    geometry = _edge_geometry(graph, edge_u, edge_v, data)
    start_coord = _coord(graph, start_node)
    if np.linalg.norm(np.asarray(geometry[-1]) - start_coord) < np.linalg.norm(
        np.asarray(geometry[0]) - start_coord
    ):
        geometry.reverse()
    return geometry


def _average_degree(graph: nx.MultiGraph) -> float:
    if graph.number_of_nodes() == 0:
        return 0.0
    return float(sum(dict(graph.degree()).values()) / graph.number_of_nodes())


def _next_node_id(graph: nx.MultiGraph) -> int:
    return max(graph.nodes, default=-1) + 1


def _merge_edge_attributes(
    graph: nx.MultiGraph,
    first: Tuple[int, int, int, dict],
    middle: int,
    second: Tuple[int, int, int, dict],
) -> Tuple[int, int, dict]:
    a_u, a_v, _, a_data = first
    b_u, b_v, _, b_data = second
    start = a_v if a_u == middle else a_u
    end = b_v if b_u == middle else b_u
    first_geometry = _oriented_geometry(graph, a_u, a_v, a_data, start)
    second_geometry = _oriented_geometry(graph, b_u, b_v, b_data, middle)
    middle_coord = _coord(graph, middle)
    if np.linalg.norm(np.asarray(first_geometry[-1]) - middle_coord) > np.linalg.norm(
        np.asarray(first_geometry[0]) - middle_coord
    ):
        first_geometry.reverse()
    if np.linalg.norm(np.asarray(second_geometry[0]) - middle_coord) > np.linalg.norm(
        np.asarray(second_geometry[-1]) - middle_coord
    ):
        second_geometry.reverse()
    geometry = first_geometry + second_geometry[1:]
    attributes = dict(a_data)
    attributes.update(
        geometry=geometry,
        length=_path_length(geometry),
        healed=bool(a_data.get("healed", False) or b_data.get("healed", False)),
    )
    repairs = [value for value in (a_data.get("repair"), b_data.get("repair")) if value]
    if repairs:
        attributes["repair"] = "+".join(sorted(set(map(str, repairs))))
    return start, end, attributes


def _set_endpoint_geometry(graph: nx.MultiGraph, u: int, v: int, attributes: dict) -> dict:
    attributes = dict(attributes)
    geometry = _edge_geometry(graph, u, v, attributes)
    if np.linalg.norm(np.asarray(geometry[0]) - _coord(graph, u)) > np.linalg.norm(
        np.asarray(geometry[-1]) - _coord(graph, u)
    ):
        geometry.reverse()
    geometry[0] = tuple(_coord(graph, u))
    geometry[-1] = tuple(_coord(graph, v))
    attributes["geometry"] = geometry
    attributes["length"] = _path_length(geometry)
    return attributes


def _node_stats(graph: nx.MultiGraph) -> Dict[str, float]:
    return {
        "node_count": int(graph.number_of_nodes()),
        "edge_count": int(graph.number_of_edges()),
        "junction_count": int(
            sum(
                1
                for node, data in graph.nodes(data=True)
                if data.get("type") == "junction" or graph.degree(node) >= 3
            )
        ),
        "average_node_degree": _average_degree(graph),
    }


def classify_skeleton_pixels(skeleton: np.ndarray) -> Dict[str, np.ndarray]:
    """Return coordinates for endpoints, degree-2 road pixels, and junction pixels."""
    skeleton = np.asarray(skeleton, dtype=bool)
    if skeleton.ndim != 2:
        raise ValueError("Skeleton must be a 2-D array")
    kernel = np.ones((3, 3), dtype=np.uint8)
    degree = cv2.filter2D(
        skeleton.astype(np.uint8), cv2.CV_16S, kernel, borderType=cv2.BORDER_CONSTANT
    ) - skeleton.astype(np.int16)
    return {
        "endpoints": np.argwhere(skeleton & (degree == 1)),
        "road_pixels": np.argwhere(skeleton & (degree == 2)),
        "junctions": np.argwhere(skeleton & (degree >= 3)),
        "isolated": np.argwhere(skeleton & (degree == 0)),
        "degree_image": degree,
    }


def _add_node_from_component(
    graph: nx.MultiGraph,
    node_id: int,
    pixels: List[Pixel],
    degree_image: np.ndarray,
    node_type: str,
    cluster_id: int,
) -> None:
    coordinate = np.mean(np.asarray(pixels, dtype=float), axis=0)
    degrees = [int(degree_image[pixel]) for pixel in pixels]
    attributes = {
        "coord": (float(coordinate[0]), float(coordinate[1])),
        "pixels": [(int(row), int(col)) for row, col in pixels],
        "type": node_type,
        "pixel_degree_max": max(degrees) if degrees else 0,
        "cluster_id": int(cluster_id),
    }
    if node_type == "junction":
        attributes["junction_blob"] = int(cluster_id)
    graph.add_node(node_id, **attributes)


def _label_node_clusters(
    graph: nx.MultiGraph, skeleton: np.ndarray, degree: np.ndarray
) -> Tuple[np.ndarray, Dict[int, int]]:
    node_labels = np.zeros(skeleton.shape, dtype=np.int32)
    label_to_node: Dict[int, int] = {}
    next_node = 0
    next_label = 1

    for node_type, mask in (
        ("junction", skeleton & (degree >= 3)),
        ("endpoint", skeleton & (degree == 1)),
        ("isolated", skeleton & (degree == 0)),
    ):
        component_count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        for component in range(1, component_count):
            pixels = [tuple(map(int, point)) for point in np.argwhere(labels == component)]
            node_labels[labels == component] = next_label
            label_to_node[next_label] = next_node
            _add_node_from_component(graph, next_node, pixels, degree, node_type, next_label)
            next_node += 1
            next_label += 1
    return node_labels, label_to_node


def extract_graph(skeleton: np.ndarray) -> nx.MultiGraph:
    """Convert an 8-connected skeleton into a chain-compressed MultiGraph.

    Connected junction-pixel blobs become one graph node each. Edge tracing only
    terminates at another endpoint, isolated node, or junction cluster, so a road
    between two meaningful graph nodes is emitted as one geometric edge.
    """
    skeleton = np.asarray(skeleton, dtype=bool)
    classes = classify_skeleton_pixels(skeleton)
    degree = classes["degree_image"]

    graph = nx.MultiGraph()
    graph.graph.update(
        coordinate_order="row_col",
        image_shape=tuple(int(value) for value in skeleton.shape),
        connectivity=8,
    )
    node_labels, label_to_node = _label_node_clusters(graph, skeleton, degree)
    visited_segments: Set[Tuple[Pixel, Pixel]] = set()

    for label, start_node in label_to_node.items():
        cluster_pixels = [tuple(map(int, p)) for p in np.argwhere(node_labels == label)]
        for start_pixel in cluster_pixels:
            for neighbor in sorted(_neighbors(start_pixel, skeleton)):
                if node_labels[neighbor] == label:
                    visited_segments.add(_segment_key(start_pixel, neighbor))
                    continue
                first_segment = _segment_key(start_pixel, neighbor)
                if first_segment in visited_segments:
                    continue

                path = [start_pixel, neighbor]
                visited_segments.add(first_segment)
                previous, current = start_pixel, neighbor
                end_node: Optional[int] = None

                while True:
                    current_label = int(node_labels[current])
                    if current_label > 0:
                        end_node = label_to_node[current_label]
                        break
                    candidates = sorted(p for p in _neighbors(current, skeleton) if p != previous)
                    if not candidates:
                        break
                    unvisited = [p for p in candidates if _segment_key(current, p) not in visited_segments]
                    if not unvisited:
                        break
                    following = unvisited[0]
                    visited_segments.add(_segment_key(current, following))
                    path.append(following)
                    previous, current = current, following

                if end_node is not None:
                    geometry = [(float(r), float(c)) for r, c in path]
                    graph.add_edge(
                        start_node,
                        end_node,
                        geometry=geometry,
                        length=_path_length(geometry),
                        healed=False,
                    )

    skeleton_pixels = [tuple(map(int, p)) for p in np.argwhere(skeleton)]
    for start in skeleton_pixels:
        for neighbor in sorted(_neighbors(start, skeleton)):
            if _segment_key(start, neighbor) in visited_segments:
                continue
            cycle_node = _next_node_id(graph)
            graph.add_node(
                cycle_node,
                coord=(float(start[0]), float(start[1])),
                pixels=[start],
                type="cycle",
                pixel_degree_max=2,
            )
            path = [start, neighbor]
            visited_segments.add(_segment_key(start, neighbor))
            previous, current = start, neighbor
            while current != start:
                candidates = sorted(p for p in _neighbors(current, skeleton) if p != previous)
                unvisited = [p for p in candidates if _segment_key(current, p) not in visited_segments]
                if not unvisited:
                    break
                following = unvisited[0]
                visited_segments.add(_segment_key(current, following))
                path.append(following)
                previous, current = current, following
            geometry = [(float(r), float(c)) for r, c in path]
            graph.add_edge(
                cycle_node,
                cycle_node,
                geometry=geometry,
                length=_path_length(geometry),
                healed=False,
            )
            break

    return graph


@dataclass
class GraphSimplificationConfig:
    """Feature switches and geometric tolerances for graph simplification."""

    contract_degree2: bool = True
    collapse_short_edges: bool = True
    deduplicate_paths: bool = True
    remove_tiny_cycles: bool = True
    short_edge_threshold: float = 4.0
    tiny_cycle_perimeter: float = 18.0
    tiny_cycle_radius: float = 4.0
    max_artifact_edge_length: float = 8.0


def _refresh_node_types(graph: nx.MultiGraph) -> None:
    for node in graph.nodes:
        original = graph.nodes[node].get("type")
        if original == "junction":
            continue
        degree = graph.degree(node)
        if degree == 0:
            node_type = "isolated"
        elif degree == 1:
            node_type = "endpoint"
        elif degree == 2:
            node_type = "road_node"
        else:
            node_type = "junction"
        graph.nodes[node]["type"] = node_type


def _cycle_nodes(graph: nx.MultiGraph) -> Set[int]:
    simple = nx.Graph()
    simple.add_nodes_from(graph.nodes)
    simple.add_edges_from((u, v) for u, v in graph.edges() if u != v)
    return {node for cycle in nx.cycle_basis(simple) for node in cycle}


def _is_protected_node(graph: nx.MultiGraph, node: int, cycle_nodes: Set[int]) -> bool:
    node_type = graph.nodes[node].get("type")
    return node_type in {"endpoint", "junction"} or graph.degree(node) != 2 or node in cycle_nodes


def _contract_degree2_nodes(graph: nx.MultiGraph) -> Tuple[nx.MultiGraph, int]:
    result = graph.copy()
    removed = 0
    changed = True
    while changed:
        changed = False
        protected_cycle_nodes = _cycle_nodes(result)
        for node in list(result.nodes):
            if node not in result or _is_protected_node(result, node, protected_cycle_nodes):
                continue
            incident = list(result.edges(node, keys=True, data=True))
            if len(incident) != 2:
                continue
            first, second = incident
            start, end, attributes = _merge_edge_attributes(result, first, node, second)
            result.remove_node(node)
            result.add_edge(start, end, **attributes)
            removed += 1
            changed = True
            break
    _refresh_node_types(result)
    return result, removed


def _same_junction_blob(graph: nx.MultiGraph, a: int, b: int) -> bool:
    blob_a = graph.nodes[a].get("junction_blob")
    blob_b = graph.nodes[b].get("junction_blob")
    return blob_a is not None and blob_a == blob_b


def _merge_nodes(graph: nx.MultiGraph, members: Iterable[int]) -> nx.MultiGraph:
    members = list(dict.fromkeys(members))
    if len(members) < 2:
        return graph.copy()
    target = min(members)
    member_set = set(members)
    result = nx.MultiGraph(**graph.graph)
    coordinates = np.asarray([_coord(graph, node) for node in members], dtype=float)
    pixels = [pixel for node in members for pixel in graph.nodes[node].get("pixels", [])]
    attributes = dict(graph.nodes[target])
    attributes.update(
        coord=tuple(np.mean(coordinates, axis=0).tolist()),
        pixels=pixels,
        type="junction",
        merged_nodes=sorted(set(sum((graph.nodes[n].get("merged_nodes", [n]) for n in members), []))),
    )
    junction_blobs = sorted(
        {graph.nodes[node].get("junction_blob") for node in members if graph.nodes[node].get("junction_blob") is not None}
    )
    if junction_blobs:
        attributes["junction_blob"] = junction_blobs[0] if len(junction_blobs) == 1 else tuple(junction_blobs)
    for node, data in graph.nodes(data=True):
        if node in member_set:
            continue
        result.add_node(node, **dict(data))
    result.add_node(target, **attributes)

    for u, v, key, data in graph.edges(keys=True, data=True):
        new_u = target if u in member_set else u
        new_v = target if v in member_set else v
        if new_u == new_v:
            continue
        edge_attrs = _set_endpoint_geometry(result, new_u, new_v, data)
        result.add_edge(new_u, new_v, **edge_attrs)
    _refresh_node_types(result)
    return result


def _edge_is_collapsible(
    graph: nx.MultiGraph, u: int, v: int, data: dict, config: GraphSimplificationConfig
) -> bool:
    if u == v:
        return False
    length = float(data.get("length", 0.0))
    if length >= config.short_edge_threshold:
        return False
    allowed_types = {"junction", "road_node"}
    if graph.nodes[u].get("type") not in allowed_types and graph.degree(u) > 2:
        return False
    if graph.nodes[v].get("type") not in allowed_types and graph.degree(v) > 2:
        return False
    if _same_junction_blob(graph, u, v):
        return True
    if graph.nodes[u].get("type") == "junction" and graph.nodes[v].get("type") == "junction":
        return False
    return np.linalg.norm(_coord(graph, u) - _coord(graph, v)) <= config.short_edge_threshold


def _collapse_short_edges(
    graph: nx.MultiGraph, config: GraphSimplificationConfig
) -> Tuple[nx.MultiGraph, int]:
    result = graph.copy()
    collapsed = 0
    changed = True
    while changed:
        changed = False
        for u, v, key, data in sorted(
            result.edges(keys=True, data=True), key=lambda edge: float(edge[3].get("length", 0.0))
        ):
            if u not in result or v not in result or key not in result[u][v]:
                continue
            if not _edge_is_collapsible(result, u, v, data, config):
                continue
            result = _merge_nodes(result, [u, v])
            collapsed += 1
            changed = True
            break
    return result, collapsed


def _edge_between(graph: nx.MultiGraph, a: int, b: int) -> Optional[Tuple[int, int, int, dict]]:
    if not graph.has_edge(a, b):
        return None
    key = min(graph[a][b], key=lambda k: graph[a][b][k].get("length", float("inf")))
    return a, b, key, graph[a][b][key]


def _deduplicate_subdivided_paths(graph: nx.MultiGraph) -> Tuple[nx.MultiGraph, int]:
    result = graph.copy()
    removed = 0
    changed = True
    while changed:
        changed = False
        for node in list(result.nodes):
            if node not in result or result.degree(node) != 2:
                continue
            if result.nodes[node].get("type") in {"endpoint", "junction"}:
                continue
            incident = list(result.edges(node, keys=True, data=True))
            if len(incident) != 2:
                continue
            neighbors = [v if u == node else u for u, v, _, _ in incident]
            if neighbors[0] == neighbors[1] or not result.has_edge(neighbors[0], neighbors[1]):
                continue
            direct = _edge_between(result, neighbors[0], neighbors[1])
            _, _, direct_key, direct_data = direct
            subdivided_length = sum(float(edge[3].get("length", 0.0)) for edge in incident)
            direct_length = float(direct_data.get("length", 0.0))
            if direct_length <= subdivided_length * 1.15:
                result.remove_node(node)
                removed += 1
                changed = True
                break
    _refresh_node_types(result)
    return result, removed


def deduplicate_edges(graph: nx.MultiGraph) -> nx.MultiGraph:
    """Remove exact parallel edges and simple subdivided duplicate paths."""
    result = graph.copy()
    removed = 0
    for u, v in list(set(result.edges())):
        if u not in result or v not in result or not result.has_edge(u, v):
            continue
        keys = list(result[u][v].keys())
        if len(keys) <= 1:
            continue
        best_key = min(keys, key=lambda k: result[u][v][k].get("length", float("inf")))
        for key in keys:
            if key != best_key and result.has_edge(u, v, key):
                result.remove_edge(u, v, key)
                removed += 1
    result, subdivided = _deduplicate_subdivided_paths(result)
    result.graph.setdefault("simplification_diagnostics", {})["removed_duplicate_paths"] = removed + subdivided
    return result


def _tiny_cycle_groups(
    graph: nx.MultiGraph, config: GraphSimplificationConfig
) -> List[List[int]]:
    simple = nx.Graph()
    simple.add_nodes_from(graph.nodes)
    simple.add_edges_from((u, v) for u, v in graph.edges() if u != v)
    groups: List[List[int]] = []
    for cycle in nx.cycle_basis(simple):
        perimeter = 0.0
        edge_lengths = []
        valid = True
        for index, u in enumerate(cycle):
            v = cycle[(index + 1) % len(cycle)]
            edge = _edge_between(graph, u, v)
            if edge is None:
                valid = False
                break
            length = float(edge[3].get("length", 0.0))
            perimeter += length
            edge_lengths.append(length)
        if not valid or perimeter >= config.tiny_cycle_perimeter:
            continue
        if any(length > config.max_artifact_edge_length for length in edge_lengths):
            continue
        coordinates = np.asarray([_coord(graph, node) for node in cycle], dtype=float)
        center = coordinates.mean(axis=0)
        radius = float(np.linalg.norm(coordinates - center, axis=1).max())
        if radius <= config.tiny_cycle_radius:
            groups.append(cycle)
    return groups


def _collapse_tiny_cycles(
    graph: nx.MultiGraph, config: GraphSimplificationConfig
) -> Tuple[nx.MultiGraph, int]:
    result = graph.copy()
    collapsed = 0
    while True:
        groups = _tiny_cycle_groups(result, config)
        if not groups:
            break
        result = _merge_nodes(result, groups[0])
        collapsed += 1
    return result, collapsed


def simplify_graph(
    graph: nx.MultiGraph, config: Optional[GraphSimplificationConfig] = None
) -> nx.MultiGraph:
    """Apply modular topology minimization passes to an extracted road graph."""
    config = config or GraphSimplificationConfig()
    result = graph.copy()
    diagnostics = {
        "before": _node_stats(result),
        "merged_junctions": 0,
        "removed_degree2_nodes": 0,
        "removed_short_edges": 0,
        "removed_tiny_cycles": 0,
        "removed_duplicate_paths": 0,
    }

    if config.remove_tiny_cycles:
        result, diagnostics["removed_tiny_cycles"] = _collapse_tiny_cycles(result, config)
    if config.collapse_short_edges:
        result, diagnostics["removed_short_edges"] = _collapse_short_edges(result, config)
    if config.contract_degree2:
        result, diagnostics["removed_degree2_nodes"] = _contract_degree2_nodes(result)
    if config.deduplicate_paths:
        before = result.number_of_edges()
        result = deduplicate_edges(result)
        diagnostics["removed_duplicate_paths"] = int(
            result.graph.get("simplification_diagnostics", {}).get("removed_duplicate_paths", 0)
        )
        diagnostics["removed_duplicate_paths"] = max(
            diagnostics["removed_duplicate_paths"], before - result.number_of_edges()
        )

    _refresh_node_types(result)
    diagnostics["after"] = _node_stats(result)
    diagnostics["merged_junctions"] = int(
        diagnostics["before"]["junction_count"] - diagnostics["after"]["junction_count"]
    )
    result.graph["simplification_diagnostics"] = diagnostics
    return result


def graph_to_records(graph: nx.MultiGraph) -> Tuple[List[dict], List[dict]]:
    """Create JSON-friendly node and edge lists from a graph."""
    nodes = [
        {
            "id": int(node),
            "row": float(data["coord"][0]),
            "col": float(data["coord"][1]),
            "type": data.get("type", "unknown"),
            "degree": int(graph.degree(node)),
        }
        for node, data in graph.nodes(data=True)
    ]
    edges = [
        {
            "u": int(u),
            "v": int(v),
            "key": int(key),
            "length": float(data.get("length", 0.0)),
            "healed": bool(data.get("healed", False)),
            "geometry": [[float(r), float(c)] for r, c in data.get("geometry", [])],
        }
        for u, v, key, data in graph.edges(keys=True, data=True)
    ]
    return nodes, edges
