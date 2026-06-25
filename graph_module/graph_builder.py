"""Compress a pixel skeleton into a geometric NetworkX graph."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

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
) -> None:
    coordinate = np.mean(np.asarray(pixels, dtype=float), axis=0)
    degrees = [int(degree_image[pixel]) for pixel in pixels]
    if max(degrees) >= 3:
        node_type = "junction"
    elif max(degrees) == 1:
        node_type = "endpoint"
    else:
        node_type = "isolated"
    graph.add_node(
        node_id,
        coord=(float(coordinate[0]), float(coordinate[1])),
        pixels=[(int(row), int(col)) for row, col in pixels],
        type=node_type,
        pixel_degree_max=max(degrees),
    )


def extract_graph(skeleton: np.ndarray) -> nx.MultiGraph:
    """Convert an 8-connected skeleton into a chain-compressed MultiGraph.

    Non-degree-2 pixel clusters become graph nodes. Each intervening degree-2
    chain becomes one edge with ``geometry`` in (row, col) order and pixel
    ``length``. Pure cycles are represented by an artificial node and self-loop.
    """
    skeleton = np.asarray(skeleton, dtype=bool)
    classes = classify_skeleton_pixels(skeleton)
    degree = classes["degree_image"]
    key_mask = skeleton & (degree != 2)
    component_count, labels = cv2.connectedComponents(
        key_mask.astype(np.uint8), connectivity=8
    )

    graph = nx.MultiGraph()
    graph.graph.update(
        coordinate_order="row_col",
        image_shape=tuple(int(value) for value in skeleton.shape),
        connectivity=8,
    )
    label_to_node: Dict[int, int] = {}
    next_node = 0
    for label in range(1, component_count):
        pixels = [tuple(map(int, point)) for point in np.argwhere(labels == label)]
        label_to_node[label] = next_node
        _add_node_from_component(graph, next_node, pixels, degree)
        next_node += 1

    visited_segments = set()

    # Trace outward from every endpoint/junction cluster.
    for label, start_node in label_to_node.items():
        cluster_pixels = [tuple(map(int, p)) for p in np.argwhere(labels == label)]
        for start_pixel in cluster_pixels:
            for neighbor in _neighbors(start_pixel, skeleton):
                if labels[neighbor] == label:
                    visited_segments.add(_segment_key(start_pixel, neighbor))
                    continue
                first_segment = _segment_key(start_pixel, neighbor)
                if first_segment in visited_segments:
                    continue

                path = [start_pixel, neighbor]
                visited_segments.add(first_segment)
                previous, current = start_pixel, neighbor
                end_node = None

                while True:
                    current_label = int(labels[current])
                    if current_label > 0:
                        end_node = label_to_node[current_label]
                        break
                    candidates = [p for p in _neighbors(current, skeleton) if p != previous]
                    if not candidates:
                        break  # Defensive: malformed skeleton after external mutation.
                    unvisited = [
                        p
                        for p in candidates
                        if _segment_key(current, p) not in visited_segments
                    ]
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

    # Components containing only degree-2 pixels are cycles and have no key node.
    skeleton_pixels = [tuple(map(int, p)) for p in np.argwhere(skeleton)]
    for start in skeleton_pixels:
        for neighbor in _neighbors(start, skeleton):
            if _segment_key(start, neighbor) in visited_segments:
                continue
            cycle_node = next_node
            next_node += 1
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
                candidates = [p for p in _neighbors(current, skeleton) if p != previous]
                unvisited = [
                    p
                    for p in candidates
                    if _segment_key(current, p) not in visited_segments
                ]
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
