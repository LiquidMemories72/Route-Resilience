"""Conservative, rule-based repairs for extracted road graphs."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np


def _length(geometry: Sequence[Sequence[float]]) -> float:
    points = np.asarray(geometry, dtype=float)
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _coord(graph: nx.MultiGraph, node: int) -> np.ndarray:
    return np.asarray(graph.nodes[node]["coord"], dtype=float)


def _endpoint_heading(
    graph: nx.MultiGraph, node: int, lookahead_points: int = 8
) -> Optional[np.ndarray]:
    """Estimate the unit vector pointing out of the existing road at an endpoint."""
    incident = list(graph.edges(node, keys=True, data=True))
    if len(incident) != 1:
        return None
    geometry = np.asarray(incident[0][3].get("geometry", []), dtype=float)
    if len(geometry) < 2:
        return None
    endpoint = _coord(graph, node)
    if np.linalg.norm(geometry[0] - endpoint) <= np.linalg.norm(geometry[-1] - endpoint):
        ordered = geometry
    else:
        ordered = geometry[::-1]
    inward = ordered[min(lookahead_points, len(ordered) - 1)]
    outward = endpoint - inward
    norm = np.linalg.norm(outward)
    return outward / norm if norm > 1e-8 else None


def merge_nearby_junctions(
    graph: nx.MultiGraph, distance_threshold: float = 3.0
) -> nx.MultiGraph:
    """Merge junction nodes whose coordinates fall within a pixel threshold."""
    if distance_threshold <= 0 or graph.number_of_nodes() < 2:
        return graph.copy()
    junctions = [
        node
        for node, data in graph.nodes(data=True)
        if data.get("type") == "junction" or graph.degree(node) >= 3
    ]
    parent = {node: node for node in junctions}

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    for index, a in enumerate(junctions):
        for b in junctions[index + 1 :]:
            if np.linalg.norm(_coord(graph, a) - _coord(graph, b)) <= distance_threshold:
                union(a, b)

    groups: Dict[int, List[int]] = {}
    for node in junctions:
        groups.setdefault(find(node), []).append(node)
    mapping = {node: root for root, members in groups.items() for node in members}
    if all(node == target for node, target in mapping.items()):
        return graph.copy()

    result = nx.MultiGraph(**graph.graph)
    all_groups: Dict[int, List[int]] = {}
    for node in graph.nodes:
        all_groups.setdefault(mapping.get(node, node), []).append(node)
    for target, members in all_groups.items():
        coordinates = np.asarray([_coord(graph, node) for node in members])
        pixels = [
            pixel
            for node in members
            for pixel in graph.nodes[node].get("pixels", [])
        ]
        attributes = dict(graph.nodes[members[0]])
        attributes.update(
            coord=tuple(np.mean(coordinates, axis=0).tolist()),
            pixels=pixels,
            type="junction" if len(members) > 1 else attributes.get("type", "unknown"),
        )
        result.add_node(target, **attributes)

    for u, v, data in graph.edges(data=True):
        new_u, new_v = mapping.get(u, u), mapping.get(v, v)
        # Edges wholly inside a merged junction blob carry no road information.
        if new_u == new_v and u != v:
            continue
        attributes = dict(data)
        geometry = [tuple(map(float, point)) for point in attributes.get("geometry", [])]
        if geometry and new_u != new_v:
            old_u, old_v = _coord(graph, u), _coord(graph, v)
            if np.linalg.norm(np.asarray(geometry[0]) - old_u) > np.linalg.norm(
                np.asarray(geometry[-1]) - old_u
            ):
                geometry.reverse()
            geometry[0] = tuple(result.nodes[new_u]["coord"])
            geometry[-1] = tuple(result.nodes[new_v]["coord"])
            attributes["geometry"] = geometry
            attributes["length"] = _length(geometry)
        result.add_edge(new_u, new_v, **attributes)
    return result


def connect_nearby_endpoints(
    graph: nx.MultiGraph,
    distance_threshold: float = 15.0,
    max_heading_angle: Optional[float] = 45.0,
    allow_same_component: bool = False,
    max_connections_per_endpoint: int = 1,
) -> nx.MultiGraph:
    """Bridge close endpoint pairs that face each other across a gap.

    Set ``max_heading_angle=None`` to use distance alone. Distances are pixels.
    """
    result = graph.copy()
    endpoints = [node for node in result.nodes if result.degree(node) == 1]
    headings = {node: _endpoint_heading(result, node) for node in endpoints}
    component = {
        node: index
        for index, nodes in enumerate(nx.connected_components(result))
        for node in nodes
    }
    candidates = []
    cosine_limit = (
        None
        if max_heading_angle is None
        else float(np.cos(np.deg2rad(max_heading_angle)))
    )
    for index, a in enumerate(endpoints):
        for b in endpoints[index + 1 :]:
            if not allow_same_component and component[a] == component[b]:
                continue
            delta = _coord(result, b) - _coord(result, a)
            distance = float(np.linalg.norm(delta))
            if distance <= 0 or distance > distance_threshold:
                continue
            if cosine_limit is not None:
                if headings[a] is None or headings[b] is None:
                    continue
                direction = delta / distance
                if np.dot(headings[a], direction) < cosine_limit:
                    continue
                if np.dot(headings[b], -direction) < cosine_limit:
                    continue
            candidates.append((distance, a, b))

    use_count = {node: 0 for node in endpoints}
    for distance, a, b in sorted(candidates):
        if (
            use_count[a] >= max_connections_per_endpoint
            or use_count[b] >= max_connections_per_endpoint
        ):
            continue
        geometry = [tuple(_coord(result, a)), tuple(_coord(result, b))]
        result.add_edge(
            a,
            b,
            geometry=geometry,
            length=distance,
            healed=True,
            repair="endpoint_bridge",
        )
        use_count[a] += 1
        use_count[b] += 1
    return result


def remove_short_dangling_branches(
    graph: nx.MultiGraph, length_threshold: float = 5.0
) -> nx.MultiGraph:
    """Iteratively prune edges shorter than the threshold at degree-1 nodes."""
    result = graph.copy()
    if length_threshold <= 0:
        return result
    changed = True
    while changed:
        changed = False
        for node in list(result.nodes):
            if node not in result or result.degree(node) != 1:
                continue
            incident = list(result.edges(node, keys=True, data=True))
            if not incident:
                continue
            u, v, key, data = incident[0]
            if float(data.get("length", 0.0)) < length_threshold:
                result.remove_edge(u, v, key)
                if node in result and result.degree(node) == 0:
                    result.remove_node(node)
                changed = True
    result.remove_nodes_from(list(nx.isolates(result)))
    return result


def refresh_node_types(graph: nx.MultiGraph) -> None:
    """Update semantic node types after graph topology changes."""
    for node in graph.nodes:
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


def heal_graph(
    graph: nx.MultiGraph,
    endpoint_distance: float = 15.0,
    max_heading_angle: Optional[float] = 45.0,
    junction_merge_distance: float = 3.0,
    dangling_length: float = 5.0,
    allow_same_component_bridges: bool = False,
) -> nx.MultiGraph:
    """Apply junction merging, endpoint bridging, then dangling-spur pruning."""
    healed = merge_nearby_junctions(graph, junction_merge_distance)
    healed = connect_nearby_endpoints(
        healed,
        distance_threshold=endpoint_distance,
        max_heading_angle=max_heading_angle,
        allow_same_component=allow_same_component_bridges,
    )
    healed = remove_short_dangling_branches(healed, dangling_length)
    refresh_node_types(healed)
    return healed
