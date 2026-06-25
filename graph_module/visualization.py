"""Matplotlib visualizations for skeleton and road graphs."""

from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

PathLike = Union[str, Path]
NODE_COLORS = {
    "endpoint": "#00d26a",
    "junction": "#ff3b30",
    "road_node": "#ffd60a",
    "cycle": "#bf5af2",
    "isolated": "#8e8e93",
}


def _finish(figure: plt.Figure, output_path: PathLike, dpi: int) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(figure)


def save_skeleton_visualization(
    skeleton: np.ndarray, output_path: PathLike, dpi: int = 180
) -> None:
    figure, axis = plt.subplots(figsize=(8, 8))
    axis.imshow(skeleton, cmap="gray")
    axis.set_title("Road skeleton")
    axis.axis("off")
    _finish(figure, output_path, dpi)


def _draw_graph(axis: plt.Axes, graph: nx.MultiGraph, show_nodes: bool = True) -> None:
    for _, _, data in graph.edges(data=True):
        geometry = np.asarray(data.get("geometry", []), dtype=float)
        if len(geometry) < 2:
            continue
        color = "#00e5ff" if data.get("healed", False) else "#ffcc00"
        axis.plot(geometry[:, 1], geometry[:, 0], color=color, linewidth=1.4, alpha=0.95)
    if show_nodes and graph.number_of_nodes():
        for node_type, color in NODE_COLORS.items():
            coordinates = np.asarray(
                [
                    data["coord"]
                    for _, data in graph.nodes(data=True)
                    if data.get("type") == node_type
                ],
                dtype=float,
            )
            if coordinates.size:
                axis.scatter(
                    coordinates[:, 1],
                    coordinates[:, 0],
                    s=18,
                    c=color,
                    edgecolors="black",
                    linewidths=0.35,
                    label=node_type,
                    zorder=4,
                )


def save_node_visualization(
    skeleton: np.ndarray,
    graph: nx.MultiGraph,
    output_path: PathLike,
    dpi: int = 180,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 8))
    axis.imshow(skeleton, cmap="gray")
    _draw_graph(axis, graph, show_nodes=True)
    if graph.number_of_nodes():
        axis.legend(loc="upper right", fontsize=7)
    axis.set_title("Endpoints, junctions, and compressed road graph")
    axis.axis("off")
    _finish(figure, output_path, dpi)


def save_graph_overlay(
    background: np.ndarray,
    graph: nx.MultiGraph,
    output_path: PathLike,
    title: str = "Road graph overlay",
    show_nodes: bool = True,
    dpi: int = 180,
) -> None:
    """Overlay a graph on a grayscale mask or RGB satellite image."""
    figure, axis = plt.subplots(figsize=(8, 8))
    image = np.asarray(background)
    axis.imshow(image, cmap="gray" if image.ndim == 2 else None)
    _draw_graph(axis, graph, show_nodes=show_nodes)
    axis.set_title(title)
    axis.axis("off")
    _finish(figure, output_path, dpi)
