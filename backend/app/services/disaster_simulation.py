from pathlib import Path
import random

import cv2
import networkx as nx
import numpy as np
from skimage.morphology import skeletonize


NEIGHBOR_OFFSETS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def _build_graph_from_mask(mask_path: Path) -> tuple[nx.Graph, np.ndarray]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise ValueError("Could not read road mask file.")

    skeleton = skeletonize(mask > 0)
    road_pixels = np.argwhere(skeleton)
    graph = nx.Graph()

    for row, col in road_pixels:
        graph.add_node((int(row), int(col)))

    skeleton_set = set(map(tuple, road_pixels))

    for row, col in road_pixels:
        for row_offset, col_offset in NEIGHBOR_OFFSETS:
            neighbor = (int(row + row_offset), int(col + col_offset))

            if neighbor in skeleton_set:
                graph.add_edge((int(row), int(col)), neighbor)

    return graph, skeleton


def _largest_component_size(graph: nx.Graph) -> int:
    if graph.number_of_nodes() == 0:
        return 0

    return max(len(component) for component in nx.connected_components(graph))


def simulate_disaster(
    mask_path: Path,
    output_path: Path,
    failure_percent: float = 10,
    simulation_type: str = "random",
    seed: int = 42,
) -> dict:
    if failure_percent < 0 or failure_percent > 90:
        raise ValueError("failure_percent must be between 0 and 90.")

    if simulation_type not in {"random", "central"}:
        raise ValueError("simulation_type must be either 'random' or 'central'.")

    graph, skeleton = _build_graph_from_mask(mask_path)
    original_node_count = graph.number_of_nodes()
    original_edge_count = graph.number_of_edges()
    original_components = nx.number_connected_components(graph) if original_node_count else 0
    original_largest_component = _largest_component_size(graph)

    failure_count = int((failure_percent / 100) * original_node_count)
    rng = random.Random(seed)

    if simulation_type == "central":
        center_row = skeleton.shape[0] / 2
        center_col = skeleton.shape[1] / 2
        ranked_nodes = sorted(
            graph.nodes,
            key=lambda node: (node[0] - center_row) ** 2 + (node[1] - center_col) ** 2,
        )
        failed_nodes = ranked_nodes[:failure_count]
    else:
        failed_nodes = rng.sample(list(graph.nodes), failure_count) if failure_count else []

    damaged_graph = graph.copy()
    damaged_graph.remove_nodes_from(failed_nodes)

    damaged_node_count = damaged_graph.number_of_nodes()
    damaged_edge_count = damaged_graph.number_of_edges()
    damaged_components = nx.number_connected_components(damaged_graph) if damaged_node_count else 0
    damaged_largest_component = _largest_component_size(damaged_graph)

    if original_largest_component:
        retained_connectivity_percent = (damaged_largest_component / original_largest_component) * 100
    else:
        retained_connectivity_percent = 0

    connectivity_loss_percent = 100 - retained_connectivity_percent
    resilience_score = max(0, min(100, retained_connectivity_percent))

    visualization = np.zeros((skeleton.shape[0], skeleton.shape[1], 3), dtype=np.uint8)

    for row, col in damaged_graph.nodes:
        visualization[row, col] = (180, 180, 180)

    for row, col in failed_nodes:
        visualization[row, col] = (0, 0, 255)

    kernel = np.ones((3, 3), np.uint8)
    visualization = cv2.dilate(visualization, kernel, iterations=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), visualization)

    return {
        "simulation_type": simulation_type,
        "failure_percent": failure_percent,
        "seed": seed,
        "failed_nodes": len(failed_nodes),
        "before": {
            "nodes": original_node_count,
            "edges": original_edge_count,
            "connected_components": original_components,
            "largest_component_nodes": original_largest_component,
        },
        "after": {
            "nodes": damaged_node_count,
            "edges": damaged_edge_count,
            "connected_components": damaged_components,
            "largest_component_nodes": damaged_largest_component,
        },
        "connectivity_loss_percent": round(connectivity_loss_percent, 2),
        "resilience_score": round(resilience_score, 2),
        "visualization_path": str(output_path),
    }


def recommend_recovery_priority(
    mask_path: Path,
    output_path: Path,
    failure_percent: float = 10,
    simulation_type: str = "random",
    seed: int = 42,
    top_k: int = 10,
    candidate_limit: int = 250,
) -> dict:
    if failure_percent < 0 or failure_percent > 90:
        raise ValueError("failure_percent must be between 0 and 90.")

    if simulation_type not in {"random", "central"}:
        raise ValueError("simulation_type must be either 'random' or 'central'.")

    graph, skeleton = _build_graph_from_mask(mask_path)
    original_largest_component = _largest_component_size(graph)
    failure_count = int((failure_percent / 100) * graph.number_of_nodes())
    rng = random.Random(seed)

    if simulation_type == "central":
        center_row = skeleton.shape[0] / 2
        center_col = skeleton.shape[1] / 2
        ranked_nodes = sorted(
            graph.nodes,
            key=lambda node: (node[0] - center_row) ** 2 + (node[1] - center_col) ** 2,
        )
        failed_nodes = ranked_nodes[:failure_count]
    else:
        failed_nodes = rng.sample(list(graph.nodes), failure_count) if failure_count else []

    damaged_graph = graph.copy()
    damaged_graph.remove_nodes_from(failed_nodes)
    damaged_largest_component = _largest_component_size(damaged_graph)
    failed_node_set = set(failed_nodes)
    priority_candidates = []
    component_sizes = {}
    failed_components = {}

    for index, component in enumerate(nx.connected_components(damaged_graph)):
        component_sizes[index] = len(component)

        for node in component:
            failed_components[node] = index

    def candidate_score(node: tuple[int, int]) -> tuple[int, int]:
        surviving_neighbors = [neighbor for neighbor in graph.neighbors(node) if neighbor not in failed_node_set]
        touched_components = {failed_components.get(neighbor) for neighbor in surviving_neighbors}
        touched_components.discard(None)
        return len(touched_components), graph.degree(node)

    candidate_nodes = sorted(failed_nodes, key=candidate_score, reverse=True)[:candidate_limit]

    for node in candidate_nodes:
        touched_components = {
            failed_components.get(neighbor)
            for neighbor in graph.neighbors(node)
            if neighbor not in failed_node_set
        }
        touched_components.discard(None)

        restored_component_size = 1 + sum(component_sizes[index] for index in touched_components)
        restored_largest_component = max(damaged_largest_component, restored_component_size)
        restoration_gain = restored_largest_component - damaged_largest_component

        if original_largest_component:
            priority_score = restoration_gain / original_largest_component
        else:
            priority_score = 0

        priority_candidates.append(
            {
                "row": node[0],
                "col": node[1],
                "restoration_gain": restoration_gain,
                "priority_score": round(priority_score, 6),
                "original_degree": graph.degree(node),
            }
        )

    priority_candidates.sort(
        key=lambda item: (item["restoration_gain"], item["original_degree"]),
        reverse=True,
    )

    recovery_priority = priority_candidates[:top_k]

    for rank, item in enumerate(recovery_priority, start=1):
        item["rank"] = rank

    visualization = np.zeros((skeleton.shape[0], skeleton.shape[1], 3), dtype=np.uint8)

    for row, col in damaged_graph.nodes:
        visualization[row, col] = (160, 160, 160)

    for row, col in failed_nodes:
        visualization[row, col] = (0, 0, 255)

    for item in recovery_priority:
        cv2.circle(visualization, (item["col"], item["row"]), 5, (0, 255, 0), -1)
        cv2.circle(visualization, (item["col"], item["row"]), 7, (255, 255, 255), 1)

    kernel = np.ones((3, 3), np.uint8)
    visualization = cv2.dilate(visualization, kernel, iterations=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), visualization)

    return {
        "simulation_type": simulation_type,
        "failure_percent": failure_percent,
        "seed": seed,
        "failed_nodes": len(failed_nodes),
        "evaluated_candidates": len(candidate_nodes),
        "candidate_limit": candidate_limit,
        "damaged_largest_component_nodes": damaged_largest_component,
        "original_largest_component_nodes": original_largest_component,
        "top_k": top_k,
        "recovery_priority": recovery_priority,
        "visualization_path": str(output_path),
    }
