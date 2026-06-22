from pathlib import Path

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

    binary_mask = mask > 0
    skeleton = skeletonize(binary_mask)

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


def generate_road_graph(mask_path: Path) -> dict:
    graph, _ = _build_graph_from_mask(mask_path)

    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()

    if node_count == 0:
        return {
            "nodes": 0,
            "edges": 0,
            "connected_components": 0,
            "largest_component_nodes": 0,
            "average_degree": 0,
            "network_density": 0,
        }

    components = list(nx.connected_components(graph))
    largest_component_nodes = max(len(component) for component in components)
    degrees = dict(graph.degree())

    average_degree = sum(degrees.values()) / node_count
    network_density = nx.density(graph)

    return {
        "nodes": node_count,
        "edges": edge_count,
        "connected_components": len(components),
        "largest_component_nodes": largest_component_nodes,
        "average_degree": round(average_degree, 2),
        "network_density": round(network_density, 6),
    }


def visualize_road_graph(mask_path: Path, output_path: Path) -> dict:
    graph, skeleton = _build_graph_from_mask(mask_path)

    visualization = np.zeros((skeleton.shape[0], skeleton.shape[1], 3), dtype=np.uint8)
    visualization[skeleton] = (255, 255, 255)

    degrees = dict(graph.degree())

    for row, col in graph.nodes:
        degree = degrees[(row, col)]

        if degree == 1:
            color = (0, 0, 255)
            radius = 2
        elif degree >= 3:
            color = (0, 255, 255)
            radius = 3
        else:
            continue

        cv2.circle(visualization, (col, row), radius, color, -1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), visualization)

    endpoint_count = sum(1 for degree in degrees.values() if degree == 1)
    intersection_count = sum(1 for degree in degrees.values() if degree >= 3)

    return {
        "visualization_path": str(output_path),
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "endpoints": endpoint_count,
        "intersections": intersection_count,
    }


def compare_road_networks(original_mask_path: Path, reconstructed_mask_path: Path) -> dict:
    original_metrics = generate_road_graph(original_mask_path)
    reconstructed_metrics = generate_road_graph(reconstructed_mask_path)

    original_components = original_metrics["connected_components"]
    reconstructed_components = reconstructed_metrics["connected_components"]
    components_reduced_by = original_components - reconstructed_components

    if original_components > 0:
        connectivity_improvement_percent = (components_reduced_by / original_components) * 100
    else:
        connectivity_improvement_percent = 0

    nodes_added = reconstructed_metrics["nodes"] - original_metrics["nodes"]
    edges_added = reconstructed_metrics["edges"] - original_metrics["edges"]

    if components_reduced_by > 0:
        status = "improved"
    elif components_reduced_by == 0 and edges_added > 0:
        status = "expanded"
    elif components_reduced_by == 0:
        status = "unchanged"
    else:
        status = "fragmented"

    return {
        "status": status,
        "original": original_metrics,
        "reconstructed": reconstructed_metrics,
        "components_reduced_by": components_reduced_by,
        "connectivity_improvement_percent": round(connectivity_improvement_percent, 2),
        "nodes_added": nodes_added,
        "edges_added": edges_added,
    }


def analyze_criticality(mask_path: Path, output_path: Path, top_k: int = 20, sample_size: int = 800) -> dict:
    graph, skeleton = _build_graph_from_mask(mask_path)
    node_count = graph.number_of_nodes()

    if node_count == 0:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), np.zeros((skeleton.shape[0], skeleton.shape[1], 3), dtype=np.uint8))

        return {
            "visualization_path": str(output_path),
            "visualization_url": None,
            "total_analyzed_nodes": 0,
            "critical_nodes": [],
        }

    degrees = dict(graph.degree())
    components = list(nx.connected_components(graph))
    largest_component = max(components, key=len)
    analysis_graph = graph.subgraph(largest_component).copy()

    candidate_nodes = [node for node in analysis_graph.nodes if degrees[node] >= 3]

    if not candidate_nodes:
        candidate_nodes = sorted(analysis_graph.nodes, key=lambda node: degrees[node], reverse=True)[:sample_size]

    if len(candidate_nodes) > sample_size:
        candidate_nodes = sorted(candidate_nodes, key=lambda node: degrees[node], reverse=True)[:sample_size]

    max_degree = max((degrees[node] for node in candidate_nodes), default=1)
    center_row = skeleton.shape[0] / 2
    center_col = skeleton.shape[1] / 2
    max_center_distance = float(np.hypot(center_row, center_col))
    ranked_nodes = []

    for node in candidate_nodes:
        degree_score = degrees[node] / max_degree if max_degree else 0
        center_distance = float(np.hypot(node[0] - center_row, node[1] - center_col))
        center_score = 1 - min(1, center_distance / max_center_distance)
        branch_score = min(1, len(list(analysis_graph.neighbors(node))) / 4)
        criticality_score = (0.55 * degree_score) + (0.30 * branch_score) + (0.15 * center_score)

        ranked_nodes.append(
            {
                "row": node[0],
                "col": node[1],
                "degree": degrees[node],
                "centrality_method": "fast_intersection_proxy",
                "criticality_score": round(criticality_score, 6),
            }
        )

    ranked_nodes.sort(key=lambda item: item["criticality_score"], reverse=True)
    critical_nodes = ranked_nodes[:top_k]

    for rank, node in enumerate(critical_nodes, start=1):
        node["rank"] = rank

    visualization = np.zeros((skeleton.shape[0], skeleton.shape[1], 3), dtype=np.uint8)
    visualization[skeleton] = (180, 180, 180)

    for node in critical_nodes:
        cv2.circle(visualization, (node["col"], node["row"]), 5, (0, 0, 255), -1)
        cv2.circle(visualization, (node["col"], node["row"]), 7, (0, 255, 255), 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), visualization)

    return {
        "visualization_path": str(output_path),
        "total_graph_nodes": node_count,
        "total_analyzed_nodes": len(candidate_nodes),
        "largest_component_nodes": analysis_graph.number_of_nodes(),
        "top_k": top_k,
        "critical_nodes": critical_nodes,
    }
