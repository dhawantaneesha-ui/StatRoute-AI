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


def _build_skeleton_graph(skeleton: np.ndarray) -> nx.Graph:
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

    return graph


def _endpoint_direction(endpoint: tuple[int, int], graph: nx.Graph, steps: int = 10) -> np.ndarray:
    path = [endpoint]
    previous = None
    current = endpoint

    for _ in range(steps):
        neighbors = [node for node in graph.neighbors(current) if node != previous]

        if not neighbors:
            break

        previous = current
        current = neighbors[0]
        path.append(current)

        if graph.degree(current) != 2:
            break

    if len(path) < 2:
        return np.array([0.0, 0.0])

    end = np.array(path[0], dtype=float)
    inner = np.array(path[-1], dtype=float)
    direction = end - inner
    norm = np.linalg.norm(direction)

    if norm == 0:
        return np.array([0.0, 0.0])

    return direction / norm


def _cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    first_norm = np.linalg.norm(first)
    second_norm = np.linalg.norm(second)

    if first_norm == 0 or second_norm == 0:
        return 0.0

    return float(np.dot(first, second) / (first_norm * second_norm))


def reconstruct_missing_roads(
    mask_path: Path,
    output_path: Path,
    max_gap_pixels: int = 35,
    min_alignment_score: float = 0.35,
) -> dict:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise ValueError("Could not read road mask file.")

    binary_mask = mask > 0
    skeleton = skeletonize(binary_mask)
    graph = _build_skeleton_graph(skeleton)
    degrees = dict(graph.degree())
    endpoints = [node for node, degree in degrees.items() if degree == 1]
    components = {node: index for index, component in enumerate(nx.connected_components(graph)) for node in component}

    reconstructed = (skeleton.astype(np.uint8)) * 255
    used_endpoints = set()
    connections = []

    for index, first in enumerate(endpoints):
        if first in used_endpoints:
            continue

        first_direction = _endpoint_direction(first, graph)
        best_candidate = None
        best_score = -1.0

        for second in endpoints[index + 1:]:
            if second in used_endpoints:
                continue

            if components.get(first) == components.get(second):
                continue

            distance = float(np.linalg.norm(np.array(first) - np.array(second)))

            if distance > max_gap_pixels:
                continue

            second_direction = _endpoint_direction(second, graph)
            first_to_second = np.array(second, dtype=float) - np.array(first, dtype=float)
            first_to_second = first_to_second / max(np.linalg.norm(first_to_second), 1.0)
            second_to_first = -first_to_second

            first_alignment = _cosine_similarity(first_direction, first_to_second)
            second_alignment = _cosine_similarity(second_direction, second_to_first)
            alignment_score = (first_alignment + second_alignment) / 2

            if alignment_score > best_score:
                best_score = alignment_score
                best_candidate = (second, distance, alignment_score)

        if best_candidate is None:
            continue

        second, distance, alignment_score = best_candidate

        if alignment_score < min_alignment_score:
            continue

        cv2.line(reconstructed, (first[1], first[0]), (second[1], second[0]), 255, 2)
        used_endpoints.add(first)
        used_endpoints.add(second)
        connections.append(
            {
                "from": {"row": first[0], "col": first[1]},
                "to": {"row": second[0], "col": second[1]},
                "distance_pixels": round(distance, 2),
                "confidence": round(min(1.0, alignment_score), 2),
            }
        )

    kernel = np.ones((3, 3), np.uint8)
    reconstructed = cv2.morphologyEx(reconstructed, cv2.MORPH_CLOSE, kernel, iterations=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), reconstructed)

    return {
        "reconstructed_path": str(output_path),
        "original_endpoints": len(endpoints),
        "connections_added": len(connections),
        "max_gap_pixels": max_gap_pixels,
        "min_alignment_score": min_alignment_score,
        "connections": connections[:25],
    }
