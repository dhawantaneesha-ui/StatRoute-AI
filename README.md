# StatRoute AI

StatRoute AI is an AI-assisted geospatial platform for extracting road networks from satellite imagery, reconstructing missing road connectivity, converting the road network into a graph, and analyzing transportation resilience under disruptive events.

The project is designed for use cases such as urban planning, disaster management, emergency response, infrastructure analysis, and route resilience assessment.

## Problem Statement

Satellite imagery is widely used for mapping road networks, but roads are often partially hidden by tree canopies, shadows, vehicles, buildings, clouds, and disaster damage. These occlusions create broken road masks and disconnected road graphs, reducing the usefulness of extracted maps.

StatRoute AI addresses this by not only extracting visible roads, but also reconstructing likely missing road segments and analyzing how resilient the network is during failures.

## Key Features

- Upload satellite images through a web dashboard.
- Extract road-like regions from imagery using image processing.
- Convert road masks into skeleton-based graph structures.
- Generate graph metrics such as nodes, edges, connected components, and largest component size.
- Visualize road skeletons, endpoints, and intersections.
- Reconstruct missing road segments using endpoint proximity and alignment.
- Compare original and reconstructed road networks.
- Identify critical road nodes using fast intersection-based scoring.
- Simulate disasters by removing road network nodes.
- Calculate connectivity loss and resilience score.
- Recommend recovery priority points after disruption.
- View outputs in separate frontend pages: Overview, Extraction, Reconstruction, Resilience, and Recovery.

## Tech Stack

### Backend

- Python
- FastAPI
- Uvicorn
- OpenCV
- NumPy
- scikit-image
- NetworkX
- Pillow
- python-multipart

### Frontend

- React
- Vite
- lucide-react
- HTML/CSS

## Project Structure

```text
StatRoute-AI/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   └── services/
│   │       ├── disaster_simulation.py
│   │       ├── graph_analysis.py
│   │       ├── road_extraction.py
│   │       └── road_reconstruction.py
│   ├── uploads/
│   ├── venv/
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── main.jsx
│   │   └── styles.css
│   ├── index.html
│   └── package.json
├── datasets/
├── notebooks/
├── outputs/
└── README.md
```

## How To Run

### 1. Start Backend

Open a terminal:

```powershell
cd C:\Users\ASUS\Desktop\StatRoute-AI\backend
.\venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Backend runs at:

```text
http://127.0.0.1:8000
```

API docs:

```text
http://127.0.0.1:8000/docs
```

### 2. Start Frontend

Open another terminal:

```powershell
cd C:\Users\ASUS\Desktop\StatRoute-AI\frontend
npm run dev
```

Frontend runs at:

```text
http://127.0.0.1:5173/
```

## Workflow

```text
Satellite Image
      ↓
Road Extraction
      ↓
Binary Road Mask
      ↓
Skeletonization
      ↓
Graph Generation
      ↓
Missing Road Reconstruction
      ↓
Network Comparison
      ↓
Criticality Analysis
      ↓
Disaster Simulation
      ↓
Recovery Priority Recommendation
```

## Algorithms Used

### Road Extraction

The prototype uses an OpenCV-based image processing pipeline:

1. Resize image.
2. Convert to grayscale.
3. Improve contrast using CLAHE.
4. Apply Gaussian blur.
5. Detect edges using Canny edge detection.
6. Use morphological closing and dilation.
7. Save binary road mask.

### Graph Creation

1. Convert the road mask into a skeleton using `skimage.morphology.skeletonize`.
2. Treat each white skeleton pixel as a graph node.
3. Check 8-neighbor connectivity around every skeleton pixel.
4. Add graph edges between neighboring road pixels.
5. Store and analyze the network using NetworkX.

### Missing Road Reconstruction

The reconstruction module uses an endpoint proximity and alignment algorithm:

1. Detect graph endpoints, where node degree equals 1.
2. Search for nearby endpoints within a maximum gap distance.
3. Estimate direction of each endpoint from its local road path.
4. Connect endpoint pairs if they are close and directionally aligned.
5. Generate a reconstructed road mask.

### Criticality Analysis

The criticality module uses fast intersection-based scoring:

1. Focus on high-degree road nodes and junction-like points.
2. Score nodes using degree, branching importance, and position.
3. Rank the most critical nodes.
4. Visualize critical nodes over the road skeleton.

### Disaster Simulation

The disaster module uses graph disruption simulation:

1. Build the road graph.
2. Remove a percentage of road nodes.
3. Support random failures and central-region failures.
4. Recalculate connected components and largest component size.
5. Calculate connectivity loss and resilience score.

### Recovery Priority

The recovery module uses connectivity restoration ranking:

1. Identify failed road nodes.
2. Check which disconnected components each failed node can reconnect.
3. Estimate restoration gain.
4. Rank repair points by gain.
5. Recommend the top recovery locations.

## Backend APIs

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Backend welcome message |
| GET | `/health` | Health check |
| POST | `/upload-image` | Upload satellite image |
| POST | `/extract-roads/{filename}` | Generate road mask |
| GET | `/outputs/{filename}` | Serve generated output image |
| POST | `/generate-graph/{mask_filename}` | Generate graph metrics |
| POST | `/visualize-graph/{mask_filename}` | Generate graph visualization |
| POST | `/reconstruct-roads/{mask_filename}` | Reconstruct missing roads |
| POST | `/compare-network` | Compare original and reconstructed networks |
| POST | `/criticality/{mask_filename}` | Detect critical road nodes |
| POST | `/simulate-disaster/{mask_filename}` | Simulate road failures |
| POST | `/recovery-priority/{mask_filename}` | Recommend repair priorities |

## Output Color Meanings

### Graph View

- White: road skeleton
- Red: endpoints
- Yellow: intersections or junction-like points

### Criticality View

- Gray: road skeleton
- Red/yellow circles: critical road nodes

### Disaster View

- Gray: surviving roads
- Red: failed or blocked road nodes

### Recovery View

- Gray: surviving roads
- Red: failed roads
- Green: recommended priority repair points

## Current Limitations

- The current road extraction model is image-processing based, not a trained deep learning model.
- Results depend on image quality, contrast, resolution, and road visibility.
- Graph nodes are pixel-level nodes, so large images can create large graphs.
- Geographic coordinates are not yet attached to outputs.
- Disaster simulation currently supports random and central failures only.

## Future Improvements

- Add U-Net, DeepLabV3+, or SegFormer for stronger road segmentation.
- Add GeoTIFF and coordinate-aware road graph export.
- Export graph as GeoJSON.
- Add OpenStreetMap comparison.
- Add report export as PDF.
- Add real disaster layers such as flood zones, landslide-prone areas, and population density.
- Add route reliability and emergency accessibility analysis.

## Short Project Pitch

StatRoute AI extracts road networks from satellite imagery, reconstructs hidden or broken road segments, converts the result into a graph, and analyzes road network resilience under disaster scenarios. Unlike basic road extraction tools, it provides reconstruction, criticality detection, disaster simulation, and recovery priority recommendations in one platform.
