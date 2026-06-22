from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app.services.disaster_simulation import recommend_recovery_priority, simulate_disaster
from app.services.graph_analysis import (
    analyze_criticality,
    compare_road_networks,
    generate_road_graph,
    visualize_road_graph,
)
from app.services.road_extraction import extract_road_mask
from app.services.road_reconstruction import reconstruct_missing_roads

APP_NAME = "StatRoute AI"

BASE_DIR = Path(__file__).resolve().parents[1]
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR.parent / "outputs"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

app = FastAPI(
    title=APP_NAME,
    description="AI-powered road extraction and disaster resilience analysis platform.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "Welcome to StatRoute AI",
        "status": "Backend is running",
        "version": "0.1.0",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": APP_NAME,
    }


@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    file_extension = Path(file.filename).suffix.lower()

    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload JPG, PNG, or TIFF satellite imagery.",
        )

    unique_filename = f"{uuid4().hex}{file_extension}"
    save_path = UPLOAD_DIR / unique_filename

    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    save_path.write_bytes(file_bytes)

    return JSONResponse(
        content={
            "message": "Image uploaded successfully",
            "original_filename": file.filename,
            "saved_filename": unique_filename,
            "saved_path": str(save_path),
        }
    )


@app.post("/extract-roads/{filename}")
def extract_roads(filename: str):
    image_path = UPLOAD_DIR / filename

    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Uploaded image not found.")

    output_filename = f"{Path(filename).stem}_road_mask.png"
    output_path = OUTPUT_DIR / output_filename

    try:
        result = extract_road_mask(image_path=image_path, output_path=output_path)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return {
        "message": "Road mask generated successfully",
        "input_filename": filename,
        "mask_filename": output_filename,
        "mask_url": f"/outputs/{output_filename}",
        "metrics": result,
    }


@app.get("/outputs/{filename}")
def get_output_file(filename: str):
    file_path = OUTPUT_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found.")

    return FileResponse(file_path)


@app.post("/generate-graph/{mask_filename}")
def generate_graph(mask_filename: str):
    mask_path = OUTPUT_DIR / mask_filename

    if not mask_path.exists():
        raise HTTPException(status_code=404, detail="Road mask file not found.")

    try:
        graph_metrics = generate_road_graph(mask_path=mask_path)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return {
        "message": "Road graph generated successfully",
        "mask_filename": mask_filename,
        "graph_metrics": graph_metrics,
    }


@app.post("/visualize-graph/{mask_filename}")
def visualize_graph(mask_filename: str):
    mask_path = OUTPUT_DIR / mask_filename

    if not mask_path.exists():
        raise HTTPException(status_code=404, detail="Road mask file not found.")

    output_filename = f"{Path(mask_filename).stem}_graph.png"
    output_path = OUTPUT_DIR / output_filename

    try:
        result = visualize_road_graph(mask_path=mask_path, output_path=output_path)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return {
        "message": "Road graph visualization generated successfully",
        "mask_filename": mask_filename,
        "graph_filename": output_filename,
        "graph_url": f"/outputs/{output_filename}",
        "metrics": result,
    }


@app.post("/reconstruct-roads/{mask_filename}")
def reconstruct_roads(mask_filename: str, max_gap_pixels: int = 35, min_alignment_score: float = 0.35):
    mask_path = OUTPUT_DIR / mask_filename

    if not mask_path.exists():
        raise HTTPException(status_code=404, detail="Road mask file not found.")

    output_filename = f"{Path(mask_filename).stem}_reconstructed.png"
    output_path = OUTPUT_DIR / output_filename

    try:
        result = reconstruct_missing_roads(
            mask_path=mask_path,
            output_path=output_path,
            max_gap_pixels=max_gap_pixels,
            min_alignment_score=min_alignment_score,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return {
        "message": "Missing road reconstruction completed successfully",
        "mask_filename": mask_filename,
        "reconstructed_filename": output_filename,
        "reconstructed_url": f"/outputs/{output_filename}",
        "metrics": result,
    }


@app.post("/compare-network")
def compare_network(original_mask_filename: str, reconstructed_mask_filename: str):
    original_mask_path = OUTPUT_DIR / original_mask_filename
    reconstructed_mask_path = OUTPUT_DIR / reconstructed_mask_filename

    if not original_mask_path.exists():
        raise HTTPException(status_code=404, detail="Original road mask file not found.")

    if not reconstructed_mask_path.exists():
        raise HTTPException(status_code=404, detail="Reconstructed road mask file not found.")

    try:
        comparison = compare_road_networks(
            original_mask_path=original_mask_path,
            reconstructed_mask_path=reconstructed_mask_path,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return {
        "message": "Road network comparison completed successfully",
        "original_mask_filename": original_mask_filename,
        "reconstructed_mask_filename": reconstructed_mask_filename,
        "comparison": comparison,
    }


@app.post("/criticality/{mask_filename}")
def criticality(mask_filename: str, top_k: int = 20, sample_size: int = 800):
    mask_path = OUTPUT_DIR / mask_filename

    if not mask_path.exists():
        raise HTTPException(status_code=404, detail="Road mask file not found.")

    output_filename = f"{Path(mask_filename).stem}_criticality.png"
    output_path = OUTPUT_DIR / output_filename

    try:
        result = analyze_criticality(
            mask_path=mask_path,
            output_path=output_path,
            top_k=top_k,
            sample_size=sample_size,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return {
        "message": "Critical road-node analysis completed successfully",
        "mask_filename": mask_filename,
        "criticality_filename": output_filename,
        "criticality_url": f"/outputs/{output_filename}",
        "metrics": result,
    }


@app.post("/simulate-disaster/{mask_filename}")
def disaster_simulation(
    mask_filename: str,
    failure_percent: float = 10,
    simulation_type: str = "random",
    seed: int = 42,
):
    mask_path = OUTPUT_DIR / mask_filename

    if not mask_path.exists():
        raise HTTPException(status_code=404, detail="Road mask file not found.")

    output_filename = f"{Path(mask_filename).stem}_{simulation_type}_{int(failure_percent)}pct_disaster.png"
    output_path = OUTPUT_DIR / output_filename

    try:
        result = simulate_disaster(
            mask_path=mask_path,
            output_path=output_path,
            failure_percent=failure_percent,
            simulation_type=simulation_type,
            seed=seed,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return {
        "message": "Disaster simulation completed successfully",
        "mask_filename": mask_filename,
        "simulation_filename": output_filename,
        "simulation_url": f"/outputs/{output_filename}",
        "metrics": result,
    }


@app.post("/recovery-priority/{mask_filename}")
def recovery_priority(
    mask_filename: str,
    failure_percent: float = 10,
    simulation_type: str = "random",
    seed: int = 42,
    top_k: int = 10,
    candidate_limit: int = 250,
):
    mask_path = OUTPUT_DIR / mask_filename

    if not mask_path.exists():
        raise HTTPException(status_code=404, detail="Road mask file not found.")

    output_filename = f"{Path(mask_filename).stem}_{simulation_type}_{int(failure_percent)}pct_recovery.png"
    output_path = OUTPUT_DIR / output_filename

    try:
        result = recommend_recovery_priority(
            mask_path=mask_path,
            output_path=output_path,
            failure_percent=failure_percent,
            simulation_type=simulation_type,
            seed=seed,
            top_k=top_k,
            candidate_limit=candidate_limit,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return {
        "message": "Recovery priority recommendation completed successfully",
        "mask_filename": mask_filename,
        "recovery_filename": output_filename,
        "recovery_url": f"/outputs/{output_filename}",
        "metrics": result,
    }
