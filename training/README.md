# StatRoute AI Model Training

This folder contains a PyTorch training pipeline for road segmentation.

## Dataset Format

Put your training dataset in this structure:

```text
datasets/
└── road_segmentation/
    ├── images/
    │   ├── sample_001.png
    │   ├── sample_002.png
    │   └── ...
    └── masks/
        ├── sample_001.png
        ├── sample_002.png
        └── ...
```

Rules:

- Every image must have a matching mask with the same filename.
- Images should be RGB satellite images.
- Masks should be binary road masks:
  - white = road
  - black = background

## Install Training Dependencies

From project root:

```powershell
cd C:\Users\ASUS\Desktop\StatRoute-AI
.\backend\venv\Scripts\activate
pip install -r training\requirements.txt
```

If PyTorch installation fails, install it from the official selector:

```text
https://pytorch.org/get-started/locally/
```

## Train U-Net

```powershell
python training\train_unet.py --data-dir datasets\road_segmentation --epochs 20 --batch-size 4 --image-size 256
```

The trained model will be saved here:

```text
models/road_unet.pt
```

## Test Prediction

```powershell
python training\predict_unet.py --model models\road_unet.pt --image path\to\satellite_image.png --output outputs\ai_road_mask.png
```

After training, the backend can be upgraded to use this model instead of the current OpenCV extraction.
