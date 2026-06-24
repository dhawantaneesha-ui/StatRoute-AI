import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from train_unet import UNet


def predict(args):
    model_path = Path(args.model)
    image_path = Path(args.image)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(model_path, map_location="cpu")
    image_size = checkpoint.get("image_size", args.image_size)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = UNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probability = torch.sigmoid(logits)[0, 0].cpu().numpy()

    mask = (probability > args.threshold).astype(np.uint8) * 255
    mask_image = Image.fromarray(mask, mode="L").resize(original_size, Image.Resampling.NEAREST)
    mask_image.save(output_path)
    print(f"Saved predicted road mask to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Predict a road mask with a trained U-Net model.")
    parser.add_argument("--model", default="models/road_unet.pt")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="outputs/ai_road_mask.png")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    predict(parse_args())
