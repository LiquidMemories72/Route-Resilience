import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# ==================================
# CONFIG
# ==================================
MODEL_PATH = "best_model.pth"
VALID_DIR = "valid"
OUTPUT_DIR = "pred_masks"

IMG_SIZE = 512
THRESHOLD = 0.5
ENCODER = "mit_b2"

# ==================================
# DEVICE
# ==================================
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

# ==================================
# TRANSFORM
# ==================================
transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
    ToTensorV2(),
])

# ==================================
# MODEL
# ==================================
model = smp.FPN(
    encoder_name=ENCODER,
    encoder_weights=None,
    in_channels=3,
    classes=1,
)

ckpt = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    ckpt["model_state_dict"]
)
model.to(device)
model.eval()

# ==================================
# FILES
# ==================================
image_files = sorted([
    f for f in os.listdir(VALID_DIR)
    if f.endswith("_sat.jpg")
])

print(f"Found {len(image_files)} images")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================================
# INFERENCE
# ==================================
with torch.no_grad():

    for filename in tqdm(image_files):

        image_path = os.path.join(
            VALID_DIR,
            filename
        )

        image = cv2.imread(image_path)
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        transformed = transform(
            image=image
        )

        image_tensor = transformed["image"]

        logits = model(
            image_tensor
            .unsqueeze(0)
            .to(device)
        )

        pred = torch.sigmoid(
            logits
        ).squeeze()

        pred = pred.cpu().numpy()

        pred = (
            pred > THRESHOLD
        ).astype(np.uint8)

        output_name = filename.replace(
            "_sat.jpg",
            "_pred.png"
        )

        cv2.imwrite(
            os.path.join(
                OUTPUT_DIR,
                output_name
            ),
            pred * 255
        )

print()
print("Done.")
print(f"Saved masks to {OUTPUT_DIR}")