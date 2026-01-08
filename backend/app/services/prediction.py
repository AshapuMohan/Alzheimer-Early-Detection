import io
import logging
import pickle
from PIL import Image
import torch
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import torch.nn as nn
from app.models.cnn import ImprovedAlzheimerCNN # Import the model class

# New imports for image splitting
from skimage import measure, filters, morphology

import base64
import httpx
from app.config import (
    DEVICE,
    MODEL_PATH,
    NUM_CLASSES,
    CLASS_LABELS,
    IMAGE_TRANSFORMS,
    NVIDIA_API_KEY # Import the API key
)

NVIDIA_INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Similarity model & transform (MUST match feature generation) ---

similarity_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# Load the fine-tuned model for feature extraction
def get_feature_extractor():
    model = models.resnet18(weights=None) # Don't use pretrained weights, we load our own
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        logger.info("Fine-tuned model loaded successfully for feature extraction.")
    except FileNotFoundError:
        logger.error(f"Fine-tuned model not found at {MODEL_PATH}. Using generic ResNet-18.")
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)


    # Remove the final classification layer
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    feature_extractor.eval()
    feature_extractor.to(DEVICE)
    return feature_extractor

resnet_model = get_feature_extractor()


import os
import pickle

# Absolute path to backend directory
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

FEATURES_PATH = os.path.join(BASE_DIR, "features.pkl")


features_db = None

try:
    with open(FEATURES_PATH, "rb") as f:
        features_db = pickle.load(f)
    logger.info("features.pkl loaded successfully.")
    logger.info(f"Loaded {len(features_db['features'])} feature vectors")
except FileNotFoundError:
    logger.error(f"features.pkl NOT found at {FEATURES_PATH}")
except Exception as e:
    logger.error(f"Error loading features.pkl: {e}")




# --- NVIDIA-based MRI Scan Validation ---
async def nvidia_validate_mri_scan(img_bytes: bytes) -> bool | None:
    """
    Uses NVIDIA's API to validate if an image is a brain MRI scan.
    """
    if not NVIDIA_API_KEY:
        logger.error("NVIDIA API key not available for MRI validation.")
        return None # Return None to indicate ambiguity

    try:
        # Convert image to RGB before validation to be consistent to model input
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        # Save the converted image back to bytes
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        rgb_img_bytes = buffer.getvalue()

        base64_image = base64.b64encode(rgb_img_bytes).decode('utf-8')
        
        payload = {
            "model": "meta/llama-4-maverick-17b-128e-instruct",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": "Is this image a SINGLE brain MRI scan showing a cross-sectional view of the brain? If it is a grid, collage, or shows multiple slices, answer 'no'. Please answer with only the word 'yes' or 'no'."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 10,
            "temperature": 0.0, # Zero temp for strict yes/no
            "top_p": 1.00,
            "stream": False
        }

        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Accept": "application/json"
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(NVIDIA_INVOKE_URL, headers=headers, json=payload, timeout=30.0)
            response.raise_for_status()
            data = response.json()
            decision = data['choices'][0]['message']['content'].strip().lower()
            
        logger.info(f"NVIDIA validation response: '{decision}'")
        
        return "yes" in decision

    except Exception as e:
        logger.warning(f"NVIDIA MRI validation unavailable: {e}. Proceeding without validation.")
        return None # Return None on API error

def predict_alzheimer_stage(img_bytes: bytes) -> dict:
    try:
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        image = IMAGE_TRANSFORMS(image).unsqueeze(0).to(DEVICE)

        model = models.resnet18()
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()

        with torch.no_grad():
            outputs = model(image)
            probabilities = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()
            predicted_class = torch.argmax(outputs, dim=1).item()

        # Get all class probabilities
        class_probabilities = {CLASS_LABELS[i]: float(probabilities[i]) for i in range(len(CLASS_LABELS))}

        return {
            "predicted_class": CLASS_LABELS[predicted_class],
            "class_index": predicted_class,
            "probabilities": class_probabilities
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"error": str(e)}


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def find_similar_image(img_bytes: bytes) -> dict:
    if features_db is None or len(features_db.get("features", [])) == 0:
        return {
            "error": "Similarity search failed. features.pkl not loaded or empty."
        }

    try:
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        image = similarity_transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            query_feature = resnet_model(image)

        query_feature = (
            query_feature
            .squeeze()
            .cpu()
            .numpy()
            .reshape(1, -1)
        )

        dataset_features = np.array(features_db["features"])

        similarities = cosine_similarity(
            query_feature,
            dataset_features
        )[0]

        best_idx = int(np.argmax(similarities))

        return {
            "most_similar_image": features_db["paths"][best_idx],
            "label": features_db["labels"][best_idx],
            "similarity_score": float(similarities[best_idx])
        }

    except Exception as e:
        logger.exception("Similarity search crashed")
        return {
            "error": "Similarity search failed due to internal error."
        }

def split_multi_view_image(img_bytes: bytes) -> list[bytes]:
    """
    Splits a multi-view MRI image into individual scans.
    If no split is detected, returns empty list.
    """
    try:
        image = Image.open(io.BytesIO(img_bytes)).convert("L")
        np_img = np.array(image)

        thresh = filters.threshold_otsu(np_img)
        binary = np_img > thresh
        cleaned = morphology.remove_small_objects(binary, min_size=500)

        labels = measure.label(cleaned)
        regions = measure.regionprops(labels)

        scans = []
        for region in regions:
            minr, minc, maxr, maxc = region.bbox
            crop = image.crop((minc, minr, maxc, maxr))

            buf = io.BytesIO()
            crop.save(buf, format="JPEG")
            scans.append(buf.getvalue())

        return scans

    except Exception as e:
        logger.error(f"Image splitting error: {e}")
        return []
