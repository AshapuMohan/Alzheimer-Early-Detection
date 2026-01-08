import os
import pickle
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
import numpy as np
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import hashlib

from app.config import MODEL_PATH, NUM_CLASSES

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
DATA_DIR = "Data"
FEATURES_PATH = "features.pkl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32 # Process images in batches for speed

# --- Model and Transform (must match prediction.py) ---
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

def get_model_hash():
    """Compute a hash of the model's state_dict to detect changes."""
    try:
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        # Convert state_dict to a string representation for hashing
        state_str = str(sorted(state_dict.items()))
        return hashlib.md5(state_str.encode()).hexdigest()
    except FileNotFoundError:
        return None

resnet_model = get_feature_extractor()
current_model_hash = get_model_hash()

# --- Custom Dataset for loading images ---
class ImagePathDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, image_path


def generate_features_db(force_regenerate=False):
    """
    Generates and saves the feature database (features.pkl) using batch processing.
    This function is resumable.
    If force_regenerate is True, it will regenerate all features regardless of existing file.
    """
    if not os.path.isdir(DATA_DIR):
        logger.error(f"Data directory '{DATA_DIR}' not found.")
        return

    # --- Load existing database or initialize a new one ---
    if os.path.exists(FEATURES_PATH) and not force_regenerate:
        logger.info(f"Found existing features file at '{FEATURES_PATH}'. Loading...")
        try:
            with open(FEATURES_PATH, "rb") as f:
                features_db = pickle.load(f)
            logger.info(f"Loaded {len(features_db['features'])} existing features.")
            
            # Check if model has changed
            stored_hash = features_db.get('model_hash')
            if stored_hash != current_model_hash:
                logger.warning("Model has changed since last feature extraction. Regenerating all features.")
                features_db = {"features": [], "labels": [], "paths": []}
            else:
                logger.info("Model hash matches. Resuming feature extraction.")
        except (pickle.UnpicklingError, EOFError, KeyError):
            logger.warning("Could not read existing features file or missing model hash. Starting fresh.")
            features_db = {"features": [], "labels": [], "paths": []}
    else:
        if force_regenerate:
            logger.info("Force regenerating all features.")
        features_db = {"features": [], "labels": [], "paths": []}

    # Store current model hash
    features_db['model_hash'] = current_model_hash

    existing_paths = set(features_db.get('paths', []))
    
    class_labels = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    logger.info(f"Found {len(class_labels)} classes: {class_labels}")

    for label in class_labels:
        class_dir = os.path.join(DATA_DIR, label)
        
        # --- Filter out already processed images ---
        all_image_files = [os.path.join(DATA_DIR, label, f).replace('\\', '/').replace('/', os.sep) for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        new_image_files = [p for p in all_image_files if p not in existing_paths]
        
        if not new_image_files:
            logger.info(f"All {len(all_image_files)} images in '{label}' are already processed. Skipping.")
            continue

        logger.info(f"Processing {len(new_image_files)} new images in '{label}' (out of {len(all_image_files)} total).")

        # --- Create dataset and dataloader for the new images ---
        dataset = ImagePathDataset([p.replace('/', os.sep) for p in new_image_files], transform=similarity_transform)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0) 

        with torch.no_grad():
            for images, paths in tqdm(dataloader, desc=f"Extracting features for {label}"):
                images = images.to(DEVICE)
                batch_features = resnet_model(images)
                
                batch_features = batch_features.squeeze().cpu().numpy()

                if batch_features.ndim == 1:
                    batch_features = batch_features.reshape(1, -1)
                
                features_db["features"].extend(batch_features)
                features_db["labels"].extend([label] * len(paths))
                relative_paths = [p.replace(os.sep, '/') for p in paths]
                features_db["paths"].extend(relative_paths)

        # --- Save progress after each class ---
        logger.info(f"Finished processing '{label}'. Saving progress...")
        try:
            with open(FEATURES_PATH, "wb") as f:
                pickle.dump(features_db, f)
            logger.info(f"Saved {len(features_db['features'])} total features to '{FEATURES_PATH}'")
        except Exception as e:
            logger.error(f"Failed to save feature database: {e}")


    if not features_db["features"]:
        logger.warning("No features were extracted. Is the data directory empty?")
        return

    # Validate features
    expected_shape = (512,)  # ResNet18 features before FC
    invalid_count = 0
    for i, feat in enumerate(features_db["features"]):
        if np.array(feat).shape != expected_shape:
            logger.warning(f"Invalid feature shape at index {i}: {np.array(feat).shape}")
            invalid_count += 1
    if invalid_count > 0:
        logger.error(f"Found {invalid_count} invalid features. Consider regenerating.")
    else:
        logger.info("All features have valid shape.")

    logger.info(f"--- Feature Generation Complete ---")
    logger.info(f"Total features in database: {len(features_db['features'])}")


if __name__ == "__main__":
    import sys
    force = len(sys.argv) > 1 and sys.argv[1] == "--force"
    logger.info(f"Using device: {DEVICE}")
    generate_features_db(force_regenerate=force)
