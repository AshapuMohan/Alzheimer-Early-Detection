import torch
import os
from dotenv import load_dotenv

# Load .env from project root (parent of backend)
# __file__ = backend/app/config.py
# dirname = backend/app
# dirname = backend
# dirname = project root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(ROOT_DIR, ".env"))


# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model configuration
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "Models", "resnet18_alzheimer_best.pth")
NUM_CLASSES = 4
CLASS_LABELS = {
    0: "Mild Dementia",
    1: "Moderate Dementia",
    2: "Non Dementia",
    3: "Very Mild Dementia"
}

# Image transformations
from torchvision import transforms

IMAGE_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# CORS configuration
CORS_ORIGINS = ["*"]  # For development, restrict in production

# NVIDIA API Configuration
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    # Warning instead of raising error, to allow app to run even if key is missing (features will just be disabled)
    print("WARNING: NVIDIA_API_KEY environment variable not set. NVIDIA API features will be unavailable.")
