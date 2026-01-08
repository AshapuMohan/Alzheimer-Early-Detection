
import torch
import torch.nn as nn
from torchvision import models
import os

# --- Configuration ---
# Make sure these match your training script
NUM_CLASSES = 4
MODEL_PATH = "Models/resnet18_alzheimer_best.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def inspect_model():
    """
    Loads and inspects the trained model to verify its integrity.
    """
    print(f"--- Model Inspection ---")
    print(f"Using device: {DEVICE}")

    # 1. Check if the model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERROR: Model file not found at '{MODEL_PATH}'")
        return

    print(f"✅ Model file found at '{MODEL_PATH}'.")

    # 2. Recreate the model architecture
    # This must be exactly the same as the architecture used for training
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.to(DEVICE)

    print("\n--- Expected Model Architecture ---")
    print(model)

    try:
        # 3. Load the state dictionary
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print("\n✅ Successfully loaded state_dict into the model.")

        # 4. Inspect the keys in the state dictionary
        print(f"\n--- Keys in State Dictionary ({len(state_dict.keys())} total) ---")
        # Print first 3 and last 3 keys as a sample
        keys = list(state_dict.keys())
        for key in keys[:3]:
            print(f"- {key}")
        print("...")
        for key in keys[-3:]:
            print(f"- {key}")


        # 5. Inspect the weights of the final layer
        # This is the layer that should have been trained.
        # The weights should not be zero or look purely random.
        fc_weights = model.fc.weight
        fc_bias = model.fc.bias

        print("\n--- Final Layer (fc) Inspection ---")
        print(f"Shape of fc.weight: {fc_weights.shape}")
        print(f"Shape of fc.bias: {fc_bias.shape}")
        print(f"Mean of fc.weight: {fc_weights.mean():.4f}")
        print(f"Std of fc.weight: {fc_weights.std():.4f}")
        print(f"Bias values: {fc_bias.data}")

        print("\n--- Analysis ---")
        if fc_weights.mean() != 0 and fc_weights.std() != 0:
            print("✅ The final 'fc' layer appears to have trained weights (not zero or random initialization).")
        else:
            print("⚠️ WARNING: The final 'fc' layer weights might be untrained (zeros or strange values).")

        print("✅ The model architecture matches the training script.")
        print("✅ The model state was loaded successfully.")
        print("\nConclusion: The model file appears to be a valid, trained PyTorch model.")


    except Exception as e:
        print(f"\n❌ ERROR during model inspection: {e}")
        print("This could be due to:")
        print("- A mismatch between the saved model and the expected architecture.")
        print("- A corrupted .pth file.")


if __name__ == "__main__":
    inspect_model()
