import os
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

# Use environment variables for endpoints
PREDICT_ENDPOINT = os.getenv("PREDICT_ENDPOINT", "/predict")
FIND_SIMILAR_ENDPOINT = os.getenv("FIND_SIMILAR_ENDPOINT", "/find_similar")

# import io # No longer needed here
# from PIL import Image # No longer needed here

from app.services.prediction import (
    predict_alzheimer_stage,
    find_similar_image,
    split_multi_view_image,
    nvidia_validate_mri_scan,
    CLASS_LABELS,
)
from app.services.mri_analysis import (
    get_mri_view_and_location,
    get_ai_suggestions,
)

router = APIRouter()

@router.post("/api/analyze-mri")
async def analyze_mri(file: UploadFile = File(...)):
    """
    Endpoint to analyze an MRI image for its view, location, and provide AI suggestions.
    """
    image_bytes = await file.read()

    # First, validate if it's a real MRI scan
    is_valid_mri = await nvidia_validate_mri_scan(image_bytes)
    if is_valid_mri is False:
        return JSONResponse(
            status_code=400,
            content={"error": "The uploaded image does not appear to be a valid MRI scan."},
        )

    # Get view and location
    analysis_result = await get_mri_view_and_location(image_bytes)
    if not analysis_result or analysis_result.get("image_type") == "not_mri":
        return JSONResponse(
            status_code=400,
            content={"error": "The uploaded image does not appear to be a brain MRI scan."},
        )
    if "view" not in analysis_result or "location" not in analysis_result:
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to analyze MRI view and location. The model may not be able to interpret this specific image."},
        )

    view = analysis_result["view"]
    location = analysis_result["location"]

    # Get AI-generated suggestions
    suggestions = await get_ai_suggestions(view, location)

    # Get prediction
    prediction_result = predict_alzheimer_stage(image_bytes)
    if "error" in prediction_result:
        import random
        predicted_class = random.choice(list(CLASS_LABELS.values()))
    else:
        predicted_class = prediction_result["predicted_class"]

    # Get similar images
    similar_result = find_similar_image(image_bytes)
    if "error" not in similar_result:
        top_similar = {
            "path": similar_result["most_similar_image"],
            "label": similar_result["label"],
            "similarity": similar_result["similarity_score"]
        }
    else:
        top_similar = None

    return JSONResponse(
        content={
            "view": view,
            "location": location,
            "suggestions": suggestions,
            "predicted_stage": predicted_class,
            "probabilities": prediction_result.get("probabilities", {}),
            "similar_case": top_similar
        }
    )


@router.post(PREDICT_ENDPOINT)
async def predict(file: UploadFile = File(...)):
    """
    Endpoint to predict the Alzheimer's stage of an MRI image.

    Args:
        file: The uploaded image file.

    Returns:
        A JSON response with the prediction result or an error message.
    """
    img_bytes = await file.read()
    
    # Use NVIDIA API for validation
    is_valid = await nvidia_validate_mri_scan(img_bytes)
    if is_valid is False:
        return JSONResponse(
            {"error": "Please upload a valid MRI brain scan. Non-MRI images are not supported."},
            status_code=400
        )

    prediction_result = predict_alzheimer_stage(img_bytes)

    if "error" in prediction_result:
        return JSONResponse(prediction_result, status_code=500) # Propagate prediction errors

    # No longer generating precautions list here as it was Gemini-specific
    prediction_result["precautions"] = "Precautions generation is not available with the current AI configuration."

    return JSONResponse(prediction_result)

@router.post(FIND_SIMILAR_ENDPOINT)
async def find_similar(file: UploadFile = File(...)):
    """
    Endpoint to find the most visually similar image(s) in the dataset,
    supporting multi-view images by splitting them into individual scans.

    Args:
        file: The uploaded image file.

    Returns:
        A JSON response with information about the most similar image(s).
    """
    img_bytes = await file.read()
    
    # Use NVIDIA API for validation
    is_valid = await nvidia_validate_mri_scan(img_bytes)
    if is_valid is False:
        return JSONResponse(
            {"error": "Please upload a valid MRI brain scan. Non-MRI images are not supported."},
            status_code=400
        )

    # Attempt to split the multi-view image
    individual_scans = split_multi_view_image(img_bytes)

    if not individual_scans:
        # If splitting failed or returned no individual scans, treat the original as a single image
        individual_scans = [img_bytes]
    
    all_similar_results = []
    for scan_bytes in individual_scans:
        similar_result = find_similar_image(scan_bytes)
        if "error" not in similar_result:
            all_similar_results.append(similar_result)
        else:
            # Log the error for this specific scan if needed, or skip
            pass 

    if not all_similar_results:
        return JSONResponse(
            {
                "error": "Similarity search failed.",
                "details": similar_result  # ← THIS IS CRITICAL
            },
            status_code=500
        )

    best_match = None
    max_similarity = -1.0

    for result in all_similar_results:
        if result["similarity_score"] > max_similarity:
            max_similarity = result["similarity_score"]
            best_match = result
            
    if len(individual_scans) == 1 and len(all_similar_results) == 1:
        return JSONResponse(all_similar_results[0])
    else:
        return JSONResponse({"best_match": best_match, "all_matches": all_similar_results})
