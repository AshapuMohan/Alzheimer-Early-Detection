import base64
import httpx
import json
import httpx
from app.config import NVIDIA_API_KEY

NVIDIA_INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"


async def get_mri_view_and_location(image_bytes: bytes):
    """
    Analyzes the MRI image using NVIDIA API to determine its view and location.
    """
    try:
        # Convert image to base64
        import base64
        import io
        from PIL import Image
        
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

        prompt = (
            "Analyze this image. It should be a brain MRI. "
            "1. Identify if it is a SINGLE slice or a MULTI-VIEW grid/collage. "
            "2. Identify the view: 'Axial', 'Coronal', 'Sagittal', or 'Unknown'. "
            "3. Identify the vertical location: 'Upper', 'Mid', 'Lower', or 'Unknown'. "
            "Return ONLY a JSON object with keys: 'image_type' (values: 'dataset_image' for single slice, 'multi_view' for grid, 'not_mri' for others), "
            "'view', and 'location'. Do not include markdown formatting."
        )

        payload = {
            "model": "meta/llama-4-maverick-17b-128e-instruct",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 100,
            "temperature": 0.2, # Low temperature for consistent JSON
            "stream": False
        }

        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Accept": "application/json"
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(NVIDIA_INVOKE_URL, headers=headers, json=payload, timeout=30.0)
            response.raise_for_status()
            content = response.json()['choices'][0]['message']['content']
            
            # Clean content to ensure valid JSON (sometimes models add ```json ... ```)
            content = content.replace("```json", "").replace("```", "").strip()
            
            try:
                result = json.loads(content)
                # Normalize keys just in case
                return {
                    "image_type": result.get("image_type", "not_mri"),
                    "view": result.get("view", "Unknown"),
                    "location": result.get("location", "Unknown")
                }
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                print(f"Failed to parse JSON from NVIDIA API: {content}")
                return {
                    "image_type": "dataset_image", # Optimistic fallback or safe failure?
                    "view": "Unknown",
                    "location": "Unknown"
                }

    except Exception as e:
        print(f"Error in get_mri_view_and_location: {e}")
        return {
            "image_type": "not_mri",
            "view": "Unknown",
            "location": "Unknown"
        }

async def get_ai_suggestions(view: str, location: str):
    """
    Generates educational suggestions and disclaimers for a given MRI view and location.

    Args:
        view: The MRI view (e.g., 'Axial').
        location: The MRI slice location (e.g., 'mid-brain region').

    Returns:
        A string containing the AI-generated suggestions.
        Returns a default error message if the generation fails.
    """
    prompt = f"An MRI image has been identified as a '{view}' view from the '{location}'. In a supportive and neutral tone, provide some general, non-medical, educational information about what this type of MRI view is typically used for. Do not mention any diseases or abnormalities. Conclude with a clear disclaimer that this is not a medical analysis and a qualified healthcare professional should be consulted for any diagnosis. Structure the response in a clear, easy-to-read format."

    try:
        payload = {
            "model": "meta/llama-4-maverick-17b-128e-instruct",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant providing general information about medical imaging."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 400,
            "temperature": 0.5,
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
            return data['choices'][0]['message']['content']
    except Exception as e:
        # In case of API error, return a static suggestion based on view and location
        static_suggestions = {
            ("Axial", "Upper"): "The Axial view from the upper region provides a top-down perspective of the brain's upper structures. This view is commonly used to assess the cerebral hemispheres and ventricles. Please note: This is not a medical diagnosis. Consult a qualified healthcare professional for any health concerns.",
            ("Axial", "Mid"): "The Axial view from the mid region shows the brain's midline structures, including the thalamus and basal ganglia. It's useful for evaluating symmetry and central brain anatomy. Please note: This is not a medical diagnosis. Consult a qualified healthcare professional for any health concerns.",
            ("Axial", "Lower"): "The Axial view from the lower region focuses on the brainstem and cerebellum. This perspective helps in examining the posterior fossa structures. Please note: This is not a medical diagnosis. Consult a qualified healthcare professional for any health concerns.",
            ("Coronal", "Upper"): "The Coronal view from the upper region displays the frontal lobes and anterior brain structures. It's often used to visualize the frontal sinuses and orbital regions. Please note: This is not a medical diagnosis. Consult a qualified healthcare professional for any health concerns.",
            ("Coronal", "Mid"): "The Coronal view from the mid region shows the temporal lobes and Sylvian fissures. This view aids in assessing the middle cerebral artery territories. Please note: This is not a medical diagnosis. Consult a qualified healthcare professional for any health concerns.",
            ("Coronal", "Lower"): "The Coronal view from the lower region highlights the cerebellum and occipital lobes. It's helpful for evaluating the posterior brain regions. Please note: This is not a medical diagnosis. Consult a qualified healthcare professional for any health concerns.",
            ("Sagittal", "Upper"): "The Sagittal view from the upper region provides a side profile of the brain's superior structures, including the corpus callosum. This view is useful for midline assessments. Please note: This is not a medical diagnosis. Consult a qualified healthcare professional for any health concerns.",
            ("Sagittal", "Mid"): "The Sagittal view from the mid region shows the brain's lateral ventricle and central structures. It's commonly used to evaluate the ventricular system. Please note: This is not a medical diagnosis. Consult a qualified healthcare professional for any health concerns.",
            ("Sagittal", "Lower"): "The Sagittal view from the lower region focuses on the brainstem and spinal cord junction. This perspective helps in examining the cervicomedullary region. Please note: This is not a medical diagnosis. Consult a qualified healthcare professional for any health concerns.",
        }
        key = (view, location)
        return static_suggestions.get(key, "This MRI view provides valuable anatomical information about the brain. The specific details depend on the exact slice location. Please note: This is not a medical diagnosis. Consult a qualified healthcare professional for any health concerns.")
