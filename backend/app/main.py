from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles # New import

from app.routes import router as api_router
from app.config import CORS_ORIGINS

app = FastAPI(title="Alzheimer MRI Prediction API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the API router
app.include_router(api_router)

# Serve static files from the root of the backend directory
app.mount("/static", StaticFiles(directory="."), name="static") # New static files mount

@app.get("/")
def read_root():
    return {"message": "Welcome to the Alzheimer MRI Prediction API"}
