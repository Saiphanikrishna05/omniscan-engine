import tempfile
import os
import shutil
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from engine import DeepfakeFusionEngine

app = FastAPI(title="Multimodal Deepfake Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

# Detect if you are on an Apple Silicon Mac (mps), Nvidia (cuda), or standard CPU
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# Initialize the Engine globally so it doesn't reload weights on every request
print("Loading Deepfake Engine into memory... This may take a moment.")
engine = DeepfakeFusionEngine(
    video_weights_path="weights/best_video_detector.pth", 
    audio_weights_path="weights/best_audio_detector_v2.pth", 
    device=DEVICE
)

@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    # FIX: Use the server's safe temporary directory
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, f"temp_{file.filename}")
    
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # 1. Determine the media type
        content_type = file.content_type
        
        # 2. Route to the correct engine function
        if content_type.startswith('video'):
            scan_results = engine.scan_media(temp_file_path)
            media_type = "Video"
        elif content_type.startswith('audio'):
            scan_results = engine.scan_audio_only(temp_file_path)
            media_type = "Audio"
        elif content_type.startswith('image'):
            scan_results = engine.scan_image_only(temp_file_path)
            media_type = "Image"
        else:
            return {"status": "Error", "diagnosis": "Unsupported file type."}
            
        return {
            "filename": file.filename,
            "media_type": media_type,
            "status": "Scan Complete",
            "diagnosis": scan_results["diagnosis"],
            "is_deepfake": scan_results["is_deepfake"],
            "video_confidence": round(scan_results["video_confidence"], 3) if scan_results["video_confidence"] is not None else None,
            "audio_confidence": round(scan_results["audio_confidence"], 3) if scan_results["audio_confidence"] is not None else None,
            "frames": scan_results.get("frames", [])
        }
        
    finally:
        # Clean up the file from the /tmp directory so the server doesn't run out of space!
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            
            
if __name__ == "__main__":
    import uvicorn
    # Hugging Face requires applications to run on port 7860
    uvicorn.run(app, host="0.0.0.0", port=7860)