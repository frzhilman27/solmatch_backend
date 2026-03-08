from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from services.vision import process_foot_image
import uvicorn
import base64
import numpy as np
import cv2

app = FastAPI(title="SoleMatch AI Engine")

# Allow the React frontend to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with the React app's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "SoleMatch AI Engine is running"}

@app.post("/api/measure")
async def measure_foot(request: dict):
    """
    Expects a JSON payload: { "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg..." }
    """
    try:
        base64_img = request.get("image")
        if not base64_img:
            raise HTTPException(status_code=400, detail="No image provided")
            
        # Clean the base64 prefix if present
        if "," in base64_img:
            base64_img = base64_img.split(",")[1]
            
        # Decode base64 to OpenCV image format
        img_data = base64.b64decode(base64_img)
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        # Process the image using Computer Vision
        result = process_foot_image(img)
        return result

    except Exception as e:
        print(f"Error processing measurement: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
