import os
import shutil
import uuid
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path

# Import palm reading functions
from tools import *
from model import *
from rectification import *
from detection import *
from classification import *
from measurement import *

app = FastAPI(title="Palm Reading API", description="API for palm reading and analysis")

def process_palm_image(input_filename: str):
    """
    Process palm image using the existing palm reading logic
    """
    path_to_input_image = f'input/{input_filename}'

    results_dir = './results'
    os.makedirs(results_dir, exist_ok=True)

    resize_value = 256
    path_to_clean_image = 'results/palm_without_background.jpg'
    path_to_warped_image = 'results/warped_palm.jpg'
    path_to_warped_image_clean = 'results/warped_palm_clean.jpg'
    path_to_warped_image_mini = 'results/warped_palm_mini.jpg'
    path_to_warped_image_clean_mini = 'results/warped_palm_clean_mini.jpg'
    path_to_palmline_image = 'results/palm_lines.png'
    path_to_model = 'checkpoint/checkpoint_aug_epoch70.pth'
    path_to_result = 'results/result.jpg'

    try:
        # 0. Preprocess image
        remove_background(path_to_input_image, path_to_clean_image)

        # 1. Palm image rectification
        warp_result = warp(path_to_input_image, path_to_warped_image)
        if warp_result is None:
            raise Exception("Failed to warp palm image")

        remove_background(path_to_warped_image, path_to_warped_image_clean)
        resize(path_to_warped_image, path_to_warped_image_clean, path_to_warped_image_mini, path_to_warped_image_clean_mini, resize_value)

        # 2. Principal line detection
        net = UNet(n_channels=3, n_classes=1)
        net.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))
        detect(net, path_to_warped_image_clean, path_to_palmline_image, resize_value)

        # 3. Line classification
        lines = classify(path_to_palmline_image)

        # 4. Length measurement
        im, contents = measure(path_to_warped_image_mini, lines)

        # 5. Save result
        save_result(im, contents, resize_value, path_to_result)

        # Create temporary file for result
        temp_fd, temp_path = tempfile.mkstemp(suffix='.jpg')
        os.close(temp_fd)  # Close the file descriptor
        
        # Copy result to temporary file
        shutil.copy2(path_to_result, temp_path)
        
        # Clean up results and input directories
        shutil.rmtree(results_dir, ignore_errors=True)
        input_dir_path = os.path.dirname(path_to_input_image)
        if os.path.exists(input_dir_path):
            shutil.rmtree(input_dir_path, ignore_errors=True)
        
        return temp_path
    except Exception as e:
        # Clean up results directory on error
        if os.path.exists(results_dir):
            shutil.rmtree(results_dir, ignore_errors=True)
        raise Exception(f"Error processing palm image: {str(e)}")

@app.post("/predict")
async def predict_palm(file: UploadFile = File(...)):
    """
    Endpoint to upload palm image and get prediction result
    """
    # Check if file is an image
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Create input directory if it doesn't exist
    input_dir = Path("input")
    input_dir.mkdir(exist_ok=True)

    # Generate unique filename to avoid conflicts
    file_extension = Path(file.filename).suffix
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    input_path = input_dir / unique_filename
    
    temp_result_path = None

    try:
        # Save uploaded file
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process the image (this will clean up input and results directories)
        temp_result_path = process_palm_image(unique_filename)

        # Check if result file exists
        if not os.path.exists(temp_result_path):
            raise HTTPException(status_code=500, detail="Failed to generate result")

        # Return the result image
        return FileResponse(
            path=temp_result_path,
            media_type='image/jpeg',
            filename='result.jpg'
        )

    except Exception as e:
        # Clean up temporary result file on error
        if temp_result_path and os.path.exists(temp_result_path):
            os.unlink(temp_result_path)
        # Clean up uploaded file if it still exists
        if input_path.exists():
            input_path.unlink()
        # Clean up input directory if it still exists
        if input_dir.exists():
            shutil.rmtree(input_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """
    Root endpoint
    """
    return {"message": "Palm Reading API is running", "endpoint": "/predict"}

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)