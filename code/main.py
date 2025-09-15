import os
import shutil
import uuid
import tempfile
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
from typing import Dict, Any

# Import palm reading functions
from tools import *
from model import *
from rectification import *
from detection import *
from classification import *
from measurement import *
from dtos.line_response import LineResponse

# In-memory storage for processed results (in production, use Redis or database)
processed_results = {}

def cleanup_expired_results():
    """Clean up expired results (older than 10 minutes)"""
    current_time = time.time()
    expired_ids = [
        result_id for result_id, data in processed_results.items()
        if current_time - data['timestamp'] > 600  # 10 minutes
    ]
    for result_id in expired_ids:
        data = processed_results.pop(result_id, {})
        if 'image_path' in data and os.path.exists(data['image_path']):
            os.unlink(data['image_path'])

app = FastAPI(title="Palm Reading API", description="API for palm reading and analysis")

def process_palm_image(input_filename: str):
    """
    Process palm image using the existing palm reading logic
    Returns both the temporary image path and structured line data
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

        # 4. Length measurement with structured data
        im, line_responses = measure_with_structured_data(path_to_warped_image_mini, lines)

        # 5. Save result using simplified function
        save_result_simple(im, path_to_result)

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
        
        return temp_path, line_responses
    except Exception as e:
        # Clean up results directory on error
        if os.path.exists(results_dir):
            shutil.rmtree(results_dir, ignore_errors=True)
        raise Exception(f"Error processing palm image: {str(e)}")

@app.post("/analyze", response_model=Dict[str, Any])
async def analyze_palm(file: UploadFile = File(...)):
    """
    Bước 1: Upload ảnh và nhận analysis JSON + session_id
    Ảnh sẽ được lưu trong buffer với session_id để lấy sau
    """
    cleanup_expired_results()
    
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
    session_id = str(uuid.uuid4())

    try:
        # Save uploaded file
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process the image
        temp_result_path, line_responses = process_palm_image(unique_filename)

        # Check if result file exists
        if not os.path.exists(temp_result_path):
            raise HTTPException(status_code=500, detail="Failed to generate result")

        # Store result in memory buffer with session_id
        processed_results[session_id] = {
            'image_path': temp_result_path,
            'line_responses': line_responses,
            'timestamp': time.time()
        }

        # Return JSON analysis with session_id
        return {
            "session_id": session_id,
            "message": "Palm reading analysis completed successfully",
            "lines": [
                {
                    "line_type": line.line_type,
                    "description": line.description,
                    "length": line.length
                } for line in line_responses
            ]
        }

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

@app.get("/result/{session_id}")
async def get_result_image(session_id: str):
    """
    Bước 2: Lấy ảnh kết quả từ buffer bằng session_id
    """
    cleanup_expired_results()
    
    if session_id not in processed_results:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    result_data = processed_results[session_id]
    image_path = result_data['image_path']
    
    if not os.path.exists(image_path):
        # Clean up invalid entry
        processed_results.pop(session_id, None)
        raise HTTPException(status_code=404, detail="Result image not found")
    
    # Return the image (file will be cleaned up by periodic cleanup or manual DELETE)
    return FileResponse(
        path=image_path,
        media_type='image/jpeg',
        filename=f'result_{session_id}.jpg'
    )

@app.delete("/result/{session_id}")
async def cleanup_result(session_id: str):
    """
    Bước 3 (Optional): Xóa kết quả khỏi buffer sau khi đã lấy xong
    """
    if session_id not in processed_results:
        raise HTTPException(status_code=404, detail="Session not found")
    
    result_data = processed_results.pop(session_id)
    
    # Clean up image file
    if 'image_path' in result_data and os.path.exists(result_data['image_path']):
        os.unlink(result_data['image_path'])
    
    return {"message": "Result cleaned up successfully"}

@app.get("/")
async def root():
    """
    Root endpoint
    """
    return {
        "message": "Palm Reading API is running",
        "workflow": {
            "step_1": "POST /analyze - Upload image, get JSON analysis + session_id",
            "step_2": "GET /result/{session_id} - Get processed image from buffer", 
            "step_3": "DELETE /result/{session_id} - Clean up (optional, auto-cleanup after 10min)"
        },
        "endpoints": {
            "/analyze": "Upload palm image and get analysis data + session_id",
            "/result/{session_id}": "Get processed result image",
            "/health": "Health check endpoint"
        },
        "example_usage": [
            "1. POST /analyze (with image) -> get session_id + analysis JSON",
            "2. GET /result/{session_id} -> download processed image", 
            "3. DELETE /result/{session_id} -> cleanup (optional)"
        ]
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)