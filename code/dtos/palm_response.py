from fastapi.responses import FileResponse
from typing import List, Optional
from .line_response import LineResponse

class PalmResponse:
    def __init__(self, image_file: FileResponse, lines: List[LineResponse], message: str = "Success"):
        self.image_file = image_file
        self.lines = lines
        self.message = message
    
    def to_dict(self):
        """Convert PalmResponse to dictionary for JSON serialization (without image file)"""
        return {
            "message": self.message,
            "lines": [
                {
                    "line_type": line.line_type,
                    "description": line.description,
                    "length": line.length
                } for line in self.lines
            ]
        }