class LineResponse:
    def __init__(self, line_type: str, description: str, length: int):
        self.line_type = line_type
        self.description = description
        self.length = length