class ObjectBox:
    def __init__(self, left: int, top: int, right: int, bottom: int):
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

class DetectedObject:
    def __init__(self, name: str, box: ObjectBox, confidence: float) :
        self.name = name
        self.box = box
        self.confidence = confidence

class DetectionResult:
    def __init__(self, inferenceTime: int, detectedObjects) :
        self.inferenceTime = inferenceTime
        self.detectedObjects = detectedObjects

