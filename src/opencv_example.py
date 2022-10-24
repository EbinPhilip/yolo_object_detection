import numpy as np
import cv2 as cv

from detector import Detector
from structs import DetectedObject

winName = 'Deep learning object detection in OpenCV'
cv.namedWindow(winName, cv.WINDOW_NORMAL)

cap = cv.VideoCapture('images/input.jpg')
outputFile = 'images/input_yolo_out_py.jpg'

modelConfiguration = "yolov3.cfg"
modelWeights = "yolov3.weights"
classesFile = "coco.names"
detector = Detector(classesFile, modelConfiguration, modelWeights)

while cv.waitKey(1) < 0:
    # get frame from the video
    hasFrame, frame = cap.read()

    # Stop the program if reached end of video
    if not hasFrame:
        print("Done processing !!!")
        print("Output file is stored as ", outputFile)
        cv.waitKey(3000)
        # Release device
        cap.release()
        break

    detectionResult = detector.detectObjects(frame)
    detector.drawBoundingBoxes(frame, detectionResult.detectedObjects.values())

    # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and
    # the timings for each of the layers(in layersTimes)
    label = 'Inference time: %.2f ms' % (detectionResult.inferenceTime)
    cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    cv.imwrite(outputFile, frame.astype(np.uint8))

    cv.imshow(winName, frame)