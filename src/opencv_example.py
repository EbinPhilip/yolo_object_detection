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

def drawPred(detectedObject: DetectedObject):
    # Draw a bounding box.
    box = detectedObject.box
    cv.rectangle(frame, (box.left, box.top), (box.right, box.bottom), (255, 178, 50), 3)
    label = '%s:%.2f' % (detectedObject.name, detectedObject.confidence)
    # Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(box.top, labelSize[1])
    cv.rectangle(frame, (box.left, top - round(1.5 * labelSize[1])), (box.left + round(1.5 * labelSize[0]), top + baseLine),
                 (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (box.left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)

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
    for obj in detectionResult.detectedObjects.values():
        drawPred(obj)

    # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and
    # the timings for each of the layers(in layersTimes)
    label = 'Inference time: %.2f ms' % (detectionResult.inferenceTime)
    cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))


    cv.imwrite(outputFile, frame.astype(np.uint8))

    cv.imshow(winName, frame)