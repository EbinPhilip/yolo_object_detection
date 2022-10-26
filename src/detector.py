import numpy as np
import cv2 as cv

from structs import DetectedObject, ObjectBox, DetectionResult

class Detector:
    def __init__(self,
                 classesFilePath: str,
                 modelConfigurationPath: str,
                 modelWeightsPath: str,
                 confThreshold=0.3,  # Confidence threshold
                 nmsThreshold=0.2,  # Non-maximum suppression threshold
                 inpWidth=416,  # Width of network's input image
                 inpHeight=416,  # Height of network's input image
                 ):
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.inpWidth = inpWidth
        self.inpHeight = inpHeight
        self.classes = None
        with open(classesFilePath, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')
        self.net = cv.dnn.readNetFromDarknet(modelConfigurationPath, modelWeightsPath)
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    # Remove the bounding boxes with low confidence using non-maxima suppression
    def _postprocess(self, frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        classIds = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > self.confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
        # Perform non maximum suppression to eliminate redundant overlapping boxes with lower confidences.
        indices = cv.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        detectedObjects = []
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            objectBox = ObjectBox(left, top, left + width, top + height)
            name = self.classes[classIds[i]]
            obj = DetectedObject(name, objectBox, confidences[i])
            detectedObjects.append(obj)
        return detectedObjects


    # Get the names of the output layers
    def _getOutputsNames(self, net):
        # Get the names of all the layers in the network
        layers_names = net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    def detectObjects(self, frame) :
        # Create a 4D blob from a frame.
        blob = cv.dnn.blobFromImage(frame, 1 / 255, (self.inpWidth, self.inpHeight), [0, 0, 0], 1, crop=False)

        # Sets the input to the network
        self.net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = self.net.forward(self._getOutputsNames(self.net))

        # Remove the bounding boxes with low confidence
        detectedObjects = self._postprocess(frame, outs)

        # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and
        # the timings for each of the layers(in layersTimes)
        t, _ = self.net.getPerfProfile()
        inferenceTime = (t * 1000.0 / cv.getTickFrequency())

        result = DetectionResult(inferenceTime, detectedObjects)
        return result

    def drawBoundingBoxes(self, frame, detectionResult):
        for detectedObject in detectionResult.detectedObjects:
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
        # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and
        # the timings for each of the layers(in layersTimes)
        label = 'Inference time: %.2f ms' % (detectionResult.inferenceTime)
        cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))