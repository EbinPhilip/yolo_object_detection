#! /usr/bin/python

import rospy
from detector import Detector
from sensor_msgs.msg import Image
from cv_bridge.core import CvBridge

from yolo_object_detection.srv import ObjectDetection
from yolo_object_detection.srv import ObjectDetectionResponse, ObjectDetectionRequest
from yolo_object_detection.msg import DetectedObjectArray, DetectedObject

modelConfiguration = "yolov3.cfg"
modelWeights = "yolov3.weights"
classesFile = "coco.names"
imagePublisher = None
cvBridge = CvBridge()

def getDetectedObjectArrayMsg(objects) -> DetectedObjectArray:
    detectedObjects = DetectedObjectArray()
    for object in objects:
        detectedObject = DetectedObject()
        detectedObject.name = object.name
        detectedObject.confidence = object.confidence
        detectedObject.box.bottom = object.box.bottom
        detectedObject.box.top = object.box.top
        detectedObject.box.left = object.box.left
        detectedObject.box.right = object.box.right
        detectedObjects.box.append(detectedObject)
    return detectedObjects

def detectObjects(request:ObjectDetectionRequest):
    image = request.image
    frame = cvBridge.imgmsg_to_cv2(image, desired_encoding='bgr8')

    detector = Detector(classesFile, modelConfiguration, modelWeights)
    detectionResult = detector.detectObjects(frame)

    detector.drawBoundingBoxes(frame, detectionResult)
    imgMsg = cvBridge.cv2_to_imgmsg(frame, encoding='bgr8')
    imagePublisher.publish(imgMsg)

    detectedObjectArray = getDetectedObjectArrayMsg(detectionResult.detectedObjects)    
    objectDetectionResponse = ObjectDetectionResponse()
    objectDetectionResponse.detectedObjects = detectedObjectArray
    return objectDetectionResponse

if __name__ == "__main__":
    rospy.init_node('object_detection_service_node')
    imagePublisher = rospy.Publisher('/detected', Image, queue_size=1)
    service = rospy.Service('object_detection_service', ObjectDetection, detectObjects)
    rospy.spin()

