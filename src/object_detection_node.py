#!/usr/bin/python

from detector import Detector
import rospy
from sensor_msgs.msg import Image
from cv_bridge.core import CvBridge
from structs import DetectedObject
from yolo_object_detection.msg import DetectedObject
from yolo_object_detection.msg import DetectedObjectArray

modelConfiguration = "/home/ebin/cv_ws/src/yolo_object_detection/yolov3.cfg"
modelWeights = "/home/ebin/cv_ws/src/yolo_object_detection/yolov3.weights"
classesFile = "/home/ebin/cv_ws/src/yolo_object_detection/coco.names"
imagePublisher = None
boundingBoxPublisher = None

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
    

def imageCallback(image: Image) :
    frame = cvBridge.imgmsg_to_cv2(image, desired_encoding='bgr8')
    detector = Detector(classesFile, modelConfiguration, modelWeights)
    detectionResult = detector.detectObjects(frame)
    detector.drawBoundingBoxes(frame, detectionResult)
    imgMsg = cvBridge.cv2_to_imgmsg(frame, encoding='bgr8')
    imagePublisher.publish(imgMsg)
    detectedObjectArray = getDetectedObjectArrayMsg(detectionResult.detectedObjects)     
    boundingBoxPublisher.publish(detectedObjectArray)

if __name__ == "__main__":
    rospy.init_node('object_detection_node')
    imagePublisher = rospy.Publisher('/detected', Image, queue_size=1)
    boundingBoxPublisher = rospy.Publisher('/boundingBoxes2d', DetectedObjectArray, queue_size=1)
    rospy.Subscriber("/camera/color/image_raw", Image, imageCallback, queue_size=1)
    rospy.spin()