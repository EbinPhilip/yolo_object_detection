#! /usr/bin/python

from detector import Detector
import rospy
from sensor_msgs.msg import Image
from cv_bridge.core import CvBridge

modelConfiguration = "yolov3.cfg"
modelWeights = "yolov3.weights"
classesFile = "coco.names"
imagePublisher = None

cvBridge = CvBridge()

def imageCallback(image: Image) :
    frame = cvBridge.imgmsg_to_cv2(image, desired_encoding='bgr8')
    detector = Detector(classesFile, modelConfiguration, modelWeights)
    detectionResult = detector.detectObjects(frame)
    detector.drawBoundingBoxes(frame, detectionResult)
    imgMsg = cvBridge.cv2_to_imgmsg(frame, encoding='bgr8')
    imagePublisher.publish(imgMsg)

if __name__ == "__main__":
    rospy.init_node('object_detection_node')
    imagePublisher = rospy.Publisher('/detected', Image, queue_size=1)
    rospy.Subscriber("/camera/color/image_raw", Image, imageCallback, queue_size=1)
    rospy.spin()