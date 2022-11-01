import rospy
from sensor_msgs.msg import Image
from cv_bridge.core import CvBridge

from yolo_object_detection.srv import ObjectDetection
from yolo_object_detection.srv import ObjectDetectionResponse, ObjectDetectionRequest
from yolo_object_detection.msg import DetectedObjectArray, DetectedObject

objectDetectorService = None
counter = 0

def callback(image):
    global counter
    request = ObjectDetectionRequest()
    request.image = image
    resp = objectDetectorService(request)
    counter = counter + 1
    if (counter>30):
        rospy.signal_shutdown("done!")

if __name__ == '__main__':
    rospy.init_node('object_detection_client_node')
    rospy.wait_for_service('object_detection_service')
    objectDetectorService = rospy.ServiceProxy('object_detection_service', ObjectDetection)
    rospy.Subscriber("/camera/color/image_raw", Image, callback)
    rospy.spin()
