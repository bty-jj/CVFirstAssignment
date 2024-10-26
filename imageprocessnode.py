import numpy as np
import rclpy
from rclpy.node import Node
from detectfaceonnx import FaceDet
from cv_bridge import CvBridge  
from std_msgs.msg import Bool

from sensor_msgs.msg import Image as RosImg
#任务一
class Detectface(Node):
    def __init__(self):
        super().__init__('track_node')
        self.model=FaceDet()
        self.res_publisher = self.create_publisher(Bool, 'detectface', 10)  
        self.hasfacemsg = Bool()  

        self.camera_subscription = self.create_subscription(RosImg, '/camera', self.image_cb, 10)
        self.detect_publisher = self.create_publisher(RosImg, '/camerares', 10)  
        self.bridge = CvBridge()  
    

    def image_cb(self, msg):
        height = msg.height
        width = msg.width
        data = np.frombuffer(msg.data, dtype=np.uint8)
        data = data.reshape((height, width, 3)) 
        resimg,hasface=self.model.detect(data)
        msg = self.bridge.cv2_to_imgmsg(resimg, encoding='bgr8')  
        self.detect_publisher.publish(msg)  
        self.hasfacemsg.data=hasface 
        self.res_publisher.publish(self.hasfacemsg)
        self.get_logger().info(f'Face detect finish, result is: {hasface}') 
  
    
def main():
    rclpy.init()

    node = Detectface()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()