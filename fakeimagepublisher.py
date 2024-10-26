import rclpy  
from rclpy.node import Node  
from sensor_msgs.msg import Image  
from cv_bridge import CvBridge  
import cv2  
import random

class ImagePublisher(Node):  
    def __init__(self):  
        super().__init__('image_publisher')  
        self.publisher_ = self.create_publisher(Image, 'camera', 10)  
        self.timer = self.create_timer(2.0, self.timer_callback)  
        self.bridge = CvBridge()
        imgpaths=['face.png','building.png']  
        
        self.cv_images = [cv2.imread(imgpath) for imgpath in imgpaths] # Replace with your image path  
        
    def timer_callback(self):  
        # Load and convert the image  
        cv_img=random.choice(self.cv_images)
        if cv_img is None:  
            self.get_logger().error('Could not read image. Check the path.')  
            return 
        # Convert OpenCV image to ROS Image message  
        msg = self.bridge.cv2_to_imgmsg(cv_img, encoding='bgr8')  
        self.publisher_.publish(msg)  
        self.get_logger().info('Publishing image')  

def main(args=None):  
    rclpy.init(args=args)  
    image_publisher = ImagePublisher()  
    
    rclpy.spin(image_publisher)  

    image_publisher.destroy_node()  
    rclpy.shutdown()  

if __name__ == '__main__':  
    main()