import rclpy  
from rclpy.node import Node  
from std_msgs.msg import Bool  
#Node 2
class BoolSubscriber(Node):  
    def __init__(self):  
        super().__init__('bool_subscriber')  
        self.subscription = self.create_subscription(  
            Bool,  
            'detectface',  
            self.listener_callback,  
            10)  
        self.subscription  # prevent unused variable warning  

    def listener_callback(self, msg):  
        self.get_logger().info(f'Received: {msg.data}')  

def main(args=None):  
    rclpy.init(args=args)  
    bool_subscriber = BoolSubscriber()  
    
    rclpy.spin(bool_subscriber)  

    bool_subscriber.destroy_node()  
    rclpy.shutdown()  

if __name__ == '__main__':  
    main()
