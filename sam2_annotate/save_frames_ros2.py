import os
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class ImageSaverNode(Node):
    def __init__(self):
        super().__init__('image_saver_node')

        self.declare_parameter('image_topic', '/rdx/camera_8')
        self.declare_parameter('output_dir', 'frames')
        self.declare_parameter('filename_format', '%05d.jpg')
        self.declare_parameter('start_index', 0)
        self.declare_parameter('encoding', 'bgr8')
        self.declare_parameter('save_stride', 10)   # save every N frames

        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.output_dir = self.get_parameter('output_dir').get_parameter_value().string_value
        self.filename_format = self.get_parameter('filename_format').get_parameter_value().string_value
        self.index = self.get_parameter('start_index').get_parameter_value().integer_value
        self.encoding = self.get_parameter('encoding').get_parameter_value().string_value
        self.save_stride = self.get_parameter('save_stride').get_parameter_value().integer_value

        os.makedirs(self.output_dir, exist_ok=True)
        self.bridge = CvBridge()
        self.frame_count = 5  # counts every incoming frame

        self.sub = self.create_subscription(
            Image, self.image_topic, self.cb, qos_profile_sensor_data
        )

        self.get_logger().info(
            f"Saving every {self.save_stride} frame(s) from '{self.image_topic}' "
            f"to '{self.output_dir}' as '{self.filename_format}' "
            f"(start={self.index}, encoding={self.encoding})"
        )

    def cb(self, msg: Image):
        self.frame_count += 1

        # Only save every Nth frame
        if (self.frame_count % self.save_stride) != 0:
            return

        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding=self.encoding)
        except Exception as e:
            self.get_logger().error(f'CvBridge conversion failed: {e}')
            return

        fname = self.filename_format % self.index if '%' in self.filename_format else f'{self.index}.jpg'
        path = os.path.join(self.output_dir, fname)

        try:
            ok = cv2.imwrite(path, cv_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if ok:
                self.get_logger().info(f'Saved image #{self.index} at frame_count={self.frame_count}: {path}')
            else:
                self.get_logger().warn(f'cv2.imwrite returned False for {path}')
        except Exception as e:
            self.get_logger().error(f'Failed to write {path}: {e}')
            return

        # Increment filename index only when we actually save
        self.index += 1

def main():
    rclpy.init()
    node = ImageSaverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
