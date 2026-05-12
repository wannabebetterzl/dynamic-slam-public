#!/usr/bin/env python3
# coding=utf-8

import os
import threading

import rospy
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry

try:
    from tf2_msgs.msg import TFMessage
    TF2_AVAILABLE = True
except Exception:
    TF2_AVAILABLE = False
    TFMessage = None


def ensure_parent_dir(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent)


class TumPoseLogger:
    def __init__(self):
        self.topic = rospy.get_param("~topic", "/orb_slam3/camera_pose")
        self.message_type = rospy.get_param("~message_type", "pose_stamped").strip().lower()
        self.output_path = rospy.get_param("~output_path", os.path.join(os.getcwd(), "orb_slam3_estimated_pose.txt"))
        self.append = bool(rospy.get_param("~append", False))
        self.tf_child_frame_id = rospy.get_param("~tf_child_frame_id", "")
        self.tf_frame_id = rospy.get_param("~tf_frame_id", "")
        self.lock = threading.Lock()
        self.fp = None
        self.message_count = 0

        ensure_parent_dir(self.output_path)
        mode = "a" if self.append else "w"
        self.fp = open(self.output_path, mode, encoding="utf-8")
        rospy.on_shutdown(self.close)

        self.subscriber = self._create_subscriber()
        rospy.loginfo(
            "TUM位姿记录器已启动: topic=%s, message_type=%s, output=%s",
            self.topic,
            self.message_type,
            self.output_path,
        )

    def _create_subscriber(self):
        if self.message_type == "pose_stamped":
            return rospy.Subscriber(self.topic, PoseStamped, self.pose_stamped_callback, queue_size=50)
        if self.message_type == "odometry":
            return rospy.Subscriber(self.topic, Odometry, self.odometry_callback, queue_size=50)
        if self.message_type == "transform_stamped":
            return rospy.Subscriber(self.topic, TransformStamped, self.transform_stamped_callback, queue_size=50)
        if self.message_type == "tf_message":
            if not TF2_AVAILABLE:
                raise RuntimeError("当前环境缺少 tf2_msgs，无法订阅 TFMessage。")
            return rospy.Subscriber(self.topic, TFMessage, self.tf_message_callback, queue_size=50)
        raise RuntimeError(
            "不支持的 message_type: {}。可选值: pose_stamped, odometry, transform_stamped, tf_message".format(
                self.message_type
            )
        )

    def _stamp_to_sec(self, header):
        if header is not None and header.stamp is not None:
            return header.stamp.to_sec()
        return rospy.Time.now().to_sec()

    def _write_line(self, timestamp, px, py, pz, qx, qy, qz, qw):
        line = "{:.6f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f}\n".format(
            timestamp, px, py, pz, qx, qy, qz, qw
        )
        with self.lock:
            self.fp.write(line)
            self.fp.flush()
            self.message_count += 1

    def pose_stamped_callback(self, msg):
        pose = msg.pose
        self._write_line(
            self._stamp_to_sec(msg.header),
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )

    def odometry_callback(self, msg):
        pose = msg.pose.pose
        self._write_line(
            self._stamp_to_sec(msg.header),
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )

    def transform_stamped_callback(self, msg):
        if self.tf_child_frame_id and msg.child_frame_id != self.tf_child_frame_id:
            return
        if self.tf_frame_id and msg.header.frame_id != self.tf_frame_id:
            return

        transform = msg.transform
        self._write_line(
            self._stamp_to_sec(msg.header),
            transform.translation.x,
            transform.translation.y,
            transform.translation.z,
            transform.rotation.x,
            transform.rotation.y,
            transform.rotation.z,
            transform.rotation.w,
        )

    def tf_message_callback(self, msg):
        for transform_msg in msg.transforms:
            if self.tf_child_frame_id and transform_msg.child_frame_id != self.tf_child_frame_id:
                continue
            if self.tf_frame_id and transform_msg.header.frame_id != self.tf_frame_id:
                continue

            transform = transform_msg.transform
            self._write_line(
                self._stamp_to_sec(transform_msg.header),
                transform.translation.x,
                transform.translation.y,
                transform.translation.z,
                transform.rotation.x,
                transform.rotation.y,
                transform.rotation.z,
                transform.rotation.w,
            )
            break

    def close(self):
        if self.fp is not None:
            rospy.loginfo("TUM位姿记录器关闭，共写入 %d 条轨迹。", self.message_count)
            self.fp.close()
            self.fp = None


def main():
    rospy.init_node("ros_pose_to_tum", anonymous=True)
    TumPoseLogger()
    rospy.spin()


if __name__ == "__main__":
    main()
