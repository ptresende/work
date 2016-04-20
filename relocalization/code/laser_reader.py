#! /usr/bin/env python
import roslib
roslib.load_manifest('relocalization')

import rospy
import cv2
import sys
import numpy as np

from cv_bridge import CvBridge

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import MapMetaData
from sensor_msgs.msg import Image


class LaserReader:
    def __init__(self):
        self.width_ = 0
        self.height_ = 0
        self.resolution_ = 0
        self.img_array_ = None
        self.img_ = None

        self.bridge_ = CvBridge()

        self.scan_sub_ = rospy.Subscriber('scan', LaserScan,
                                          self.laser_cb)
        self.map_sub_ = rospy.Subscriber('map_metadata', MapMetaData,
                                         self.map_cb)
        self.img_pub_ = rospy.Publisher('scan_image', Image,
                                        queue_size=10)

    def map_cb(self, data):
        self.width_ = data.width
        self.height_ = data.height
        self.resolution_ = data.resolution

        self.img_array_ = np.empty((self.width_, self.height_), np.uint8)
        self.img_array_.fill(255)

    def laser_cb(self, data):
        if self.img_array_ is None:
            return

        self.img_array_ = np.empty((self.width_, self.height_), np.uint8)
        self.img_array_.fill(255)

        angle_min = data.angle_min
        angle_inc = data.angle_increment

        curr_angle = angle_min

        for i in data.ranges:
            ran_in_m = i
            ran_in_px = ran_in_m / self.resolution_

            self.write_pixel_to_image(ran_in_px, curr_angle)

            curr_angle += angle_inc

        # self.view_image()
        np.savetxt('jjj.txt', self.img_array_)
        self.img_pub_.publish(self.bridge_.cv2_to_imgmsg(self.img_array_))
                              # encoding="mono8"))

    def write_pixel_to_image(self, ran, ang):
        c_x = self.width_ / 2
        c_y = self.height_ / 2

        d_x = ran * np.cos(ang)
        d_y = ran * np.sin(ang)

        p_x = np.round(c_x - d_x)
        p_y = np.round(c_y - d_y)

        self.img_array_[p_x, p_y] = 0

    def view_image(self):
        cv2.imshow('Image', self.img_array_)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        sys.exit(0)


def main():
    rospy.init_node('laser_reader')
    LaserReader()
    rospy.spin()


if __name__ == '__main__':
    main()
