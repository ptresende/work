#! /usr/bin/env python
import roslib
roslib.load_manifest('relocalization')

import rospy
import numpy as np
import cv2
import copy
import matplotlib.pyplot as plt

from cv_bridge import CvBridge

from sensor_msgs.msg import Image


def print_image(img):
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def print_image_forever(img):
    cv2.imshow('Images', img)
    cv2.waitKey(0)


def print_kp(img1, kp1, img2, kp2):
    for i in kp1:
        cv2.circle(img1,
                   (int(np.round(i.pt[0])), int(np.round(i.pt[1]))),
                   5, (0, 0, 255))

    for i in kp2:
        cv2.circle(img2,
                   (int(np.round(i.pt[0])), int(np.round(i.pt[1]))),
                   5, (0, 0, 255))

    print_image_forever(img1)
    print_image_forever(img2)


def draw_circles_at(img, pts):
    for pt in pts:
        cv2.circle(img,
                   (int(np.round(pt[0])), int(np.round(pt[1]))),
                   5, (0, 0, 255))

    print_image_forever(img)


class Relocalization:
    def __init__(self):
        self.map_img_ = 0
        self.laser_img_ = 0

        # self.detector_ = cv2.ORB()
        self.detector_ = cv2.BRISK()
        # self.detector_ = cv2.Canny()

        self.bridge_ = CvBridge()

        self.scan_sub_ = rospy.Subscriber('scan_image', Image,
                                          self.laser_cb)

        self.read_map()

    def read_map(self):
        filename = rospy.get_param("map_filename")
        self.map_img_ = cv2.imread(filename)

    def laser_cb(self, data):
        self.laser_img_ = self.bridge_.imgmsg_to_cv2(data, "passthrough")

        print_image(self.laser_img_)

        if self.laser_img_ is not None:
            self.relocalize()
            # self.relocalize_with_harris()

    def relocalize_with_harris(self):
        map_gray = cv2.cvtColor(self.map_img_, cv2.COLOR_BGR2GRAY)
        laser_gray = self.laser_img_

        map_corners = cv2.cornerHarris(map_gray, 2, 3, 0.04)
        map_corners = cv2.dilate(map_corners, None)
        map_corners = map_corners > 0.01 * map_corners.max()

        laser_corners = cv2.cornerHarris(laser_gray, 2, 3, 0.04)
        laser_corners = cv2.dilate(laser_corners, None)
        laser_corners = laser_corners > 0.01 * laser_corners.max()

        map_to_match = []
        laser_to_match = []
        matched_corners = []

        for x in xrange(0, map_gray.shape[0]):
            for y in xrange(0, map_gray.shape[1]):
                print "(x, y) =", (x, y)
                if map_corners[x + y] is True and laser_corners[x + y] is True:
                    print "asdasdas"

        # draw_circles_at(self.map_img_, matched_corners)

    def relocalize(self):
        map_gray = cv2.cvtColor(self.map_img_, cv2.COLOR_BGR2GRAY)
        laser_gray = self.laser_img_

        cv2.imwrite("laser_gray.png", laser_gray)
        print_image(map_gray)

        kp_map, des_map = self.detector_.detectAndCompute(map_gray, None)
        kp_laser, des_laser = self.detector_.detectAndCompute(laser_gray, None)

        print_kp(self.map_img_, kp_map, laser_gray, kp_laser)

        print "#Map Kps:", len(kp_map)
        print "#Laser Kps:", len(kp_laser), "\n"

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        matches = bf.match(des_map, des_laser)
        matches = sorted(matches, key=lambda x: x.distance)

        print "#Matches:", len(matches), "\n"

        i = 0
        for m in matches:
            kp_l = kp_laser[m.trainIdx]
            kp_m = kp_map[m.queryIdx]

            pt_l = kp_l.pt
            ang_l = kp_l.angle

            pt_m = kp_m.pt
            ang_m = kp_m.angle

            x_diff = np.abs(pt_m[0] - pt_l[0])
            y_diff = np.abs(pt_m[1] - pt_l[1])
            ang_diff = np.abs(ang_l - ang_m)

            cv2.circle(self.map_img_,
                       (int(np.round(x_diff)), int(np.round(y_diff))),
                       5, (0, 0, 255))

            print "Match #" + str(i)
            print "\tDist: " + str(m.distance)
            print "\td_x = " + str(x_diff)
            print "\td_y = " + str(y_diff)
            print "\td_a = " + str(ang_diff)

            i += 1

        print_image(self.map_img_)


def main():
    rospy.init_node('relocalization')
    Relocalization()
    rospy.spin()


if __name__ == '__main__':
    main()
