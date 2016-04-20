import cv2
import numpy as np

map_filename = '/home/quadbase/Work/bot_simulator/maps/ISR_navigation_map.pgm'

################## MAP ######################
img_map = cv2.imread(map_filename)
gray = cv2.cvtColor(img_map, cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv2.cornerHarris(gray, 2, 3, 0.04)

# result is dilated for marking the corners, not important
dst = cv2.dilate(dst, None)

# Threshold for an optimal value, it may vary depending on the image.
img_map[dst > 0.01 * dst.max()] = [0, 0, 255]







laser_filename = '/home/quadbase/Work/relocalization/laser_gray.png'

################## LASER ######################
img_laser = cv2.imread(laser_filename)
gray = cv2.cvtColor(img_laser, cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv2.cornerHarris(gray, 2, 3, 0.04)

# result is dilated for marking the corners, not important
dst = cv2.dilate(dst, None)

# Threshold for an optimal value, it may vary depending on the image.
img_laser[dst > 0.01 * dst.max()] = [0, 0, 255]


cv2.imshow('dst', img_map)
cv2.imshow('dst2', img_laser)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
