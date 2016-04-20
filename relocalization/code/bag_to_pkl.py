import rosbag
import numpy as np
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer

from nav_msgs.msg import OccupancyGrid, MapMetaData, Odometry
from sensor_msgs.msg import LaserScan

import support

map_bag = rosbag.Bag('../files/map3.bag')
readings_bag = rosbag.Bag('../files/relocalization_dataset4.bag')

map_array = []

for topic, msg, t in map_bag.read_messages(topics=['/map_metadata', '/map']):
    if topic == '/map_metadata':
        pass
    elif topic == '/map':
        r = msg.info.resolution
        w = msg.info.width
        h = msg.info.height
        ox = msg.info.origin.position.x
        oy = msg.info.origin.position.y

        map_i = np.asarray(msg.data)
        map_ = map_i.reshape(w, h)

        objs = np.where(map_ >= 65)
        objs_array = np.vstack((objs[1], objs[0])).T
        objs_fixed = objs_array * r
        objs_final = objs_fixed + np.array([ox, oy])

        map_array = objs_final
    else:
        print "Something went wrong"
        break


poses = {}
stamps = []
for topic, msg, t in readings_bag.read_messages(topics=['/amcl_pose']):
    t = 0
    r = 0

    readings = 0

    if topic == '/amcl_pose':
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        t = np.array([x, y])

        pre_t = t

        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w

        r = np.arctan2(2 * (qx * qy + qz * qw),
                       1 - 2 * (qy ** 2 + qz ** 2))

        stamp = msg.header.stamp
        stamps.append(stamp)

        poses[stamp] = (t, r)
    else:
        pass

scans = {}
for topic, msg, t in readings_bag.read_messages(topics=['/scan']):
    if topic == '/scan':
        a_min = msg.angle_min
        a_max = msg.angle_max
        a_inc = msg.angle_increment

        laser = np.asarray(msg.ranges)
        laser_i = np.arange(0, len(laser))
        corrs = laser_i * a_inc
        corrs_ = corrs + a_min
        x = laser * np.cos(corrs_)
        y = laser * np.sin(corrs_)

        readings = np.vstack((x, y)).T
        stamp = msg.header.stamp

        if stamp in stamps:
            scans[stamp] = readings

            x1 = laser * np.cos(corrs_ + poses[stamp][1]) + poses[stamp][0][0]
            y1 = laser * np.sin(corrs_ + poses[stamp][1]) + poses[stamp][0][1]
            readings1 = np.vstack((x1, y1)).T

            # support.plot_double(map_array, readings, block=True)
            # support.plot_double(map_array, readings1, block=True)
    else:
        pass

ds = SupervisedDataSet(len(readings.flatten()), 3)

# Do we even need the map?...
first_stamp = 0
i = 0
for stamp, reading in scans.iteritems():
    t = poses[stamp][0]
    r = poses[stamp][1]
    pose = np.asarray([t[0], t[1], r])

    if i == 0:
        first_stamp = stamp

    i += 1

    ds.addSample(reading.flatten(), pose)

print "Trained the net with", i, "samples."

net = buildNetwork(len(readings.flatten()), 3, 3, bias=True, hiddenclass=TanhLayer)
trainer = BackpropTrainer(net, ds)

trainer.train()

print "Test:", net.activate(scans[first_stamp].flatten())
print "The result should be:", poses[first_stamp]
