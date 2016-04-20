#! /usr/bin/env python

import copy
import time
import pdb
import icp
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


def plot_super(map_obstacles, laser_obstacles, title):
    f = plt.figure()
    sp = f.add_subplot(111)
    sp.plot(map_obstacles[:, 0], map_obstacles[:, 1], '.')
    sp.plot(laser_obstacles[:, 0], laser_obstacles[:, 1], '.r')
    sp.set_title(title)
    plt.draw()


def kabsch(p, q):
    p, q = expand(p, q)
    cov = np.dot(p.T, q)
    U, s, V = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(np.multiply(V.T, U.T)))
    t = np.dot(V.T, np.matrix([[1, 0], [0, d]]))
    r = np.dot(t, U.T)

    return r


def expand(p, q):
    diff = np.max([p.shape[0], q.shape[0]]) - \
           np.min([p.shape[0], q.shape[0]])

    if p.shape[0] >= q.shape[0]:
        for i in xrange(0, diff):
            q = np.vstack([q, [0, 0]])
    else:
        for i in xrange(0, diff):
            p = np.vstack([p, [0, 0]])

    return p, q


def rotate(p, theta):
    rot = np.matrix([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])
    return np.dot(p, rot.T)


def translate(p, t):
    return np.add(p, t.T)


def bounding_box(v):
    min_x, min_y = np.min(v, axis=0)
    max_x, max_y = np.max(v, axis=0)

    return np.array([(min_x, min_y), (max_x, min_y),
                     (max_x, max_y), (min_x, max_y)])


def do_kdtree(source, query):
    mytree = sp.spatial.cKDTree(source)
    dist, indexes = mytree.query(query)
    return indexes


def find_initial_poses(map_obstacles, num_points, num_closest_points):
    map_bounding = bounding_box(map_obstacles)
    x_it = map_bounding[1][0] / num_points
    y_it = map_bounding[2][1] / num_points
    x = 0
    y = 0

    maptree = sp.spatial.cKDTree(map_obstacles)

    result = {}

    print "#Points:", num_points
    print "#Closest Points:", num_closest_points

    for x_i in xrange(num_points):
        x += x_it
        for y_i in xrange(num_points):
            y += y_it

            print "Analyzing (x, y):", (x, y)

            dist, idx = maptree.query([x, y], k=5)
            result[(x, y)] = map_obstacles[idx]

    print "Result:", result
    return result


def relocalize(map_obstacles, laser_obstacles):
    map_bounding = bounding_box(map_obstacles)
    starting = True

    find_initial_poses(map_obstacles, 3, 5)

    while True:
        # Randomize translation
        x = np.random.uniform(map_bounding[0][0],
                              map_bounding[1][0])

        y = np.random.uniform(map_bounding[0][1],
                              map_bounding[2][1])

        if starting:
            t = np.array([0, 0])
            laser_trans = laser_obstacles
        else:
            # Apply translation
            t = np.array([x, y])
            laser_trans = translate(laser_obstacles, t)

        # Find optimal rotation
        r = kabsch(map_obstacles, laser_trans)
        theta = np.arccos(r.item(0, 1))

        # Apply rotation
        laser_trans_rot = rotate(laser_trans, theta)

        nn_idx = do_kdtree(map_obstacles, laser_trans_rot)
        nn = map_obstacles[nn_idx]

        mse = (np.square(laser_trans_rot - nn)).mean(axis=None)
        print "MSE:", mse

        if mse <= 0.05:
            # m = icp.icp(map_obstacles, laser_trans_rot)
            # print "ICP Result:", m
            break

        starting = False

    # pdb.set_trace()

    return t, r


def main():
    map_obstacles = np.loadtxt('obstacles_map.txt')
    laser_obstacles = np.loadtxt('obstacles_laser.txt')

    true_rotation = np.pi / 10
    true_translation = np.array([5, -5])

    laser_rot = rotate(laser_obstacles, true_rotation)
    laser_trans = translate(laser_rot, true_translation)

    t, r = relocalize(map_obstacles, laser_trans)
    theta = np.arccos(r.item(0, 1))

    print "True Rotation:", true_rotation
    print "True Translation:", true_translation
    print "-------------------------------------"
    print "Estimated Rotation:", theta
    print "Estimated Translation:", t
    print "-------------------------------------"
    print "Rotation Error:", np.abs(true_rotation - theta)
    print "Translation Error:", true_translation - t

    laser_reloc = rotate(laser_rot, -np.arccos(r.item(0, 1)))

    plot_super(map_obstacles, laser_obstacles, "Original")
    plot_super(map_obstacles, laser_trans, "Measure misaligned with map")
    plot_super(map_obstacles, laser_reloc, "Measure realigned with map")
    plt.show()


if __name__ == '__main__':
    main()
