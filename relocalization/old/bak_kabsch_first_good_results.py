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


def center(v):
    mean = np.mean(v, axis=0)
    v -= mean

    return v


def bounding_box(v):
    v = np.asarray(v)
    min_x, min_y = np.min(v, axis=0)
    max_x, max_y = np.max(v, axis=0)

    return np.array([(min_x, min_y), (max_x, min_y),
                     (max_x, max_y), (min_x, max_y)])


def do_kdtree(source, query):
    mytree = sp.spatial.cKDTree(source)
    dist, indexes = mytree.query(query)
    return indexes


# Returns the num_closest_points closest points to the subdivision of
# map_obstacles by num_points * num_points
def partition_space(map_obstacles, map_tree, num_points, num_closest_points):
    map_bounding = bounding_box(map_obstacles)
    x_min = map_bounding[0][0]
    x_max = map_bounding[1][0]
    y_min = map_bounding[0][1]
    y_max = map_bounding[2][1]
    x_it = (x_max - x_min) / num_points
    y_it = (y_max - y_min) / num_points
    x = x_min
    y = y_min

    result = {}

    while round(x, 5) <= round(x_max, 5):
        y = y_min
        while round(y, 5) <= round(y_max, 5):
            dist, idx = map_tree.query([x, y], k=num_closest_points)
            result[(x, y)] = map_obstacles[idx]

            y += y_it

        x += x_it

    return result


def compute_error(laser, map_obstacles, map_tree, t, num_closest_points):
    laser_tree = sp.spatial.cKDTree(laser)
    laser_dist, laser_idx = laser_tree.query(t, k=num_closest_points)
    laser_pts = laser[laser_idx]

    map_dist, map_idx = map_tree.query(laser_pts, k=1)

    # print map_dist

    # map_dist, map_idx = map_tree.query(t, k=num_closest_points)
    # map_pts = map_obstacles[map_idx]

    # min_dists = []
    # for l in laser_pts:
    #     dists = []
    #     for m in map_pts:
    #         dist = np.sqrt(np.square(l.item(0) - m.item(0)) + np.square(l.item(1) - m.item(1)))
    #         dists.append(dist)

    #     min_dist_idx = np.argmin(dists)
    #     min_dist = dists[min_dist_idx]
    #     min_dists.append(min_dist)

    return np.sum(map_dist) / len(map_dist)
    # return np.sum(min_dists) / len(min_dists)


def relocalize(map_obstacles, laser_obstacles, num_closest_points):
    map_tree = sp.spatial.cKDTree(map_obstacles)
    # laser_bounding = bounding_box(laser_obstacles)
    # laser_bounding_mean = np.mean(laser_bounding)
    # laser_centered = laser_obstacles - laser_bounding_mean
    # laser_centered = center(laser_bounding)
    laser_centered = center(laser_obstacles)
    laser_tree = sp.spatial.cKDTree(laser_centered)
    dist, idx = laser_tree.query([0, 0], k=num_closest_points)
    laser_nn = laser_centered[idx]

    # plot_super(map_obstacles, laser_centered, "Laser Centered")
    # plt.show()

    num_points = 10
    mse_thresh = 0.1

    while True:
        print "Running with #Points:", num_points
        map_nn = partition_space(map_obstacles, map_tree, num_points, num_closest_points)

        for pt, nn in map_nn.items():
            t = np.asarray(pt)
            laser_trans = translate(laser_centered, t)
            r = kabsch(map_obstacles, laser_trans)
            theta = np.arccos(r.item(0, 1))
            laser_trans_rot = rotate(laser_trans, theta)

            mse = compute_error(laser_trans_rot, map_obstacles, map_tree, t, num_closest_points)

            if mse <= mse_thresh:
                break

        if mse <= mse_thresh:
            print "\nMSE:", mse, "\n"
            break

        num_points += 1

    return t, r


def main():
    map_obstacles = np.loadtxt('obstacles_map.txt')
    laser_obstacles = np.loadtxt('obstacles_laser.txt')

    true_rotation = np.pi / 10
    true_translation = np.array([15, -5])

    laser_rot = rotate(laser_obstacles, true_rotation)
    laser_trans = translate(laser_rot, true_translation)

    t, r = relocalize(map_obstacles, laser_trans, 30)
    theta = np.arccos(r.item(0, 1))

    print "True Rotation:", true_rotation
    print "True Translation:", true_translation
    print "-------------------------------------"
    print "Estimated Rotation:", theta
    print "Estimated Translation:", t
    print "-------------------------------------"
    print "Rotation Error:", np.abs(true_rotation - theta)
    print "Translation Error:", np.abs(true_translation - t)

    laser_reloc = rotate(laser_rot, -np.arccos(r.item(0, 1)))

    plot_super(map_obstacles, laser_obstacles, "Original")
    plot_super(map_obstacles, laser_trans, "Measure misaligned with map")
    plot_super(map_obstacles, laser_reloc, "Measure realigned with map")
    plt.show()


if __name__ == '__main__':
    main()
