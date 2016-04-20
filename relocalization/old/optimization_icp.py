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
    # p, q = expand(p, q)
    # print "P:", p.shape
    # print "Q:", q.shape
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
            # qq = np.vstack([q, q[0][0]])  # [0, 0]])
    else:
        for i in xrange(0, diff):
            p = np.vstack([p, [0, 0]])
            # p = np.vstack([p, p[0][0]])  # [0, 0]])

    return p, q


def rotate(p, theta):
    rot = np.matrix([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])
    return np.dot(p, rot.T)


def translate(p, t):
    return p + t


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
def partition_space(map_obstacles, map_tree, num_points):
    map_bounding = bounding_box(map_obstacles)
    x_min = map_bounding[0][0]
    x_max = map_bounding[1][0]
    y_min = map_bounding[0][1]
    y_max = map_bounding[2][1]
    x_it = (x_max - x_min) / num_points
    y_it = (y_max - y_min) / num_points
    x = x_min
    y = y_min

    result = []

    while round(x, 5) <= round(x_max, 5):
        y = y_min
        while round(y, 5) <= round(y_max, 5):
            result.append((x, y))

            y += y_it

        x += x_it

    return result


def compute_error(laser, map_obstacles, map_tree, t, num_closest_points):
    laser_tree = sp.spatial.cKDTree(laser)
    laser_dist, laser_idx = laser_tree.query(t, k=num_closest_points)
    laser_pts = laser[laser_idx]
    # map_dist, map_idx = map_tree.query(laser_pts, k=1)
    map_dist, map_idx = map_tree.query(laser, k=1)

    return np.sum(map_dist) / len(map_dist)


def relocalize(M, L):
    # map_tree = sp.spatial.cKDTree(map_obstacles)

    t_i = np.array([0, 0])
    # r_i = np.matrix([[0, 0], [0, 0]])
    r_i = np.array([[0, 0], [0, 0]], ndmin=2)

    dist = lambda L, M, R, t, i, j: np.linalg.norm(np.add(np.subtract(M[i, :], np.multiply(L[j, :], R.reshape((2, 2)))), t))

    # R = r_i
    # t = t_i
    # i = 0
    # j = 0
    # # print R * L[j, :]
    # print dist(L, M, R, t, i, j)

    fun = lambda R, t: [np.min([dist(L, M, R, t, i, j) for j in xrange(0, len(L))]) for i in xrange(0, len(M))]

    res = sp.optimize.minimize(fun, r_i, t_i, options={'maxiter': 10, 'disp': True})

    print res

    t = np.array([0, 0])
    r = 0

    return t, r


def main():
    map_obstacles = np.loadtxt('obstacles_map.txt')
    laser_obstacles = np.loadtxt('obstacles_laser.txt')

    true_rotation = np.pi / 8
    true_translation = np.array([5, 5])

    laser_rot = rotate(laser_obstacles, true_rotation)
    laser_trans = translate(laser_rot, true_translation)

    start = time.time()
    t, r = relocalize(map_obstacles, laser_trans)
    end = time.time()

    theta = np.arccos(r.item(0, 1))

    print "True Rotation:", true_rotation
    print "True Translation:", true_translation
    print "-------------------------------------"
    print "Estimated Rotation:", theta
    print "Estimated Translation:", t
    print "-------------------------------------"
    print "Rotation Error:", np.abs(true_rotation - theta)
    print "Translation Error:", np.abs(true_translation - t)
    print "-------------------------------------"
    print "Time Taken:", str(end - start)

    laser_reloc = rotate(laser_rot, -np.arccos(r.item(0, 1)))

    plot_super(map_obstacles, laser_obstacles, "Original")
    plot_super(map_obstacles, laser_trans, "Measure misaligned with map")
    plot_super(map_obstacles, laser_reloc, "Measure realigned with map")
    plt.show()


if __name__ == '__main__':
    main()
