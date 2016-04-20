#! /usr/bin/env python

import numpy as np
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


def relocalize(map_obstacles, laser_obstacles):
    t = np.array([0, 0])
    r = kabsch(map_obstacles, laser_obstacles)

    return t, r


def main():
    map_obstacles = np.loadtxt('obstacles_map.txt')
    laser_obstacles = np.loadtxt('obstacles_laser.txt')

    true_rotation = np.pi / 10
    true_translation = np.array([0, 0])

    laser_rot = rotate(laser_obstacles, true_rotation)
    laser_trans = translate(laser_rot, true_translation)

    t, r = relocalize(map_obstacles, laser_rot)
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
