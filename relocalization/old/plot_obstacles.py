#! /usr/bin/env python

import copy
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


def relocalize(map_obstacles, map_bounding, laser_centered, laser_bounding):
    map_min = map_bounding[0]
    laser_min = laser_bounding[0]
    diff = laser_min - map_min

    print "Diff=", diff

    while True:
        laser_temp = translate(laser_centered, diff)

        map_bounding = bounding_box(map_obstacles)
        map_segment = get_segment(map_obstacles, map_bounding)

        plot_super(map_obstacles, map_segment, "Crop")

        r = kabsch(map_obstacles, laser_temp)
        theta = np.arccos(r.item(0, 1))

        laser_rot = rotate(laser_temp, theta)

        laser_rot, map_segment = expand(laser_rot, map_segment)
        if np.mean(np.abs(laser_rot - map_segment)) < 0.5:
            break

        break

    return diff, r


def get_segment(v, bounding):
    v_copy = copy.deepcopy(v)

    idx = 0
    for i in v_copy:
        if i[0] < bounding[0][0] or \
           i[0] > bounding[1][0]:
            del v_copy[idx]
            continue
        if i[1] < bounding[0][1] or \
           i[1] > bounding[2][1]:
            del v_copy[idx]
            continue

    return v_copy


def bounding_box(v):
    min_x, min_y = np.min(v, axis=0)
    max_x, max_y = np.max(v, axis=0)

    return np.array([(min_x, min_y), (max_x, min_y),
                     (max_x, max_y), (min_x, max_y)])


def center(v):
    mean = np.mean(v, axis=0)
    v -= mean

    return v


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


def main():
    map_obstacles = np.loadtxt('obstacles_map.txt')
    laser_obstacles = np.loadtxt('obstacles_laser.txt')

    plot_super(map_obstacles, laser_obstacles, "Original")

    true_rotation = 0  # np.pi / 10
    true_translation = np.array([0, 0])

    laser_rot = rotate(laser_obstacles, true_rotation)
    laser_trans = translate(laser_rot, true_translation)

    plot_super(map_obstacles, laser_trans,
               "Laser Readings Translated and Rotated")

    laser_centered = center(laser_obstacles)
    laser_bounding = bounding_box(laser_centered)
    # plot_super(laser_obstacles, laser_bounding, "Laser Bounding Box")

    map_centered = center(map_obstacles)
    map_bounding = bounding_box(map_centered)
    # plot_super(map_obstacles, map_bounding, "Map Bounding Box")

    plt.show(block=False)

    print "Starting Relocalization..."

    # trans, rot = relocalize(map_obstacles, map_bounding,
    #                         laser_centered, laser_bounding)

    r = kabsch(laser_obstacles, laser_obstacles)
    rot = np.arccos(r.item(0, 1))

    print "True Rotation:", true_rotation
    # print "Estimated Translation:", trans
    print "Estimated Rotation:", rot

    laser_reloc = rotate(laser_rot, rot)
    plot_super(map_obstacles, laser_reloc, "Measure realigned with map")
    plt.show()


if __name__ == '__main__':
    main()
