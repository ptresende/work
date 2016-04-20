#! /usr/bin/env python

import numpy as np
import scipy.cluster.vq as sp
import scipy.stats.mstats as stats
import matplotlib.pyplot as plt


def plot_super(map_obstacles, laser_obstacles, title):
    f = plt.figure()
    sp = f.add_subplot(111)
    sp.plot(map_obstacles[:, 0], map_obstacles[:, 1], '.')
    sp.plot(laser_obstacles[:, 0], laser_obstacles[:, 1], '.r')
    sp.set_title(title)
    plt.draw()


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


def compute_distances(v):
    num_points = len(v)
    mat = np.zeros((num_points, num_points))

    i_idx = 0
    j_idx = 0

    for i in v:
        j_idx = 0

        for j in v:
            dist = np.linalg.norm(i - j)
            mat[i_idx, j_idx] = dist

            j_idx += 1

        i_idx += 1

    return mat


def fit_points(points, target):
    new_points = np.array((len(points), 1))
    new_target = np.array((len(target), 1))

    n = np.min(np.abs(new_points - new_target))
    n_idx = np.argmin(np.abs(new_points - new_target))

    print n, n_idx


def main():
    map_obstacles = np.loadtxt('obstacles_map.txt')
    laser_obstacles = np.loadtxt('obstacles_laser.txt')

    # Center the data
    map_obstacles = center(map_obstacles)
    laser_obstacles = center(laser_obstacles)

    # Translate and rotate the measurements
    true_rotation = np.pi / 10
    true_translation = np.array([10, 10])

    laser_trans = translate(laser_obstacles, true_translation)
    laser_rot = rotate(laser_trans, true_rotation)

    # Run k-means
    map_code, map_dist = sp.kmeans(map_obstacles, 10)
    code, dist = sp.vq(laser_obstacles, map_code)

    m = stats.mode(code)
    pos = map_code[m[0][0]]

    print "Code:", map_code
    print "Dist:", dist
    print "Most likely cluster:", m
    print "Which corresponds to:", pos

    # plot_super(map_obstacles, laser_obstacles, "Original")
    plot_super(map_obstacles, map_code, "Map's k-means centers")
    plot_super(map_obstacles, np.vstack([laser_obstacles, pos]), "The Cluster")
    # plot_super(laser_obstacles, laser_code, "Laser's k-means centers")
    # plot_super(map_obstacles, laser_trans, "Measure misaligned with map")
    # plot_super(map_obstacles, laser_reloc, "Measure realigned with map")
    plt.show()


if __name__ == '__main__':
    main()
