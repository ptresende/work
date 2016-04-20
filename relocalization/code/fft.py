#! /usr/bin/env python

import support

import copy
import time
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt


def main():
    noise = True

    if not noise:
        tri_orig = np.loadtxt('../files/obstacles_map.txt')
        tri_alt = np.loadtxt('../files/obstacles_laser.txt')
    else:
        m = np.array([0, 0])
        c = np.array([[1, 0], [0, 1]])
        tri_orig = np.loadtxt('../files/obstacles_map.txt')
        tri_alt = np.loadtxt('../files/obstacles_laser.txt')
        tri_alt = tri_alt + \
                  np.random.multivariate_normal(m, c, len(tri_alt)) * 0.1

    true_rotation = np.pi / 4
    true_translation = np.array([0, 0])

    tri_alt = support.rotate(tri_alt, true_rotation)
    tri_alt = support.translate(tri_alt, true_translation)

    tri_orig_centered = support.center(tri_orig)
    tri_alt_centered = support.center(tri_alt)
    support.plot_double(tri_orig_centered, tri_alt_centered, "Original")

    start = time.time()

    map_tree = spatial.cKDTree(tri_orig)

    num_points = 5
    results = []

    while True:
        map_partition = support.partition_space(tri_orig, map_tree, num_points)
        
        for pt in map_partition:
            tri_alt_trans = support.translate(tri_alt_centered, pt)
            orig_dists = support.measure_distance(tri_orig_centered)
            alt_dists = support.measure_distance(tri_alt_trans)
    
            # FFT
            orig_fft, orig_angle = support.compute_fft(orig_dists)
            alt_fft, alt_angle = support.compute_fft(alt_dists)
            ps = orig_angle - alt_angle
            results.append(ps[0])

        break

    print "Estimated PS:", min(results)
    print "Real PS:", true_rotation

    end = time.time()

    print "Time:", str(end - start)

    plt.show()


if __name__ == '__main__':
    main()
