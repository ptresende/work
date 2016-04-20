#! /usr/bin/env python

import support

import copy
import time
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


def main():
    tri = True
    noise = True

    if not tri:
        if not noise:
            tri_orig = np.loadtxt('../files/obstacles_laser.txt')
            tri_alt = copy.deepcopy(tri_orig)
        else:
            m = np.array([0, 0])
            c = np.array([[1, 0], [0, 1]])
            tri_orig = np.loadtxt('../files/obstacles_laser.txt')
            tri_alt = copy.deepcopy(tri_orig) + \
                      np.random.multivariate_normal(m, c, len(tri_orig)) * 0.1
    else:
        if not noise:
            tri_orig = support.create_triangle()
            tri_alt = copy.deepcopy(tri_orig)
        else:
            m = np.array([0, 0])
            c = np.array([[1, 0], [0, 1]])
            tri_orig = support.create_triangle()
            tri_alt = copy.deepcopy(tri_orig) + \
                      np.random.multivariate_normal(m, c, len(tri_orig)) * 0.5

    true_rotation = np.pi
    true_translation = np.array([0, 0])

    tri_alt = support.rotate(tri_alt, true_rotation)
    tri_alt = support.translate(tri_alt, true_translation)

    tri_orig_centered = support.center(tri_orig)
    tri_alt_centered = support.center(tri_alt)

    support.plot_double(tri_orig_centered, tri_alt_centered, "Original")

    start = time.time()

    orig_dists = support.measure_distance(tri_orig_centered)
    alt_dists = support.measure_distance(tri_alt_centered)
    support.plot_double(orig_dists, alt_dists, "D(alpha)")

    # FFT
    orig_abs, orig_angle = support.compute_fft(orig_dists)
    alt_abs, alt_angle = support.compute_fft(alt_dists)
    # support.plot_double(orig_fft_cte, alt_fft_cte, "FT Constant Terms")

    ps = orig_angle - alt_angle
    abs_diff = orig_abs - alt_abs

    weight_n = [1 / n if n != 0 else 1 for n in xrange(len(ps))]
    weight_n_sq = [1 / np.square(n) if n != 0 else 1 for n in xrange(len(ps))]

    print "PS[0]:", ps[0], "\t\t\tABS[0]:", abs_diff[0]
    print "---------------------------------------------------------------"
    print "PS[1]:", ps[1], "\t\tABS[0]:", abs_diff[1]
    print "PS[2]:", ps[2], "\t\tABS[0]:", abs_diff[2]
    print "PS[3]:", ps[3], "\t\tABS[0]:", abs_diff[3]
    print "PS[4]:", ps[4], "\t\tABS[0]:", abs_diff[4]
    print "PS[5]:", ps[5], "\t\tABS[0]:", abs_diff[5]
    print "PS[6]:", ps[6], "\t\tABS[0]:", abs_diff[6]
    print "PS[7]:", ps[7], "\t\tABS[0]:", abs_diff[7]
    print "PS[8]:", ps[8], "\t\tABS[0]:", abs_diff[8]
    print "PS[9]:", ps[9], "\t\tABS[0]:", abs_diff[9]
    print "PS[10]:", ps[10], "\t\tABS[0]:", abs_diff[10]
    print "---------------------------------------------------------------"
    print "PS Sum:", np.sum(ps), "\t\tABS Sum:", np.sum(abs_diff)
    print "PS Sum W(1 / n):", np.sum(np.dot(ps, weight_n)), \
        "\t\tABS Sum W(1 / n):", np.sum(np.dot(abs_diff, weight_n))
    print "PS Sum W(1 / sq(n)):", np.sum(np.dot(ps, weight_n_sq)), \
        "\t\tABS Sum W(1 / sq(n)):", np.sum(np.dot(abs_diff, weight_n_sq))
    print "---------------------------------------------------------------"
    print "Real PS:", true_rotation

    end = time.time()

    tri_alt_corrected = support.rotate(tri_alt, -ps[1])
    support.plot_double(tri_orig, tri_alt_corrected, "Corrected Shape")

    print "\n\nTime:", str(end - start)

    plt.show()


if __name__ == '__main__':
    main()
