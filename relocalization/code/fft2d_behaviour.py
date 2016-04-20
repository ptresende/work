#! /usr/bin/env python

import support

import copy
import time
import numpy as np
import scipy as sp
import scipy.signal as signal
import scipy.misc as misc
import matplotlib.pyplot as plt


def main():
    tri = True
    noise = False
    plots = False

    if not tri:
        if not noise:
            tri_orig = np.loadtxt('../files/obstacles_laser.txt')
            tri_cpy = copy.deepcopy(tri_orig)
        else:
            m = np.array([0, 0])
            c = np.array([[1, 0], [0, 1]])
            tri_orig = np.loadtxt('../files/obstacles_laser.txt')
            tri_cpy = copy.deepcopy(tri_orig) + \
                      np.random.multivariate_normal(m, c, len(tri_orig)) * 1.5
    else:
        if not noise:
            tri_orig = support.create_triangle()
            tri_cpy = copy.deepcopy(tri_orig)
        else:
            m = np.array([0, 0])
            c = np.array([[1, 0], [0, 1]])
            tri_orig = support.create_triangle()
            tri_cpy = copy.deepcopy(tri_orig) + \
                      np.random.multivariate_normal(m, c, len(tri_orig)) * 0.05

    true_rotation = 0
    true_translation = np.array([0, 0])

    tri_rot = copy.deepcopy(support.rotate(tri_cpy, true_rotation))
    tri_alt = copy.deepcopy(support.translate(tri_rot, true_translation))

    if plots:
        support.plot_double(tri_orig, tri_alt, "Original")

    # tri_orig_fun = support.function_from_pc(tri_orig)
    # tri_alt_fun = support.function_from_pc(tri_alt)

    x, y, tri_orig_fun, tri_alt_fun = support.get_2_2d_functions(tri_orig, tri_alt)
    # tri_orig_fun, tri_alt_fun = support.zero_pad(tri_orig_fun, tri_alt_fun)

    if plots:
        # support.plot_contour(x, y, tri_orig_fun)
        # support.plot_contour(x, y, tri_alt_fun)
        support.plot_contour_single(tri_orig_fun)
        support.plot_contour_single(tri_alt_fun)

    start = time.time()

    # Function FFT
    orig_fft = support.compute_fft2d(tri_orig_fun, plot=plots,
                                     title="Original")
    alt_fft = support.compute_fft2d(tri_alt_fun, plot=plots,
                                    title="Alternative")
    orig_f, orig_log_base = support.logpolar(tri_orig_fun)
    alt_f, alt_log_base = support.logpolar(tri_alt_fun)
    orig_lp_fft = support.compute_fft2d(orig_f)
    alt_lp_fft = support.compute_fft2d(alt_f)

    print "Orig FFT, LP:", orig_fft.shape, orig_f.shape, orig_log_base
    print "Alt FFT, LP:", alt_fft.shape, alt_f.shape, alt_log_base

    # Pointcloud FFT
    pc_orig_fft = support.compute_fft2d(tri_orig)
    pc_alt_fft = support.compute_fft2d(tri_alt)

    # Function-to-Pointcloud ratios
    rx = 1 / 100.0  # np.max(tri_orig[:, 0]) / tri_orig_fun.shape[0]
    ry = 1 / 100.0  # np.max(tri_orig[:, 1]) / tri_orig_fun.shape[1]

    # Cross-correlation
    # corr = signal.correlate2d(tri_orig_fun, tri_alt_fun, mode='same')
    # print "Corr;", np.unravel_index(np.argmax(corr), corr.shape)

    # Translation estimation
    et = support.phase_correlation(orig_fft, alt_fft, rx, ry)
    er = 0
    # support.numeric_translation(orig_fft, alt_fft, tri_orig_fun)

    # Rotation estimation
    orig_abs = np.abs(orig_fft)
    alt_abs = np.abs(alt_fft)

    oxy = np.where(orig_abs >= 0)
    oxy = np.vstack((oxy[1], oxy[0])).T * rx

    axy = np.where(alt_abs >= 0)
    axy = np.vstack((axy[1], axy[0])).T * rx

    print "O:", oxy.shape
    print "A:", axy.shape

    r = support.kabsch(oxy, axy)
    print r
    er = np.arccos(r.item(0, 1))

    print "\nTrue Translation:", true_translation
    print "True Rotation:", true_rotation

    print "\nEstimated Translation:", et
    print "Estimated Rotation:", er

    print "\nTranslation Error:", np.abs(true_translation - et)
    print "Rotation Error:", np.abs(true_rotation - er)

    end = time.time()

    print "\n\nTime:", str(end - start)

    plt.show()


if __name__ == '__main__':
    main()
