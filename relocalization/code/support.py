#! /usr/bin/env python

import copy
import time
import pdb
import sys
import numpy as np
import scipy as sp
from scipy import spatial as spatial
from scipy.ndimage.interpolation import rotate as sprotate
import matplotlib.pyplot as plt

try:
    import scipy.ndimage.interpolation as ndii
except ImportError:
    import ndimage.interpolation as ndii


def rotate(p, theta):
    rot = np.matrix([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])

    p = np.dot(p, rot.T)

    new = []
    for pt in p:
        new_pt = np.array([pt[0, 0], pt[0, 1]])
        new.append(new_pt)

    return np.asarray(new)


def translate(p, t):
    return p + t


def center(v):
    mean = np.mean(v, axis=0)
    v -= mean

    return v


def centroid(v):
    return np.mean(v, axis=0)


def kabsch(p, q):
    cov = np.dot(p.T, q)
    U, s, V = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(np.multiply(V.T, U.T)))
    t = np.dot(V.T, np.matrix([[1, 0], [0, d]]))
    r = np.dot(t, U.T)

    return r


def kabsch2(P, Q):
    """
    The optimal rotation matrix U is calculated and then used to rotate matrix
    P unto matrix Q so the minimum root-mean-square deviation (RMSD) can be
    calculated.
    Using the Kabsch algorithm with two sets of paired point P and Q,
    centered around the center-of-mass.
    Each vector set is represented as an NxD matrix, where D is the
    the dimension of the space.
    The algorithm works in three steps:
    - a translation of P and Q
    - the computation of a covariance matrix C
    - computation of the optimal rotation matrix U
    http://en.wikipedia.org/wiki/Kabsch_algorithm
    Parameters:
    P -- (N, number of points)x(D, dimension) matrix
    Q -- (N, number of points)x(D, dimension) matrix
    Returns:
    U -- Rotation matrix
    """

    # Computation of the covariance matrix
    C = np.dot(np.transpose(P), Q)

    # Computation of the optimal rotation matrix
    # This can be done using singular value decomposition (SVD)
    # Getting the sign of the det(V)*(W) to decide
    # whether we need to correct our rotation matrix to ensure a
    # right-handed coordinate system.
    # And finally calculating the optimal rotation matrix U
    # see http://en.wikipedia.org/wiki/Kabsch_algorithm
    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    # Create Rotation matrix U
    U = np.dot(V, W)

    return U


def bounding_box(v):
    v = np.asarray(v)
    min_x, min_y = np.min(v, axis=0)
    max_x, max_y = np.max(v, axis=0)

    return np.array([(min_x, min_y), (max_x, min_y),
                     (max_x, max_y), (min_x, max_y)])


def plot_xy_single(x, y, title="Untitled", block=False):
    f = plt.figure()
    subp = f.add_subplot(111)
    subp.plot(x, y, '.')
    subp.set_title(title)
    plt.show(block)


def plot_xy_double(x1, y1, x2, y2, title="Untitled", block=False):
    f = plt.figure()
    subp = f.add_subplot(111)
    subp.plot(x1, y1, '.')
    subp.plot(x2, y2, 'x')
    subp.set_title(title)
    plt.show(block)


def plot_single(v, title="Untitled", block=False):
    f = plt.figure()
    subp = f.add_subplot(111)

    try:
        v_nr, v_nc = v.shape
        subp.plot(v[:, 0], v[:, 1], '.')
    except:
        v_nr = v.size
        v_x = xrange(0, v_nr)
        subp.plot(v_x, v, '.')

    subp.set_title(title)
    plt.show(block)


def plot_double(v, w, title="Untitled", block=False):
    f = plt.figure()
    subp = f.add_subplot(111)

    try:
        v_nr, v_nc = v.shape
        subp.plot(v[:, 0], v[:, 1], '.')
    except:
        v_nr = v.size
        v_x = xrange(0, v_nr)
        subp.plot(v_x, v, '.')

    try:
        w_nr, w_nc = w.shape
        subp.plot(w[:, 0], w[:, 1], 'x')
    except:
        w_nr = w.size
        w_x = xrange(0, w_nr)
        subp.plot(w_x, w, 'x')

    subp.set_title(title)
    plt.show(block)


def plot_contour_single(x, title="Untitled", block=False):
    f = plt.figure()
    subp = f.add_subplot(111)
    plt.contour(x)
    subp.set_title(title)
    plt.show(block)


def plot_contour(x, y, z, title="Untitled", block=False):
    f = plt.figure()
    subp = f.add_subplot(111)
    plt.contourf(x, y, z)
    subp.set_title(title)
    plt.show(block)


def plot_fft(abs_, angle_, title="Untitled", block=False):
    plt.figure()
    plt.suptitle(title)

    plt.subplot(121)
    plt.contour(abs_)
    plt.title("Abs{FFT}")

    plt.subplot(122)
    plt.contour(angle_)
    plt.title("Angle{FFT}")

    plt.show(block)


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


def R2P(x):
    return np.array([(np.abs(i), np.angle(i)) for i in x]).reshape((len(x), 2))


def correct_phase(phase):
    while phase > 2 * np.pi:
        phase -= 2 * np.pi

    return phase / np.pi


def create_line(x_max, alpha, num_pts, start_pt=[0, 0]):
    if alpha == np.pi / 2:
        x = np.zeros((num_pts,))
        y = np.linspace(0, x_max, num_pts)
        return np.vstack((x, y)).T
        
    if alpha == 3 * np.pi / 2:
        x = np.zeros((num_pts,))
        y = np.linspace(0, -x_max, num_pts)
        return np.vstack((x, y)).T

    m = np.tan(alpha)

    alpha = correct_phase(alpha)

    if alpha >= 3 * np.pi / 2 or alpha < np.pi / 2:
        x = np.linspace(0, x_max, num_pts)
    elif alpha >= np.pi / 2 and alpha < 3 * np.pi / 2:
        x = np.linspace(-x_max, 0, num_pts)
    else:
        print "There was a problem creating the line."
        print "The given angle was", alpha
        sys.exit()

    b = start_pt[1] - (m * start_pt[0])

    y = m * x + b

    return np.vstack((x, y)).T


def create_triangle():
    x_0 = np.linspace(-2, 2, 500)
    y_0 = np.ones_like(x_0) * -2

    x_1 = np.linspace(-2, 0, 250)
    y_1 = 3 * x_1 + 4

    x_2 = np.linspace(0, 2, 250)
    y_2 = -3 * x_2 + 4

    x = np.hstack((x_0, x_1, x_2))
    y = np.hstack((y_0, y_1, y_2))

    return np.vstack((x, y)).T


def intersect(kdtree, l):
    dist, indexes = kdtree.query(l, k=1)

    min_idx = np.argmin(dist)
    idx = indexes[min_idx]

    return idx


def measure_distance(v):
    kdtree = spatial.cKDTree(v)

    alpha_range = np.arange(0, np.pi * 2, np.pi / 250)

    distances = []

    for alpha in alpha_range:
        l = create_line(5, alpha, v.shape[0])
        i = intersect(kdtree, l)
        distances.append(np.sqrt(np.square(v[i, 0]) + np.square(v[i, 1])))

    distances = np.asarray(distances)

    return distances


def get_2_2d_functions(orig, alt):
    r = 0.01

    o_x_min, o_x_max = np.min(orig[:, 0]), np.max(orig[:, 0])
    o_y_min, o_y_max = np.min(orig[:, 1]), np.max(orig[:, 1])

    a_x_min, a_x_max = np.min(alt[:, 0]), np.max(alt[:, 0])
    a_y_min, a_y_max = np.min(alt[:, 1]), np.max(alt[:, 1])

    x_min = np.min([o_x_min, a_x_min])
    x_max = np.max([o_x_max, a_x_max])
    y_min = np.min([o_y_min, a_y_min])
    y_max = np.max([o_y_max, a_y_max])

    x, y = np.meshgrid(np.arange(x_min, x_max, r),
                       np.arange(y_min, y_max, r))

    o_f = np.zeros((x.shape[0], x.shape[1]))
    a_f = np.zeros((x.shape[0], x.shape[1]))

    for pt in orig:
        px, py = pt

        ix = np.abs(x[0, :] - px).argmin()
        iy = np.abs(y[:, 0] - py).argmin()

        o_f[iy, ix] = 1

    for pt in alt:
        px, py = pt

        ix = np.abs(x[0, :] - px).argmin()
        iy = np.abs(y[:, 0] - py).argmin()

        a_f[iy, ix] = 1

    return x, y, o_f, a_f


def get_2d_function(pc):
    x = []
    y = []

    for pt in pc:
        try:
            x.append(pt[0])
            y.append(pt[1])
        except:
            print pt.shape
            e = sys.exc_info()[0]
            print e
            sys.exit()

    x = np.asarray(x)
    x_shifted = copy.deepcopy(x)
    x_shifted = np.roll(x_shifted, 1)
    x_diff = np.abs(x - x_shifted)
    x_diff = np.min(x_diff[x_diff > 0.0001])

    y = np.asarray(y)
    y_shifted = copy.deepcopy(y)
    y_shifted = np.roll(y_shifted, 1)
    y_diff = np.abs(y - y_shifted)
    y_diff = np.min(y_diff[y_diff > 0.0001])

    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    x_ = np.arange(x_min, x_max, x_diff)
    x_ = np.round(x_, 4)
    x_ *= 1000
    x_ = np.floor(x_)
    x_ /= 1000

    y_ = np.arange(y_min, y_max, y_diff)
    y_ = np.round(y_, 4)
    y_ *= 1000
    y_ = np.floor(y_)
    y_ /= 1000

    ff = np.zeros((len(y_) + np.round(np.abs(y_min)),
                   len(x_) + np.round(np.abs(x_min))))

    for i in xrange(0, len(pc)):
        x_t = np.round(x[i], 4)
        x_t *= 1000
        x_t = np.floor(x_t)
        x_t /= 1000

        y_t = np.round(y[i], 4)
        y_t *= 1000
        y_t = np.floor(y_t)
        y_t /= 1000

        try:
            x_f = np.abs(x_ - x_t).argmin()
            y_f = np.abs(y_ - y_t).argmin()

            ff[y_f + np.round(np.abs(y_min)),
               x_f + np.round(np.abs(x_min))] = 1
        except:
            e = sys.exc_info()[0]
            print e
            print "FF Shape:", ff.shape
            print "X:", y_f + np.round(y_min)
            print "Y:", x_f + np.round(x_min)
            sys.exit()

    return ff


def function_from_pc(pc):
    x = []
    y = []

    for pt in pc:
        try:
            x.append(pt[0])
            y.append(pt[1])
        except:
            print pt.shape
            e = sys.exc_info()[0]
            print e
            sys.exit()

    x = np.asarray(x)
    x_shifted = copy.deepcopy(x)
    x_shifted = np.roll(x_shifted, 1)
    x_diff = np.abs(x - x_shifted)
    x_diff = np.min(x_diff[x_diff > 0.0001])

    y = np.asarray(y)
    y_shifted = copy.deepcopy(y)
    y_shifted = np.roll(y_shifted, 1)
    y_diff = np.abs(y - y_shifted)
    y_diff = np.min(y_diff[y_diff > 0.0001])

    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    x_ = np.arange(x_min, x_max, x_diff)
    x_ = np.round(x_, 4)
    x_ *= 1000
    x_ = np.floor(x_)
    x_ /= 1000

    y_ = np.arange(y_min, y_max, y_diff)
    y_ = np.round(y_, 4)
    y_ *= 1000
    y_ = np.floor(y_)
    y_ /= 1000

    ff = np.zeros((len(y_), len(x_)))

    for i in xrange(0, len(pc)):
        x_t = np.round(x[i], 4)
        x_t *= 1000
        x_t = np.floor(x_t)
        x_t /= 1000

        y_t = np.round(y[i], 4)
        y_t *= 1000
        y_t = np.floor(y_t)
        y_t /= 1000

        try:
            x_f = np.abs(x_ - x_t).argmin()
            y_f = np.abs(y_ - y_t).argmin()

            ff[y_f, x_f] = 1
        except:
            e = sys.exc_info()[0]
            print e
            sys.exit()

    return ff


# The Fourier transform of a function of time itself is a complex-valued
# function of frequency, whose absolute value represents the amount of that
# frequency present in the original function, and whose complex argument is
# the phase offset of the basic sinusoid in that frequency.
def compute_fft(v):
    fft = sp.fft(v)

    angle_ = np.angle(fft)
    abs_ = np.abs(fft)

    return abs_, angle_


def compute_fft2d(v, plot=False, title=""):
    fft = np.fft.fft2(v)
    fft_abs = np.abs(fft)
    fft_angle = np.angle(fft)

    if plot:
        plot_fft(fft_abs, fft_angle, title)

    return fft


def zero_pad(orig, alt):
    x_orig, y_orig = orig.shape
    x_alt, y_alt = alt.shape

    print "Before padding:"
    print "\tOrig S:", orig.shape
    print "\tAlt S:", alt.shape

    if x_orig > x_alt:
        pad = np.zeros((1, y_alt))

        for i in xrange(0, x_orig - x_alt):
            alt = np.vstack((alt, pad))
    else:
        pad = np.zeros((1, y_orig))

        for i in xrange(0, x_alt - x_orig):
            orig = np.vstack((orig, pad))

    x_orig, y_orig = orig.shape
    x_alt, y_alt = alt.shape

    if y_orig > y_alt:
        pad = np.zeros((x_alt, 1))

        for i in xrange(0, y_orig - y_alt):
            alt = np.hstack((alt, pad))
    else:
        pad = np.zeros((x_orig, 1))

        for i in xrange(0, y_alt - y_orig):
            orig = np.hstack((orig, pad))

    print "After padding:"
    print "\tOrig S:", orig.shape
    print "\tAlt S:", alt.shape

    return orig, alt


def rotate_complex(a, angle, reshape=True):
    r = sprotate(a.real, angle, reshape=reshape, mode='wrap')
    i = sprotate(a.imag, angle, reshape=reshape, mode='wrap')

    return r + 1j * i


def numeric_translation(fft_orig, fft_alt, fun_alt):
    print "Orig FFT:", fft_orig.shape
    print "Alt FFT:", fft_alt.shape

    # Compute abs and angle for both ffts
    abs_orig = np.abs(fft_orig)
    angle_orig = np.angle(fft_orig)

    abs_alt = np.abs(fft_alt)
    angle_alt = np.angle(fft_alt)

    # Measure abs and angle diffs
    abs_diff = abs_orig - abs_alt
    angle_diff = angle_orig - angle_alt

    # Abs and Angle sums
    abs_sum_x = np.sum(abs_diff, axis=1)
    abs_sum_y = np.sum(abs_diff, axis=0)
    angle_sum_x = np.sum(angle_diff, axis=1)
    angle_sum_y = np.sum(angle_diff, axis=0)

    w_x = [1 / float(n) if n != 0 else 0 for n in xrange(len(abs_sum_x))]
    w_y = [1 / float(n) if n != 0 else 0 for n in xrange(len(abs_sum_y))]

    # f0 stuff
    abs_x_f0 = abs_diff[:, 1]
    abs_y_f0 = abs_diff[1, :]
    angle_x_f0 = angle_diff[:, 1]
    angle_y_f0 = angle_diff[1, :]

    # cte stuff
    abs_x_cte = abs_diff[:, 0]
    abs_y_cte = abs_diff[0, :]
    angle_x_cte = angle_diff[:, 0]
    angle_y_cte = angle_diff[0, :]

    print "\n\n"
    print "PS[0, 0], Abs[0, 0]:", angle_diff[0, 0], abs_diff[0, 0]
    print "PS[1, 1], Abs[1, 1]:", angle_diff[1, 1], abs_diff[1, 1]
    print "-------------------------------------------------------------------"
    print "Abs Sum (x, y):", np.sum(abs_sum_x), np.sum(abs_sum_y)
    print "Abs WSum (x, y):", np.dot(abs_sum_x, w_x), np.dot(abs_sum_y, w_y)
    print "-------------------------------------------------------------------"
    print "Angle Sum (x, y):", np.sum(angle_sum_x), np.sum(angle_sum_y)
    print "Angle WSum (x, y):", np.dot(angle_sum_x, w_x), np.dot(angle_sum_y, w_y)
    print "-------------------------------------------------------------------"
    print "f0:"
    print "\tAbs x Sum, WSum:", np.sum(abs_x_f0), np.dot(abs_x_f0, w_x)
    print "\tAbs y Sum, WSum:", np.sum(abs_y_f0), np.dot(abs_y_f0, w_y)
    print "\tAngle x Sum, WSum:", np.sum(angle_x_f0), np.dot(angle_x_f0, w_x)
    print "\tAngle y Sum, WSum:", np.sum(angle_y_f0), np.dot(angle_y_f0, w_y)
    print "\tAngle x Corr:", correct_phase(np.sum(angle_x_f0)), correct_phase(np.dot(angle_x_f0, w_x))
    print "\tAngle y Corr:", correct_phase(np.sum(angle_y_f0)), correct_phase(np.dot(angle_y_f0, w_y))
    print "-------------------------------------------------------------------"
    print "cte:"
    print "\tAbs x Sum, WSum:", np.sum(abs_x_cte), np.dot(abs_x_cte, w_x)
    print "\tAbs y Sum, WSum:", np.sum(abs_y_cte), np.dot(abs_y_cte, w_y)
    print "\tAngle x Sum, WSum:", np.sum(angle_x_cte), np.dot(angle_x_cte, w_x)
    print "\tAngle y Sum, WSum:", np.sum(angle_y_cte), np.dot(angle_y_cte, w_y)
    print "\tAngle x Corr:", correct_phase(np.sum(angle_x_cte)), correct_phase(np.dot(angle_x_cte, w_x))
    print "\tAngle y Corr:", correct_phase(np.sum(angle_y_cte)), correct_phase(np.dot(angle_y_cte, w_y))

    rfft = rotate_complex(fft_alt, -np.pi / 2, reshape=False)
    abs_rfft = np.abs(rfft)
    angle_rfft = np.angle(rfft)

    # plot_contour_single(relevant_diff)
    # plot_fft(abs_orig, angle_orig, title="Original")
    # plot_fft(abs_alt, angle_alt, title="Modified")
    # plot_fft(abs_rfft, angle_rfft, title="Modified Rotated Back")
    # plot_fft(abs_diff, angle_diff)

    # diff_11 = fft_orig[1, 1] - fft_alt[1, 1]
    # diff_11 = np.log(diff_11)
    # diff_11 /= 1 / (2j * np.pi)

    # diff = fft_orig - fft_alt
    # diff = np.log(diff)
    # diff /= 1 / (2j * np.pi)

    # diff_c = fun_alt / fft_orig
    # diff_c = np.log(diff)
    # diff_c /= 1 / (2j * np.pi)

    # print "-------------------------------------------------------------------"
    # print "(Tx, Ty)[1, 1]:", diff_11
    # print "-------------------------------------------------------------------"
    # print "(Tx, Ty):", diff
    # print "-------------------------------------------------------------------"
    # print "(Tx, Ty)c:", diff_c

    orig_measure = np.array([abs_orig[1, :], abs_orig[:, 1]]).T
    alt_measure = np.array([abs_alt[1, :], abs_alt[:, 1]]).T
    print "Con Shape:", orig_measure.shape, alt_measure.shape
    r = kabsch(orig_measure, alt_measure)
    theta = np.arccos(r.item(0, 1))
    print theta


def phase_correlation(fft_orig, fft_alt, rx, ry):
    # Use phase correlation to get translation
    alt_conj = np.ma.conjugate(fft_alt)
    print "Conj:", alt_conj.shape
    T = fft_orig * alt_conj
    T /= np.absolute(T)
    print "T:", T.shape
    t = np.fft.ifft2(T).real
    print "t:", t.shape
    print "t[t >= 0]:", t[t >= 0].shape

    ty, tx = np.unravel_index(np.argmax(t), t.shape)

    # x_max_for_y = np.argmax(t, axis=0)
    # y_max_for_x = np.argmax(t, axis=1)
    # tx = np.argmax(x_max_for_y)
    # ty = np.argmax(y_max_for_x)
    # print "Argmax, axis 0:", np.argmax(t, axis=0).shape
    # print "Argmax, axis 1:", np.argmax(t, axis=1).shape
    # tx = np.argmax(t[:, 0])
    # ty = np.argmax(t[:, 1])

    print "Estimated translation:", tx, ty
    print "Estimated translation (corrected):", tx * rx, ty * ry

    return np.array([tx * rx, ty * ry])


def compute_translation(im0, im1):
    shape = im0.shape

    f0 = np.fft.fft2(im0)
    f1 = np.fft.fft2(im1)

    ir = np.abs(np.fft.ifft2((f0 * f1.conjugate()) /
                             (np.abs(f0) * np.abs(f1))))

    t0, t1 = np.unravel_index(np.argmax(ir), shape)

    if t0 > shape[0] // 2:
        t0 -= shape[0]
    if t1 > shape[1] // 2:
        t1 -= shape[1]

    print "Computed Translation:", t0, t1
    return [t0, t1]


def logpolar(v, angles=None, radii=None):
    shape = v.shape
    center = shape[0] / 2, shape[1] / 2

    if angles is None:
        angles = shape[0]
    if radii is None:
        radii = shape[1]

    theta = np.empty((angles, radii), dtype=np.float64)
    theta.T[:] = -np.linspace(0, np.pi, angles, endpoint=False)
    # d = radii
    d = np.hypot(shape[0]-center[0], shape[1]-center[1])
    log_base = 10.0 ** (np.log(d) / (radii))
    radius = np.empty_like(theta)
    radius[:] = np.power(log_base, np.arange(radii,
                                             dtype=np.float64)) - 1.0
    x = radius * np.sin(theta) + center[0]
    y = radius * np.cos(theta) + center[1]
    output = np.empty_like(x)
    ndii.map_coordinates(v, [x, y], output=output)

    return output, log_base
