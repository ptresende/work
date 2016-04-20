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
    lena_orig = misc.lena()
    lena_alt = copy.deepcopy(lena_orig)

    lena_alt = lena_alt[100:400, 100:400]

    orig_fft = np.fft.fft2(lena_orig)
    alt_fft = np.fft.fft2(lena_alt)

    support.phase_correlation(orig_fft, alt_fft, 1, 1)


if __name__ == '__main__':
    main()
