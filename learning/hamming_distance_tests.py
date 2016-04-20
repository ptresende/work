#! /usr/bin/env python
import PyQt4
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import time
import pdb
import sys

from sequence_generator import *


def hamming_distance(s1, s2):
    if len(s1) != len(s2):
        raise ValueError("Undefined for sequences of unequal length")
    return sum(ch1 != ch2 for ch1, ch2 in zip(s1, s2))

def main():
    Tw = 100  # data width
    #seq_gen = FixedSeqGenerator()
    seq_gen = RandomWithPatternsGenerator()
    #seq_gen = RandomWithSeveralPatternsGenerator()

    #s1 = FixedSeqGenerator([1, 2, 3])
    #s2 = NoisySeqGenerator([99])
    #seq_gen = NoisySeqMixer([s1, s2],[2])
    
    signal = []
    ham_signal = []
    auto_signal = []

    for t in xrange(0, Tw):
        # pdb.set_trace()
        y = seq_gen.step()

        symbol = y[0]
        signal.append(symbol)

        autocorrelation = np.correlate(signal - np.mean(signal),
                                       signal - np.mean(signal), 'full')

        auto_signal.append(np.sum(autocorrelation))

    ham = hamming_distance(signal, signal)

    for i in xrange(0, len(signal)):
        ham_signal.append(hamming_distance(signal, np.roll(signal, i)))

    print "Simulation finished."
    print "Autocorrelation:", autocorrelation
    print "Hamming Distance:", ham
    print "Hamming Distance Rolled:", ham_signal

    f = plt.figure()
    spy = f.add_subplot(411)
    spy.stem(signal, label="signal", linefmt='b-', markerfmt='bo')
    plt.ylabel('Signal')
    plt.xlabel('step')
    spa = f.add_subplot(412)
    spa.stem(ham_signal, label="ham_roll", linefmt='r-', markerfmt='ro')
    plt.ylabel('Hamming Roll')
    plt.xlabel('step')
    spr = f.add_subplot(413)
    spr.stem(autocorrelation, label="autocovariance_rise", linefmt='g-', markerfmt='go')
    plt.ylabel('Autocovariance')
    plt.xlabel('step')
    spr = f.add_subplot(414)
    spr.stem(auto_signal, label="autocorrelation", linefmt='k-', markerfmt='kd')
    plt.ylabel('Full Autocorrelation')
    plt.xlabel('phase')
    # plt.xlim([0, 10])
    # plt.ylim([0, 2000])
    plt.show()                  ### --> So assim consigo manter a imagem aberta


if __name__ == '__main__':
    main()
