#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import pdb
import sys

from sequence_generator import *
from markov_single_chain import *


def chunks(STM, n):
    return [STM[i:i + n] for i in xrange(0, len(STM)) if (i + n) <= len(STM)]


def get_possible_patterns(STM, symbols):
    possible_patterns = {}

    for symbol in symbols:
        possible_patterns[symbol] = []
        for i in xrange(0, len(STM)):
            if STM[i] == symbol:
                possible_patterns[symbol].append(STM[i:])

    return possible_patterns


def chain_exists(pat, chains):
    for chain in chains:
        if chain._chain == pat:
            return True

    return False


def main():
    Tw = 30  # data width
    # seq_gen = FixedSeqGenerator()
    # seq_gen = RandomWithSeveralPatternsGenerator()
    seq_gen = SeqGenerator()

    # s1 = FixedSeqGenerator([1, 2, 3])
    # s2 = NoisySeqGenerator([99])
    # seq_gen = NoisySeqMixer([s1, s2],[2])

    STM = []
    symbol = -1
    chains = []
    distances = []
    symbols = []

    for t in xrange(0, Tw):
        # pdb.set_trace()
        y = seq_gen.step()

        symbol = y[0]
        if symbol not in symbols:
            if symbol != 0:
                symbols.append(symbol)

        STM.append(symbol)

    possible_patterns = get_possible_patterns(STM, symbols)

    print possible_patterns

    first = True
    curr_chain = 0
    for root, pats in possible_patterns.iteritems():
        first = True
        for pat in pats:
            if first is True:
                curr_chain = Chain(pat)
                chains.append(curr_chain)
                first = False
            else:
                curr_chain.add_chain(pat)

    for chain in chains:
        chain.print_chain()
        # distances.append(chain.update(STM))

    # print "Distances information:"
    # print "\tMax:", np.max(distances)
    # print "\tMin:", np.min(distances)
    # print "\tAverage:", np.average(distances)

    print "\n\n\n"
    print "Do you want to do anything?"
    print '\t"t active_symbol time time_to_test" for testing state distribution'
    print '\t"p T" for printing the transition matrix'
    print '\t"plot" for plotting'
    print '\t"s" for performing a sanity check'
    print '\t"d" for debugging'
    print '\t"q" for quitting'

    # while 1:
    #     inp = raw_input('--> ')

    #     if inp == "q":
    #         sys.exit()
    #     elif inp == "plot":
    #         f = plt.figure()
    #         spy = f.add_subplot(211)
    #         spy.stem(STM, label="STM", linefmt='b-', markerfmt='bo')
    #         plt.ylabel('STM')
    #         plt.xlabel('step')
    #         spa = f.add_subplot(212)
    #         spa.stem(distances, label="distances", linefmt='r-', markerfmt='ro')
    #         plt.ylabel('Distances')
    #         plt.xlabel('Pattern')
    #         # spr = f.add_subplot(413)
    #         # spr.stem(autocov_behaviour, label="autocovariance_rise", linefmt='g-', markerfmt='go')
    #         # plt.ylabel('Autocovariance Rise')
    #         # plt.xlabel('step')
    #         # spr = f.add_subplot(414)
    #         # spr.stem(autocorrelation, label="autocorrelation", linefmt='k-', markerfmt='kd')
    #         # plt.ylabel('Full Autocorrelation')
    #         # plt.xlabel('phase')
    #         # plt.xlim([0, 10])
    #         # plt.ylim([0, 2000])
    #         plt.show()
    #     elif inp[0] == "p":
    #         l = inp.split()

    #         what = l[1]

    #         if what == "T":
    #             print "Transition Matrix:\n\t", chain.compute_transition_matrix()
    #     elif inp[0] == "d":
    #         pdb.set_trace()
    #     elif inp[0] == "s":
    #         chain.sanity_check()
    #     elif inp[0] == "t":
    #         l = inp.split()

    #         symbol = int(l[1])
    #         time = int(l[2])
    #         time_to_test = int(l[3])

    #         chain.get_state_distribution(symbol, time, time_to_test)
    #     else:
    #         print "Unrecognized command. Please repeat it"


if __name__ == '__main__':
    main()
