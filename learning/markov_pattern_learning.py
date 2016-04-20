#! /usr/bin/env python

# import PyQt4
# import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# import scipy as sp
# import time
import pdb
import sys

from sequence_generator import *
# from sequence_generator_2 import *

from markov_chain import *


def hamming_distance(s1, s2):
    if len(s1) != len(s2):
        raise ValueError("Undefined for sequences of unequal length")
    return sum(ch1 != ch2 for ch1, ch2 in zip(s1, s2))


def autocov_rise(signal, autocov):
    if len(signal) >= 2:
        if autocov[-1] > autocov[-2]:
            return True
        else:
            return False
    else:
        return False


def main():
    Tw = 5000  # data width
    # seq_gen = FixedSeqGenerator()
    seq_gen = RandomWithSeveralPatternsGenerator()

    # s1 = FixedSeqGenerator([1, 2, 3])
    # s2 = NoisySeqGenerator([99])
    # seq_gen = NoisySeqMixer([s1, s2],[2])
    
    signal = []
    symbol = -1
    node = -1
    last_symbol = -1
    last_node = -1

    chain = 0
    chain_list = []
    inited = False

    for t in xrange(0, Tw):
        # pdb.set_trace()
        y = seq_gen.step()

        symbol = y[0]
        signal.append(symbol)

        if node == -1 and last_node == -1:
            # First iteration
            last_node = Node(symbol)
            continue

        if not inited:
            # Second iteration
            if symbol == last_node._symbol:
                # The second symbol is equal to the first one
                chain = Chain(last_node, last_node)
                chain_list.append(chain)
            else:
                # Otherwise
                node = Node(symbol)
                chain = Chain(last_node, node)
                chain_list.append(chain)

            inited = True
        else:
            node = chain.get_node_from_symbol(symbol)
            if not node:
                node = Node(symbol)

        chain.add_node(last_node, node)

        last_symbol = symbol
        last_node = chain.get_node_from_symbol(last_symbol)
        if not last_node:
            last_node = Node(last_symbol)

    autocorrelation = np.correlate(signal-np.mean(signal),
                                   signal-np.mean(signal), 'full')

    chain.print_chain()
    # tree.sanity_check()

    print("Simulation finished.")

    print "\nSize of chains in memory:"
    print "\tNormal chain ->", sys.getsizeof(chain), "bytes"

    print "\n\n\n"
    print "Do you want to do anything?"
    print '\t"t active_symbol time_later" \
          for testing state distribution'
    print '\t"p T" for printing the transition matrix'
    print '\t"plot" for plotting'
    print '\t"s" for performing a sanity check'
    print '\t"d" for debugging'
    print '\t"q" for quitting'

    while 1:
        inp = raw_input('--> ')

        if inp == "q":
            sys.exit()
        elif inp == "plot":
            f = plt.figure()
            spy = f.add_subplot(411)
            spy.stem(signal, label="signal", linefmt='b-', markerfmt='bo')
            plt.ylabel('Signal')
            plt.xlabel('step')
            spa = f.add_subplot(412)
            spa.stem(autocov, label="autocovariance", linefmt='r-',
                     markerfmt='ro')
            plt.ylabel('Autocovariance')
            plt.xlabel('step')
            spr = f.add_subplot(413)
            spr.stem(autocov_behaviour, label="autocovariance_rise",
                     linefmt='g-', markerfmt='go')
            plt.ylabel('Autocovariance Rise')
            plt.xlabel('step')
            spr = f.add_subplot(414)
            spr.stem(autocorrelation, label="autocorrelation",
                     linefmt='k-', markerfmt='kd')
            plt.ylabel('Full Autocorrelation')
            plt.xlabel('phase')
            plt.xlim([0, 10])
            plt.ylim([0, 2000])
            plt.show()
        elif inp == "":
            pass
        elif inp[0] == "p":
            l = inp.split()

            what = l[1]

            if what == "T":
                print "Transition Matrix:\n\t", \
                    chain.compute_transition_matrix()
        elif inp[0] == "d":
            pdb.set_trace()
        elif inp[0] == "s":
            chain.sanity_check()
        elif inp[0] == "t":
            l = inp.split()

            symbol = int(l[1])
            time_later = int(l[2])

            chain.get_state_distribution(symbol, time_later)
        else:
            print "Unrecognized command. Please repeat it"


if __name__ == '__main__':
    main()
