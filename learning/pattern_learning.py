
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
#from sequence_generator_2 import *

from tree_learning import *

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
    Tw = 500  # data width
    #seq_gen = FixedSeqGenerator()
    seq_gen = RandomWithSeveralPatternsGenerator()

    #s1 = FixedSeqGenerator([1, 2, 3])
    #s2 = NoisySeqGenerator([99])
    #seq_gen = NoisySeqMixer([s1, s2],[2])
    
    signal = []
    autocov = []
    autocov_behaviour = []
    symbol = 0
    last_symbol = 0

    tree = Tree("Tree")
    #autocov_tree = Tree("Autocovariance-Filtered Tree")

    for t in xrange(0, Tw):
        # pdb.set_trace()
        y = seq_gen.step()

        symbol = y[0]

        signal.append(symbol)
        #autocov.append(np.cov(signal))
        #autocov_behaviour.append(autocov_rise(signal, autocov))

        if symbol == 0 and last_symbol == 0:
            continue

        tree.update(last_symbol, symbol)

        #if autocov_rise(signal, autocov):
        #    autocov_tree.update(last_symbol, symbol)

        last_symbol = symbol

    #autocov_tree.print_tree()
    #autocov_tree.sanity_check()

    autocorrelation = np.correlate(signal-np.mean(signal),
                                   signal-np.mean(signal), 'full')

    print "\n\n"

    tree.print_tree()
    tree.sanity_check()

    # if autocov_tree._rises == len([i for i in autocov_behaviour if i != 0]):
    #     print "Checks out"
    #     print "Rises:", autocov_tree._rises
    #     print "Len:", len([i for i in autocov_behaviour if i != 0])
    # else:
    #     print "Rises:", autocov_tree._rises
    #     print "Len:", len([i for i in autocov_behaviour if i != 0])

    print("Simulation finished.")

    print "\nSize of trees in memory:"
    #print "\tAutocovariance-filtered tree ->", sys.getsizeof(autocov_tree), "bytes"
    print "\tNormal tree ->", sys.getsizeof(tree), "bytes"

    print "\n\n\n"
    print "Do you want to do anything?"
    print "\tt parent_name for testing the tree"
    print "\tplot for plotting"
    print "\tq for quitting"

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
            spa.stem(autocov, label="autocovariance", linefmt='r-', markerfmt='ro')
            plt.ylabel('Autocovariance')
            plt.xlabel('step')
            spr = f.add_subplot(413)
            spr.stem(autocov_behaviour, label="autocovariance_rise", linefmt='g-', markerfmt='go')
            plt.ylabel('Autocovariance Rise')
            plt.xlabel('step')
            spr = f.add_subplot(414)
            spr.stem(autocorrelation, label="autocorrelation", linefmt='k-', markerfmt='kd')
            plt.ylabel('Full Autocorrelation')
            plt.xlabel('phase')
            plt.xlim([0, 10])
            plt.ylim([0, 2000])
            plt.show()                  ### --> So assim consigo manter a imagem aberta
            #f.show()                   ### --> Codigo original para mostrar a imagem
            #g = plt.figure()
        elif inp[0] == "t":
            parent = inp[2:]

            tree.get_pattern_from_parent(parent)
        else:
            print "Unrecognized command. Please repeat it"



if __name__ == '__main__':
    main()
