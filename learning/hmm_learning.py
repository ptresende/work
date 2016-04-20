#! /usr/bin/env python

import numpy as np
import hmmestimate as hmm

h1 = hmm.HMM()

h1.pi = np.array([0.5, 0.5])

h1.A = np.array([[0.85, 0.15],
                 [0.12, 0.88]])

h1.B = np.array([[0.8, 0.1, 0.1],
                 [0.0, 0.0, 1]])

observations, states = h1.simulate(1000)


learned = hmm.HMM()

learned.pi = np.array([0.5, 0.5])

learned.A = np.array([[0.5, 0.5],
                      [0.5, 0.5]])

learned.B = np.array([[0.3, 0.3, 0.4],
                      [0.2, 0.5, 0.3]])

learned.train(observations, 0.0001, graphics=True)

print "Learned:"
print "\tPi:", learned.pi, "\n\n"
print "\tA:", learned.A, "\n\n"
print "\tB:", learned.B, "\n\n"

