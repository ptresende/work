#! /usr/bin/env python
import PyQt4
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
import pdb
from sequence_generator import SeqGenerator
from reward_generator import RewardGenerator
from agent import Agent


def main():
    Tw = 100  # data width
    Y = []
    A = []
    R = []
    seq_gen = SeqGenerator()
    rew_gen = RewardGenerator()
    agent = Agent(3, 2)
    total_r = 0
    for t in range(Tw):
        (y, l) = seq_gen.step()
        a = agent.step(y)
        r = rew_gen.evaluate(l, a)

        total_r += r
        Y.append(y)
        A.append(a)
        R.append(r)

    print("Simulation finished.")
    print("Average Reward:", total_r / float(Tw), " ", total_r)

    f = plt.figure()
    spy = f.add_subplot(311)
    spy.stem(Y, label="observations", linefmt='b-', markerfmt='bo')
    plt.ylabel('Observations')
    plt.xlabel('step')
    spa = f.add_subplot(312)
    spa.stem(A, linefmt='r-', markerfmt='ro')
    plt.ylabel('Actions')
    plt.xlabel('step')
    spr = f.add_subplot(313)
    spr.stem(R, linefmt='g-', markerfmt='go')
    plt.ylabel('Rewards')
    plt.xlabel('step')
    plt.show()                  ### --> So assim consigo manter a imagem aberta
    #f.show()                   ### --> Codigo original para mostrar a imagem
    g = plt.figure()





if __name__ == '__main__':
    main()
