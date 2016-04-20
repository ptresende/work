import numpy as np
from random import randint
import math

# Patterns: 2
sq = {0: [1,2,3]}
#sq = {0: [1, 2, 3], 1: [1, 3, 4]}
#sq = {0: [1, 1, 1, 2], 1: [1, 1, 2, 3]}
#sq = {0: [1, 1, 1, 1, 1], 1: [1, 1, 1, 1, 10]}

# Patterns: 3
#sq = {0: [1, 1, 1, 2], 1: [1, 1, 2, 3], 2: [1, 2, 4, 8]}

# Patterns: 10
#sq = {0: [1, 1, 1, 2], 1: [1, 1, 2, 3], 2: [1, 1, 4, 8], 3: [1, 1, 2, 2], \
#      4: [1, 1, 9, 5], 5: [1, 1, 11, 2], 6: [1, 1, 8, 4], 7: [1, 1, 4, 4], \
#      8: [1, 1, 3, 2], 9: [1, 1, 5, 7]}

# Patterns: 100
#sq = {}

#for i in xrange(0, 100):
#    sq[i] = [1, 1, i + 1, int(math.floor(i / 2))]

#sq[0] = [1, 1, 1, 2]


Tseq = 10


class SeqGenerator():
    def __init__(self):
        self._sq = sq  # could be randomized
        self._tseq = Tseq  # could be randomized or asynchronous
        self._t = 0
        self._current_seq = 0

    # With zeroes
    def step(self):
        m = self._t % self._tseq
        
        if m == 0:
            #r = np.random.random_sample()
            #self._current_seq = r > 0.5
            r = randint(0, len(self._sq) - 1)
            self._current_seq = r
            
        if m < len(sq[self._current_seq]):
            s = sq[self._current_seq]
            y = s[m]
            l = self._current_seq
        else:
            y = 0
            l = None

        self._t += 1
        return (y, l)

class RandomGenerator():
    def __init__(self):
        self._sq = np.random.randint(0, 10, 500)
        self._t = 0

    def step(self):
        l = 1
        y = self._sq[self._t]

        self._t += 1

        return (y, l)

class RandomWithPatternsGenerator():
    def __init__(self):
        self._sq = np.random.randint(0, 100, 500)
        self._pattern = [1, 2, 3]
        self._t = 0
        self._i = 0
        self._pattern_mode = False

    def step(self):
        l = 1

        ran = np.random.random_sample()

        if self._pattern_mode:
            y = self._pattern[self._i]
            self._i += 1
            self._t += 1

            if self._i >= len(self._pattern):
                self._pattern_mode = False
                self._i = 0

            return (y, l)

        if ran > 0.5:
            self._pattern_mode = True
            y = self._pattern[self._i]
            self._i += 1
        else:
            y = self._sq[self._t]

        self._t += 1

        return (y, l)

class RandomWithSeveralPatternsGenerator():
    def __init__(self):
        self._sq = np.random.randint(0, 25, 5000)
        # self._patterns = np.matrix([[1, 2, 3, 4], [6, 7, 8, 9], [111, 1000, 555, 1906], [2000, 2001, 2002, 2003], [3001, 3002, 3003, 3004]])
        self._patterns = np.matrix([[100, 101, 100, 103]])
        self._pattern = 0
        self._t = 0
        self._i = 0
        self._pattern_mode = False

    def step(self):
        l = 1

        ran = np.random.random_sample()

        if self._pattern_mode:
            y = self._patterns.item(self._pattern, self._i)
            self._i += 1
            self._t += 1

            if self._i >= self._patterns.shape[1]:
                self._pattern_mode = False
                self._i = 0

            return (y, l)

        if ran > 0.5:
            self._pattern_mode = True
            self._pattern = np.random.randint(0, self._patterns.shape[0])

            y = self._patterns.item(self._pattern, self._i)

            self._i += 1
        else:
            y = self._sq[self._t]

        self._t += 1

        return (y, l)

class FixedSeqGenerator():
    def __init__(self):
        self._sq = [2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self._sq2 = [2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self._sq3 = [6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self._t = 0
        self.last_y = 0
        self._i = 0

    # With zeroes
    def step(self):
        l = 1
            
        if self._i < 10:
            y = self._sq[self._t]
        elif self._i < 20:
            y = self._sq2[self._t]
        else:
            y = self._sq3[self._t]

        self.last_y = y
        self._t += 1

        if self._t >= len(self._sq):
            self._i += 1
            self._t = 0

        return (y, l)

class FixedSeqGeneratorWithReward():
    def __init__(self):
        self._sq = [2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self._sq2 = [2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self._t = 0
        self.last_y = 0
        self._i = 0

    # With zeroes
    def step(self):
        l = 1
        r = 0
            
        if self._i < 10:
            y = self._sq[self._t]
        else:
            y = self._sq2[self._t]

        self.last_y = y
        self._t += 1

        if self._t >= len(self._sq):
            self._i += 1
            self._t = 0

        return (y, l, r)

class OnesGenerator():
    def step(self):
        return (1,1)

