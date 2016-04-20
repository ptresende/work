import numpy as np
import pdb

class Sequence(object):
    """
    Base class for all sequences.
    Can be stepped, and can be queried to see if it's over.
    Can also produce a label for supervised training.
    """
    def __init__(self, seq, label = None):
        self._seq = seq
        self._l = label
        self._t = 0

    def draw(self):
        return None
    
    def step(self):
        self._t = self._t%len(self._seq)
        y = self.draw()
        self._t += 1
        return y
    
    def finished(self):
        return self._t >= len(self._seq)
    
    def get_label(self):
        return self._l

class FixedSeqGenerator(Sequence):
    """
    Outputs a fixed noiseless sequence.
    """
    def __init__(self, seq, label = None):
        super(FixedSeqGenerator,self).__init__(seq, label)

    def draw(self):
        return self._seq[self._t]
    
class NoisySeqGenerator(Sequence):
    """
    Outputs a fixed noisy sequence - one in which each there is a probability distribution
    over possible symbols at each step.
    """
    def __init__(self, pseq, label = None):
        super(NoisySeqGenerator,self).__init__(pseq, label)
        #pseq is a list of numpy float arrays describing a probability distribution over symbols
            
    def draw(self):
        N = np.size(self._seq[self._t])
        P = self._seq[self._t]
        if np.size(self._seq[self._t]) == 1:
            N = np.arange(self._seq[self._t])
            P = None
        return np.random.choice(N, p=P)

class SeqMixer(Sequence):
    """
    Abstract sequence of sequences (of sequences..)
    """
    def __init__(self, mseqs, mix_order, sep=None):
        super(SeqMixer,self).__init__(mix_order)
        self._mseqs = mseqs
        self._current = self.draw()
        self._sep = sep
        self._at_sep = False
    
    def step(self):
        self._t = self._t%len(self._seq)
        if self._at_sep is True:
            self._at_sep = False
            return self._sep
        else:
            y = self._current.step()
            if self._current.finished():
                self._current = self.draw()
                self._t += 1
                if self._sep is not None:   
                    self._at_sep = True
            return y

    def draw(self):
        raise Exception
    
    def get_label(self):
        return self._current.get_label()


class FixedSeqMixer(SeqMixer):
    """
    Outputs a fixed mix of (not necessarily fixed) sequences
    """
    def __init__(self, mseqs, mix_order, sep=None):
        super(FixedSeqMixer,self).__init__(mseqs, mix_order, sep)

    def draw(self):
        return self._mseqs[self._seq[self._t]]

class NoisySeqMixer(SeqMixer):
    """
    Outputs a noisy mix of sequences
    """
    def __init__(self, mseqs, pmix_order, sep=None):
        super(NoisySeqMixer,self).__init__(mseqs, pmix_order, sep)

    def draw(self):
        N = np.size(self._seq[self._t])
        P = self._seq[self._t]
        if np.size(self._seq[self._t]) == 1:
            N = np.arange(self._seq[self._t])
            P = None
        return self._mseqs[np.random.choice(N, p=P)]

class OnesGenerator():
    """
    For backwards compatibility.. not really needed
    """
    def step(self):
        return (1,1)

        