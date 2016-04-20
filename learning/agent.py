import numpy as np
from units import *


class Agent:
    def __init__(self, ninputs, noutputs):
        self.net = [] 
        """ Tem de ser uma lista de listas, em que cada elemento e um nivel.
            So assim consigo processar um nivel de cada vez.
            O "nivel" de uma unidade pode ter de ser recalculado, dependendo das entradas. """
                          
        self.inputs = []
        self.outputs = []
        """
        NOTE: E aqui que e feito o mapeamento implicito entre inputs / unidades
        """
        for i in range(ninputs):
            su = SUnit()
            self.net.append(su)
            self.inputs.append(su)
        self.input_map = [None]+range(ninputs)

        for i in range(noutputs):
            au = AUnit()
            self.net.append(au)
            self.outputs.append(au)

        self.stm = []

        self.outputs[1].connect(self.inputs[0], 100.0)

    def step(self, y):

        
        """ ----------------------PERCEPTION--------------------------- """
        
        """
        Activating the input units according to sensor data
        """
        #self.input_map[y].activate()

        """
        Waking all units
        """
#        for level in self.net:
#            for u in level:
#                u.evaluate() 
        """------------????????---------------------------------------- """
        
        
        
        """ ------------------------ACTION----------------------------- """

        """
        Sampling output units according to their activation levels (i.e. Values).
        """
        na = len(self.outputs)

        pa = np.zeros(len(self.outputs))
        any_positive = False
        for i in range(len(self.outputs)):
            v = self.outputs[i].evaluate()
            if v > 0:
                any_positive = True
                pa[i] = v

        if not any_positive:
            return None

        pa = pa/np.sum(pa)
        print 'pa: ', pa

        a = np.random.choice(range(0, na), p=pa)
        
        """
        Cleaning up
        """
        for su in self.inputs:
            su.deactivate()

        return a

