import copy
import math
import numpy as np
import matplotlib.pyplot as plt

import management
import drawing_tools as dt
from matplotlib.legend_handler import HandlerLine2D

class Unit(object):
    """
    A base for all classes of computational 'units'. Units are objects that have 
    inputs and outputs, which perform some computational function, and that have
    connections to other units.
    """
    def __init__(self):
        """
        A list of references to other units that can influence the output of this unit.
        """
        self._inputs = set()
        
        """
        A list of references to other units that are influenced by the output of this unit.
        """
        self._outputs = set()
        
        """
        A boolean attribute that represents whether the unit is being used at a given time.
        """
        self._active = False
                
        """
        A boolean attribute that represents a prediction that this unit will be activated
        in the next time.
        """
        self._primed = False

        """
        A real-valued measure of confidence regarding this unit's 'primed'-ness
        """
        self._prime_confidence = 0.0

        """
        A real-valued threshold over the confidence with which this unit is primed
        """        
        self._prime_thresh = 0.0

    def get_prime_confidence(self):
        return self._prime_confidence

    def set_prime_confidence(self, value):
        self._primed = self._prime_confidence > self._prime_thresh
        
    prime_confidence = property(get_prime_confidence,
                                set_prime_confidence)

    """
    Return whether this unit is primed.
    """
    def is_primed(self):
        return self._primed

    """
    A function that is used to update the internal attributes of a unit (regardless of whether
    or not it is active).
    """
    def update(self):
        pass
    
    """
    Activate this unit.
    """
    def activate(self, timetable):
        self._active = True
        for o in self._outputs:
            o.activate(timetable)
        
    """
    De-activate this unit.
    """
    def deactivate(self):
        self._active = False

    """
    Return whether this unit is active.
    """
    def is_active(self):
        return self._active

    def link_to_outputs(self, unit):
        self._outputs.add(unit)
        
    def link_to_inputs(self, unit):
        self._inputs.add(unit)



class SUnit(Unit):
    """
    A class that represents a unit that relies on 'sensorial' input, i.e. coming directly from the
    real-time sensor data of an agent.
    """
    def __init__(self, content):
        super(SUnit, self).__init__()
        self._content = content # TODO: Think about this.

    """
    TODO: Define overloaded operators for element-based comparison
    """
    def equals(self, test):
        return self._content == test._content

    """
    Prints relevant info about the unit in human-readable format.
    """
    def print_info(self):
        print("\tSUnit:")
        print("\t\tContent: ", self._content, "\tPrimed: ", self._primed, "\n")



class PUnit(Unit):
    """
    A recognizer of a specific sequence of input symbols. A C-unit can have multiple inputs,
    and learns how to predict the next symbol at its input given its execution history. 
    """

    def __init__(self, pattern):
        super(PUnit, self).__init__()
        
        """
        The temporal pattern that this unit is supposed to learn how to recognize and predict
        """
        self.alpha = 0.05
        self.lbd = 0.9
        self.eps = 0.5
        self.Tw = 10000  # data width
        self.Yh = []
        self.Lh = []
        self.Ah = []
        self.Rh = []
        self.Sh = []
        self.Q0hs = []
        self.Q1hs = []
        self.Q0hmc = []
        self.Q1hmc = []
        self.Q0hq = []
        self.Q1hq = []
        self.Alphah = []
        self.Epsh = []
        self.traj = []

        self.pattern = pattern
        self.S = {'match': {'ndef' : 0,'fail' : 1, 'win': 2},
                  'time': range(0,len(self.pattern))}
        self.A = ['idle', 'prime']
        
        self.Q_s = np.zeros([len(self.S['match'])*len(self.S['time']),len(self.A)])
        self.Q_q = np.zeros([len(self.S['match'])*len(self.S['time']),len(self.A)])
        self.Qmc = np.zeros([len(self.S['match'])*len(self.S['time']),len(self.A)])
        self.Nmc = np.zeros([len(self.S['match'])*len(self.S['time']),len(self.A)])
        self.et = np.zeros([len(self.S['match'])*len(self.S['time']),len(self.A)])

        self.s0 = ('ndef',0)
        self.s = self.s0
        self.s_next = self.s0
        self.t_mdp = 0
        self.t_seq = 0
        self.t_to_update = 1
        self.t_lifetime = 0
        self.a = 0
        self.a_next = 0
        self.seq_active = False
        self.greedy = False
        self.primed = False
        np.random.seed()

    def equals(self, test):
        return self._STM.equals(test._STM) and self._symbol.equals(test._symbol)


    def activate(self, timetable):
        print("P-Unit Activated. STM: ", len(self._pattern))
        #TODO: insert unit into timetable if needed
        self._active = True
        for o in self._outputs:
            o.activate(timetable)

    """
    Prints relevant info about the unit in human-readable format.
    """
    def print_info(self):
        print "Learning pattern:", self.pattern
        print "Final Policy:", self.pol

    def draw(self):
        self.draw_sequences()
        #self.draw_history()

    def draw_sequences(self):
        f = plt.figure()
        spy = f.add_subplot(511)
        spy.stem(self.Yh[-100:], label="observations", linefmt='b-')
        plt.ylabel('Observations')
        plt.xlabel('step')
        
        spa = f.add_subplot(512)
        spa.stem(self.Ah[-100:], linefmt='r-', markerfmt='ro')
        plt.ylabel('Actions')
        plt.xlabel('step')
        
        spr = f.add_subplot(513)
        spr.stem(self.Rh[-100:], linefmt='g-', markerfmt='go')
        plt.ylabel('Rewards')
        plt.xlabel('step')
        
        sps = f.add_subplot(514)
        sps.stem(self.Sh[-100:], linefmt='k-', markerfmt='ko')
        plt.ylabel('State')
        plt.xlabel('step')
        
        spy.set_title("Pattern: %s"%str(self.pattern))
        
        f.show()
    
    def draw_history(self):
        fq = plt.figure()
        line_Q0h, = plt.plot(self.Q0hs, color='r', label="SARSA")
        plt.plot(self.Q0hmc, color='b', label="Monte Carlo")
        plt.plot(self.Q0hq, color='g', label="Q-Learning")
        plt.title('State-Action pair: [0,1] = 0')
        plt.ylabel('Q-Value')
        plt.xlabel('Step')
        plt.legend(handler_map={line_Q0h: HandlerLine2D(numpoints=4)})
        
        fqmc = plt.figure()
        line_Q1h, = plt.plot(self.Q1hs, color='r', label="SARSA")
        plt.plot(self.Q1hmc, color='b', label="Monte Carlo")
        plt.plot(self.Q1hq, color='g', label="Q-Learning")
        plt.title('State-Action pair: [2,1] = 0.33')
        plt.ylabel('Q-Value')
        plt.xlabel('Step')
        plt.legend(handler_map={line_Q1h: HandlerLine2D(numpoints=4)})
        
        falpha = plt.figure()
        line_alpha_epsilon, = plt.plot(self.Alphah, color='b', label="Alpha")
        plt.plot(self.Epsh, color='r', label="Epsilon")
        plt.title('Alpha/Epsilon')
        plt.ylabel('Alpha/Epsilon')
        plt.xlabel('Step')
        plt.legend(handler_map={line_alpha_epsilon: HandlerLine2D(numpoints=4)})
    
        fq.show()
        fqmc.show()
        falpha.show()
        

    def reward_function(self, s, a_name):
        if a_name == 'prime':
            if s[0] == 'win':
                return 1 - s[1] / float(len(self.pattern))
            elif s[0] == 'fail':
                return -(1 - s[1] / float(len(self.pattern)))
        return 0
        
    def update_q(self, s_now, a, s_next, r):
        self.Q_q[s_now,a] = self.Q_q[s_now,a] + self.alpha * (r + np.max(self.Q_q[s_next,:]) - self.Q_q[s_now,a])
    
    def update_sarsa(self, s_now, a_now, r, s_next, a_next):
        delta = r + self.Q_s[s_next, a_next] - self.Q_s[s_now, a_now]
        self.et[s_now, a_now] += 1
        
        self.Q_s = self.Q_s + self.alpha * delta * self.et
        self.et = self.lbd * self.et
    
    def update_monte_carlo(self, r):
        B = {}
        for sa in self.traj:
            s = sa[0]
            a = sa[1]
            if not B.has_key((s,a)):
                B[(s,a)] = 1
                self.Qmc[s,a] = (self.Qmc[s,a] * self.Nmc[s,a] + r) / float(self.Nmc[s,a] + 1)
                self.Nmc[s,a] += 1
    
    def joint(self, s):
        m = self.S['match']
        return m[s[0]]*len(self.S['time']) + s[1]
    
    def update_state(self, y):
        s_next = ('ndef', 0)
        if self.s[0] != 'ndef':
            return s_next
            
        if self.A[self.a] == 'prime':
            s_next = (self.s[0], self.t_mdp)
        elif self.t_mdp < len(self.pattern)-1:
            s_next = (self.s[0], self.s[1] + 1)
            
        if y == self.pattern[self.t_seq]:
            if self.t_seq == len(self.pattern) - 1:
                s_next = ('win', s_next[1])
        else:
            s_next = ('fail', s_next[1])
        
        return s_next

    def new_symbol(self, y, l):
        self.Yh.append(y)
        self.Lh.append(l)
        
        r = 0
        
        if self.seq_active is False and y == self.pattern[0]:
            self.seq_active = True
        
        if self.seq_active is True:
            self.t_to_update -= 1

            if self.t_to_update == 0 or y != self.pattern[self.t_seq]:
                    
                if self.t_seq > 0:
                    self.s_next = self.update_state(y)
                    
                    self.js = self.joint(self.s)
                    self.jn = self.joint(self.s_next)
                    
                    r = self.reward_function(self.s_next, self.A[self.a])
                    self.update_q(self.js, self.a, self.jn, r)
                    
                    self.t_mdp += 1
                    
                if self.s_next[0] == 'ndef':
                    ran = np.random.uniform(0,1)
                    if ran > self.eps:
                        self.greedy = True
                        self.a_next = np.argmax(self.Q_q[self.joint(self.s_next),:])
                    else:
                        ran = np.random.uniform(0,1)
                        self.greedy = False
                        self.a_next = int(ran > 0.5)

                if self.t_seq > 0:
                    self.update_sarsa(self.js, self.a, r, self.jn, self.a_next)
                
                if self.A[self.a] == 'idle' and self.A[self.a_next] == 'prime':
                    self.t_to_update = (len(self.pattern) - 1) - self.t_seq
                    self.primed = True
                else:                   
                    self.t_to_update = 1
                
                self.a = self.a_next
                self.s = self.s_next
                self.traj.append((self.joint(self.s), self.a))
                
  
            self.t_seq += 1
        
        self.Ah.append(self.a)
        self.Sh.append(self.joint(self.s_next))
        
        if self.s_next[0] == 'win' or self.s_next[0] == 'fail' or self.t_seq == len(self.pattern):
            if self.s_next[0] == "win" and self.primed and self.greedy:
                self.eps = 0.9 * self.eps
            elif self.s_next[0] == "fail" and self.primed and self.greedy:
                self.eps = 1 - ((1 - self.eps) * 0.9)
            
            self.Alphah.append(self.alpha)
            self.Epsh.append(self.eps)
            
            self.a = 0
            self.s_next = self.s0
            self.t_mdp = 0
            self.t_seq = 0
            self.seq_active = False
            self.primed = False
            self.update_monte_carlo(r)
                
            self.traj = []
        
        self.Q0hs.append(self.Q_s[0,1])
        self.Q1hs.append(self.Q_s[2,1])
        self.Q0hmc.append(self.Qmc[0,1])
        self.Q1hmc.append(self.Qmc[2,1])
        self.Q0hq.append(self.Q_q[0,1])
        self.Q1hq.append(self.Q_q[2,1])
        self.Rh.append(r)

        self.pol = np.argmax(self.Q_s, 1)
        self.q_pol = np.argmax(self.Q_q, 1)
        self.mc_pol = np.argmax(self.Qmc, 1)
        
        self.t_lifetime += 1









# class AUnit(Unit):
#     def __init__(self):
#         self._weights = []
#         self.level = 0
#         ##this baseline should be adapted w / time. I don't want the robot spazzing out just because there is an unknown input
#         self.__baseline = 0
# 
#     def activate(self):
#         pass
# 
#     def evaluate(self):
#         if len(self._connections) == 0:
#             val = 0
#         else:
#             c_vals = np.zeros(len(self._connections))
#             for i in range(len(self._connections)):
#                 if self._connections[i].is_active():
#                     c_vals[i] = 1.0
# 
#             val = np.inner(np.asarray(self._weights), c_vals)
# 
#         self.level = 1.0 / (1.0 + np.exp(val))
# 
#         self._active = False
#         return self.level
# 
#     def connect(self, unit, weight):
#         if unit not in self._connections:
#             self._connections.append(unit)
#             self._weights.append(weight)

"""
A class that represents a network of Units, and that can itself be viewed as a Unit.
"""
class Net(Unit):
#TODO: overload python operators
    def __init__(self):
        super(Net, self).__init__()
        self._net = []

    def add_unit(self, unit):
        self._net.append(unit)

    def is_empty(self):
        if not self._net:
            return True
        else:
            return False

    def contains(self, unit):
        for u in self._net:
            if u.equals(unit):
                return True

        return False

    def get_unit(self, unit):
        for u in self._net:
            if u.equals(unit):
                return u

        management.unit_not_found()

    def print_net(self):
        print("Net:\n")

        for u in self._net:
            u.print_info()

        print("\n")

    def draw_net(self):
        graph = []
        it = 0

        for u in self._net:
            last_it = len(u._net)
            for s in u._net:
                if it == 0:
                    graph[it] = []
                    graph[it].append(s._content)
                elif it == last_it:
                    graph[it-1].append(s._content)
                else:
                    graph[it-1].append(s._content)

                    graph[it] = []
                    graph[it].append(s._content)

                it += 1

            print("GRAPH:", graph)
            it = 0
