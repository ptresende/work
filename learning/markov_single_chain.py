#! /usr/bin/env python

import pdb

import matplotlib.pyplot as plt
import numpy as np
import drawing_tools as dt
import operator
import copy
import sys

DEBUG = False


class Node:
    _ID = 0

    def __init__(self, symbol):
        self._symbol = symbol
        self._total_transitions_from = 0.0
        self._id = Node._ID

        Node._ID += 1

    def new_transition_from(self):
        self._total_transitions_from += 1.0


class Transition:
    def __init__(self, from_node, to_node):
        self._from_node = from_node
        self._to_node = to_node
        self._ocurrences = 1.0

    def new_occurence(self):
        self._ocurrences += 1.0


class Chain:
    def __init__(self, chain):
        self._nodes = []
        self._transitions = []
        self._chain = chain
        self._size = len(chain)

        for n in chain:
            self._nodes.append(Node(n))

        for i in xrange(0, len(self._nodes) - 1):
            self._transitions.append(Transition(self._nodes[i], self._nodes[i + 1]))

        if DEBUG:
            print "Nodes:", [n._symbol for n in self._nodes]
            for t in self._transitions:
                print "T:", t._from_node._symbol, "->", t._to_node._symbol

        self._prob_thr = 0.45
        self._dist_thr = 0.30

    def print_chain(self):
        # T = self.compute_transition_matrix()

        print "Nodes:", [n._symbol for n in self._nodes]
        print "Transitions:"
        for t in self._transitions:
            print "\t", t._from_node._symbol, "->", t._to_node._symbol

        if DEBUG:
            print "\nTransitions:", T

        # self.get_likely_patterns(T)

    def add_chain(self, chain):
        if chain[0] != self._chain[0]:
            print "Trying to add a chain with a different root!"
            sys.exit()

        for i in xrange(0, len(chain) - 1):
            for t in self._transitions:
                if t._from_node == chain[i] and t._to_node == chain[i + 1]:
                    # Transition already exists in the chain
                    pass
                elif t._from_node == chain[i] and t._to_node != chain[i + 1] or t._from_node != chain[i] and t._to_node == chain[i + 1]:
                    # Transition does not yet exist
                    if self.get_node_from_symbol(chain[i + 1]) is False:
                        # Node for symbol at i + 1 does not yet exist
                        n = Node(chain[i + 1])
                        self._transitions.append(self.get_node_from_symbol(chain[i]),
                                                 n)
                        continue

                    if self.get_node_from_symbol(chain[i]) is False:
                        # Node for symbol at i does not yet exist
                        n = Node(chain[i])
                        self._transitions.append(n,
                                                 self.get_node_from_symbol(chain[i]))
                        continue

                    # Both symbols already exist
                    self._transitions.append(self.get_node_from_symbol(chain[i]),
                                             self.get_node_from_symbol(chain[i + 1]))

    def update(self, STM):
        recent_occ = STM[-self._size:]
        dist, idx = self.hamming_distance(recent_occ, self._chain)

        if dist == 0:
            self.reinforce_all_transitions()
        elif dist != 0 and dist <= self._dist_thr * self._size:
            print "CHAIN:", self._chain
            print "RECENT OCC:", recent_occ
            print "IDX:", idx
            print "DIST:", dist
            print "\n\n"

        return dist

    def reinforce_all_transitions(self):
        for t in self._transitions:
            t.new_occurence()
            t._from_node.new_transition_from()

    def hamming_distance(self, s1, s2):
        if len(s1) != len(s2):
            raise ValueError("Undefined for sequences of unequal length")

        hd = [ch1 != ch2 for ch1, ch2 in zip(s1, s2)]

        idx = []
        for i in xrange(0, len(s1)):
            if s1[i] != s2[i]:
                idx.append(i)

        return sum(hd), idx

    def get_likely_patterns(self, T):
        pattern = {}
        max_over_transition = -1
        max_idx = -1
        r_num = 0
        c_num = 0

        for r in T:
            max_over_transition = -1
            max_idx = -1
            c_num = 0

            for c in r:
                v = c
                if v >= max_over_transition:
                    max_over_transition = v
                    max_idx = (r_num, c_num)

                pattern[max_idx] = max_over_transition
                c_num += 1

            r_num += 1

        print "\nPatterns:"

        for p, v in pattern.iteritems():
            from_node = self.get_node_from_id(p[0])._symbol
            to_node = self.get_node_from_id(p[1])._symbol

            if v >= self._prob_thr:
                print "\tFrom " + str(from_node) + " to " + str(to_node) + " with probability " + str(v)

        print "\n"

    def compute_transition_matrix(self):
        num_node = len(self._nodes)
        T = np.zeros((num_node, num_node))

        for t in self._transitions:
            r = t._from_node._id
            c = t._to_node._id

            if DEBUG:
                print "Transition:"
                print "\tFrom:", t._from_node._symbol, "(ID =", str(t._from_node._id) + ")"
                print "\tTo:", t._to_node._symbol, "(ID =", str(t._to_node._id) + ")"
                print "\tOcurrences:", t._ocurrences
                print "\tTotal Ocurrences from Parent:", t._from_node._total_transitions_from

            T[r, c] = t._ocurrences / t._from_node._total_transitions_from

        return T

    def compute_distribution(self, T, x, time, time_to_test):
        if time == time_to_test:
            return np.dot(x, T)
        else:
            return self.compute_distribution(T, x, time, time_to_test - 1)

    def get_state_distribution(self, symbol, time, time_to_test):
        T = self.compute_transition_matrix()
        node = self.get_node_from_symbol(int(symbol))
        x = np.zeros(len(self._nodes))
        x[node._id] = 1.0

        power = time_to_test - time
        res = np.dot(x, np.linalg.matrix_power(T, power))

        print "\nThe result is:\n\t", res
        print "\nWhich means that the active symbol will most likely be", self.get_node_from_id(np.argmax(res))._symbol

    def add_node(self, parent_node, new_node):
        try:
            if new_node._symbol in [s._symbol for s in self._nodes]:
                # Node already exists
                transition_added = False

                for t in self._transitions:
                    if t._from_node._symbol == parent_node._symbol and t._to_node._symbol == new_node._symbol:
                        # Transition between nodes already exists
                        t.new_occurence()
                        t._from_node.new_transition_from()

                        transition_added = True

                if not transition_added:
                    # Transition between nodes does not yet exist
                    new_transition = Transition(parent_node, new_node)
                    self._transitions.append(new_transition)
                    new_transition._from_node.new_transition_from()

                    transition_added = True
            else:
                # Node is new
                self._nodes.append(new_node)
                self._transitions.append(Transition(parent_node, new_node))
                parent_node.new_transition_from()
        except:
            print "Encontra la o bug"
            pdb.set_trace()

    def get_node_from_symbol(self, symbol):
        for s in self._nodes:
            if symbol == s._symbol:
                return s
        return False

    def get_node_from_id(self, id_):
        for s in self._nodes:
            if id_ == s._id:
                return s
        return False

    def sanity_check(self):
        l_sum = 0
        c_sum = 0

        T = self.compute_transition_matrix()

        for r in T:
            c_sum = 0
            for c in r:
                c_sum += c

            if (1.0 - c_sum) > 0.01:
                print "The math does not add up... Sum =", c_sum, "for line", r
            else:
                l_sum += 1

        if l_sum == T.shape[0]:
            print "Very bueno mathematiques!"
