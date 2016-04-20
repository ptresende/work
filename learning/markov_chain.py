#! /usr/bin/env python

import pdb
# import matplotlib.pyplot as plt
import numpy as np
# import drawing_tools as dt
# import operator
# import copy
# import sys

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
    def __init__(self, root, second):
        self._nodes = []
        self._transitions = []

        self._nodes.append(root)
        self._nodes.append(second)

        self._transitions.append(Transition(root, second))
        root.new_transition_from()

        self._prob_thr = 0.45

    def print_chain(self):
        T = self.compute_transition_matrix()

        print "Nodes:", [n._symbol for n in self._nodes]

        if DEBUG:
            print "\nTransitions:", T

        self.get_likely_patterns(T)

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
                print "\tFrom " + str(from_node) + " to " + str(to_node) \
                    + " with probability " + str(v)

        print "\n"

    def compute_transition_matrix(self):
        num_node = len(self._nodes)
        T = np.zeros((num_node, num_node))

        for t in self._transitions:
            r = t._from_node._id
            c = t._to_node._id

            if DEBUG:
                print "Transition:"
                print "\tFrom:", t._from_node._symbol, \
                    "(ID =", str(t._from_node._id) + ")"
                print "\tTo:", t._to_node._symbol, \
                    "(ID =", str(t._to_node._id) + ")"
                print "\tOcurrences:", t._ocurrences
                print "\tTotal Ocurrences from Parent:", \
                    t._from_node._total_transitions_from

            T[r, c] = t._ocurrences / t._from_node._total_transitions_from

        return T

    def compute_distribution(self, T, x, time, time_to_test):
        if time == time_to_test:
            return np.dot(x, T)
        else:
            return self.compute_distribution(T, x, time, time_to_test - 1)

    def get_state_distribution(self, symbol, time_later):
        T = self.compute_transition_matrix()
        node = self.get_node_from_symbol(int(symbol))
        x = np.zeros(len(self._nodes))
        x[node._id] = 1.0

        power = time_later
        res = np.dot(x, np.linalg.matrix_power(T, power))

        print "\nThe result is:\n\t", res
        print "\nWhich means that the active symbol will most likely be", \
            self.get_node_from_id(np.argmax(res))._symbol

    def add_node(self, parent_node, new_node):
        try:
            if new_node._symbol in [s._symbol for s in self._nodes]:
                # Node already exists
                transition_added = False

                for t in self._transitions:
                    if t._from_node._symbol == parent_node._symbol and \
                       t._to_node._symbol == new_node._symbol:
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

    # def identify_patterns(self):
    #     print "\nIdentified Patterns for", self._name + ":"

    #     for p in self._tree:
    #         for c, v in p._children.iteritems():
    #             if v[1] >= self._thr:
    #                 print str(p._symbol) + " -> " + str(c) + "\n"
    #                 self._patterns[p._symbol] = c

    # def get_pattern_from_parent(self, parent):
    #     pattern = []
    #     max_val = self._thr
    #     max_symbol = -1

    #     parent = self.get_parent(parent)

    #     pattern.append(parent)

    #     if parent == None:
    #         print "The parent you specified does not exist."
    #         return
    #     else:
    #         print "The current parent (" + str(parent._symbol) \
    #    + ") will probably lead to the following pattern:"

    #     while 1:
    #         if parent == None:
    #             pattern.pop()
    #             break

    #         if parent._children:
    #             for c, v in parent._children.iteritems():
    #                 if v[1] >= max_val:
    #                     max_val = v[1]
    #                     max_symbol = c

    #             parent = self.get_parent(max_symbol)
    #             pattern.append(parent)

    #             max_val = self._thr
    #             max_symbol = -1
    #         else:
    #             break

    #     print [p._symbol for p in pattern]

    # def parent_in_tree(self, parent):
    #     for p in self._tree:
    #         if p._symbol == parent:
    #             return True

    #     return False

    # def get_parent(self, parent):
    #     for p in self._tree:
    #         if p._symbol == int(parent):
    #             return p

    # def update_all_children(self):
    #     for p in self._tree:
    #         p.update_children(self._rises)

    # def update(self, parent, child):
    #     self._rises += 1

    #     if self.parent_in_tree(parent):
    #         p = self.get_parent(parent)

    #         if p.has_child(child):
    #             p.new_child_occurence(child, self._rises)
    #         else:
    #             p.add_child(child, self._rises)
    #     else:
    #         p = Parent(parent)
    #         self._tree.append(p)
    #         p.add_child(child, self._rises)

    #     self.update_all_children()


