#! /usr/bin/env python

import pdb

import matplotlib.pyplot as plt
import numpy as np
import drawing_tools as dt
import operator
import copy



class Parent:
    def __init__(self, symbol):
        self._symbol = symbol
        self._children = {}

    def print_parent(self):
        print "\tParent:", self._symbol
        print "\t\tChildren:", self._children

    def add_child(self, child, rises):
        self._children[child] = [1.0, 1.0 / rises]

    def new_child_occurence(self, child, rises):
        self._children[child][0] += 1.0
        self._children[child][1] = self._children[child][0] / rises

    def has_child(self, child):
        if child in self._children:
            return True
        else:
            return False

    def update_children(self, rises):
        for c, v in self._children.iteritems():
            v[1] = v[0] / rises


class Tree:
    def __init__(self, name):
        self._tree = []
        self._rises = 0
        self._name = name
        self._patterns = {}
        self._thr = 0.05

    def print_tree(self):
        print self._name + ":"

        for p in self._tree:
            p.print_parent()

    def sanity_check(self):
        p_sum = 0

        for parent in self._tree:
            for c, v in parent._children.iteritems():
                p_sum += v[1]

        if p_sum != 1.0:
            print "The math does not add up... Sum =", p_sum
        else:
            print "Very bueno mathematiques!"

    def identify_patterns(self):
        print "\nIdentified Patterns for", self._name + ":"

        for p in self._tree:
            for c, v in p._children.iteritems():
                if v[1] >= self._thr:
                    print str(p._symbol) + " -> " + str(c) + "\n"
                    self._patterns[p._symbol] = c

    def get_pattern_from_parent(self, parent):
        pattern = []
        max_val = self._thr
        max_symbol = -1

        parent = self.get_parent(parent)

        pattern.append(parent)

        if parent == None:
            print "The parent you specified does not exist."
            return
        else:
            print "The current parent (" + str(parent._symbol) + ") will probably lead to the following pattern:"

        while 1:
            if parent == None:
                pattern.pop()
                break

            if parent._children:
                for c, v in parent._children.iteritems():
                    if v[1] >= max_val:
                        max_val = v[1]
                        max_symbol = c

                parent = self.get_parent(max_symbol)
                pattern.append(parent)

                max_val = self._thr
                max_symbol = -1
            else:
                break

        print [p._symbol for p in pattern]

    def parent_in_tree(self, parent):
        for p in self._tree:
            if p._symbol == parent:
                return True

        return False

    def get_parent(self, parent):
        for p in self._tree:
            if p._symbol == int(parent):
                return p

    def update_all_children(self):
        for p in self._tree:
            p.update_children(self._rises)

    def update(self, parent, child):
        self._rises += 1

        if self.parent_in_tree(parent):
            p = self.get_parent(parent)

            if p.has_child(child):
                p.new_child_occurence(child, self._rises)
            else:
                p.add_child(child, self._rises)
        else:
            p = Parent(parent)
            self._tree.append(p)
            p.add_child(child, self._rises)

        self.update_all_children()


