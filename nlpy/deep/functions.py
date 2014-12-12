#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 NLPY.ORG
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import theano
import theano.tensor as T
# from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
import re
import copy

FLOATX = theano.config.floatX

global_rand = np.random.RandomState()

def make_float_matrices(*names):
    ret = []
    for n in names:
        ret.append(T.matrix(n, dtype=FLOATX))
    return ret

def make_float_vectors(*names):
    ret = []
    for n in names:
        ret.append(T.vector(n, dtype=FLOATX))
    return ret

def monitor_var(value, name="", disabled=False):
    if disabled:
        return value
    else:
        return theano.printing.Print(name)(value)

def back_grad(jacob, err_g):
    return T.dot(jacob, err_g)
    #return (jacob.T * err_g).T

def replace_graph(node, dct, deepcopy=True):
    """
    Replace nodes in a computational graph (Safe).
    """
    if not hasattr(node, 'owner'):
        return node
    if not hasattr(node.owner, 'inputs'):
        return node

    if deepcopy:
        new_node = copy.deepcopy(node)
        new_node.owner.inputs = copy.copy(node.owner.inputs)
        node = new_node

    owner = node.owner

    for i, elem in enumerate(owner.inputs):
        if elem in dct:
            owner.inputs[i] = dct[elem]
        else:
            owner.inputs[i] = replace_graph(elem, dct)
    return node

def build_node_name(n):
    if "owner" not in dir(n) or "inputs" not in dir(n.owner):
        return str(n)
    else:
        op_name = str(n.owner.op)
        if "{" not in op_name:
            op_name = "Elemwise{%s}" % op_name
        if "_" in op_name:
            op_name = re.sub(r"\{[^}]+_([^_}]+)\}", "{\\1}", op_name)
        return "%s(%s)" % (op_name, ",".join([build_node_name(m) for m in n.owner.inputs]))


def smart_replace_graph(n, dct, name_map=None, deepcopy=True):
    """
    Replace nodes in a computational graph (Smart).
    """
    if not name_map:
        name_map = {}
        for src, dst in dct.items():
            name_map[build_node_name(src)] = dst

    if not hasattr(n, 'owner'):
        return n
    if not hasattr(n.owner, 'inputs'):
        return n

    if deepcopy:
        new_node = copy.deepcopy(n)
        new_node.owner.inputs = copy.copy(n.owner.inputs)
        n = new_node

    owner = n.owner

    for i, elem in enumerate(owner.inputs):
        if elem in dct:
            owner.inputs[i] = dct[elem]
        elif build_node_name(elem) in name_map:
            owner.inputs[i] = name_map[build_node_name(elem)]
        else:
            owner.inputs[i] = smart_replace_graph(elem, dct, name_map, deepcopy)
    return n

class VarMap():

    def __init__(self):
        self.varmap = {}

    def __get__(self, instance, owner):
        if instance not in self.varmap:
            return None
        else:
            return self.varmap[instance]

    def __set__(self, instance, value):
        self.varmap[instance] = value

    def __contains__(self, item):
        return item in self.varmap

    def update_if_not_existing(self, name, value):
         if name not in self.varmap:
             self.varmap[name] = value

    def get(self, name):
        return self.varmap[name]

    def set(self, name, value):
        self.varmap[name] = value
