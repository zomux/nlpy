#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 NLPY.ORG
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

from networks.basic_nn import NeuralNetwork
import cPickle as pickle

def dump_network_params(network, path):
    pickle.dump(network.params(), open(path, 'w'))

def load_network_params(network, path):
    network.set_params(pickle.load(open(path)))