#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 NLPY.ORG
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


class NetworkConfig(object):

    def __init__(self):
        self.layers = [3,2,1]
        self.tied_weights = False
        self.decode_from = 1
        self.no_learn_biases = False

        # Activation
        self.hidden_activation = "sigmoid"
        self.output_activation = "softmax"

        # Noise
        self.input_noise = 0.
        self.input_dropouts = 0.
        self.hidden_noise = 0.
        self.hidden_dropouts = 0.