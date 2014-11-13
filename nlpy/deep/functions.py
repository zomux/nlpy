#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 NLPY.ORG
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import theano
import theano.tensor as T

def softmax(x):
    # TT.nnet.softmax doesn't work with the HF trainer.
    z = T.exp(x.T - x.T.max(axis=0))
    return (z / z.sum(axis=0)).T

FLOATX = theano.config.floatX