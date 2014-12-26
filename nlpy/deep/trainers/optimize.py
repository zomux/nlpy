#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 NLPY.ORG
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import theano
import theano.tensor as T
import numpy as np
from collections import OrderedDict
from nlpy.deep.functions import FLOATX


def optimize_parameters(params, gparams, shapes=None, max_norm = 5.0, lr = 0.01, eps= 1e-6, rho=0.95, method="ADADELTA",
                        beta=0.0):
    """
    Optimize by SGD, AdaGrad, or AdaDelta.
    Returns the shared variables for the gradient caches,
    and the updates dictionary for compilation by a
    theano function.

    :param params: parameters to optimize
    :param gparams: gradients
    :param max_norm: cap on excess gradients
    :param lr: base learning rate for adagrad and SGD
    :param eps: numerical stability value to not divide by zero sometimes
    :param rho: adadelta hyperparameter
    :param method: 'ADAGRAD', 'ADADELTA', or 'SGD'

    :returns updates: the updates to pass to theano function
    :returns gsums: gradient caches for Adagrad and AdaDelta
    :returns xsums: gradient caches for AdaDelta only
    :returns lr: theano shared : learning rate
    :returns max_norm theano_shared : normalizing clipping value for excessive gradients (exploding)
    """
    # lr = theano.shared(np.float64(lr).astype(FLOATX))
    if not shapes:
        shapes = params
    # eps = np.float64(eps).astype(FLOATX)
    # rho = np.float64(rho).astype(FLOATX)
    # if max_norm is not None and max_norm is not False:
    #     max_norm = theano.shared(np.float64(max_norm).astype(FLOATX))
    oneMinusBeta = 1 - beta



    gsums   = [theano.shared(np.zeros_like(param.get_value(borrow=True), dtype=FLOATX), name="gsum_%s" % param.name) if (method == 'ADADELTA' or method == 'ADAGRAD') else None for param in shapes]
    xsums   = [theano.shared(np.zeros_like(param.get_value(borrow=True), dtype=FLOATX), name="xsum_%s" % param.name) if method == 'ADADELTA' else None for param in shapes]

    # Fix for AdaGrad, init gsum to 1
    if method == 'ADAGRAD':
        for gsum in gsums:
            gsum.set_value(gsum.get_value() ** 0)

    updates = OrderedDict()

    for gparam, param, gsum, xsum in zip(gparams, params, gsums, xsums):
        # clip gradients if they get too big
        # if max_norm is not None and max_norm is not False:
        #     grad_norm = gparam.norm(L=2)
        #     gparam = (T.minimum(max_norm, grad_norm)/ grad_norm) * gparam

        if method == 'ADADELTA':
            updates[gsum] = rho * gsum + (1. - rho) * (gparam **2)
            dparam = -T.sqrt((xsum + eps) / (updates[gsum] + eps)) * gparam
            updates[xsum] =rho * xsum + (1. - rho) * (dparam **2)
            updates[param] = param * oneMinusBeta + dparam
        elif method == 'ADAGRAD':
            updates[gsum] =  gsum + (gparam ** 2)
            updates[param] =  param * oneMinusBeta - lr * (gparam / (T.sqrt(updates[gsum] + eps)))
        else:
            updates[param] = param * oneMinusBeta - gparam * lr

    return updates.items()
    #return updates, gsums, xsums, lr, max_norm