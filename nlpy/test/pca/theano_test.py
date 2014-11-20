#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 NLPY.ORG
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import unittest
import theano.tensor as T
import theano
import numpy as np

class TheanoTest(unittest.TestCase):

    def _test_function1(self):

        def f(x):
            return x ** 4 + 12 * x

        def u():
            print 1

        x = T.lscalar("x")
        func = theano.function([x], f(x), updates={theano.shared(1, "g"): f(x)})

        print func(1)
        print func(2)

    def test_function2(self):

        def f(x):
            return x ** 4 + 12 * x

        x = T.lscalar("x")
        k = f(x)
        y = f(k)

        func = theano.function([x], [k ,y])
        theano.printing.debugprint(func)

        print func(1)

    def _test_scan1(self):

        def scan_func(x, p1):
            print "A"
            return p1 + x

        x = T.lscalar("x")

        result, update = theano.scan(fn=scan_func, outputs_info=x, sequences=np.array([1,2,3]))
        print update
        func = theano.function(inputs=[x], outputs=result[-1], updates=update)

        print func(1)


    def _test_scan2(self):

        def step(a, b):
            return a + b, b

        h = T.lscalar("h")
        x = T.lscalar("x")


        [cui, bui], _ = theano.scan(step, sequences=np.array([1,2,3]), outputs_info=[theano.shared(value=0, name='W_in'), None])

        func = theano.function([], cui)
        print func()



if __name__ == '__main__':
    unittest.main()