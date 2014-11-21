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

    def _test_function2(self):

        def f(x):
            return x ** 4 + 12 * x

        x = T.lscalar("x")
        k = f(x)
        y = f(k)

        func = theano.function([x], [k ,y])
        theano.printing.debugprint(func)

        print func(1)

    def _test_function3(self):

        h1 = theano.shared(1.0, name="h1")
        w = theano.shared(1.0, name="w")
        h2 = theano.shared(2.0, name="h2")
        x1 = theano.shared(3.0, name="x1")
        x = T.dscalar("x")

        h = (x1 * h2 * w) + (x * h1 * w)

        err = (h**2 - 200)**2
        grad = T.grad(err, w)

        func = theano.function(inputs=[x], outputs=[h**2, h, grad], updates=[(x1, x), (h2, h1), (h1, h), (w, w-grad)])

        for k in [1.0, 2.0]:
            print "w", w.get_value()
            print "f", func(k)
            print "x1", x1.get_value()
            print "h1", h1.get_value()
            print "h2", h2.get_value()
            print "w", w.get_value()

    def _test_function4(self):

        w = theano.shared(1.0, name="w")

        w1 = theano.shared(1.0, name="w1")
        w2 = theano.shared(1.0, name="w2")
        w3 = theano.shared(1.0, name="w3")


        def joke(a, b, c):
            k = a + c[a] * b
            return k

        x = T.dscalar("x")
        hs, _ = theano.scan(joke, sequences=[np.array([0.0, 1.0, 2.0])], outputs_info=[np.float64(1.0)],
                            non_sequences=[[w1, w2, w3]] )
        print hs, _
        g1 = 0
        g2 = T.grad(hs[-1], w1)
        # g, _ = theano.map(lambda c, d: T.grad(c, d), sequences=hs, non_sequences=x)
        # g = theano.map(lambda c, d: T.grad(c, d), sequences=hs, non_sequences=[w])
        # g = T.grad(hs, [x])
        # g = T.grad(hs[-1], x)

        func = theano.function(inputs=[], outputs=[hs[0], hs[-1],  g2], updates=[])
        print func()
        print w.get_value()

    def test_function5(self):

        w = theano.shared(1.0, name="w")


        def joke(a, b):
            k = w * a
            # g = 0.01 * T.grad((k - 1)**2, w)
            return k, {w: w - 1.0}

        x = T.dscalar("x")
        hs, _ = theano.scan(joke, sequences=[np.array([1.0, 2.0, 3.0])], outputs_info=[np.float64(1.0)] )

        print hs, _

        def upd(h):
            return T.grad(hs[h], w)

        gs, up = theano.map(upd, sequences=[T.arange(hs.shape[0])])
        print gs, up
        # print hs, _
        # print gs, up

        func = theano.function(inputs=[], outputs=gs, updates= [])
        print func()
        print w.get_value()

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