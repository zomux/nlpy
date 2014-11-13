#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 NLPY.ORG
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import unittest

from nlpy.basic.contentfulness import ContentfullnessEstimator

class ContentfullnessEstimatorTest(unittest.TestCase):

    def test(self):
        ce = ContentfullnessEstimator()
        print ce.estimate(["China", "travel"])
        print ce.estimate(["country", "travel"])
        print ce.estimate(["yes", "know"])
        print ce.estimate(["cook", "cake"])