#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 NLPY.ORG
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


import unittest
from nlpy.util import FeatureContainer
from nlpy.util import internal_resource

class FeatureContainerTest(unittest.TestCase):


    def test(self):
        fc = FeatureContainer(internal_resource("dataset/heart_scale.txt"))
        print len(fc.data)
        print len(fc.targets)