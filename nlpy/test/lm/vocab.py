#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 NLPY.ORG
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

from nlpy.lm import Vocab
from nlpy.util import internal_resource
import unittest
from nlpy.lm.data_generator import RNNDataGenerator

class VocabTest(unittest.TestCase):

    def _test_vocab(self):
        data_path = internal_resource("lm_test/valid")
        v = Vocab()
        v.load(data_path)
        print v.size
        print v.binvector("ergerrghwegr")

    def test_generator(self):
        data_path = internal_resource("lm_test/valid")
        v = Vocab()
        v.load(data_path)
        c = 0
        g = RNNDataGenerator(v, data_path, history_len=0)
        for d in g:
            print d
            c += 1
            if c > 100:
                break