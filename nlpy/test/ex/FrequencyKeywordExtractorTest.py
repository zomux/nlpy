#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 NLPY.ORG
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

from nlpy.ex.keyword import FrequencyKeywordExtractor
import unittest

class FrequencyKeywordExtractorTest(unittest.TestCase):

    def test_extract(self):
        testcase = "I used to live in Japan".split()
        ex = FrequencyKeywordExtractor()
        self.assertEqual(ex.extract(testcase), ["Japan", "live"])

if __name__ == '__main__':
    unittest.main()

