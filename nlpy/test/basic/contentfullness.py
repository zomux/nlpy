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

    def test_ranking(self):
        ce_f = ContentfullnessEstimator()
        ce_r = ContentfullnessEstimator(source='ranking')
        ce_s = ContentfullnessEstimator(source='/home/hadoop/works/chat/resources/topic_based_questions/just_questions.txt')
        testcases = ["hobby", "China", "Philippines"]
        for case in testcases:
            print case, ce_f.estimate_word(case), ce_r.estimate_word(case), ce_s.estimate_word(case)