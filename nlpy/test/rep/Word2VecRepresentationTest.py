#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 NLPY.ORG
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import unittest
from nlpy.rep import Word2VecRepresentation

class Word2VecRepresentationTest(unittest.TestCase):

    def setUp(self):
        self.rep = Word2VecRepresentation()

    def test_find_similar_words(self):
        testcase = "China"
        most_similiar = self.rep.similar_words(testcase)[0]
        self.assertEqual(most_similiar, "Chinese")
