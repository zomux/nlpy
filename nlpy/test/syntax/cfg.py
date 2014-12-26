#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 NLPY.ORG
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import unittest
from nlpy.syntax.cfg import StanfordCFGParser
from nlpy.basic import DefaultTokenizer
import corenlp

class StanfordCFGParserTest(unittest.TestCase):


    def test_parse(self):
        testcase = "I cook rice"

        tk = DefaultTokenizer()
        p = StanfordCFGParser()
        tree = p.parse(tk.tokenize(testcase))
        print tree

    def test_terminals(self):
        testcase = "i cook rice."

        tk = DefaultTokenizer()
        p = StanfordCFGParser()
        tree = p.parse(tk.tokenize(testcase))
        print p.extract_terminals(tree)
