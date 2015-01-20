#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 NLPY.ORG
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import unittest
from nlpy.syntax.cfg import StanfordCFGParser, BatchStanfordCFGParser
from nlpy.basic import DefaultTokenizer
import corenlp
from nlpy.util import LineIterator

class StanfordCFGParserTest(unittest.TestCase):


    def _test_parse(self):
        testcase = "One difference from C: I wrote a little wrapper around malloc/free, cymem."

        tk = DefaultTokenizer()

        p = StanfordCFGParser()

        tree = p.parse(tk.tokenize(testcase))
        print tree

    def test_bath_parse(self):
        tk = DefaultTokenizer()
        p = BatchStanfordCFGParser()
        testcases = ["it turns out good", "it will work (so it is)"]
        tokenized_cases = []
        for case in testcases:
            tokenized_cases.append(tk.tokenize(case))

        p.cache(tokenized_cases)
        p.save("/tmp/jjsjsj.gz")
        p.load("/tmp/jjsjsj.gz")
        print p.parse(tk.tokenize("it will work (so it is)"))

    def _test_terminals(self):
        testcase = "i cook rice."

        tk = DefaultTokenizer()
        p = StanfordCFGParser()
        tree = p.parse(tk.tokenize(testcase))
        print p.extract_terminals(tree)
