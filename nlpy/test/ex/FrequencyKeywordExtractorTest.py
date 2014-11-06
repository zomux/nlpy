#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 NLPY.ORG
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

from nlpy.ex.keyword import FrequencyKeywordExtractor

if __name__ == '__main__':
    testcase = "I used to live in Japan".split()
    ex = FrequencyKeywordExtractor()
    print ex.extract(testcase)