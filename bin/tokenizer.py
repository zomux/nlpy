#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 NLPY.ORG
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

from nlpy.basic import DefaultTokenizer
import sys, os

if __name__ == '__main__':
    tok = DefaultTokenizer()
    for l in sys.stdin.xreadlines():
        l = l.strip()
        print " ".join(tok.tokenize(l))