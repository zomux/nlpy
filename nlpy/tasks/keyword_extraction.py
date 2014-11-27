#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 NLPY.ORG
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

from nlpy.ex.keyword import DefaultKeywordExtractor
from nlpy.basic import DefaultLemmatizer
from nlpy.basic import DefaultRecaser
from nlpy.basic import DefaultTokenizer

class KeywordExtractor(object):

    def __init__(self):
        self._kwex = DefaultKeywordExtractor()
        self._lem = DefaultLemmatizer()
        self._recaser = DefaultRecaser()
        self._tokenizer = DefaultTokenizer()

    def extract(self, sent):
        keywords = self._kwex.extract(
                map(self._recaser.recase,
                    map(self._lem.lemmatize,
                        map(str.lower,
                            self._tokenizer.tokenize(sent))))
            )
        return keywords

    def extract_weighted(self, sent):
        keywords = self._kwex.extract_weighted(
                map(self._recaser.recase,
                    map(self._lem.lemmatize,
                        map(str.lower,
                            self._tokenizer.tokenize(sent))))
            )
        return keywords

    @staticmethod
    def serve(params):
        global keyword_extractor
        if "keyword_extractor" not in globals():
            keyword_extractor = KeywordExtractor()

        return {"output": keyword_extractor.extract(params['input'])}