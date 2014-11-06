#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 NLPY.ORG
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

from nlpy.util import NBestList
from nlpy.rep import Word2VecRepresentation
from nlpy.basic import DefaultTokenizer
from nlpy.ex.keyword import DefaultKeywordExtractor
from nlpy.basic import DefaultLemmatizer
from nlpy.basic import DefaultRecaser

class SemanticSearcher(object):

    def __init__(self, tokenizer=None, vec=None):
        self._vec = vec if vec else Word2VecRepresentation()
        self._tokenizer = tokenizer if tokenizer else DefaultTokenizer()
        self._kwex = DefaultKeywordExtractor()
        self._lem = DefaultLemmatizer()
        self._recaser = DefaultRecaser()
        self._data = []

    def load_data(self, data):
        """
        :type data: list of str
        """
        self._data = data

    def search(self, sent):
        """
        :type keywords: list of str
        :return: str
        """

        return self.searchMany(sent, N=1)[0][1]

    def searchMany(self, sent, N=100):
        """
        return n best results
        :param sent: query
        :rtype: list of (str, float)
        """
        nbest = NBestList(N)
        keywords = self._lemmatized_keywords(sent)[:5]
        print keywords

        # Find best sentence
        best_score = 0
        best_sent = ""
        for d in self._data:
            keywords_d = self._lemmatized_keywords(d)[:5]
            total_score = 0.0
            count = 0
            for kw_d, wt_d in keywords_d:
                for kw, wt in keywords:
                    if kw_d not in self._vec._model.vocab:
                        continue
                    total_score += self._vec.similarity(kw_d, kw) * wt * wt_d
                    count += 1
            if not count:
                continue
            average_score = total_score / count
            nbest.add(average_score, d)

        if nbest.is_empty():
            nbest.add(0, "")

        return nbest.get()


    def _lemmatized_keywords(self, sent):
        keywords = self._kwex.extract_weighted(
                map(self._recaser.recase,
                    map(self._lem.lemmatize,
                        map(str.lower,
                            self._tokenizer.tokenize(sent))))
            )
        return keywords



