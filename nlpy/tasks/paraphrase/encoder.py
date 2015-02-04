#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 NLPY.ORG
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import numpy as np

from sequentialize import CFGSequencer
from nlpy.basic import DefaultTokenizer
from nlpy.syntax import StanfordCFGParser
from nlpy.basic import DefaultRecaser

class ParaphraseEncoder(object):

    def __init__(self, network, vec=None, parser=None, pooling_size=15, regs_allowed=5):
        self._vec = vec
        self._network = network
        self._parser = StanfordCFGParser() if not parser else parser
        self._tokenizer = DefaultTokenizer()
        self._recaser = DefaultRecaser()
        self.pooling_size = pooling_size
        self.regs_allowed = regs_allowed

    def encode(self, text, tokenized=False):
        if tokenized:
            toks = text
        else:
            toks = self._tokenizer.tokenize(text)
        if len(toks) <= 1:
            return [self._get_word_vec(t) for t in toks]
        else:
            tree = self._parser.parse(toks)
            seq = list(CFGSequencer(tree))
            if max([x[2] for x in seq]) >= self.regs_allowed:
                return None
            token_data, seq_data = self._build_data(seq)
            return self._network.convert(token_data, seq_data)

    def _build_data(self, seq):
        tokens = []
        sequence = []
        max_reg = 0
        for left, right, target in seq:
            if type(left) != int:
                tokens.append(left[1:-1])
                left = - len(tokens)
            if type(right) != int:
                tokens.append(right[1:-1])
                right = - len(tokens)
            sequence.append((left, right, target))
            if max(left, right) > max_reg:
                max_reg = max(left, right)

        token_data = [np.zeros(300, dtype='float32')]
        for tok in tokens:
            tok_vec = self._get_word_vec(tok)
            token_data.append(tok_vec)
        return token_data, sequence

    def _get_word_vec(self, tok):
        tok = self._recaser.recase(tok)
        if tok not in self._vec._model.vocab:
            tok_vec = np.zeros(300, dtype='float32')
        else:
            tok_id = self._vec._model.vocab[tok].index
            tok_vec = self._vec._model.syn0norm[tok_id].astype('float32')
        return tok_vec

    def _distance(self, rep1, rep2):
        return np.sqrt(np.sum((rep1 - rep2)**2))

    def _min_block(self, matrix, x_begin, x_end, y_begin, y_end):
        min_value = matrix[x_begin][y_begin]
        for x in range(x_begin, x_end):
            for y in range(y_begin, y_end):
                val = matrix[x][y]
                if val < min_value:
                    min_value = val
        return min_value

    def dynamic_pool(self, reps1, reps2):
        # Initialize matrices
        sim_matrix = []
        for _, rep1 in enumerate(reps1):
            sims = []
            for _, rep2 in enumerate(reps2):
                sims.append(self._distance(rep1, rep2))
            sim_matrix.append(sims)
        pooling_matrix = []
        for _ in range(self.pooling_size):
            pooling_matrix.append([0]*self.pooling_size)
        # Pooling
        h_span = float(len(reps1)) / self.pooling_size
        v_span = float(len(reps2)) / self.pooling_size
        for i in range(self.pooling_size):
            for j in range(self.pooling_size):
                min_val = self._min_block(sim_matrix,
                                          int(i*h_span), int((i+1)*h_span),
                                          int(j*v_span), int((j+1)*v_span))
                pooling_matrix[i][j] = min_val
        return np.array(pooling_matrix)

    def make_pooling_matrix(self, text1, text2, reps1=None, reps2=None):
        toks1, toks2 = map(self._tokenizer.tokenize, (text1, text2))
        tok_reps1 = np.array(map(self._get_word_vec, toks1))
        tok_reps2 = np.array(map(self._get_word_vec, toks2))
        reps1 = self.encode(toks1, tokenized=True) if reps1 is None else reps1
        reps2 = self.encode(toks1, tokenized=True) if reps2 is None else reps2
        if reps1 == None or reps2 == None:
            return None
        pooling_matrix = self.dynamic_pool(np.concatenate((tok_reps1, reps1)), np.concatenate((tok_reps2, reps2)))
        return pooling_matrix

    def detect(self, text1, text2):
        return self.make_pooling_matrix(text1, text2)