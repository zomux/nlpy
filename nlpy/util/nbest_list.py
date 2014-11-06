#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 NLPY.ORG
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

class NBestList(object):

    def __init__(self, N, reverse=False):
        self.N = N
        self._list = []
        self._count = 0
        self._reverse = reverse

    def add(self, score, item):
        """
        Add an item with score.
        :param item: item
        :param score: score
        """
        self._list.append((score, item))
        self._count += 1
        if len(self._list) > self.N and len(self._list) > self._count / 10:
            self._list.sort(reverse=not self._reverse)
            del self._list[self.N: len(self._list)]

    def get(self):
        """
        Get result.
        :rtype: list of (float, score)
        """
        self._list.sort(reverse=not self._reverse)
        if len(self._list) > self.N:
            del self._list[self.N: len(self._list)]
        return self._list

    def get_copy(self):
        """
        Get a copy of result.
        :rtype: list of (float, score)
        """
        return list(self.get())

    def is_empty(self):
        """
        Is empty.
        :rtype: bool
        """
        return len(self._list) == 0
