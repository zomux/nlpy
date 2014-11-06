#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 NLPY.ORG
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import sys

if sys.version_info[:2] < (2, 6):
    raise Exception('This version of gensim needs Python 2.6 or later. ')


from distutils.core import setup

setup(
    name='nlpy',
    version='0.0.1',
    description='Natural Language Processing on Python',

    author='Raphael Shu',
    author_email='raphael1@uaca.com',

    url='http://nlpy.org',
    download_url='http://pypi.python.org/pypi/nlpy',

    keywords=' Natural Language Processing '
        ' Natural Language Understanding '
        ' Semantic Representation '
        ' Machine Translation ',

    license='LGPL',
    platforms='any',

    zip_safe=False,

    classifiers=[ # from http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Text Processing :: Linguistic',
    ],

    setup_requires = [
        'numpy >= 1.3',
        'gensim >= 0.10.0'
    ],
    install_requires=[
        'numpy >= 1.3',
        'gensim >= 0.10.0'
    ],

    extras_require={
    },

    include_package_data=True,
)