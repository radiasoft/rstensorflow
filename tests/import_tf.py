# -*- coding: utf-8 -*-
#py.test tests/import_tensorflow.py
u"""PyTest for :mod:`rstensorflow.import_tf`

:copyright: Copyright (c) 2017 RadiaSoft LLC.  All Rights Reserved.
:license: http://www.apache.org/licenses/LICENSE-2.0.html
"""
import pytest

import tensorflow as tf

def test():
    hello = tf.constant('Hello, TensorFlow!')
    sess = tf.Session()
    print(sess.run(hello))

# Hello, TensorFlow!
#    assert 0
