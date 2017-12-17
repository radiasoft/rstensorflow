# -*- coding: utf-8 -*-
u"""PyTest for basic import of TensorFlow

:copyright: Copyright (c) 2017 RadiaSoft LLC.  All Rights Reserved.
:license: http://www.apache.org/licenses/LICENSE-2.0.html
"""
import os
import tensorflow as tf
import pytest

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def test():
    hello = tf.constant('Hello, TensorFlow!')
    sess = tf.Session()
    print(sess.run(hello))
