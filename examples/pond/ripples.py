# -*- coding: utf-8 -*-
u"""TensorFlow example: simulating ripples on a pond

Taken from TensorFlow tutorials, https://www.tensorflow.org/tutorials/pdes

Regarding IP,
    Except as otherwise noted, the content of this page is licensed 
    under the Creative Commons Attribution 3.0 License, and code 
    samples are licensed under the Apache 2.0 License. For details, 
    see our Site Policies. Java is a registered trademark of Oracle 
    and/or its affiliates.

The original code from TensorFlow has been modified.
In particular, the plotting is significantly different.

:license: http://www.apache.org/licenses/LICENSE-2.0.html
"""
from __future__ import absolute_import, division, print_function

#Imports for data and computation
import tensorflow as tf
import numpy as np

#Imports for visualization
from matplotlib import pyplot as plt
from matplotlib import animation

#Prevent complaints about avx instruction sets
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Define the PDE
# Our pond is a 2D square with n_cell * n_cell cells.
n_cell = 128

# Number of iterations to skip between frames
n_skip = 10

# The total number of frames to plot
n_plot = 20

# Number of iterations of the PDE
n_iter = n_skip * n_plot

# Define the limits of the simulation domain
x_min = -0.1
x_max =  0.1
length = x_max - x_min

# Define functions required for the dynamics
def make_kernel(_a):
  """Transform a 2D array into a convolution kernel"""
  _a = np.asarray(_a)
  _a = _a.reshape(list(_a.shape) + [1,1])
  return tf.constant(_a, dtype=1)

def simple_conv(_x, _k):
  """A simplified 2D convolution operation"""
  _x = tf.expand_dims(tf.expand_dims(_x, 0), -1)
  _y = tf.nn.depthwise_conv2d(_x, _k, [1, 1, 1, 1], padding='SAME')
  return _y[0, :, :, 0]

def laplace(_x):
  """Compute the 2D laplacian of an array"""
  laplace_k = make_kernel([[0.5, 1.0, 0.5],
                           [1.0, -6., 1.0],
                           [0.5, 1.0, 0.5]])
  return simple_conv(_x, laplace_k)

# Initial Conditions -- some rain drops hit a pond

# Set everything to zero
u_init = np.zeros([n_cell, n_cell], dtype=np.float32)
v_init = np.zeros([n_cell, n_cell], dtype=np.float32)

# Some rain drops hit a pond at random points
for n in range(int(7+n_cell/20)):
  array_a, array_b = np.random.randint(0, n_cell, 2)
  u_init[array_a, array_b] = np.random.uniform()

u_min = np.min(np.min(u_init, axis=1), axis=0)
u_max = np.max(np.max(u_init, axis=1), axis=0)

# Now let's specify the details of the differential equation.

# Parameters:
# eps -- time resolution
# damping -- wave damping
eps = tf.placeholder(tf.float32, shape=())
damping = tf.placeholder(tf.float32, shape=())

# Create variables for simulation state
U = tf.Variable(u_init)
V = tf.Variable(v_init)

# Discretized PDE update rules
U_ = U + eps * V
V_ = V + eps * (laplace(U) - damping * V)

# Operation to update the state
step = tf.group(
  U.assign(U_),
  V.assign(V_))

# Initialize state to initial conditions
sess = tf.Session()
with sess.as_default():
  tf.global_variables_initializer().run()

# simulate the ripples on the pond
u_plot = [u_init]
for i in range(n_plot):
  with sess.as_default():
    for j in range(n_skip):
      step.run({eps: 0.03, damping: 0.035})
    u_plot.append(U.eval())

# Now work on generating the animation

# First set up the figure, the axis, and the plot element we want to animate
my_figure = plt.figure()

# ax = my_figure.add_subplot(111)
# delta = length / (n_cell)
# x = y = np.arange(x_min, x_max, delta)
# X, Y = np.meshgrid(x, y)

# calculate the min and max values
for i in range(n_plot):
  tmp_min = np.min(np.min(u_plot[i], axis=1), axis=0)
  tmp_max = np.max(np.max(u_plot[i], axis=1), axis=0)
  u_min = min(u_min, tmp_min)
  u_max = max(u_max, tmp_max)

# initialization function: plot the background of each frame
def init():
  my_image = plt.imshow(u_plot[0], interpolation='bilinear',
                        extent=[x_min,x_max,x_min,x_max],
                        vmax=u_max, vmin=u_min)
  return my_image,

# create each new frame
def update(frame):
  my_image = plt.imshow(u_plot[frame], interpolation='bilinear',
                        extent=[u_min,u_max,u_min,u_max],
                        vmax=u_max, vmin=u_min)
  return my_image,

# create the animation.
#    blit=True means only redraw pixels that change
anim = animation.FuncAnimation(my_figure, update, init_func=init,
                               frames=n_plot, interval=1, blit=True)

# save the animation as mp4
anim.save('pond.mp4', fps=30, writer="ffmpeg", extra_args=['-vcodec', 'libx264'])

# render the animation
plt.show()
