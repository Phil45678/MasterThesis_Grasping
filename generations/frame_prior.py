# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 18:14:39 2020

@author: norman marlier
"""
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


class FramePrior():
    """Prior distribution for the object frame, p(pO)
    
    The frame pO is defined as (x, y, rz).
    (x, y) is the position on the table and rz is the rotation around
    the vertical axis.
    """
    
    def __init__(self, xmin, xmax, ymin, ymax):
        """Constructor."""
        # R2 random variables x-y
        # Parameters of the distribution
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

        # Distribution
        self.pose_distrib = tfp.distributions.Uniform([self.xmin, self.ymin, -np.pi],
                                                  [self.xmax, self.ymax, np.pi])
        self.joint_distribution = tfp.distributions.Independent(self.pose_distrib, reinterpreted_batch_ndims=1)
    
    @tf.function
    def sample_from_joint(self, n):
        return self.joint_distribution.sample(n)

    @staticmethod
    def from_network(frame):
        """Change (cos, sin) representation to real representation."""
        pos = frame[:, 0:2]
        angle = tf.math.atan2(frame[:, 3:4], frame[:, 2:3])
        frame = tf.concat([pos, angle], axis=-1)
        return frame
    
    @staticmethod
    def to_network(frame):
        """Change real representation to (cos, sin) representation."""
        cos_angle = tf.math.cos(frame[:, 2:])
        sin_angle = tf.math.sin(frame[:, 2:])
        return tf.concat([frame[:, 0:2], cos_angle, sin_angle], axis=-1)

