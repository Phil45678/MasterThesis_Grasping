# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 15:29:06 2020

@author: norman marlier

"""
import unittest
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("../..")

from DexterousManipulation.model.grasp_pose_estimator import get_inputs_outputs, GraspPoseEstimator
from DexterousManipulation.controllers.grasp_planner import GraspPlanner
from DexterousManipulation.generations.hand_prior import HandPrior
from DexterousManipulation.generations.object_prior import ObjectPrior
from DexterousManipulation.generations.frame_prior import FramePrior
from DexterousManipulation.generations.sample_from_simulators import sample_obs


class TestGraspPlanner(unittest.TestCase):
    """Test Grasp planner."""

    def test_plan_mle(self):
        """Test plan function."""
        # Prior distrib
        hand_prior = HandPrior(-0.25, 0.25, -0.25, 0.25, 0.001, 0.25, 20)
        frame_prior = FramePrior(-0.25, 0.25, -0.4, 0.4)
        object_prior = ObjectPrior("../generations/shapenet_config.json")
        # Fix a object
        idx = 0
        obj = object_prior.get_obj(idx)
        # Fix frame
        true_frame = tf.zeros([1, 3])
        # Simulate an observation
        depth_img = sample_obs(obj, true_frame.numpy()[0]).reshape((480, 640, 1))
        depth_img = cv2.resize(depth_img, (128, 96),
                           interpolation=cv2.INTER_CUBIC).reshape((1, 96, 128, 1))
        
        # Model
        in_net, out_net = get_inputs_outputs()
        grasp_pose_estimator = GraspPoseEstimator(in_net, out_net)
        # Grasp Planner
        gp = GraspPlanner(grasp_pose_estimator, hand_prior, frame_prior)
        
        # Runs
        h, xO = gp.plan_mle([tf.ones([1, 1]), depth_img])
        print("h", h)
        print("xO", xO)
    
    def test_plan_map(self):
        """Test plan_h function."""
        # Prior distrib
        hand_prior = HandPrior(-0.25, 0.25, -0.25, 0.25, 0.001, 0.25, 20)
        frame_prior = FramePrior(-0.25, 0.25, -0.4, 0.4)
        object_prior = ObjectPrior("../generations/shapenet_config.json")
        # Fix a object
        idx = 0
        obj = object_prior.get_obj(idx)
        # Fix frame
        true_frame = tf.zeros([1, 3])
        # Simulate an observation
        depth_img = sample_obs(obj, true_frame.numpy()[0]).reshape((480, 640, 1))
        depth_img = cv2.resize(depth_img, (128, 96),
                           interpolation=cv2.INTER_CUBIC).reshape((1, 96, 128, 1))
        
        # Model
        in_net, out_net = get_inputs_outputs()
        grasp_pose_estimator = GraspPoseEstimator(in_net, out_net)
        # Grasp Planner
        gp = GraspPlanner(grasp_pose_estimator, hand_prior, frame_prior)
        
        # Runs
        h = gp.plan_map([tf.ones([1, 1]), depth_img])
        print("h", h)


if __name__ == "__main__":
    unittest.main()
        

