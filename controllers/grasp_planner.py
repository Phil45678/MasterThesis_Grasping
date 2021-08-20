# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 15:06:58 2020

@author: norman marlier

Gras planner
"""
import tensorflow as tf
import numpy as np
import sys
sys.path.append("../..")
from pymanopt.solvers import SteepestDescent, ConjugateGradient
from pymanopt.solvers.conjugate_gradient import BetaTypes
from pymanopt import Problem

from DexterousManipulation.optimization.cost_function import MAPCost, MLECost
from DexterousManipulation.optimization.manifold_function import HandFunction, PoseFunction, Pose4DFunction
from DexterousManipulation.optimization.searching import find_initial_point


class GraspPlanner():
    """Grasp planner.

    The grasp planner solves the equation
    h = argmax f(Sg, Sr, i, h)
    posterior = r(Sg=1, Sr=1, i|h)p(h)
    likelihood = r(Sg=1, Sr=1, i|h)
    """
    def __init__(self, gp_estimator, hand_prior):
        """Constructor."""
        # Model
        self.gp_estimator = gp_estimator
        # Prior distributions
        self.hand_prior = hand_prior
        # Cost functions
        self.map_fun = MAPCost(None, None, None)
        self.mle_fun = MLECost(None, None)
        # Manifold functions
        self.hand_fun = HandFunction(self.map_fun, None)
        # Grasp type
        if len(self.hand_prior.preshape_probs) == 1:
            self.grasp_type = [tf.constant([[1.]])]
        else:
            self.grasp_type = (tf.constant([[1., 0., 0.]]),
                               tf.constant([[0., 1., 0.]]),
                               tf.constant([[0., 0., 1.]]))

    def plan_grasp(self, img, fun_type="map", nb_pts=10, optimizer=True,
                   verbosity=0):
        """Find the hand configuration given an image.

        h = arg max f <=> h = arg min -log f
        """
        if fun_type == "map":
            cost_fun = self.map_fun
        else:
            cost_fun = self.mle_fun
        # Update hand function
        if img is None:
            m_params = [tf.ones([nb_pts, 1]), tf.ones([nb_pts, 1])]
            params = [tf.ones([1, 1]), tf.ones([1, 1])]
        else:
            m_params = [tf.ones([nb_pts, 1]), tf.ones([nb_pts, 1]), tf.tile(img, [nb_pts, 1, 1, 1])]
            params = [tf.ones([1, 1]), tf.ones([1, 1]), img]
        state = {"cost_fun": cost_fun,
                 "fun": {"model": self.gp_estimator, "prior": self.hand_prior}}
        self.hand_fun.update_state(state)
        
        # Initial point : x, R ~ p(x)p(R|x)
        samples = self.hand_prior.joint_distribution.sample(nb_pts)
        samples = self.hand_prior.to_network(samples)
        
        # Iterate over the grasp type
        min_val = np.inf
        for g in self.grasp_type:
            # Update hand mode and multiples inputs
            self.hand_fun.update_state({"hand_mode": g,
                                        "fun": {"x": m_params}})

            # Concatenate x, R and g
            candidates = tf.concat([samples[:, 0:12], tf.tile(g, [nb_pts, 1])],
                                   axis=-1)
            # Find good initial point
            init_point = find_initial_point(self.hand_fun.fun, candidates).numpy()
            
            # Update parameters - only one sample
            state = {"fun": {"x": params}}
            self.hand_fun.update_state(state)
            # Use optimization
            if optimizer:
                # Create the problem
                problem = Problem(manifold=self.hand_fun.get_manifold(),
                                  cost=self.hand_fun.get_cost(),
                                  egrad=self.hand_fun.get_egrad(),
                                  verbosity=verbosity)
                # Solver
                solver = ConjugateGradient(BetaTypes.PolakRibiere,
                                           logverbosity=verbosity,
                                           maxiter=15,
                                           maxtime=3)
                x_init = [init_point[0, 0:3], init_point[0, 3:12].reshape((3, 3))]
                # Optimize it
                if verbosity == 2:
                    Opt, opt_log = solver.solve(problem, x=x_init)
                else:
                    Opt = solver.solve(problem, x=x_init)
                    opt_log = None
            else:
                # Initial point is the solution
                Opt = [init_point[0, 0:3], init_point[0, 3:12].reshape((3, 3))]
                opt_log = None
            # Better solution
            c = self.hand_fun.get_cost()(Opt[0], Opt[1])
            # Can be outside of the domain
            if c == np.nan:
                Opt = [init_point[0, 0:3], init_point[0, 3:12].reshape((3, 3))]
                # Better solution
                c = self.hand_fun.get_cost()(Opt[0], Opt[1])
            
            if c <= min_val:
                # Reconstruct the solution
                h_x = tf.reshape(tf.convert_to_tensor(Opt[0], dtype=tf.float32), [1, -1])
                h_R = tf.reshape(tf.convert_to_tensor(Opt[1], dtype=tf.float32), [1, -1])
                Hopt = tf.concat([h_x, h_R, g], axis=1)
                # Update min val
                min_val = c
            
            
        return [Hopt, opt_log]


class PosePlanner():
    """Pose planner.

    The pose planner solves the equations
    pO = argmax r(i|pO)p(pO)
    pO, z = argmax r(i|pO, z)p(pO)p(z)
    """
    def __init__(self, pose_estimator, frame_prior, pose3d):
        """Constructor."""
        # Model
        self.pose_estimator = pose_estimator
        # Prior distributions
        self.frame_prior = frame_prior
        # Cost functions
        self.map_fun = MAPCost(None, None, None)
        self.mle_fun = MLECost(None, None)
        # Manifold functions
        self.pose3d = pose3d
        if pose3d:
            self.pose_fun = PoseFunction(self.map_fun, None)
        else:
            self.pose_fun = Pose4DFunction(self.map_fun, None)

    
    def estimate_pose(self, img, fun_type="map", nb_pts=10, optimizer=True,
                      verbosity=0):
        """Estimate the position of the object.

        pO = arg max f <=> pO = arg min - log f
        
        pO, z = arg max f <=> pO, z = arg min -log f
        """
        if fun_type == "map":
            cost_fun = self.map_fun
        else:
            cost_fun = self.mle_fun
        # Update pose function
        m_params = [tf.tile(img, [nb_pts, 1, 1, 1])]
        params = [img]
        state = {"cost_fun": cost_fun,
                 "fun": {"model": self.pose_estimator,
                         "prior": self.frame_prior,
                         "x": m_params}}
        self.pose_fun.update_state(state)
        # Initial point
        samples = self.frame_prior.joint_distribution.sample(nb_pts)
        samples = self.frame_prior.to_network(samples)
        # Find good initial point
        init_point = find_initial_point(self.pose_fun.fun, samples)
        init_pO = init_point[0]
        cos_phi = init_pO[0, 2]
        sin_phi = init_pO[0, 3]
        R2 = np.array([[cos_phi, -sin_phi],
                       [sin_phi, cos_phi]])
        if self.pose3d:
            t = np.array([init_pO[0, 0], init_pO[0, 1]])
        else:
            t = np.array([init_pO[0, 0], init_pO[0, 1], init_point[1][0, 0]])
        # Use optimization
        if optimizer:
            # Update parameters - only one sample
            state = {"fun": {"x": params}}
            self.pose_fun.update_state(state)
            # Create the problem
            problem = Problem(manifold=self.pose_fun.get_manifold(),
                              cost=self.pose_fun.get_cost(),
                              egrad=self.pose_fun.get_egrad(),
                              verbosity=verbosity)
            # Solver
            solver = ConjugateGradient(BetaTypes.PolakRibiere,
                                       logverbosity=verbosity,
                                       maxiter=20,
                                       maxtime=2)
            x_init = [t, R2]
            # Optimize it
            if verbosity == 2:
                Opt, opt_log = solver.solve(problem, x=x_init)
            else:
                Opt = solver.solve(problem, x=x_init)
                opt_log = None
            # Reconstruct the solution
            if self.pose3d:
                t = tf.reshape(tf.convert_to_tensor(Opt[0], dtype=tf.float32),
                               [1, -1])
                R = tf.reshape(tf.convert_to_tensor(Opt[1], dtype=tf.float32),
                               [1, -1])[0:1, 0:3:2]
                Popt = tf.concat([t, R], axis=1)
            else:
                xy = tf.reshape(tf.convert_to_tensor(Opt[0][0:2], dtype=tf.float32),
                               [1, -1])
                R = tf.reshape(tf.convert_to_tensor(Opt[1], dtype=tf.float32),
                               [1, -1])[0:1, 0:3:2]
                z = tf.reshape(tf.convert_to_tensor(Opt[0][2], dtype=tf.float32),
                               [1, -1])
                Popt = [tf.concat([xy, R], axis=1), z]
        else:
            opt_log = None
            # Reconstruct the solution
            if self.pose3d:
                Popt = tf.constant(init_point)
            else:
                Popt = [tf.constant(init_point[0]), tf.constant(init_point[1])]

        return [Popt, opt_log]


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import sys
    sys.path.append("../..")
    from DexterousManipulation.model.object_model import ObjectDataset
    from DexterousManipulation.model.height_object_prior import PriorPose4D, height_prior
    from DexterousManipulation.model.ensemble import EnsembleModel
    from DexterousManipulation.generations.frame_prior import FramePrior
    from DexterousManipulation.generations.sample_from_simulators import generate_image_sample
    from DexterousManipulation.simulations.obs_simulator import KinectSimulator, DepthSimulator
    from DexterousManipulation.simulations.grasp_simulator import GraspSimulator
    from DexterousManipulation.simulations.utils_sim import get_background
    from DexterousManipulation.training.dataset import filter_img_tf
    from DexterousManipulation.generations.object_prior import ObjectPrior
    obj_prior = ObjectPrior("../generations/ds_obj_config.json")
    obj = obj_prior.get_obj(2)
    print("Heigh of the object:", obj["z"])
    # Height prior
    prior_z = height_prior(obj_prior.object_dataset)
    # Frame prior
    pO = FramePrior(-0.2, 0.2, -0.35, 0.35)
    # 4D pose prior
    prior4D = PriorPose4D(prior_z, pO)
    
    # Model
    p_model_file = "../results/p_ur5_model"
    if p_model_file is None:
        model_p = None
    else:
        if "min_val" in p_model_file:
            files = [p_model_file]
        else:
            files = [p_model_file + "/model_" + str(i) + "/min_val" for i in range(1, 5)]
        model_p = EnsembleModel(model_files=files)
    pp = PosePlanner(model_p, prior4D, False)
    
    import pybullet as pb
    # Create the Obs Simulator
    # Background
    background = get_background()
    # Simulator
    obs_sim = KinectSimulator(pb, background, (480, 640))
    #obs_sim = DepthSimulator(pb, background, (480, 640))
    # Img
    pO_init = [0., 0., -np.pi/2]
    inputs_sim = [obj, pO_init]
    img = obs_sim.sample(inputs_sim, True, True, engine_connection="GUI").reshape((1, 480, 640, 1))
    img = filter_img_tf(img)
    
    X = pp.estimate_pose(img, nb_pts=1000, optimizer=True)
    
    print("X", X)
    
    
    