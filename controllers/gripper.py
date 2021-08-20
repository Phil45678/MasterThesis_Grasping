# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 12:39:35 2020

@author: norman marlier
"""
import pybullet as pb
import numpy as np

from scipy.spatial.transform import Rotation as R


class Gripper():
    """Abstract class for gripper in pybullet."""
    
    def __init__(self, gripper_filename, *args, **kwargs):
        
        # General variables
        self.args = args
        self.kwargs = kwargs
        
        # Gripper URDF 
        self.gripper_filename = gripper_filename
    
    def load(self, parameters: dict, engine, id_server=None):
        """Load the gripper."""
        base_ori = R.from_quat(parameters["base_ori"])*self.gripper_inv
        gripper_id = engine.loadURDF(self.gripper_filename,
                                     basePosition=parameters["base_pos"],
                                     baseOrientation=base_ori.as_quat(),
                                     useFixedBase=parameters["fixed_base"],
                                     flags=engine.URDF_USE_SELF_COLLISION,
                                     globalScaling=1,
                                     physicsClientId=id_server)
        self.gripper_id = gripper_id
        
        # Filter the collisions
        self.filter_collision(engine, parameters, id_server=id_server)
    
    def get_transform_eef(self):
        """Return the transfrom to align with eef frame."""
        raise NotImplementedError
    
    def filter_collision(self, engine, parameters: dict, id_server=None):
        """Filter the collision due to the self collision flag."""
        raise NotImplementedError
    
    def get_mode(self, mode):
        """Return the mode as string for the controller."""
        raise NotImplementedError


class Robotiq3fGripper(Gripper):
    """Robotiq3f Gripper - not attach to a robot"""
    def __init__(self, gripper_filename, *args, **kwargs):
        # Parents
        super(Robotiq3fGripper, self).__init__(gripper_filename, *args, **kwargs)
        
        self.gripper_mode = {"basic": 0, "pinch": 1, "wide": 2}
        self.gripper_inv = R.from_quat([0., 0., 0., 1.]).inv()
    
    def get_transform_eef(self):
        
        Tg = [0, 0, 0]
        Qg = [0, 0, -np.sin(np.pi/4), np.cos(np.pi/4)]
        #Qg = [0., 0., 0., 1.]
        
        return Tg, Qg
    
    def filter_collision(self, engine, parameters: dict, id_server=None):
        
        # Delete self collision between fingers of the gripper and the palm
        f1 = [-1, -1, 0, 1, 1, 2, -1, -1, 4, 5, 5, 6, -1, -1, 8,  9, 9, 10]
        f2 = [ 0,  1, 1, 2, 3, 3,  4,  5, 5, 6, 7, 7,  8,  9, 9, 10, 11, 11]
        enableCollision = 0
        
        for f_i, f_j in zip(f1, f2):
            engine.setCollisionFilterPair(self.gripper_id, self.gripper_id,
                                          f_i, f_j, enableCollision)
        
        # Filter the collision of a specific link because it goes outside of
        # the palm
        engine.setCollisionFilterPair(self.gripper_id, parameters["plane_id"],
                                      4, -1, enableCollision)
        engine.setCollisionFilterPair(self.gripper_id, parameters["plane_id"],
                                      -1, -1, 1)
    
    def get_mode(self, mode_index):

        for mode, index in self.gripper_mode.items():
            if index == mode_index:
                return mode


class SawyerElectricGripper(Gripper):
    """Sawyer Elctric Gripper - not attach to a robot"""
    def __init__(self, gripper_filename, *args, **kwargs):
        # Parents
        super(SawyerElectricGripper, self).__init__(gripper_filename, *args, **kwargs)
        
        self.gripper_mode = {"basic": 0}
        
        self.gripper_inv = R.from_euler("xyz", [-np.pi/2, np.pi, 0.]).inv()
    
    def get_transform_eef(self):
        
        Tg = [0, 0, 0]
        Qg1 = R.from_quat([0, 0, np.sin(-np.pi/4), np.cos(-np.pi/4)])
        Qg2 = R.from_quat([np.sin(np.pi/4), 0., 0., np.cos(np.pi/4)])
        Qg = Qg2*Qg1
        
        return Tg, Qg.as_quat()
    
    def filter_collision(self, engine, parameters: dict, id_server=None):
        
        # Delete self collision between fingers
        f1 = [1]
        f2 = [3]
        enableCollision = 0
        
        for f_i, f_j in zip(f1, f2):
            engine.setCollisionFilterPair(self.gripper_id, self.gripper_id,
                                          f_i, f_j, enableCollision)
        
    
    def get_mode(self, mode_index):

        for mode, index in self.gripper_mode.items():
            if index == mode_index:
                return mode
        
        
    
    
        

