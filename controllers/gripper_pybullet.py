# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 15:52:19 2019

@author: norman marlier

Class for the virtual Robotiq 3-fingers in Pybullet
Services are:
    -basic control operations on the gripper (open, close)
    -get contact points and friction force
    -get the pose of the gripper
"""
import numpy as np

class Gripper():
    
    #### SOME DATASHEET
    #### Min force = 15N
    #### Max force = 60N
    #### Envelop = 20 to 155 mm
    
    def __init__(self, physics_id=0, joint_index=[9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20], movable_joint_index=[10, 11, 12, 14, 15, 16, 18, 19, 20], step=0.05):
        
        # Max and min forces
        self.MAX_FORCE = 2.

        # List which contains the index of the join for the gripper
        self.joint_index = joint_index
        self.movable_joint_index = movable_joint_index
        
        # gripper joint state
        self.gripper_pose = {self.joint_index[0]: -0.0162, self.joint_index[1]: 0.0495,
                             self.joint_index[2]: 0.0, self.joint_index[3]: -0.052,
                             self.joint_index[4]: 0.0162, self.joint_index[5]: 0.0495,
                             self.joint_index[6]: 0.0, self.joint_index[7]: -0.052,
                             self.joint_index[8]: 0.0495, self.joint_index[9]: 0.0,
                             self.joint_index[10]: -0.052}
        
        # pose of the closed gripper
        self.close_gripper_poses = {"basic": [-0.0162, 1.221, 1.57, -0.95, 0.0162, 1.221, 1.57, -0.95, 1.221, 1.57, -0.95],
                                    "pinch": [-0.157, 0.916, 0.0, -0.968, 0.157, 0.924, 0.0, -0.977, 0.924, 0.0, -0.977],
                                    "wide": [0.176, 1.221, 1.57, -0.95, -0.176, 1.221, 1.57, -0.95, 1.221, 1.57, -0.95],
                                    "scissor": [-0.178, 0.0495, 0.0, -0.0523, 0.178, 0.0495, 0.0, -0.0523, 0.0495, 0.0, -0.0523]}
        
        # Preshape configuration
        self.hand_preshapes = {"basic": 0, "pinch": 1, "wide": 2, "scissor": 3}
        self.hand_preshapes_inv = {0: "basic", 1: "pinch", 2: "wide", 3: "scissor"}
        self.hand_mode = self.hand_preshapes["basic"]

        # step for motion
        # You can only move the gripper by 255 steps
        # TODO: does not take into account the 255 steps
        self.count = 255 
        self.step = step
        
        self.physicsClientId = physics_id
        
    def activate_ft_sensor(self, p, robot, index):
        p.enableJointForceTorqueSensor(robot, index, physicsClientId=self.physicsClientId)
        
    def get_ft_sensor(self, p, robot, index):
        _, _, ft, _ = p.getJointState(robot, index, physicsClientId=self.physicsClientId)
        return ft
        
    def get_joint_index(self):
        return self.joint_index
    
    def get_mode(self):
        return self.hand_preshapes_inv[self.hand_mode]
    
    def set_mode(self, p, gripper_id, mode):      
        self.hand_mode = self.hand_preshapes[mode]
        if self.hand_preshapes_inv[self.hand_mode] == "basic":
            self.gripper_pose[self.joint_index[0]] = -0.0162
            self.gripper_pose[self.joint_index[4]] = 0.0162
        elif self.hand_preshapes_inv[self.hand_mode] == "pinch":
            self.gripper_pose[self.joint_index[0]] = -0.157
            self.gripper_pose[self.joint_index[4]] = 0.157
        elif self.hand_preshapes_inv[self.hand_mode] == "wide":
            self.gripper_pose[self.joint_index[0]] = 0.176
            self.gripper_pose[self.joint_index[4]] = -0.176
        elif self.hand_preshapes_inv[self.hand_mode] == "scissor":
            self.gripper_pose[self.joint_index[0]] = 0.191
            self.gripper_pose[self.joint_index[4]] = -0.191
        else:
            print("Wrong mode... go in mode basic")
            self.hand_mode = self.hand_preshapes['basic']
        
        # set the pose
        p.setJointMotorControlArray(gripper_id, self.joint_index, controlMode=p.POSITION_CONTROL, targetPositions=list(self.gripper_pose.values()),
                                    velocityGains=[0.6]*11, forces=[self.MAX_FORCE]*11, physicsClientId=self.physicsClientId)
        
    def reset_mode(self, p, gripper_id, mode):      
        self.hand_mode = self.hand_preshapes[mode]
        if self.hand_preshapes_inv[self.hand_mode] == "basic":
            self.gripper_pose[self.joint_index[0]] = -0.0162
            self.gripper_pose[self.joint_index[4]] = 0.0162
        elif self.hand_preshapes_inv[self.hand_mode] == "pinch":
            self.gripper_pose[self.joint_index[0]] = -0.157
            self.gripper_pose[self.joint_index[4]] = 0.157
        elif self.hand_preshapes_inv[self.hand_mode] == "wide":
            self.gripper_pose[self.joint_index[0]] = 0.176
            self.gripper_pose[self.joint_index[4]] = -0.176
        elif self.hand_preshapes_inv[self.hand_mode] == "scissor":
            self.gripper_pose[self.joint_index[0]] = 0.191
            self.gripper_pose[self.joint_index[4]] = -0.191
        else:
            print("Wrong mode... go in mode basic")
            self.hand_mode = self.hand_preshapes['basic']
        
        # Force the pose
        for i, joint_id in enumerate(self.movable_joint_index):
            p.resetJointState(gripper_id, joint_id, targetValue=self.gripper_pose[self.joint_index[i]],
                              targetVelocity=0, physicsClientId=self.physicsClientId)
        
    def set_pose(self, pose):
        
        if self.hand_preshapes_inv[self.hand_mode] == "scissor":
            # Update the pose in the entire pose
            self.gripper_pose[self.joint_index[0]] = pose[0]
            self.gripper_pose[self.joint_index[4]] = pose[4]
        else:
            # Take only the useful joint - movable joint
            #pose = pose[[1, 2, 3, 5, 6, 7, 8, 9, 10]]
            intersec = self.intersection(self.joint_index, self.movable_joint_index)
            indices = []
            for x in intersec:
                indices.append(self.joint_index.index(x))
            pose = pose[indices]
            # Update the pose in the entire pose
            for i, index in enumerate(self.movable_joint_index):
                self.gripper_pose[index] = pose[i]
            
    def close_gripper(self, p, gripper_id):
        # Close the gripper

        # Get the current grasp pose
        current_grasp_poses = p.getJointStates(gripper_id, self.joint_index, physicsClientId=self.physicsClientId)
        current_grasp_poses = np.array(current_grasp_poses)
        
        close_poses = self.close_gripper_poses[self.hand_preshapes_inv[self.hand_mode]]
    
        # Difference between the current and the wanted grasp pose
        diff_poses = -np.array(current_grasp_poses[:, 0]) + np.array(close_poses)
        target_poses = diff_poses*self.step
        target_poses += current_grasp_poses[:, 0]
        
        # Update the gripper pose
        self.set_pose(target_poses)
        
        # Move the joints
        p.setJointMotorControlArray(gripper_id, self.joint_index, controlMode=p.POSITION_CONTROL,
                                    targetPositions=list(self.gripper_pose.values()), velocityGains=[0.6]*11,
                                    forces=[self.MAX_FORCE]*11, physicsClientId=self.physicsClientId)
        
    def reset_close_gripper(self, p, gripper_id):
        # Close the gripper without the physic engine
        
        close_pose = self.close_gripper_poses[self.hand_preshapes_inv[self.hand_mode]]
        
        # Force the pose
        for i, joint_id in enumerate(self.movable_joint_index):
            p.resetJointState(gripper_id, joint_id, targetValue=close_pose[i],
                              targetVelocity=0, physicsClientId=self.physicsClientId)
            
        
    def open_gripper(self, p, gripper_id):
        # Open the gripper
        # Get the current grasp pose
        current_grasp_poses = p.getJointStates(gripper_id, self.joint_index, physicsClientId=self.physicsClientId)
        current_grasp_poses = np.array(current_grasp_poses)
    
        # Difference between the current and the wanted grasp pose
        target_poses = current_grasp_poses[:, 0]* (1-self.step)
        
        # Update gripper pose
        self.set_pose(target_poses)
        
        # Move the joints
        p.setJointMotorControlArray(gripper_id, self.joint_index, controlMode=p.POSITION_CONTROL,
                                    targetPositions=list(self.gripper_pose.values()), velocityGains=[0.6]*11,
                                    forces=[self.MAX_FORCE]*11, physicsClientId=self.physicsClientId)
        
        
    @staticmethod
    def intersection(lst1, lst2): 
        lst3 = [value for value in lst1 if value in lst2] 
        return lst3    
        

            
 

