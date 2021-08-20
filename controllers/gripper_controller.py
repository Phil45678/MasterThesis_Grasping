# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 12:14:58 2020

@author: norman marlier

Abstract for Gripper in Pybullet

"""
import numpy as np
import copy


def inverse_mapping(f):
    return f.__class__(map(reversed, f.items()))


class GripperController():
    """Gripper controller abstract class."""

    def __init__(self, gripper_id, urdf, id_server, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

        # Parameters
        self.id_server = id_server
        self.id = gripper_id
        self.urdf = urdf

        self.activate_state = False
        self.joint_target = None

        self.joints = {}
        self.joint_states = []
        self.joint_vel = []
        self.joint_index = []
        self.joint_limits = []
    
    def get_gripper_ID(self): 
        return self.id

    def activate(self, engine, id_server=None):
        """Activate the force/torque sensor and the joint motors."""
        # Get the joint indices
        for i in range(engine.getNumJoints(self.id, physicsClientId=self.id_server)):
            info = engine.getJointInfo(self.id, i, physicsClientId=self.id_server)
            if info[1] in self.joint_names:
                # Add the joint
                index = self.joint_names.index(info[1])
                self.joints[self.joint_names[index]] = info[0]
                # Activate F/T sensor
                engine.enableJointForceTorqueSensor(self.id, info[0], True,
                                                    physicsClientId=self.id_server)
                # Actuated joint
                if info[2] != 4:
                    # Joint state
                    state = engine.getJointState(self.id, info[0], physicsClientId=self.id_server)
                    self.joint_states.append(state[0])
                    self.joint_vel.append(state[1])
                    # Add index
                    self.joint_index.append(info[0])
                    # Add limits
                    self.joint_limits.append([info[8], info[9]])
                    
        self.activate_state = True

    def open_gripper(self):
        """Open the gripper by changing the internal state."""
        raise NotImplementedError

    def close_gripper(self):
        """Close the gripper by changin the internal state."""
        raise NotImplementedError

    def set_joint_target(self, joint_state):
        """Set the joint states target."""
        raise NotImplementedError

    def update_joint_states(self, engine, id_server=None):
        """Get the joint states."""
        for i in range(len(self.joint_states)):
            # Joint state
            state = engine.getJointState(self.id, self.joint_index[i], physicsClientId=self.id_server)
            self.joint_states[i] = state[0]
            self.joint_vel[i] = state[1]

    def set_params(self, parameters: dict, engine, id_server=None):
        """Set parameters."""
        raise NotImplementedError

    def reset(self, gripper_id, id_server):
        """Reset all the internal states."""
        # Reset gripper_id
        self.id = gripper_id
        # Id server
        self.id_server = id_server
        # Reset mode
        self.mode = "basic"
        # Reset target joints
        self.target = None
        # Reset activate state
        self.activate_state = False
        # Reset joint information
        self.joints = {}
        self.joint_states = []
        self.joint_index = []

    def step(self, engine, id_server=None):
        """Make one step with the physic engine."""
        raise NotImplementedError

    def filter_collision(self, engine, parameters: dict, id_server=None):
        """Filter the collision due to the self collision flag."""
        raise NotImplementedError

    def get_mode(self, mode):
        """Return the mode as string for the controller."""
        raise NotImplementedError
    
    def force_open(self, engine, id_server):
        """Force the gripper to be open."""
        raise NotImplementedError


class Robotiq3fControllerSimple(GripperController):
    """Robotiq 3-finger controller."""

    def __init__(self, gripper_id, urdf, id_server, *args, **kwargs):
        super(Robotiq3fControllerSimple, self).__init__(gripper_id, urdf,
                                                        id_server, *args,
                                                        **kwargs)

        # Joints
        # URDF - Hardware
        # From the gripper urdf
        if self.urdf == "gripper":
            self.joint_names = [b'palm',
                                b'palm_finger_1_joint',
                                b'finger_1_joint_1',
                                b'finger_1_joint_2',
                                b'finger_1_joint_3',
                                b'palm_finger_2_joint',
                                b'finger_2_joint_1',
                                b'finger_2_joint_2',
                                b'finger_2_joint_3',
                                b'palm_finger_middle_joint',
                                b'finger_middle_joint_1',
                                b'finger_middle_joint_2',
                                b'finger_middle_joint_3']
        # From the UR5 urdf
        elif self.urdf == "ur5":
            self.joint_names = [b'l_palm',
                                b'l_palm_finger_1_joint',
                                b'l_finger_1_joint_1',
                                b'l_finger_1_joint_2',
                                b'l_finger_1_joint_3',
                                b'l_palm_finger_2_joint',
                                b'l_finger_2_joint_1',
                                b'l_finger_2_joint_2',
                                b'l_finger_2_joint_3',
                                b'l_palm_finger_middle_joint',
                                b'l_finger_middle_joint_1',
                                b'l_finger_middle_joint_2',
                                b'l_finger_middle_joint_3']
        else:
            raise ValueError("Urdf not valid:", self.urdf)

        # Gripper parameters
        # Have to to be very low otherwise simulation will be not accurate
        self.force = 0.5
        self.params = {"mode": ["basic", "pinch", "wide"],
                       "inv_mode": {0: "basic", 1: "pinch", 2: "wide"},
                       "force": [10*self.force, 10*self.force, self.force, self.force,
                                 10*self.force, 10*self.force, self.force, self.force,
                                 20*self.force, 2*self.force, 2*self.force],
                       "res": 50}
        self.mode = "basic"

    def __str__(self):

        return "Robotiq3f controller - Activate : {} - Mode : {} - Joint Names : {}".format(self.activate_state, self.mode, self.joint_names)

    def activate(self, engine, id_server=None):

        if not self.activate_state:
            
            super(Robotiq3fControllerSimple, self).activate(engine, id_server)

            # Close position for state
            self.close_joint_states = {"basic": [-0.0162, 1.221, 1.57, -0.95, 0.0162, 1.221, 1.57, -0.95, 1.221, 1.57, -0.95],
                                       "pinch": [-0.156, 0.916, 0.0, -0.968, 0.156, 0.932, 0.0, -0.986, 0.932, 0.0, -0.986],
                                       "wide": [0.176, 1.221, 1.57, -0.95, -0.176, 1.221, 1.57, -0.95, 1.221, 1.57, -0.95]}
            # Open position for state
            self.open_joint_states = {"basic": [-0.0162, 0.049, 0, -0.052, 0.0162, 0.049, 0, -0.052, 0.049, 0, -0.052],
                                       "pinch": [-0.157, 0.049, 0.0, -0.052, 0.157, 0.049, 0.0, -0.052, 0.049, 0.0, -0.052],
                                       "wide": [0.176, 0.049, 0., -0.052, -0.176, 0.049, 0., -0.052, 0.049, 0., -0.052]}
        else:
            pass

    def set_joint_target(self, target):
        # Set the target
        self.target = target

    def update_joint_states(self, engine, id_server):

        if self.activate_state:
            super(Robotiq3fControllerSimple, self).update_joint_states(engine, id_server)
        else:
            return None

    def step(self, engine, id_server=None):

        if self.activate_state:
            # If target exists
            if self.target:
                """
                # Limit overhead but call once
                engine.setJointMotorControlArray(self.id, self.joint_index,
                                                 controlMode=engine.POSITION_CONTROL,
                                                 targetPositions=self.target,
                                                 forces=self.params["force"],
                                                 physicsClientId=id_server)
                """
                for i, joint in enumerate(self.joint_index):
                    engine.setJointMotorControl2(self.id, joint,
                                                 controlMode=engine.POSITION_CONTROL,
                                                 targetPosition=self.target[i],
                                                 maxVelocity=1,
                                                 force=self.params["force"][i],
                                                 physicsClientId=id_server)
        else:
            pass
        
    def close_gripper(self):
        # Set target as close configuration
        self.target = self.close_joint_states[self.mode]
    
    def open_gripper(self):

        # TODO: Register open_joint_states
        # Set target as open configuration
        self.target = self.open_joint_states[self.mode]
    
    def set_params(self, mode, engine, id_server=None):
        
        if self.activate_state:
            # Set the mode of the gripper
            # Check the mode
            if mode in self.params["mode"]:
                # Change the mode
                self.mode = mode
                # Change the motor position only for joint linked to the palm
                self.update_joint_states(engine, id_server)
                target = self.joint_states
                target[0] = self.open_joint_states[self.mode][0]
                target[4] = self.open_joint_states[self.mode][4]
                # Set the traget
                self.set_joint_target(target)
            # Not appropariate mode ? pass
            else:
                pass
        else:
            pass
    
    def get_mode(self, mode_index):
        """
        for mode, index in self.params["inv_mode"].items():
            if index == mode_index:
                return mode
        """
        if mode_index in self.params["inv_mode"].keys():
            return self.params["inv_mode"][mode_index]
        
    def filter_collision(self, engine, id_server=None):
        # Delete self collision between fingers
        f1 = [8, 8, 8, 12, 12, 16]
        f2 = [12, 16, 20, 16, 20, 20]
        enableCollision = 1

        for f_i, f_j in zip(f1, f2):
            engine.setCollisionFilterPair(self.id, self.id,
                                          f_i, f_j, enableCollision)


class Robotiq3fControllerRegular(GripperController):
    """Robotiq 3-finger controller."""
    def __init__(self, gripper_id, urdf, id_server, *args, **kwargs):
        super(Robotiq3fControllerRegular, self).__init__(gripper_id, urdf, id_server, 
                                                         *args, **kwargs)
        # Joints
        # URDF - Hardware
        # From the gripper urdf
        if self.urdf == "gripper":
            self.joint_names = [b'palm',
                                b'palm_finger_1_joint',
                                b'finger_1_joint_1',
                                b'finger_1_joint_2',
                                b'finger_1_joint_3',
                                b'palm_finger_2_joint',
                                b'finger_2_joint_1',
                                b'finger_2_joint_2',
                                b'finger_2_joint_3',
                                b'palm_finger_middle_joint',
                                b'finger_middle_joint_1',
                                b'finger_middle_joint_2',
                                b'finger_middle_joint_3']
        # From the UR5 urdf
        elif self.urdf == "ur5":
            self.joint_names = [b'l_palm',
                                b'l_palm_finger_1_joint',
                                b'l_finger_1_joint_1',
                                b'l_finger_1_joint_2',
                                b'l_finger_1_joint_3',
                                b'l_palm_finger_2_joint',
                                b'l_finger_2_joint_1',
                                b'l_finger_2_joint_2',
                                b'l_finger_2_joint_3',
                                b'l_palm_finger_middle_joint',
                                b'l_finger_middle_joint_1',
                                b'l_finger_middle_joint_2',
                                b'l_finger_middle_joint_3']
        else:
            raise ValueError("Urdf not valid:", self.urdf)

        # Gripper parameters
        # Have to to be very low otherwise simulation will be not accurate
        self.force = 0.5
        self.params = {"mode": ["basic", "pinch", "wide"],
                       "inv_mode": {0: "basic", 1: "pinch", 2: "wide"},
                       "force": [10*self.force, 2*self.force, self.force, self.force,
                                 10*self.force, 2*self.force, self.force, self.force,
                                 4*self.force, 2*self.force, 2*self.force],
                       "res": 50}
        self.mode = "basic"
        
        self.gripper = "3_fingers"

    def __str__(self):
        
        return "Robotiq3f controller - Activate : {} - Mode : {} - Joint Names : {}".format(self.activate_state, self.mode, self.joint_names)

    def activate(self, engine, id_server=None):

        if not self.activate_state:
            # Activate
            super(Robotiq3fControllerRegular, self).activate(engine, id_server)
            """
            # Close position for state
            self.close_joint_states = {"basic": [-0.0162, 1.221, 1.57, -0.95, 0.0162, 1.221, 1.57, -0.95, 1.221, 1.57, -0.95],
                                       "pinch": [-0.156, 0.916, 0.0, -0.968, 0.156, 0.932, 0.0, -0.986, 0.932, 0.0, -0.986],
                                       "wide": [0.176, 1.221, 1.57, -0.95, -0.176, 1.221, 1.57, -0.95, 1.221, 1.57, -0.95]}
            # Open position for state
            
            self.open_joint_states = {"basic": [-0.0162, 0.049, 0, -0.052, 0.0162, 0.049, 0, -0.052, 0.049, 0, -0.052],
                                       "pinch": [-0.157, 0.049, 0.0, -0.052, 0.157, 0.049, 0.0, -0.052, 0.049, 0.0, -0.052],
                                       "wide": [0.176, 0.049, 0., -0.052, -0.176, 0.049, 0., -0.052, 0.049, 0., -0.052]}
            """
            # Close position for state
            self.close_joint_states = {"basic": [-0.0162, 1.221, 1.57, -0.052, 0.0162, 1.221, 1.57, -0.052, 1.221, 1.57, -0.052],
                                       "pinch": [-0.156, 0.916, 0.0, -0.052, 0.156, 0.932, 0.0, -0.052, 0.932, 0.0, -0.052],
                                       "wide": [0.176, 1.221, 1.57, -0.052, -0.176, 1.221, 1.57, -0.052, 1.221, 1.57, -0.052]}
            # Open position for state
            
            self.open_joint_states = {"basic": [-0.0162, 0.049, 0, -1.22, 0.0162, 0.049, 0, -1.22, 0.049, 0, -1.22],
                                       "pinch": [-0.157, 0.049, 0.0, -0.9, 0.157, 0.049, 0.0, -0.9, 0.049, 0.0, -0.9],
                                       "wide": [0.176, 0.049, 0., -0.9, -0.176, 0.049, 0., -0.9, 0.049, 0., -0.9]}
            
        else:
            pass

    def set_joint_target(self, target):
        # Set the target
        self.target = target

    def update_joint_states(self, engine, id_server):

        if self.activate_state:
            super(Robotiq3fControllerRegular, self).update_joint_states(engine, id_server)
        else:
            return None

    def step(self, engine, id_server=None):

        if self.activate_state:
            # If target exists
            if self.target:
                target = self.set_step_m(engine, id_server)
                # Limit overhead but call once
                engine.setJointMotorControlArray(self.id, self.joint_index,
                                                 controlMode=engine.POSITION_CONTROL,
                                                 targetPositions=target,
                                                 forces=self.params["force"],
                                                 physicsClientId=id_server)
                
        else:
            pass
    
    def check_vel_joint(self, engine, id_server, joint_index):
        """Check if a joint moves."""
        # Update joint states
        self.update_joint_states(engine, id_server)
        # Check
        if np.abs(self.joint_vel[joint_index]) > 0.001:
            return True
        else:
            return False
    
    def set_step_m(self, engine, id_server):
        """Set the joint to a desired state.
        
        Because of underactuated fingers, each joint moves only when the
        previous reaches its maximum value.
        """
        # Update joint states
        self.update_joint_states(engine, id_server)
        # Target
        target = copy.copy(self.target)
        # First angle moves
        for (joint_1, joint_2, joint_3) in zip([1, 5, 8], [2, 6, 9], [3, 7, 10]):
            # If stuck
            cond_vel = self.check_vel_joint(engine, id_server, joint_1)
            cond_init = self.joint_states[joint_1] == self.open_joint_states[self.mode][joint_1]
            if (not cond_vel) and (not cond_init):
                # Do nothing for the joint 1
                # Go to other joint
                target[joint_2] = self.target[joint_2]
                # If stuck
                cond_vel = self.check_vel_joint(engine, id_server, joint_2)
                if self.mode == "pinch":
                    cond_init = False
                else:
                    cond_init = self.joint_states[joint_2] != self.open_joint_states[self.mode][joint_2]
                if (not cond_vel) and (not cond_init):
                    # Go to other joints
                    target[joint_3] = self.target[joint_3]
                else:
                    target[joint_3] = self.joint_states[joint_3]
            else:
                # Static other joints
                target[joint_2] = self.joint_states[joint_2]
                target[joint_3] = self.joint_states[joint_3]
    
        return target
        
    def close_gripper(self):
        # Set target as close configuration
        self.target = self.close_joint_states[self.mode]
    
    def open_gripper(self):

        # TODO: Register open_joint_states
        # Set target as open configuration
        self.target = self.open_joint_states[self.mode]
    
    def set_params(self, mode, engine, id_server=None):
        
        if self.activate_state:
            # Set the mode of the gripper
            # Check the mode
            if mode in self.params["mode"]:
                # Change the mode
                self.mode = mode
                # Change the motor position only for joint linked to the palm
                self.update_joint_states(engine, id_server)
                target = self.joint_states
                target[0] = self.open_joint_states[self.mode][0]
                target[4] = self.open_joint_states[self.mode][4]
                # Set the traget
                self.set_joint_target(target)
            # Not appropariate mode ? pass
            else:
                pass
        else:
            pass
    
    def get_mode(self, mode_index):
        """
        for mode, index in self.params["inv_mode"].items():
            if index == mode_index:
                return mode
        """
        if mode_index in self.params["inv_mode"].keys():
            return self.params["inv_mode"][mode_index]
    
    def filter_collision(self, engine, parameters: dict, id_server=None):
        
        # Delete self collision between fingers of the gripper and the palm
        shift_index = 9
        palm = -1
        f1 = np.array([0, 1, 2, 3])
        f2 = np.array([4, 5, 6, 7])
        f3 = np.array([8, 9, 10, 11])
        palm += shift_index
        f1 += shift_index
        f2 += shift_index
        f3 += shift_index
        ph0 = [f1[0], f2[0], f3[0]]
        ph1 = [f1[1], f2[1], f3[1]]
        ph2 = [f1[2], f2[2], f3[2]]
        ph3 = [f1[3], f2[3], f3[3]]
        """
        f1 = np.array([-1, -1, 0, 1, 1, 1, 1,2, -1, -1, 4, 5, 5, 6, -1, -1, 8,  9, 9, 10])
        f2 = np.array([ 0,  1, 1, 2, 3, 3,  4,  5, 5, 6, 7, 7,  8,  9, 9, 10, 11, 11])
        f1 += 9
        f2 += 9
        for f_i, f_j in zip(f1, f2):
            engine.setCollisionFilterPair(self.id, self.id,
                                          f_i, f_j, enableCollision)
        """
        # Only contact with the end of the fingers and palm is admitted
        enableCollision = 0
        
        for i in f1:
            for j in f2:
                engine.setCollisionFilterPair(self.id, self.id,
                                              i, j, enableCollision)
        
        for i in f1:
            for k in f3:
                engine.setCollisionFilterPair(self.id, self.id,
                                              i, k, enableCollision)
        
        for j in f2:
            for k in f3:
                engine.setCollisionFilterPair(self.id, self.id,
                                              j, k, enableCollision)
        for p in ph0:
            engine.setCollisionFilterPair(self.id, self.id,
                                          p, palm, enableCollision)
        
        for p in ph1:
            engine.setCollisionFilterPair(self.id, self.id,
                                          p, palm, enableCollision)
        for p in ph2:
            engine.setCollisionFilterPair(self.id, self.id,
                                          p, palm, enableCollision)
        
        # Enable contact between ph3
        for i, joint in enumerate(ph3):
            for j in ph3[i+1:]:
                engine.setCollisionFilterPair(self.id, self.id,
                                              joint, j, 1)
                    
        
    
    def force_open(self, engine, id_server):
        
        for i, joint in enumerate(self.joint_index):
            engine.resetJointState(self.id, joint,
                                   self.open_joint_states[self.mode][i], 0.,
                                   physicsClientId=id_server)


class SawyerElectricController(GripperController):

    """Sawyer electric gripper controller."""

    def __init__(self, gripper_id, urdf, id_server, *args, **kwargs):
        super(SawyerElectricController, self).__init__(gripper_id, urdf, id_server, 
                                                  *args, **kwargs)

        # Joints
        # URDF - Hardware
        if urdf == "gripper":
            self.joint_names = [b'left_endpoint',
                                b'l_gripper_l_finger_joint',
                                b'l_gripper_l_finger_tip_joint',
                                b'l_gripper_r_finger_joint',
                                b'l_gripper_r_finger_tip_joint']
        elif urdf == "sawyer":
            self.joint_names = [b'right_electric_gripper_base_joint',
                                b'right_gripper_base_joint',
                                b'right_gripper_l_finger_joint',
                                b'right_gripper_l_finger_tip_joint',                             
                                b'right_gripper_r_finger_joint',
                                b'right_gripper_r_finger_tip_joint',
                                b'right_gripper_tip_joint']

        # Gripper parameters
        self.params = {"mode": ["basic"],
                       "inv_mode": {0: "basic"},
                       "force": 1.,
                       "res": 50}
        self.mode = "basic"
        self.gripper = "eletric_jaw"

    def __str__(self):
        
        return "Sawyer electric gripper controller - Activate : {} - Mode : {} - Joint Names : {}".format(self.activate_state, self.mode, self.joint_names)

    def activate(self, engine, id_server=None):

        if not self.activate_state:
            super(SawyerElectricController, self).activate(engine, id_server)
            # Close position for state
            self.open_joint_states = {"basic": [0.020832, -0.020832]}
            # Open position for state
            self.close_joint_states = {"basic": [0., 0.]}
        else:
            pass


    def set_joint_target(self, target):
        # Set the target
        self.target = target

    def update_joint_states(self, engine, id_server):

        if self.activate_state:
            super(SawyerElectricController, self).update_joint_states(engine, id_server)
        else:
            pass

    def step(self, engine, id_server=None):

        if self.activate_state:
            # If target exists
            if self.target:               
                engine.setJointMotorControlArray(self.id, self.joint_index,
                                                 controlMode=engine.POSITION_CONTROL,
                                                 targetPositions=self.target, 
                                                 velocityGains=[0.6]*len(self.joint_states),
                                                 forces=[self.params["force"]]*len(self.joint_states),
                                                 physicsClientId=self.id_server)
        else:
            pass

    def close_gripper(self):

        # Set target as close configuration
        self.target = self.close_joint_states[self.mode]

    def open_gripper(self):

        # TODO: Register open_joint_states
        # Set target as open configuration
        self.target = self.open_joint_states[self.mode]
    
    def set_params(self, mode, engine, id_server=None):
        # Set the mode of the gripper
        # Check th emode
        pass

    def get_mode(self, mode_index):
        if mode_index in self.params["inv_mode"].keys():
            return self.params["inv_mode"][mode_index]

    def filter_collision(self, engine, id_server=None):
        
        # Delete self collision between fingers
        f1 = [22]
        f2 = [24]
        enableCollision = 0
        
        for f_i, f_j in zip(f1, f2):
            engine.setCollisionFilterPair(self.id, self.id,
                                          f_i, f_j, enableCollision)
    
    def force_open(self, engine, id_server):
        
        for i, joint in enumerate(self.joint_index):
            engine.resetJointState(self.id, joint,
                                   self.open_joint_states[self.mode][i], 0.,
                                   physicsClientId=id_server)
        

if __name__ == "__main__":
    import time
    import pybullet as pb
    from gripper import Robotiq3fGripper
    id_server = pb.connect(pb.GUI)
    pb.setRealTimeSimulation(True)
    
    gripper = Robotiq3fGripper("../data/robotiq_3f_gripper_visualization/cfg/robotiq-3f-gripper_articulated.urdf")
    
    gripper.load({"base_pos": [0., 0., 0.],
                  "base_ori": [0., 0., 0., 1.],
                  "fixed_base": True}, pb, id_server)
    ctrl = Robotiq3fController(gripper.gripper_id, id_server)
    
    print(ctrl)
    ctrl.activate(pb)
    
    print(ctrl)

    print("Basic")
    input("Close")
    ctrl.close_gripper()
    ctrl.step(pb)
        
    input("Open")
    ctrl.open_gripper()
    ctrl.step(pb)
    
    input("Pinch")
    ctrl.set_params("pinch", pb)
    ctrl.step(pb)
    input("Close")
    ctrl.close_gripper()
    ctrl.step(pb)
        
    input("Open")
    ctrl.open_gripper()
    ctrl.step(pb)
    
    input("Wide")
    ctrl.set_params("wide", pb)
    ctrl.step(pb)
    input("Close")
    ctrl.close_gripper()
    ctrl.step(pb)
        
    input("Open")
    ctrl.open_gripper()
    ctrl.step(pb)
    
    input("ok")
    
    ctrl.reset(gripper, id_server)
    
    print(ctrl)
    """
    sawyer_gripper = pb.loadURDF("../data/rethink_ee_description/urdf/electric_gripper/sawyer_gripper.urdf",
                          useFixedBase=True,
                          flags=pb.URDF_USE_SELF_COLLISION)
    ctrl = SawyerElectricController(sawyer_gripper, id_server)
    
    print(ctrl)
    
    ctrl.activate(pb)
    
    print(ctrl)
    
    
    input("close")
    ctrl.close_gripper()
    ctrl.step(pb)
    
    input("open")
    ctrl.open_gripper()
    ctrl.step(pb)
    
    input("ok")
    """
    
    
    
    