# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 10:19:28 2020

@author: norman marlier
"""
import pybullet
import pybullet_planning
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys
sys.path.append("../..")

# Simulator
from DexterousManipulation.simulations.simulator import Simulator
from DexterousManipulation.simulations.grasp_simulator import GraspSimulator
from DexterousManipulation.simulations.utils_sim import is_contact, show_pose, load_vhacd_body
# Controllers
from DexterousManipulation.controllers.gripper_pybullet import Gripper
# Planner
from pybullet_planning.motion_planners import rrt_connect, birrt
from pybullet_planning.interfaces.planner_interface.joint_motion_planning import get_distance_fn, get_extend_fn, get_sample_fn, check_initial_end
from pybullet_planning.interfaces.robots import get_collision_fn
from pybullet_planning.interfaces.robots.joint import get_joint_positions



class TrajectorySimulator(Simulator):
    """Simulator of the reachability.
    
    It consists in a robotic arm, a gripper and a
    background.
    """
    def __init__(self, pybullet_module, robot_dict, background, 
                 sim_parameters):
        """Constructor."""
        # Pybullet module
        self.p = pybullet_module
        # Robot filename
        self.robot_dict = robot_dict
        # Parameters
        self.sim_parameters = sim_parameters
        # Background
        self.background = background
        
        self.RRT_ITERATIONS = 30
        self.RRT_RESTARTS = 10
        self.RRT_SMOOTHING = 20
    
    def start_sim(self, parameters, engine_connection="DIRECT"):
        # Start the simulation
        # Connection to a pybullet server
        if engine_connection == "GUI":
            self.engine_connection = self.p.GUI
        else:
            self.engine_connection = self.p.DIRECT
        # DEBUG MODE
        # The DEBUG MODE allows to print some variables and interacts with
        # the simulation
        if self.engine_connection == self.p.GUI:
            self.debug = 1
        else:
            self.debug = 0
        # Connect to the server and put it into no realtime mode
        self.id_server = self.p.connect(self.engine_connection)

        self.p.setRealTimeSimulation(0, physicsClientId=self.id_server)
        self.p.setPhysicsEngineParameter(numSolverIterations=150,
                                         physicsClientId=self.id_server)
        if self.debug:
            print(self.p.getPhysicsEngineParameters(physicsClientId=self.id_server))
        # Put the camera at the same place
            self.p.resetDebugVisualizerCamera(2.3, 180, -41, [0., 0, 0.7],
                                              physicsClientId=self.id_server)
        # Apply gravity
        self.p.setGravity(0., 0., -9.81)
        # Setup the world
        self.setup_world(parameters)

    def setup_world(self, parameters: list):
        """Load the background.
        
        Parameters
        ----------
        sclae of the table
        """
        self.obj_id = {"id": [], "name": []}
        for obj in self.background:
            if "urdf" in obj.keys():
                if "table" in obj["urdf"]:
                    # Compute scaling
                    scale = parameters[0]
                    self.table_height = obj["table_height"]
                else:
                    scale = 1.
                # Load object
                self.obj_id["id"].append(self.p.loadURDF(obj["urdf"],
                                                         globalScaling=scale,
                                                         basePosition=obj["base_pos"],
                                                         baseOrientation=obj["base_ori"],
                                                         physicsClientId=self.id_server))
                self.obj_id["name"].append(obj["urdf"])
        # Robot
        # TODO: Add stand
        # Visual mesh
        visualShapeId = self.p.createVisualShape(shapeType=self.p.GEOM_BOX,
                                                 halfExtents=[0.4, 0.06, 0.02],
                                                 rgbaColor=[221/255, 211/255, 211/255, 1],
                                                 physicsClientId=self.id_server)
        # Collision mesh
        collisionShapeId = self.p.createCollisionShape(shapeType=self.p.GEOM_BOX,
                                                       halfExtents=[0.4, 0.06, 0.02],
                                                       physicsClientId=self.id_server)
        # Rigid body
        stand1 = self.p.createMultiBody(baseMass=0,
                                        baseCollisionShapeIndex=collisionShapeId,
                                        baseVisualShapeIndex=visualShapeId,
                                        basePosition=[-0.96, 0., 0.84],
                                        baseOrientation=[0., 0., 0., 1.],
                                        useMaximalCoordinates=False,
                                        physicsClientId=self.id_server)
        self.obj_id["id"].append(stand1)
        self.obj_id["name"].append("small_extrusion")
        # Visual mesh
        visualShapeId2 = self.p.createVisualShape(shapeType=self.p.GEOM_BOX,
                                                 halfExtents=[0.55/2, 0.78/2, 0.80/2],
                                                 rgbaColor=[221/255, 211/255, 211/255, 1],
                                                 physicsClientId=self.id_server)
        # Collision mesh
        collisionShapeId2 = self.p.createCollisionShape(shapeType=self.p.GEOM_BOX,
                                                       halfExtents=[0.55/2, 0.78/2, 0.80/2],
                                                       physicsClientId=self.id_server)
        # Rigid body
        stand2 = self.p.createMultiBody(baseMass=0,
                                        baseCollisionShapeIndex=collisionShapeId2,
                                        baseVisualShapeIndex=visualShapeId2,
                                        basePosition=[-0.96+0.14-0.275, 0., 0.4],
                                        baseOrientation=[0., 0., 0., 1.],
                                        useMaximalCoordinates=False,
                                        physicsClientId=self.id_server)
        self.obj_id["id"].append(stand2)
        self.obj_id["name"].append("big_extrusion")
        self.robot_init_pos = self.robot_dict["base_pos"]
        self.robot_init_ori = self.robot_dict["base_ori"]
        self.robot = self.p.loadURDF(self.robot_dict["urdf"],
                                     basePosition=self.robot_init_pos,
                                     baseOrientation=self.robot_init_ori,
                                     useFixedBase=True)
        
        if self.debug:
            show_pose(self.p, self.robot_init_pos, [0, 0, 0, 1], self.id_server, scale=0.4)
        # Initial joint pose
        self.robot_joints = self.robot_dict["robot_joints"]
        self.init_joint_pose = self.robot_dict["init_joint_pose"]
        for i in range(len(self.robot_joints)):
            self.p.resetJointState(self.robot,
                                   self.robot_joints[i],
                                   self.init_joint_pose[i])
    
    def place_object(self, pO, obj):
        """Put an object on the table.
        
        obj is object dictionnary
        pO is a 2D pose, pO ~ p(pO)
        """
        # Set position and orientation BEFORE rotate around z axis
        obj_pos = [pO[0], pO[1], self.table_height]
        # Load the object
        parameters_obj = obj.copy()
        parameters_obj["base_pos"] = obj_pos
        parameters_obj["useFixedBase"] = True
        # Load object
        self.grasped_object = load_vhacd_body(self.p, parameters_obj,
                                              server_id=self.id_server)
        # Rotate around z-axis
        obj_pos, obj_ori = self.p.getBasePositionAndOrientation(self.grasped_object,
                                                                physicsClientId=self.id_server)
        _, obj_ori = self.p.multiplyTransforms([0, 0, 0], [0., 0., np.sin(pO[2]/2), np.cos(pO[2]/2)],
                                               obj_pos, obj_ori)
        self.p.resetBasePositionAndOrientation(self.grasped_object, obj_pos, obj_ori,
                                               physicsClientId=self.id_server)
        
        return obj_pos, obj_ori

    def close_sim(self):
        """Close the simulation."""
        self.p.resetSimulation(self.id_server)
        self.p.disconnect(self.id_server)

    def forward(self, inputs: list, parameters: list, *args, **kwargs):
        """Evaluate Sg knowing h and O in a deterministic way Sg = f(h, pO, O).

        Inputs
        ------
        h represents the hand configuration:
            -position in space, x
            -orientation, R
            -finger configuration, g

        O represents an object with its properties:
            -scaling
            -orientation
            -geometric mesh
        
        pO represents the 2D pose of the object:
            -Position in space
            -Rotation in 2D

        Parameters
        ----------
        planner


        inputs: [object, hand configuration, object_pose]
        parameters: [planner]

        **kwargs:
            -engine_connection
        """
        # Variables
        obj = inputs[0]
        h = inputs[1]
        pO = inputs[2]
            
        # Start sim
        self.start_sim([1.], engine_connection=kwargs['engine_connection'])
        
        # Position and orientation of the object - World coordinate
        obj_pos = [pO[0], pO[1], self.table_height+ obj["z"]]
        obj_ori = [0., 0., np.sin(pO[2]/2), np.cos(pO[2]/2)]
        if self.debug:
            show_pose(self.p, obj_pos, obj_ori, self.id_server)
        # Grasp pose to end effector - World coordinate
        # Rotation matrix to quaternions
        hRot = R.from_matrix(h[0, 3:12].reshape((3, 3)))
        trans_gripper = self.robot_dict["trans_gripper"]
        trans_coupler = self.robot_dict["trans_coupler"]
        #trans_coupler = 0
        Tg, Qg = self.p.multiplyTransforms(h[0, 0:3], hRot.as_quat(), [trans_gripper + trans_coupler
                                                , 0., 0.], [0., 0., 0., 1.])
        
        # Hand pose - World coodinate
        table_grasp_pos, table_grasp_ori = self.p.multiplyTransforms(obj_pos, obj_ori,
                                                                     Tg, Qg)
        if self.debug:
            show_pose(self.p, table_grasp_pos, table_grasp_ori, self.id_server) 
            
        # Frame used by the IK solver
        jd = [0.1]*self.p.getNumJoints(self.robot)
        target_joints = self.p.calculateInverseKinematics(self.robot,
                                                          self.robot_dict["eef_link"],
                                                          table_grasp_pos,
                                                          table_grasp_ori,
                                                          jointDamping=jd)
        # Check if the solution given by the IK is reachable
        # Put the robot to the solution found by IK
        for i in range(len(self.robot_joints)):
            self.p.resetJointState(self.robot, self.robot_joints[i],
                                   target_joints[i])
        
        # Get the frame
        wanted_eef_frame = self.p.getLinkState(self.robot, self.robot_dict["eef_link"])[0:2]
        # Diff in translation
        diff_trans = np.linalg.norm(np.array(wanted_eef_frame[0])-np.array(table_grasp_pos))
        if self.debug:
            print("Wanted frame trans", np.array(wanted_eef_frame))
            print("H xyz", np.array(table_grasp_pos))
            print("Diff in translation", diff_trans)
        if  diff_trans > np.random.default_rng().normal(1e-3, 1e-4):
            # Not reachable
            Sr = 0.
            # Close sim
            self.close_sim()
            return Sr
        # TODO : Check orientation
        
        # Replace the robot in the inital configuration
        for i in range(len(self.robot_joints)):
            self.p.resetJointState(self.robot, self.robot_joints[i],
                                   self.init_joint_pose[i])
            
        # Place the object
        obj_pos, obj_ori = self.place_object(pO, obj)
        # Inpsect the env
        if self.debug:
            input("inspect")
        
        # Motion planning - Functions
        obstacles = [self.grasped_object] + self.obj_id["id"]
        if self.debug:
            print(obstacles)
        sample_fn = get_sample_fn(self.robot, self.robot_joints)
        distance_fn = get_distance_fn(self.robot, self.robot_joints)
        extend_fn = get_extend_fn(self.robot, self.robot_joints)
        collision_fn = get_collision_fn(self.robot, self.robot_joints, obstacles=obstacles, self_collisions=True)
        
        # Initial conf
        start_conf = get_joint_positions(self.robot, self.robot_joints)
        # End conf
        end_conf = target_joints[0:self.robot_dict["dof"]]
        if self.debug:
            print("Initial conf", start_conf)
            print("Target Grasp joint:", target_joints)
        # Planner
        planner = parameters[0]
        if planner == "rrt_connect":
            res = rrt_connect(start_conf, end_conf, distance_fn, sample_fn, extend_fn, collision_fn,
                              iterations=self.RRT_ITERATIONS, tree_frequency=2,
                              max_time=1)
        elif planner == "birrt":
            res = birrt(start_conf, end_conf, distance_fn, sample_fn, extend_fn, collision_fn,
                        restarts=self.RRT_RESTARTS, smooth=self.RRT_SMOOTHING,
                        max_time=2)
        else:
            sys.exit("Error : not good planner")
        
        if res is None:
            # No trajectory found
            Sr = 0
        else:
            # The target is reachable
            Sr = 1.
            if self.debug:
                print("Distance function", distance_fn(res[-1], end_conf))
                print("Trajectory points", len(res))
                # Reset the object
                self.p.resetBasePositionAndOrientation(self.grasped_object,
                                                    obj_pos, obj_ori,
                                                    physicsClientId=self.id_server)
                for i in range(len(self.robot_joints)):
                    self.p.resetJointState(self.robot,
                                           self.robot_joints[i],
                                           self.init_joint_pose[i])
                joint_traj_sim = np.zeros((10*len(res), self.robot_dict["dof"]))
                joint_traj_computed = np.zeros((len(res), self.robot_dict["dof"]))
                speed_traj_sim = np.zeros((10*len(res), self.robot_dict["dof"]))
                input("start")
                for i, step in enumerate(res):
                    joint_traj_computed[i] = step
                    self.p.setJointMotorControlArray(self.robot, self.robot_joints,
                                                     controlMode=self.p.POSITION_CONTROL,
                                                     targetPositions=step, 
                                                     velocityGains=[0.8]*len(self.robot_joints),
                                                     physicsClientId=self.id_server)
                    for j in range(10):
                        joint_traj_sim[i*10+j] = get_joint_positions(self.robot, self.robot_joints)
                        states = self.p.getJointStates(self.robot, self.robot_joints, physicsClientId=self.id_server)
                        speed_traj_sim[i*10+j] = [x[1] for x in states]
                        self.p.stepSimulation()
                        time.sleep(1./240.)
                for step in reversed(res):
                    self.p.setJointMotorControlArray(self.robot, self.robot_joints,
                                                     controlMode=self.p.POSITION_CONTROL,
                                                     targetPositions=step, 
                                                     velocityGains=[0.8]*len(self.robot_joints),
                                                     physicsClientId=self.id_server)
                    for j in range(10):
                        self.p.stepSimulation()
                        time.sleep(1./240.)
                input("stop")
                # Print the joint trajectory
                fig = plt.figure()
                for j in range(self.robot_dict["dof"]):
                    plt.subplot(3, 2, j+1)
                    plt.plot(10*np.arange(len(res)), joint_traj_computed[:, j], label=r"desired value", marker="o")
                    #plt.plot(range(len(res)), joint_traj_computed[:, 0], ":")
                    plt.plot(range(10*len(res)), joint_traj_sim[:, j], ":", label=r"sim value")
                    plt.legend()
                # Print the joint velocity
                fig = plt.figure()
                for j in range(self.robot_dict["dof"]):
                    plt.subplot(3, 2, j+1)
                    plt.plot(range(10*len(res)), speed_traj_sim[:, j], ":", label=r"sim value")
                    plt.ylabel("Velocity at joint " + str(j))
                    plt.legend()
                plt.show()
        
        self.close_sim()
        
        return Sr
        

if __name__ == "__main__":
    from DexterousManipulation.simulations.utils_sim import get_background, get_robot
    from DexterousManipulation.generations.object_prior import ObjectPrior
    # Object
    obj_prior = ObjectPrior("../generations/ds_obj_config.json")
    obj = obj_prior.get_obj(17)
    # Background
    background = get_background("ur5")
    robot_dict = get_robot("ur5")
    # Trajectory simulator
    traj_sim = TrajectorySimulator(pybullet, robot_dict, background, None)
    engine_connection = "GUI"
    # Hand
    Rot1 = R.from_matrix(np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]))
    Rot2 = R.from_quat([0., 0., np.sin(np.pi/4), np.cos(np.pi/4)])
    #Rot = Rot2*Rot1
    #Rot = R.from_quat([0., 0., 0., 1.])
    Rot = Rot1
    h = np.concatenate(([[0., 0., 0.17]], Rot.as_matrix().reshape((1, -1)), [[1, 0., 0.]]), axis=-1)
    data = np.load("D:/generated_data/dataset_sandbox/train_ds/samples_0.npz")
    hand = data["hand"]
    frame = data["frame"]
    # Object frame
    #frame = [0., 0., 0.]
    # Inputs
    inputs = [obj, h, frame]
    # Sample
    nb_samples = 10
    Sr_bi = np.zeros((nb_samples, 1))
    Sr_connect = np.zeros((nb_samples, 1))
    import time
    start_time = time.time()
    for i, (hd, pO) in enumerate(zip(hand[0:nb_samples], frame[0:nb_samples])):
        print("i", i)
        h = np.array([hd])
        #print("Hand", h)
        #pO_center = [0., 0., 0.1]
        inputs = [obj, h, pO]
        Sr_bi[i] = traj_sim.forward(inputs, parameters=["birrt"], engine_connection=engine_connection)
        #Sr_connect[i] = traj_sim.forward(inputs, parameters=["rrt_connect"], engine_connection=engine_connection)
    print(time.time() - start_time) 
    print(np.mean(Sr_bi))
    print(np.mean(Sr_connect))
    
        
            