# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 10:19:28 2020
Adapted from: 
@author: norman marlier


Used for generating the grasp data 
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
from DexterousManipulation.simulations.utils_sim import is_contact, show_pose, load_vhacd_body
from DexterousManipulation.simulations.utils_sim import get_grasp_matrix, get_wrench_space, get_links
from DexterousManipulation.simulations.metrics import grasp_isotropy, VolumeCH
# Planner
from pybullet_planning.motion_planners import rrt_connect, birrt
from pybullet_planning.interfaces.planner_interface.joint_motion_planning import get_distance_fn, get_extend_fn, get_sample_fn, check_initial_end
from pybullet_planning.interfaces.robots import get_collision_fn
from pybullet_planning.interfaces.robots.joint import get_joint_positions



class RobotSimulator(Simulator):
    """Simulator of the reachability and grasping.
    
    It consists in a robotic arm, a gripper and a
    background.
    """
    def __init__(self, pybullet_module, robot_dict, background, 
                 gripper_controller, sim_parameters):
        """Constructor."""
        # Pybullet module
        self.p = pybullet_module
        # Robot filename
        self.robot_dict = robot_dict
        # Gripper controller
        self.gripper_ctrl = gripper_controller
        # Parameters
        self.sim_parameters = sim_parameters
        # Background
        self.background = background

        self.robot = 0 
        
        # NOTE: this will change ! (one for now == joint_distance)
        self.output_dim = 2
        
        self.RRT_RESTARTS = 30
        self.RRT_SMOOTHING = 20
    
    def start_sim(self, parameters, engine_connection, verbosity):
        # Start the simulation
        # Connection to a pybullet server
        if engine_connection == "GUI":
            self.engine_connection = self.p.GUI
        else:
            self.engine_connection = self.p.DIRECT
        # DEBUG MODE
        # The DEBUG MODE allows to show the simulation
        if self.engine_connection == self.p.GUI:
            self.debug = 1
        else:
            self.debug = 0
        # Connect to the server and put it into no realtime mode
        self.id_server = self.p.connect(self.engine_connection)

        # Apply gravity
        self.p.setGravity(0., 0., -9.81)
        self.p.setRealTimeSimulation(0, physicsClientId=self.id_server)
        self.p.setPhysicsEngineParameter(numSolverIterations=150,
                                         physicsClientId=self.id_server)
        if verbosity >= 1:
            print(self.p.getPhysicsEngineParameters(physicsClientId=self.id_server))
        # Put the camera at the same place
            self.p.resetDebugVisualizerCamera(1., 180, -41, [0., 0, 0.7],
                                              physicsClientId=self.id_server)
        
        # Setup the world
        self.setup_world(parameters[0], parameters[1], verbosity)

    def setup_world(self, parameters: list, obj_to_grasp, verbosity):
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
        # Robot Stand - if UR5
        self.robot_init_pos = self.robot_dict["base_pos"]
        self.robot_init_ori = self.robot_dict["base_ori"]
        if self.robot_dict["robot"] == "ur5":
            stand_x_pos = self.robot_init_pos[0] - 0.26
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
                                            basePosition=[stand_x_pos, 0., 0.84],
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
                                            basePosition=[stand_x_pos+0.14-0.275, 0., 0.4],
                                            baseOrientation=[0., 0., 0., 1.],
                                            useMaximalCoordinates=False,
                                            physicsClientId=self.id_server)
            self.obj_id["id"].append(stand2)
            self.obj_id["name"].append("big_extrusion")

        # Load the robot
        self.robot = self.p.loadURDF(self.robot_dict["urdf"],
                                     basePosition=self.robot_init_pos,
                                     baseOrientation=self.robot_init_ori,
                                     useFixedBase=True,
                                     flags=self.p.URDF_USE_SELF_COLLISION)
        # Self collision at loading
        self.p.stepSimulation()
        for pts in self.p.getContactPoints(self.robot):
            index_i = pts[3]
            index_j = pts[4]
            self.p.setCollisionFilterPair(self.robot, self.robot,
                                          index_i, index_j,
                                          0, physicsClientId=self.id_server)

        if verbosity == 2:
            get_links(self.p, self.robot, server_id=self.id_server)
            for joint in range(self.p.getNumJoints(self.robot)):
                print(self.p.getJointInfo(self.robot, joint))

        if self.debug:
            show_pose(self.p, self.robot_init_pos, [0, 0, 0, 1], self.id_server, scale=0.4)
        # Initial joint pose
        self.robot_joints = self.robot_dict["robot_joints"]
        if verbosity >= 1:
            print("Robot joint")
            print(self.robot_joints)
        self.init_joint_pose = self.robot_dict["init_joint_pose"]
        for i in range(len(self.robot_joints)):
            self.p.resetJointState(self.robot,
                                   self.robot_joints[i],
                                   self.init_joint_pose[i])

        # Load the object to grasp in [0., 0., 0.]
        parameters_obj = obj_to_grasp.copy()
        parameters_obj["base_pos"] = [0., 0., 0.]
        parameters_obj["useFixedBase"] = True
        # Load object
        self.grasped_object = load_vhacd_body(self.p, parameters_obj,
                                              server_id=self.id_server)


    def place_object(self, pO, obj):
        """Put an object on the table.

        obj is object dictionnary
        pO is a 2D pose, pO ~ p(pO)
        """
        # Set position and orientation BEFORE rotate around z axis
        obj_pos = [pO[0], pO[1], self.table_height + obj["z"]]
        #obj_pos = [0, 0, self.table_height + obj["z"]]
        """
        # Load the object
        parameters_obj = obj.copy()
        parameters_obj["base_pos"] = obj_pos
        parameters_obj["useFixedBase"] = True
        # Load object
        self.grasped_object = load_vhacd_body(self.p, parameters_obj,
                                              server_id=self.id_server)
        """

        
        # Set the position
        self.p.resetBasePositionAndOrientation(self.grasped_object,
                                               posObj=obj_pos,
                                               ornObj=obj["base_ori"],
                                               physicsClientId=self.id_server)
        # Rotate around z-axis
        obj_pos, obj_ori = self.p.getBasePositionAndOrientation(self.grasped_object,
                                                                physicsClientId=self.id_server)
        _, obj_ori = self.p.multiplyTransforms([0, 0, 0], [0., 0., np.sin(pO[2]/2), np.cos(pO[2]/2)],
                                               obj_pos, obj_ori)
        self.p.resetBasePositionAndOrientation(self.grasped_object, obj_pos, obj_ori,
                                               physicsClientId=self.id_server)
        #print("Printing object position:" , obj_pos)
        #print("Printing object orientation:" , obj_ori)

        return obj_pos, obj_ori

    def close_sim(self):
        """Close the simulation."""
        self.p.resetSimulation(self.id_server)
        self.p.disconnect(self.id_server)

    def forward(self, inputs: list, parameters: list, *args, **kwargs):
        # NOTE: This will change 
        """
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
        parameters: [lat_coef, spin_coef, planner]

        **kwargs:
            -engine_connection
            -verbosity
        """
        # Variables
        obj = inputs[0]
        h = inputs[1]
        pO = inputs[2]
        # Parameters
        m = obj["density"]*obj["volume"] #add condition on this 
        lat_coef = parameters[0]
        spin_coef = parameters[1]
        planner = parameters[2]
        table_scale = parameters[3]
        # Hyper parameters
        engine_connection = kwargs["engine_connection"]
        verbosity = kwargs["verbosity"]
        if "save_file" in kwargs.keys():
            save_file = kwargs["save_file"]
        else:
            save_file = None

        # Metrics
        # NOTE: add your own metrics 
        joint_distance = 0 
        # Sr
        Sr = 0.

        # Start sim
        self.start_sim([[table_scale], obj], 
                       engine_connection,
                       verbosity)
        
        # Position and orientation of the object - World coordinate
        obj_pos = [pO[0], pO[1], self.table_height + obj["z"]]
        obj_ori = [0., 0., np.sin(pO[2]/2), np.cos(pO[2]/2)]
        if self.debug:
            show_pose(self.p, obj_pos, obj_ori, self.id_server)
        # Grasp pose to end effector - World coordinate
        # Rotation matrix to quaternions
        hRot = R.from_matrix(h[0, 3:12].reshape((3, 3)))
        trans_gripper = self.robot_dict["trans_gripper"]
        trans_coupler = self.robot_dict["trans_coupler"]
        Tg, Qg = self.p.multiplyTransforms(h[0, 0:3], hRot.as_quat(), [trans_gripper + trans_coupler
                                                , 0., 0.], [0., 0., 0., 1.])
        if verbosity >= 1:
            print("Tg", Tg)
            print("Norm of Tg", np.linalg.norm(Tg))
        # Hand pose - World coodinate
        obj_pos_unbiased = [pO[0], pO[1], self.table_height] # no adding the object height ... 
        table_grasp_pos, table_grasp_ori = self.p.multiplyTransforms(obj_pos_unbiased, obj_ori,
                                                                     Tg, Qg)
        if self.debug:
            show_pose(self.p, table_grasp_pos, table_grasp_ori, self.id_server)
        
        # Inpsect the env
        if verbosity == 2:
            input("inspect")

        # Frame used by the IK solver
        jd = [0.01]*self.p.getNumJoints(self.robot)
        target_joints = self.p.calculateInverseKinematics(self.robot,
                                                          self.robot_dict["eef_link"],
                                                          table_grasp_pos,
                                                          table_grasp_ori,
                                                          jointDamping=jd)
        # Check if the solution given by the IK is reachable
        # Put the robot to the solution found by IK
        for i in range(len(self.robot_joints)):
            self.p.resetJointState(self.robot, self.robot_joints[i],
                                   target_joints[self.robot_dict["robot_index"][i]])
        
        # Get the frame
        wanted_eef_frame = self.p.getLinkState(self.robot, self.robot_dict["eef_link"])[0:2]
        if self.debug:
            show_pose(self.p, wanted_eef_frame[0], wanted_eef_frame[1], self.id_server)
        # Inpsect the env
        if verbosity == 2:
            input("inspect")
        # Diff in translation
        diff_trans = np.linalg.norm(np.array(wanted_eef_frame[0])-np.array(table_grasp_pos))
        if verbosity >= 1:
            print("Target Grasp joint:", target_joints)
            print("Wanted frame trans", np.array(wanted_eef_frame))
            print("H xyz", np.array(table_grasp_pos))
            print("Diff in translation", diff_trans)
        if  diff_trans > np.random.default_rng().normal(1e-3, 1e-4):
            # Close sim
            self.close_sim()
            return np.array([Sr, joint_distance], dtype=np.float32)
        # TODO : Check orientation
        
        # Replace the robot in the inital configuration
        for i in range(len(self.robot_joints)):
            self.p.resetJointState(self.robot, self.robot_joints[i],
                                   self.init_joint_pose[i])
            
        # Place the object
        obj_pos, obj_ori = self.place_object(pO, obj)
        for i in range(100):
            self.p.stepSimulation()
        # Inpsect the env
        if verbosity == 2:
            # Draw a circle 
            input("inspect")

         
        # Open the gripper in order to avoid collision
        # Reset the controller
        self.gripper_ctrl.reset(self.robot, self.id_server)
        # Activate gripper controller
        self.gripper_ctrl.activate(self.p, self.id_server)
        self.gripper_ctrl.filter_collision(self.p, self.id_server)

        # Set the gripper to a preshape
        grasp_mode = np.argmax(h[0, 12:])
        self.gripper_ctrl.set_params(self.gripper_ctrl.get_mode(grasp_mode),
                                     self.p, id_server=self.id_server)
        # Open the gripper - force it
        #self.gripper_ctrl.open_gripper()
        self.gripper_ctrl.force_open(self.p, self.id_server)
        # Some simulation steps to stabilize numerical issues
        for steps in range(50):
            # Send signal to the gripper
            self.gripper_ctrl.step(self.p, self.id_server)
            self.p.stepSimulation(self.id_server)
            
        # Motion planning - Functions
        obstacles = [self.grasped_object] + self.obj_id["id"]
        if verbosity >= 1:
            print(obstacles)
        sample_fn = get_sample_fn(self.robot, self.robot_joints)
        distance_fn = get_distance_fn(self.robot, self.robot_joints)
        extend_fn = get_extend_fn(self.robot, self.robot_joints)
        collision_fn = get_collision_fn(self.robot, self.robot_joints, obstacles=obstacles, self_collisions=False)
        
        # Initial conf
        start_conf = get_joint_positions(self.robot, self.robot_joints)
        # End conf
        end_conf = target_joints[0:self.robot_dict["dof"]]
        if verbosity >= 1:
            print("Initial conf", start_conf)
            print("Target Grasp joint:", end_conf)
        # Motion planningPlanner
        if planner == "rrt_connect":
            res = rrt_connect(start_conf, end_conf, distance_fn, sample_fn, extend_fn, collision_fn,
                              iterations=self.RRT_ITERATIONS, tree_frequency=2,
                              max_time=1)
        elif planner == "birrt":
            res = birrt(start_conf, end_conf, distance_fn, sample_fn, extend_fn, collision_fn,
                        restarts=self.RRT_RESTARTS, smooth=self.RRT_SMOOTHING,
                        max_time=3)
        else:
            sys.exit("Error : not good planner")
        
        if res is None:
            # No trajectory found
            if verbosity >= 1:
                print("No trajectory found")
            self.close_sim()
            return np.array([Sr, joint_distance], dtype=np.float32)
        else:
            # The target is reachable
            Sr = 1.
            
            # Show the trajectory
            if self.debug:
                if verbosity >= 1:
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
                if verbosity == 2:
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
                        self.p.stepSimulation()
                        time.sleep(1./240.)
                if verbosity == 2:
                    input("stop")
                    # Print the joint trajectory
                    for j in range(self.robot_dict["dof"]):
                        plt.subplot(4, 2, j+1)
                        plt.plot(10*np.arange(len(res)), joint_traj_computed[:, j], label=r"desired value", marker="o")
                        #plt.plot(range(len(res)), joint_traj_computed[:, 0], ":")
                        plt.plot(range(10*len(res)), joint_traj_sim[:, j], ":", label=r"sim value")
                        plt.legend()
                    plt.show()
            # Else, put the robot to the solution found by IK
            else:
                for i in range(len(self.robot_joints)):
                    self.p.resetJointState(self.robot, self.robot_joints[i],
                                           target_joints[i])

        # Grasp test
        # Graspable metric
        # The graspable metric is a binary variable
        # which inform about the capabilities of the pose
        # for a specific gripper to grasp an object.
        # It is based on geometric informations.
        # Qgii is used here.
        # S = 1 if coll-free(pose_before_closing) and Qgii > threshold
        #   = 0 otherwise
        # Some simulation steps to stabilize numerical issues
        for steps in range(50):
            self.p.stepSimulation(self.id_server)
            if self.debug:
                time.sleep(1./240.)


        # Change its dynamic properties
        # Source
        # https://www.cds.caltech.edu/~murray/books/MLS/pdf/mls94-complete.pdf p218
        self.p.changeDynamics(self.grasped_object, -1, lateralFriction=lat_coef,
                              spinningFriction=spin_coef, mass=0.,
                              restitution=self.sim_parameters["restitution"])
        if verbosity >= 1:
            print("Density", obj["density"])
            print("Volume", obj["volume"])
            print("Mass", m)
            print("Lateral coefficient", lat_coef)
            print("Spinning coefficient", spin_coef)

        if verbosity == 2:
            input(">Press a key to close the gripper")

        # Performs steps for closing the gripper
        nb_steps = 300
        self.gripper_ctrl.close_gripper()
        #self.gripper_ctrl.step(self.p, self.id_server)
        gripper_state = np.zeros((nb_steps+1, len(self.gripper_ctrl.joint_states)))
        self.gripper_ctrl.update_joint_states(self.p, self.id_server)
        gripper_state[0] = self.gripper_ctrl.joint_states
        for steps in range(nb_steps):
            self.p.setGravity(0., 0., -9.81)
            self.p.stepSimulation(self.id_server)
            self.gripper_ctrl.step(self.p, self.id_server)
            self.gripper_ctrl.update_joint_states(self.p, self.id_server)
            gripper_state[steps+1] = self.gripper_ctrl.joint_states
            if self.debug:
                time.sleep(1./240.)

        if verbosity == 2:
            for j in range(len(self.gripper_ctrl.joint_states)):
                plt.subplot(2, np.ceil(len(self.gripper_ctrl.joint_states)/2), j+1)
                plt.plot(np.arange(nb_steps+1), gripper_state[:, j], label=r"sim value", marker="o")
                plt.ylabel("joint " + str(j))
                plt.hlines(self.gripper_ctrl.close_joint_states[self.gripper_ctrl.mode][j], 0, nb_steps)
                plt.legend()
            plt.show()
        # Let the gripper in the closing configuration - only 3 finger
        """
        if self.gripper_ctrl.gripper == "3_fingers":
            self.gripper_ctrl.set_joint_target(self.gripper_ctrl.joint_states)
            self.gripper_ctrl.step(self.p, self.id_server)
            self.p.stepSimulation()
        """

        #test = self.gripper_ctrl.joint_states 
        #print(test)
        """
        # Determine distance between both joints
        num_joints = self.p.getNumJoints(self.robot)

        for joint in range(num_joints): 
            print("Joint info:", pybullet.getJointInfo(self.robot, joint))
        """

        right_finger_dist = pybullet.getJointState(self.robot, 24)[0] # < 0 
        left_finger_dist = pybullet.getJointState(self.robot, 22)[0] # > 0 

        joint_distance = left_finger_dist - right_finger_dist
        

        # input("Press a key to stop ")
              
        self.close_sim()
        
        return np.array([Sr, joint_distance], dtype=np.float32)
    
    def restore(self, bullet_file, parameters, verbosity):
        """Restore the simulation."""
        
        self.start_sim(parameters, "GUI", verbosity)
        
        print(self.p.getNumBodies())
        self.p.restoreState(fileName=bullet_file)
        
        #input("stop")
        
        self.close_sim()
        

if __name__ == "__main__":
    from DexterousManipulation.simulations.utils_sim import get_background, get_robot
    from DexterousManipulation.generations.object_prior import ObjectPrior
    from DexterousManipulation.controllers.gripper_controller import Robotiq3fControllerRegular, Robotiq3fControllerSimple, SawyerElectricController
    
    sim = "sawyer"
    # Object
    obj_prior = ObjectPrior("../generations/egad.json")
    obj = obj_prior.get_obj(20)
    print(obj)
    # Background
    background = get_background(sim)
    robot_dict = get_robot(sim)
    # Gripper
    if sim == "ur5":
        gripper_ctrl = Robotiq3fControllerRegular(None, sim, None)
    elif sim =="sawyer":
        gripper_ctrl = SawyerElectricController(None, sim, None)
    # Simulation parameters
    sim_parameters = {"model": "soft",
                      "ng": 20,
                      "restitution": 0,
                      "threshold": 1e-3}
    # Robot simulator
    robot_sim = RobotSimulator(pybullet, robot_dict, background, gripper_ctrl,
                               sim_parameters)
    
    engine_connection = "GUI"
    # Hand
    Rot1 = R.from_matrix(np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]))
    Rot2 = R.from_quat([0., 0., np.sin(np.pi/4), np.cos(np.pi/4)])
    #Rot = Rot2*Rot1
    #Rot = R.from_quat([0., 0., 0., 1.])
    Rot = Rot1
    h = np.concatenate(([[0., 0., 0.25]], Rot.as_matrix().reshape((1, -1)), [[0, 1, 0]]), axis=-1)
    parameters = [1.5, 0.7*0.002, "birrt", 1.]
    # Object frame
    frame = [0., 0., 0.]
    # Inputs
    inputs = [obj, h, frame]
    # Sample
    nb_samples = 1
    Sr_bi = np.zeros((nb_samples, 4))
    Sr_connect = np.zeros((nb_samples, 4))
    import time
    start_time = time.time()
    bullet_file = "test.bullet"
    for i in range(nb_samples):
        print("i", i)
        inputs = [obj, h, frame]
        Sr_bi[i] = robot_sim.forward(inputs, parameters, engine_connection=engine_connection, verbosity=1, save_file=bullet_file)
        #Sr_connect[i] = robot_sim.forward(inputs, parameters=["rrt_connect"], engine_connection=engine_connection)
    print(time.time() - start_time) 
    print(np.mean(Sr_bi[:, 0]))
    print(np.mean(Sr_connect))
    #robot_sim.restore(bullet_file, [[parameters[3]], obj])