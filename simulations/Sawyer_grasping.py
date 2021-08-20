"""
Simulation of the Sawyer gripper

Authors: 
    - Norman Marlier
    - Philippe Schneider 


"""



import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../..")
from scipy.spatial.transform import Rotation as R
from DexterousManipulation.simulations.simulator import Simulator
from DexterousManipulation.simulations.utils_sim import load_vhacd_body, is_contact, get_grasp_matrix, get_wrench_space, get_links
from DexterousManipulation.simulations.utils_sim import show_pose, show_pose_matrix, show_friction_cone, linearized_friction_cone
from DexterousManipulation.simulations.metrics import min_singular, grasp_isotropy, VolumeCH
from DexterousManipulation.generations.hand_prior import HandPrior



class GraspSimulator(Simulator):
    """Grasp simulator.

    Evaluate a metric Sg knowing the hand configuration and the object.

    Implement the forward pass:
    Sg = f(h, T)
    """

    def __init__(self, pybullet_module, gripper, gripper_ctrl, sim_parameters, output_dim, *args, **kwargs):
        """Constructor."""
        super(GraspSimulator, self).__init__(output_dim, *args, **kwargs)
        # Pybullet module
        self.p = pybullet_module
        # Gripper
        self.gripper = gripper
        # Gripper controller
        self.gripper_ctrl = gripper_ctrl
        # Intrinsic parameters
        self.sim_parameters = sim_parameters

    def forward(self, inputs: list, parameters: list, *args, **kwargs):
        """Evaluate Sg knowing h and O in a deterministic way Sg = f(h, O).

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

        Parameters
        ----------
        friction coefficient µ
        torsion friction coefficient gamma
        mass m

        inputs: [object, hand configuration]
        parameters: [µ, gamma, m]

        **kwargs:
            -engine_connection
        """
        # Inputs
        obj = inputs[0]
        hand_configuration = inputs[1]
        # Parameters
        m = parameters[0]
        lat_coef = parameters[1]
        spin_coef = parameters[2]
        # Hyper parameters
        engine_connection = kwargs["engine_connection"]
        noisy = kwargs["noisy"]
        # Output
        metric = np.zeros(shape=self.output_dim)
        # Connection to a pybullet server
        if engine_connection == "GUI":
            mode = self.p.GUI
        else:
            mode = self.p.DIRECT
        # DEBUG MODE
        # The DEBUG MODE allows to print some variables and interacts with
        # the simulation
        if mode == self.p.GUI:
            debug = 1
        else:
            debug = 0
        
        self.verbosity = kwargs["verbosity"]
        # Connect to the server and put it into no realtime mode
        self.id_server = self.p.connect(mode)

        self.p.setRealTimeSimulation(0, physicsClientId=self.id_server)
        self.p.setPhysicsEngineParameter(numSolverIterations=150,
                                         physicsClientId=self.id_server)
        if debug:
            print(self.p.getPhysicsEngineParameters(physicsClientId=self.id_server))
        # Put the camera at the same place
        if debug:
            self.p.resetDebugVisualizerCamera(1.3, 180, -41, [0., -0.2, -0.33],
                                         physicsClientId=self.id_server)
        # Gravity
        self.p.setGravity(0., 0., -9.81)
        # Stable configuration
        plane = self.p.loadURDF("../data/plane.urdf")
        # Got the hand configuration
        grasp_trans = hand_configuration[0, 0:3]
        grasp_trans[2] += obj["z"]
        r = R.from_matrix(hand_configuration[0, 3:12].reshape((3, 3)))
        grasp_ori = r.as_quat()
        grasp_mode = np.argmax(hand_configuration[0, 12:15])
        # Need to change the orientation because it is not the same convention
        if debug:
            show_pose(self.p, grasp_trans, grasp_ori, self.id_server)
            print("Gripper ori:", grasp_ori)
            print("Grasp pos in world coordinates: ", grasp_trans)
        Tg, Qg = self.gripper.get_transform_eef()
        gripper_trans, gripper_ori = self.p.multiplyTransforms(grasp_trans, grasp_ori,
                                                               Tg, Qg)
        if debug:
            print(gripper_ori)
        # Load the gripper
        gripper_params = {"base_pos": gripper_trans, "base_ori": gripper_ori,
                          "fixed_base": False, "plane_id": plane}
        if debug:
            print(gripper_params["base_ori"])
        self.gripper.load(gripper_params, self.p, self.id_server)
        gripper = self.gripper.gripper_id
        get_links(self.p, gripper, server_id=self.id_server)
        # Create a constraint for the gripper
        constraint_id = self.p.createConstraint(gripper, -1, -1, -1, self.p.JOINT_FIXED,
                                 [0, 0, 0], [0, 0, 0], gripper_trans,
                                 childFrameOrientation=gripper_ori,
                                 physicsClientId=self.id_server)
        # Reset the controller
        self.gripper_ctrl.reset(gripper, self.id_server)
        # Activate gripper controller
        self.gripper_ctrl.activate(self.p, self.id_server)
        # Set the gripper to a preshape
        self.gripper_ctrl.set_params(self.gripper.get_mode(grasp_mode),
                                     self.p, id_server=self.id_server)
        # Open the gripper
        self.gripper_ctrl.open_gripper()
        # Send signal to the gripper
        self.gripper_ctrl.step(self.p, self.id_server)
        # Change the color
        if debug:
            nb_joints = self.p.getNumJoints(gripper)
            for i in range(-1, nb_joints):
                self.p.changeVisualShape(gripper, i, rgbaColor=(0.2, 0.2, 0.2, 1))
    
        # Graspable metric
        # The graspable metric is a binary variable
        # which inform about the capabilities of the pose
        # for a specific gripper to grasp an object.
        # It is based on geometric informations.
        # Qgii is used here.
        # S = 1 if coll-free(pose_before_closing) and Qgii > threshold
        #   = 0 otherwise
        # Collision before ?
        # Some simulation steps to stabilize numerical issues
        for steps in range(10):
            self.p.stepSimulation(self.id_server)
        # Load the object to grasp
        grasped_object = load_vhacd_body(self.p, obj, server_id=self.id_server)
        pos, _ = self.p.getBasePositionAndOrientation(grasped_object, physicsClientId=self.id_server)
        # Some simulation steps to stabilize numerical issues
        for steps in range(10):
            self.p.stepSimulation(self.id_server)
        # Change its dynamic properties
        # Mass = density*volume
        #m = density*obj["volume"]
        # Source
        # https://www.cds.caltech.edu/~murray/books/MLS/pdf/mls94-complete.pdf p218
        self.p.changeDynamics(grasped_object, -1, lateralFriction=lat_coef,
                              spinningFriction=spin_coef, mass=m, 
                              restitution=self.sim_parameters["restitution"])
        if debug:
            print("Density", m/obj["volume"])
            print("Mass", m)
            print("Lateral coefficient", lat_coef)
            print("Spinning coefficient", spin_coef)
        # Coll-free(pose_before_closing) with the object
        contact_pts = self.p.getContactPoints(gripper, grasped_object,
                                              physicsClientId=self.id_server)
        collide_before = is_contact(contact_pts)
        # Coll-free(pose_before_closing) with the support
        stand_contact_pts = self.p.getContactPoints(gripper, plane,
                                                    physicsClientId=self.id_server)
        if debug:
            print(stand_contact_pts)
        collide_before = is_contact(stand_contact_pts) or collide_before
        # Qgii
        Qgii = 0.
        #Vgws
        Vgws = 0.
        # Sg
        Sg = 0.
        # St
        St = 0.
        if debug:
            print("Collide before: ", collide_before)
            print("Show contact distance")
            for i, pts in enumerate(contact_pts):
                print("Contact distance", pts[8])
            input(">Press a key to close the gripper")

        # Performs steps for closing the gripper
        # 255 is chosen because the gripper has 255 steps max
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
            if debug:
                time.sleep(1./240.)
        """
        if self.verbosity == 2:
            for j in range(len(self.gripper_ctrl.joint_states)):
                plt.subplot(2, np.ceil(len(self.gripper_ctrl.joint_states)/2), j+1)
                plt.plot(np.arange(nb_steps+1), gripper_state[:, j], label=r"sim value", marker="o")
                plt.ylabel("joint " + str(j))
                plt.hlines(self.gripper_ctrl.close_joint_states[self.gripper_ctrl.mode][j], 0, nb_steps)
                plt.legend()
            plt.show()

        """
        # Let the gripper in the closing configuration - only 3 finger
        if self.gripper_ctrl.gripper == "3_fingers":
            self.gripper_ctrl.set_joint_target(self.gripper_ctrl.joint_states)
            self.gripper_ctrl.step(self.p, self.id_server)
            self.p.stepSimulation()

        # Compute the Grasp matrix
        contact_pts = self.p.getContactPoints(grasped_object, gripper,
                                              physicsClientId=self.id_server)
        base_pos, _ = self.p.getBasePositionAndOrientation(grasped_object)
        G = get_grasp_matrix(base_pos, contact_pts,
                             model=self.sim_parameters["model"])
        # Qgii
        Qgii = grasp_isotropy(G)
        # GWS
        GWS = get_wrench_space(contact_pts, base_pos,
                               lat_coef, spin_coef,
                               ng=self.sim_parameters["ng"],
                               model=self.sim_parameters["model"])
        Vgws = VolumeCH(GWS)
        if self.verbosity >= 1:
            np.set_printoptions(precision=3)
            print("Qgii", Qgii)
            if len(contact_pts) != 0:
                print("G Rank", np.linalg.matrix_rank(G))
                print("GWS rank:", np.linalg.matrix_rank(GWS))
                print("Volume of Convex Hull:", Vgws)
        if self.verbosity == 2:
            input(">Press a key to end the simulation")
    
        # Update the metric
        # metric = coll-free() & Qgii > threshold
        Sg = not(collide_before) and (Qgii > self.sim_parameters["threshold"])
        
        metric = np.array([Sg, Qgii, Vgws], dtype=np.float32)
        # Reset simulation
        self.p.resetSimulation(self.id_server)
    
        # Close connection
        self.p.disconnect(self.id_server)
    
        return metric
    
    def sample(self, inputs, sim_noise, parameters_noise, *args, **kwargs):
        """Evaluate Sg knowing h and O, i.e Sg ~ p(Sg|h, O, alpha, beta).

        inputs = [h, O]
        sim_noise (bool): add noise to the simulation itself
        parameters_noise (bool): add noise to the parameters


        **kwargs:
            -engine_connection
        """
        # Parameters
        if parameters_noise:
            mass = np.random.default_rng().uniform(0.05, 2., 1) # kg
            lat_coef = np.random.default_rng().uniform(0.1, 1.5, 1)
            spin_coef = lat_coef*np.random.default_rng().normal(2*1e-3, 1e-4) # gamma = µ*radius, radius ~ N(0.002, 0.0001)
        else:
            mass = 0.7 # kg
            lat_coef = 0.5
            spin_coef = lat_coef*2*1e-3 # gamma = µ*radius, radius ~ N(0.002, 0.0001)
        parameters = [mass, lat_coef, spin_coef]
        # Hyperparameters
        engine_connection = kwargs["engine_connection"]
        verbosity = kwargs["verbosity"]
        # Deterministic simulation
        outputs = self.forward(inputs, parameters, noisy=sim_noise,
                               engine_connection=engine_connection,
                               verbosity=verbosity)

        return outputs.reshape((1, self.output_dim))


if __name__ == "__main__":
    import time
    from DexterousManipulation.controllers.gripper_controller import Robotiq3fControllerSimple, Robotiq3fControllerRegular, SawyerElectricController
    from DexterousManipulation.controllers.gripper import Robotiq3fGripper, SawyerElectricGripper
    from DexterousManipulation.generations.object_prior import ObjectPrior
    # Importing object
    obj_prior = ObjectPrior("../generations/egad.json")
    obj = obj_prior.get_obj(10)
    obj["scale"] = [1.]*3
    obj_prior.object_dataset.set_geometrical_attributs(obj)
    print(obj)

    Rot1 = R.from_matrix(np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]))
    Rot2 = R.from_quat([0., 0., np.sin(np.pi/4), np.cos(np.pi/4)])
    #Rot = Rot2*Rot1
    #Rot = R.from_quat([0., 0., 0., 1.])
    Rot = Rot1
    h = np.concatenate(([[0., 0., 0.148]], Rot.as_matrix().reshape((1, -1)), [[0., 1., 0.]]), axis=-1)

    # Using new prior here 
    pos_parameters = {"distrib": "shell",
                        "rdistrib": "truncnorm", 
                        "rlow": 0.24, 
                        "rhigh" : 0.26, 
                      "rmin": 0.23,
                      "rmax": 0.27,
                      "shell": 0.0}
    
    hand_prior = HandPrior(pos_parameters, {"kappa": 20, "dof" : 3}, {"preshape": [1./3]*3})
    test = hand_prior.joint_distribution.sample(1)
    for element in test: 
        print(element)
    h_sim = hand_prior.to_network(test).numpy()
    
    h = h_sim

    # Hyperparameters
    engine_connection = "GUI"
    # Gripper
    #gripper = Robotiq3fGripper("../data/robotiq_3f_gripper_visualization/cfg/robotiq-3f-gripper_articulated.urdf")
    #gripper_ctrl = Robotiq3fControllerRegular(None, "gripper", None)
    gripper = SawyerElectricGripper("../data/rethink_ee_description/urdf/electric_gripper/sawyer_gripper.urdf")
    gripper_ctrl = SawyerElectricController(None, "gripper", None)
    # Simulation parameters
    sim_parameters = {"model": "soft",
                      "ng": 20,
                      "restitution": 0,
                      "threshold": 1e-3}
    # Create the grasp simulator
    import pybullet
    grasp_sim = GraspSimulator(pybullet, gripper, gripper_ctrl, sim_parameters, 3)
    ts = time.time()
    res = grasp_sim.sample([obj, h], False, False,
                           engine_connection=engine_connection,
                           verbosity=2)
    print("Time:", time.time()-ts)
    print("Res:", res)
   
