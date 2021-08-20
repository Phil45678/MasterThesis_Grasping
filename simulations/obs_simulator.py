# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 16:05:49 2020

@author: norman marlier
"""
import numpy as np
import trimesh
import sys
sys.path.append("../..")
from scipy.spatial.transform import Rotation as R
from kinect_pybullet.kinect_scanner import Kinect
from DexterousManipulation.simulations.simulator import Simulator
from DexterousManipulation.simulations.utils_sim import is_contact, show_pose, load_vhacd_body, show_pts, show_pose_matrix


class CameraSimulator(Simulator):
    """Camera simulator class."""
    def __init__(self, pybullet, background, output_dim, *args, **kwargs):
        """Constructor."""
        super(CameraSimulator, self).__init__(output_dim, *args, **kwargs)
        # Pybullet module
        self.p = pybullet
        # Background
        self.background = background
    
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
    
    def show_env(self):
        """ Show the background."""
        # GUI mode
        mode = self.p.GUI
        # Connect to the server
        self.id_server = self.p.connect(mode)
        # Setup background - no scale
        self.setup_world([1.])
        # Wait until getting the stop signal
        input("Stop inspection")
        # Disconnect
        self.p.disconnect(self.id_server)
    
    
class KinectSimulator(CameraSimulator):
    """Kinect simulator.
    
    Simulate an observation from an object T with a pose Ft
    o = f(T, Ft)
    """
    
    def __init__(self, pybullet, background, output_dim, *args, **kwargs):
        """Constructor.
        
        Background is a list of dic (urdf, base_pos, base_ori)
        """
        super(KinectSimulator, self).__init__(pybullet, background,
                                              output_dim, *args, **kwargs)
        
    def forward(self, inputs: list, parameters: list, *args, **kwargs):
        """Evaluate i knowing pO and O in a deterministic way i = f(pO, O).
        
        Inputs
        ------
        pO represents the 2D pose of the object:
            -Position in space
            -Rotation in 2D

        O represents an object with its properties:
            -scaling
            -orientation
            -geometric mesh
        
        Parameters
        ----------
        xk represents the position of the Kinect in space
        qk represents the orientation of the Kinect (quaternions)
        table_scale represents the dimension scaling of the table, ~ N(1, 0.005)
        

        inputs: [object, object pose]
        parameters: [d_xk, d_qk, table_scale]

        **kwargs:
            -noisy
            -engine_connection
        """
        # Inputs
        obj = inputs[0]
        pO = inputs[1]
        # Parameters
        d_xk = parameters[0]
        d_qk = parameters[1]
        table_scale = parameters[2]
        # Hyperparameters
        engine_connection = kwargs["engine_connection"]
        noisy = kwargs["noisy"]
        
        # Connection to a pybullet server
        if engine_connection == "GUI":
            mode = self.p.GUI
        else:
            mode = self.p.DIRECT
        # DEBUG MODE
        # The DEBUG MODE allows to print some variables and interacts with
        # the simulation
        if mode == self.p.GUI:
            debug = True
        else:
            debug = False
        # Connect to the server
        self.id_server = self.p.connect(mode)
        # Setup background
        self.setup_world([table_scale])
        # Kinect sensor
        kinect_ori_true = self.background[2]["base_ori"]
        # Add small perturbation
        _, kinect_ori = self.p.multiplyTransforms([0.]*3, d_qk,
                                                  [0.]*3, kinect_ori_true)
        # Position
        xk_true = self.background[2]["base_pos"]
        show_pose(self.p, xk_true, kinect_ori, self.id_server)
        xk = xk_true + d_xk
        self.kinect = Kinect(self.p,
                             xk,
                             kinect_ori,
                             self.id_server)        
        # Set position and orientation
        # pO gives information on x,y position and rotation around z-axis
        obj_pos = [pO[0], pO[1], self.table_height]
        _, obj_ori = self.p.multiplyTransforms([0, 0, 0], [0., 0., np.sin(pO[2]/2), np.cos(pO[2]/2)],
                                              [0., 0., 0], obj["base_ori"])
        parameters_obj = obj.copy()
        parameters_obj["base_pos"] = obj_pos
        parameters_obj["base_ori"] = list(obj_ori)
        # Load object
        grasped_object = load_vhacd_body(self.p, parameters_obj, server_id=self.id_server)
        
        if debug:
            input("Start sim")

        # Scan
        obs = self.kinect.scan(self.p, debug, add_noise=noisy, add_reflectivity=noisy)
        obs = np.flipud(obs)
        
        # Debug
        if debug:
            input("press a key to end >")
        
        # Reset simulation
        self.p.resetSimulation(self.id_server)
    
        # Close connection
        self.p.disconnect(self.id_server)

        return obs
    
    def sample(self, inputs, sim_noise, parameters_noise, *args, **kwargs):
        """Generate observation i from an object and a frame.

        i ~ p(i|xO, O)

        Parameters and hyperparameters are set here.

        inputs = [h, O]
        sim_noise (bool): add noise to the simulation itself
        parameters_noise (bool): add noise to the parameters


        **kwargs:
            -engine_connection
        """
        # Sample
        outputs = np.zeros((1, 480, 640))
        # Parameters
        if parameters_noise:
            # Paper:
            # Learning ambidextrous robot grasping policies
            # Supp material
            # Position in space of the Kinect : 5x5x5cm
            x_box = 0.05
            y_box = 0.05
            z_box = 0.05
            Xk_x = np.random.default_rng().uniform(-x_box/2, x_box/2, 1)
            Xk_y = np.random.default_rng().uniform(-y_box/2, y_box/2, 1)
            Xk_z = np.random.default_rng().uniform(-z_box/2, z_box/2, 1)
            d_xk = np.array([Xk_x, Xk_y, Xk_z]).reshape(-1)
            # Orientation of the Kinect
            # Draw from a multivariate gaussian a increment of rotation
            # Mean and Covariance matrix
            mean = [0., 0., 0.]
            std_rx = 0.002
            std_ry = 0.01
            std_rz = 0.002
            cov = [[std_rx**2, 0., 0.],
                   [0., std_ry**2, 0.],
                   [0., 0., std_rz**2]]
            # Sample
            incr_rot = KinectSimulator.sample_incr_rot(mean, cov)
            incr_rot = R.from_matrix(incr_rot)
            d_qk = incr_rot.as_quat()
            # Scale of the table
            table_scale = np.random.default_rng().normal(1., 0.005, 1)
        else:
            d_xk = np.zeros(3)
            d_qk = np.array([0., 0., 0., 1.])
            table_scale = 1
        parameters = [d_xk, d_qk, table_scale]
        engine_connection = kwargs["engine_connection"]
        outputs[0] = self.forward(inputs, parameters, noisy=sim_noise,
                                  engine_connection=engine_connection)
        return outputs

    @staticmethod
    def sample_incr_rot(mean, cov):
        """R = R exp_map(û), u ~ N(mean, cov)."""
        # Increment and its norm
        u = np.random.default_rng().multivariate_normal(mean, cov, 1)[0]
        norm_u = np.linalg.norm(u)
        # Tilde operator
        u_tilde = np.array([[0., -u[2], u[1]],
                            [u[2], 0., -u[0]],
                            [-u[1], u[0], 0.]])
        # Exponential map
        incr_rot = np.eye(3) + (np.sin(norm_u)/norm_u)*u_tilde + ((1-np.cos(u))/norm_u**2)*u_tilde@u_tilde
        
        return incr_rot


class DepthSimulator(CameraSimulator):
    """Class which uses the perfect depth from OpenGL buffer.

    Simulate an observation from an object T with a pose Ft
    o = f(T, Ft)
    """

    def __init__(self, pybullet, background, output_dim, *args, **kwargs):
        """Constructor.
        
        Background is a list of dic (urdf, base_pos, base_ori)
        """
        super(DepthSimulator, self).__init__(pybullet, background,
                                             output_dim, *args, **kwargs)
        # Features of the depth sensor
        # Copy to Kinect
        self.xres = 480
        self.yres = 640
        self.fov = 43.1845
        self.aspect = self.yres / self.xres
        self.min_dist = 0.4
        self.max_dist = 7
        self.flength = 4.73
        
        # Renderer - Use of TINY RENDERER to be used in cluster
        self.renderer = self.p.ER_TINY_RENDERER
        
    def forward(self, inputs: list, parameters: list, *args, **kwargs):
        """Evaluate i knowing pO and O in a deterministic way i = f(pO, O).
        
        Inputs
        ------
        pO represents the 2D pose of the object:
            -Position in space
            -Rotation in 2D

        O represents an object with its properties:
            -scaling
            -orientation
            -geometric mesh
        
        Parameters
        ----------
        xk represents the position of the depth senosr in space
        qk represents the orientation of the depth sensor (quaternions)
        table_scale represents the dimension scaling of the table, ~ N(1, 0.005)
        

        inputs: [object, object pose]
        parameters: [d_xk, d_qk, table_scale]

        **kwargs:
            -noisy
            -engine_connection
        """
        # Inputs
        obj = inputs[0]
        pO = inputs[1]
        # Parameters
        d_xk = parameters[0]
        d_qk = parameters[1]
        table_scale = parameters[2]
        # Hyperparameters
        engine_connection = kwargs["engine_connection"]
        noisy = kwargs["noisy"]

        # Connection to a pybullet server
        if engine_connection == "GUI":
            mode = self.p.GUI
        else:
            mode = self.p.DIRECT
        # DEBUG MODE
        # The DEBUG MODE allows to print some variables and interacts with
        # the simulation
        if mode == self.p.GUI:
            debug = True
        else:
            debug = False
        # Connect to the server
        self.id_server = self.p.connect(mode)
        # Setup background
        self.setup_world([table_scale])
        
        # Depth sensor features
        if noisy:
            sigma_min_max = 0.05
            error = 0.02
        else:
            sigma_min_max = 0.
            error = 0.
            
        near = np.random.default_rng().normal(self.min_dist, sigma_min_max, 1)
        far = np.random.default_rng().normal(self.max_dist, sigma_min_max, 1)
        low = (1 - error) * self.flength
        high = (1 + error) * self.flength
        flength = np.random.default_rng().uniform(low, high, 1)
        
        # View matrix
        # Position
        xk_true = self.background[2]["base_pos"]
        xk = xk_true + d_xk
        # Orientation
        qz = [0., 0., np.sin(np.pi/2), np.cos(np.pi/2)]
        _, qk = self.p.multiplyTransforms([0.]*3, d_qk, [0.]*3, qz)
        rot = R.from_quat(qk)
        z = rot.as_matrix()[:, 2]
        if debug:
            print("z", z)
        
        # Set position and orientation of the target
        inv_xk, inv_qk = self.p.invertTransform(xk, qk)
        target_pos, target_ori = [flength, 0., 0.], [0., 0., 0., 1.]
        target_pos, _ = self.p.multiplyTransforms(inv_xk, inv_qk, target_pos, target_ori)
        
        # pO gives information on x,y position and rotation around z-axis
        obj_pos = [pO[0], pO[1], self.table_height]
        _, obj_ori = self.p.multiplyTransforms([0, 0, 0], [0., 0., np.sin(pO[2]/2), np.cos(pO[2]/2)],
                                              [0., 0., 0], obj["base_ori"])
        
        parameters_obj = obj.copy()
        parameters_obj["base_pos"] = obj_pos
        parameters_obj["base_ori"] = list(obj_ori)
        # Load the object
        grasped_object = load_vhacd_body(self.p, parameters_obj, server_id=self.id_server)
        
        if debug:
            input("Start sim")
        view_matrix = self.p.computeViewMatrix(xk, target_pos, z)

        projection_matrix = self.p.computeProjectionMatrixFOV(self.fov, 
                                                              self.aspect, 
                                                              near,
                                                              far)
        
        # Scan
        images = self.p.getCameraImage(self.yres,
                                       self.xres,
                                       view_matrix,
                                       projection_matrix,
                                       shadow=True,
                                       renderer=self.renderer)
        rgb_opengl = np.reshape(images[2], (self.xres, self.yres, 4)) * 1. / 255.
        depth_buffer_opengl = np.reshape(images[3], [self.xres, self.yres])
        obs = far * near / (far - (far - near) * depth_buffer_opengl)
        
        # Reset simulation
        self.p.resetSimulation(self.id_server)
    
        # Close connection
        self.p.disconnect(self.id_server)

        return obs
    
    def sample(self, inputs, sim_noise, parameters_noise, *args, **kwargs):
        """Generate observation i from an object and a frame.

        i ~ p(i|xO, O)

        Parameters and hyperparameters are set here.

        inputs = [h, O]
        sim_noise (bool): add noise to the simulation itself
        parameters_noise (bool): add noise to the parameters


        **kwargs:
            -engine_connection
        """
        # Sample
        outputs = np.zeros((1, 480, 640))
        # Parameters
        if parameters_noise:
            # Paper:
            # Learning ambidextrous robot grasping policies
            # Supp material
            # Position in space of the Kinect : 5x5x5cm
            x_box = 0.05
            y_box = 0.05
            z_box = 0.05
            Xk_x = np.random.default_rng().uniform(-x_box/2, x_box/2, 1)
            Xk_y = np.random.default_rng().uniform(-y_box/2, y_box/2, 1)
            Xk_z = np.random.default_rng().uniform(-z_box/2, z_box/2, 1)
            d_xk = np.array([Xk_x, Xk_y, Xk_z]).reshape(-1)
            # Orientation of the Kinect
            # Draw from a multivariate gaussian a increment of rotation
            # Mean and Covariance matrix
            mean = [0., 0., 0.]
            std_rx = 0.002
            std_ry = 0.01
            std_rz = 0.002
            cov = [[std_rx**2, 0., 0.],
                   [0., std_ry**2, 0.],
                   [0., 0., std_rz**2]]
            # Sample
            incr_rot = KinectSimulator.sample_incr_rot(mean, cov)
            incr_rot = R.from_matrix(incr_rot)
            d_qk = incr_rot.as_quat()
            # Scale of the table
            table_scale = np.random.default_rng().normal(1., 0.005, 1)
        else:
            d_xk = np.zeros(3)
            d_qk = np.array([0., 0., 0., 1.])
            table_scale = 1
        parameters = [d_xk, d_qk, table_scale]
        engine_connection = kwargs["engine_connection"]
        outputs[0] = self.forward(inputs, parameters, noisy=sim_noise,
                                  engine_connection=engine_connection)
        return outputs

    @staticmethod
    def sample_incr_rot(mean, cov):
        """R = R exp_map(û), u ~ N(mean, cov)."""
        # Increment and its norm
        u = np.random.default_rng().multivariate_normal(mean, cov, 1)[0]
        norm_u = np.linalg.norm(u)
        # Tilde operator
        u_tilde = np.array([[0., -u[2], u[1]],
                            [u[2], 0., -u[0]],
                            [-u[1], u[0], 0.]])
        # Exponential map
        incr_rot = np.eye(3) + (np.sin(norm_u)/norm_u)*u_tilde + ((1-np.cos(u))/norm_u**2)*u_tilde@u_tilde
        
        return incr_rot
    
if __name__ == "__main__":
    import pybullet
    import matplotlib.pyplot as plt
    from DexterousManipulation.training.dataset import filter_img_tf
    
    import time
    import open3d as o3d
    from DexterousManipulation.generations.object_prior import ObjectPrior
    from DexterousManipulation.simulations.utils_sim import get_background, get_obj_pts_from_kinect_depth
    from DexterousManipulation.simulations.utils_sim import load_pts3d, get_pts_from_kinect_depth
    obj_prior = ObjectPrior("../generations/ds_obj_config.json")
    obj = obj_prior.get_obj(17)
    print(obj)
    frame = [0., 0., 0]
    engine_connection = "DIRECT"
    # Background
    background = get_background("ur5")
    print("Camera", background[2])
    # Simulator
    kinect_sim = KinectSimulator(pybullet, background, (480, 640))
    
    t = time.time()
    kinect_origin = kinect_sim.sample([obj, frame],
                                      parameters_noise=False, 
                                      sim_noise=True,
                                      engine_connection=engine_connection)[0]
    """
    kinect_origin = np.load("../experiments/imgs/ur5/train_ds/035_power_drill/img_00.npz")["img"]
    """
    kinect_origin = np.nan_to_num(kinect_origin)
    
    kinect_origin = np.where(kinect_origin < 1.1, kinect_origin, np.zeros(shape=kinect_origin.shape))
    kinect_origin[400:] = 0.
    kinect_origin_o3d_pts = get_pts_from_kinect_depth(kinect_origin)
    o3d.io.write_point_cloud("kinect_point_cloud.ply", kinect_origin_o3d_pts, print_progress=True)
    kinect_o3d_pts = get_obj_pts_from_kinect_depth(kinect_origin)
    o3d.io.write_point_cloud("object_point_cloud.ply", kinect_o3d_pts, print_progress=True)
    kinect_o3d_pts.translate([frame[0], frame[1], 0.625 + obj["z"]], False)
    
    plt.imshow(kinect_origin)
    plt.show()
    
    pts = np.asarray(kinect_o3d_pts.points).copy()
    print(pts.shape)
    plt.hist(pts[:, 2])
    plt.show()
    plt.hist(pts[:, 0])
    plt.show()
    plt.hist(pts[:, 1])
    plt.show()
    
    server_id = pybullet.connect(pybullet.GUI)
    
    background = get_background("ur5")
    obj_id = {"id": [], "name": []}
    for obj in background:
        if "urdf" in obj.keys():
            if "table" in obj["urdf"]:
                # Compute scaling
                scale = 1.
                table_height = obj["table_height"]
            else:
                scale = 1.
            # Load object
            obj_id["id"].append(pybullet.loadURDF(obj["urdf"],
                                globalScaling=scale,
                                basePosition=obj["base_pos"],
                                baseOrientation=obj["base_ori"],
                                physicsClientId=server_id))
            obj_id["name"].append(obj["urdf"])
    pybullet.stepSimulation()
    pts3d_id = load_pts3d(pybullet, pts, server_id)
    input('stop')
    pybullet.disconnect()
    
    """
    kinect = kinect_origin.reshape((1, 480, 640, 1))
    kinect_filter = filter_img_tf(kinect)
    #kinect_scale_pts3d = kinect_sim.kinect.get_pts3d_from_depth(kinect_filter.numpy()[0, :, :, 0])
    
    mesh = trimesh.Trimesh(vertices=kinect_pts3d)
    mesh.export("test_kinect.ply")
    
    
    obs_sim = DepthSimulator(pybullet, background, (480, 640))
    # Sample
    t = time.time()
    samples = obs_sim.sample([obj, frame], True, False,
                             engine_connection=engine_connection)
    samples = samples.reshape((1, 480, 640, 1))
    samples = filter_img_tf(samples)
    print(time.time()-t)

    # Show
    plt.subplot(121)
    plt.imshow(kinect_inv_filter[0, :, :, 0])
    #plt.axis("off")
    plt.subplot(122)
    plt.imshow(kinect[0, :, :, 0])
    plt.show()
    """

