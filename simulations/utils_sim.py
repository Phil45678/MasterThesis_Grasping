# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 12:15:06 2019

@author: norman marlier
"""
import numpy as np
import os, inspect
try:
    import trimesh
except ImportError:
    print("Cannot import trimesh")
finally:
    pass
try:
    import tensorflow as tf
except ImportError:
    print("Cannot import tensorflow")
finally:
    pass
try:
    import open3d as o3d
except ImportError:
    print("Cannot import open3d")
finally:
    pass
from scipy.spatial.transform import Rotation as R
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.join(currentdir, "../gym")


def get_background(background="ur5"):
    """Return a  list of dic of urdf for background."""
    plane_dic = {"urdf": "../data/plane.urdf",
                 "base_pos": [0., 0., 0.],
                 "base_ori": [0., 0., 0., 1.]}
    if background == "ur5":
        table_dic = {"urdf": "../data/table_labo.urdf",
                     "base_pos": [0., 0., 0.],
                     "base_ori": [0, 0, 0, 1.],
                     "table_height": 0.625}
        # Camera orientation
        R1 = R.from_quat([0., 0., np.sin(np.pi/4), np.cos(np.pi/4)])
        y_angle = 70.
        R2 = R.from_quat([0, np.sin(y_angle/180*np.pi/2), 0., np.cos(np.pi/6)])
        Rtot = R2 * R1
        camera = {"id": "camera",
                  "base_pos": np.array([0.72, 0.02, 1]),
                  "base_ori": Rtot.as_quat()}
        
    elif background == "sawyer":
        table_dic = {"urdf": "../data/table_sawyer.urdf",
                     "base_pos": [0., 0., 0.],
                     "base_ori": [0, 0, 0, 1.],
                     "table_height": 0.783}
        # Camera orientation
        R1 = R.from_quat([0., 0., np.sin(np.pi/4), np.cos(np.pi/4)])
        y_angle = 55.
        R2 = R.from_quat([0, np.sin(y_angle/180*np.pi/2), 0., np.cos(np.pi/6)])
        Rtot = R2 * R1
        camera = {"id": "camera",
                  "base_pos": np.array([0.61, 0.0, 1.215]),
                  "base_ori": Rtot.as_quat()}
    background = [plane_dic, table_dic, camera]
    
    return background

def get_robot(robot="ur5"):
    """Return a dict with information about the robot."""
    if robot == "ur5":
        robot_dict = {"urdf": "../data/ur5_eef.urdf",
                     "base_pos": [-0.6, 0., 0.88], # True values [-0.7, 0., 0.88]
                     "base_ori": [0., 0., np.sin(np.pi*3/8), np.cos(np.pi*3/8)],
                     "robot_joints": [1, 2, 3, 4, 5, 6],
                     "robot_index": [0, 1, 2, 3, 4, 5],
                     "init_joint_pose": [np.pi/4, -np.pi/2, 0., -np.pi/2, np.pi/2, 0],
                     "trans_gripper": -0.054,
                     "trans_coupler": -0.036,
                     "eef_link": 7,
                     "robot": "ur5"}
    elif robot == "sawyer":
        robot_dict = {"urdf": "../data/sawyer_eef.urdf",
                     "base_pos": [-0.7, 0., 0.95],
                     "base_ori": [0., 0., 0., 1.],
                     "robot_joints": [2, 7, 8, 9, 10, 12, 15],
                     "robot_index": [0, 1, 2, 3, 4, 5, 6],
                     "init_joint_pose": [0., -np.pi/4, 0., 0., 0., 0., 0.],
                     "trans_gripper": 0.,
                     "trans_coupler": 0.,
                     "eef_link": 17,
                     "robot": "sawyer"}
    else:
        raise ValueError("This robot is not known")
        
    robot_dict["dof"] = len(robot_dict["robot_joints"])
    
    return robot_dict
    
    
def get_total_friction_force(contact_pts):
    """Compute the total friction force in "world coordinates".

    Parameters
    ----------
    contact_pts: a list of contact pts

    Returns
    -------
    total_lateral_friction_force: the total friction force
    """
    total_lateral_friction_force = [0, 0, 0]
    for i, pts in enumerate(contact_pts):
        total_lateral_friction_force[0] += pts[11][0] * pts[10] + pts[13][0] *pts[12]
        total_lateral_friction_force[1] += pts[11][1] * pts[10] + pts[13][1] *pts[12]
        total_lateral_friction_force[2] += pts[11][2] * pts[10] + pts[13][2] *pts[12]
    return total_lateral_friction_force


def get_grasp_matrix_(body_pos, contact_pts):
    """Compute the grasp matrix G.

    Parameters
    ----------
    -body_pos: an list of 3float, the position of the object in world
    cooridnates
    -contact_pts: a list, the contact pts
    -physicsClientId(optional): the physics client id

    Returns
    -------
    -G: an numpy array(6x3N), the grasp matrix
    """
    # Grasp matrix
    # G = [I3, S(r)] for one contact point
    G = np.zeros((6, 3*len(contact_pts)))

    # Fill the G matrix
    for i, pts in enumerate(contact_pts):
        T = np.array(pts[5]) - np.array(body_pos)
        Gi = np.zeros((6, 3))
        Gi[0:3, 0:3] = np.eye(3)
        Gi[3, 1] = -T[2]
        Gi[3, 2] = T[1]
        Gi[4, 0] = T[2]
        Gi[4, 2] = -T[0]
        Gi[5, 0] = -T[1]
        Gi[5, 1] = T[0]
        G[:, i*3:(i+1)*3] = Gi

    return G


def get_grasp_matrix(body_pos, contact_pts, model="hard"):
    """Compute the grasp matrix G.

    Parameters
    ----------
    -body_pos: an list of 3float, the position of the object in world
    cooridnates
    -contact_pts: a list, the contact pts
    -physicsClientId(optional): the physics client id

    Returns
    -------
    -G: an numpy array(6x3N), the grasp matrix
    """
    # Grasp matrix
    # G = [I3, S(r)] for one contact point
    G = np.zeros((6, 1))
    
    # Wrench basis
    Wb = wrench_basis(model=model)
    # Fill the G matrix
    for i, pts in enumerate(contact_pts):
        # No normal force ?
        if pts[9] == 0.:
            continue
        else:
            # Position of the contact points
            pc = np.array(pts[5]) - np.array(body_pos)
            # Cross product matrix form
            pc_cp = np.array([[0., -pc[2], pc[1]],
                              [pc[2], 0., -pc[0]],
                              [-pc[1], pc[0], 0.]])
            # Rotation
            x_axis = pts[11]
            y_axis = pts[13]
            z_axis = pts[7]
            Rc = np.vstack((x_axis, y_axis, z_axis)).T
            #print(Rc)
            # Wrench maps
            Wm = np.block([[Rc, np.zeros((3, 3))],
                           [pc_cp@Rc, Rc]])
            Gi = Wm@Wb
            G = np.concatenate((G, Gi), axis=1)

    return G[:, 1:]


def get_wrench_space(contact_pts, obj_frame, fric_coef, fric_moment, ng, model="hard"):
    """Return the discretize set of wrench a contact can exert as:
        
        Wi = [wi_1,..., wil,..., wi_ng], ng = approximation of the friction cone
        
        W = [Wi, .., Wn], n = number of contact points
    """
    W = np.zeros((6, 1))
    for pts in contact_pts:
        # Check if there exists a normal force
        if np.isclose(pts[9], 0.):
            continue
        else:
            # Grasp matrix
            G = get_grasp_matrix(obj_frame, [pts], model=model)
            # Linearized friction cone
            edges = linearized_friction_cone(pts, ng, fric_coef=fric_coef, fric_moment=fric_moment, model=model)
            edges[:, 0:3] /= np.linalg.norm(edges[:, 0:3], axis=1).reshape((-1, 1))
            # Wrench space of the i contact point
            Wi = G@edges.T
            # Total Wrench space
            W = np.concatenate((W, Wi), axis=1)
    return W[:, 1:]
    
def linearized_friction_cone(contact_pts, ng, fric_coef, fric_moment, model="hard"): 
    """Return a linearized friction cone."""
    if model == "soft":
        edges = np.zeros((ng, 4)) # ng x [fn, ft, ft, mn]
        S = np.zeros((4, ng+2))
        S[0, 0:ng] = fric_coef*np.cos(np.linspace(1, ng, num=ng, endpoint=True)*2*np.pi/ng)
        S[1, 0:ng] = fric_coef*np.sin(np.linspace(1, ng, num=ng, endpoint=True)*2*np.pi/ng)
        S[2, :] = 1.
        S[3, ng] = fric_moment
        S[3, ng+1] = -fric_moment
        # Rotation
        x_axis = contact_pts[11]
        y_axis = contact_pts[13]
        z_axis = contact_pts[7]
        R = np.vstack((x_axis, y_axis, z_axis))
        Rc = np.zeros((4, 4))
        Rc[0:3, 0:3] = R
        Rc[3, 3] = 1.
    elif model == "hard":
        edges = np.zeros((ng, 3)) # ng x [fn, ft, ft, mn]
        S = np.ones((3, ng))
        S[0] = fric_coef*np.cos(np.linspace(1, ng, num=ng, endpoint=True)*2*np.pi/ng)
        S[1] = fric_coef*np.sin(np.linspace(1, ng, num=ng, endpoint=True)*2*np.pi/ng)
        # Rotation
        x_axis = contact_pts[11]
        y_axis = contact_pts[13]
        z_axis = contact_pts[7]
        Rc = np.vstack((x_axis, y_axis, z_axis))
    # Edges
    edges = S.T@Rc
    
    return edges

def wrench_basis(model="hard"):
    """Wrench basis for friction point contact.

    Murray, R., Li, Z., & Sastry, S. (1994).
    A mathematical introduction to robotic manipulation.
    Boca Ratón, FL: CRC Press.
    
    Parameters
    ----------
    
    -model: string, model of the contact point (hard or soft)

    Return
    ------
    -Wb: an numpy array (6x3), the wrench basis
    """
    if model == "soft":
        return np.block([[np.eye(3), np.zeros((3, 1))],
                         [np.zeros((3, 3)), np.array([[0., 0., 1.]]).T]])
    else:
        return np.vstack((np.eye(3), np.zeros((3, 3))))


def test_grasp_matrix():
    """Test grasp matrix in a basic case.
    
    Murray, R., Li, Z., & Sastry, S. (1994).
    A mathematical introduction to robotic manipulation.
    Boca Ratón, FL: CRC Press.
    
    Page 221
    """
    # Define contact pts
    r = 0.5    
    pts_1 = (0., 0., 0., 0, 0., [0., -r, 0.], 0., [0., 1., 0.], 0., 0., 0., [0., 0., 1.], 0., [1., 0., 0.])
    pts_2 = (0., 0., 0., 0, 0., [0., r, 0.], 0., [0., -1., 0.], 0., 0., 0., [1., 0., 0.], 0., [0., 0., 1.])
    contact_pts = (pts_1, pts_2)
    
    # Body pos
    body_pos = [0., 0., 0.]
    
    # Grasp matrix
    G = get_grasp_matrix(body_pos, contact_pts)
    G_ = get_grasp_matrix_(body_pos, contact_pts)
    
    print(G)
    print(G_)
    
    return G, G_


def get_contact_forces(contact_pts):
    """Compute the contact forces fi.

    Parameters
    ----------
    -contact_pts: a list(N), the contact points

    Return
    ------
    -f: a numpy array(3Nx1), the forces [fx, fy, fz]
    """
    # Contact forces fi
    fc = np.zeros((3*len(contact_pts), 1))

    # Fill the G matrix
    for i, pts in enumerate(contact_pts):
        fx = pts[7][0]*pts[9] + pts[11][0]*pts[10] + pts[13][0]*pts[12]
        fy = pts[7][1]*pts[9] + pts[11][1]*pts[10] + pts[13][1]*pts[12]
        fz = pts[7][2]*pts[9] + pts[11][2]*pts[10] + pts[13][2]*pts[12]
        fc[i*3:(i+1)*3] = np.array([[fx, fy, fz]]).T
    return fc


def get_rotation(u, v, orthogonal_vector=None):
    # Get the rotation between "first_axis" and "end_axis"
    # PARAMETERS:
    # u: an array of float,1x3 and norm(u) = 1
    # v: an array of float, 1x3 and norm(v) = 1
    # orthogonal_vector: an array of float, 1x3 and norm(orthogonal_vector) =1
    #                    specific axis of rotation when u=-v
    # RETURN:
    # qr: an array of float, 1x4 and norm(qr) = 1
    k_cos_theta = np.dot(u, v)
    k = np.sqrt(np.linalg.norm(u) * np.linalg.norm(v))
    
    quats = np.zeros(4)

    if (k_cos_theta / k == -1):
        # 180 degree rotation around any orthogonal vector
        if orthogonal_vector is None:
            quats[0:3] = get_orthogonal(u)
        else:
            quats[0:3] = orthogonal_vector
        return quats
    quats[0:3] = np.cross(u, v)
    quats[3] = k_cos_theta + k
    quats /= np.linalg.norm(quats)
    
    return quats
    
    
def get_orthogonal(vector):
    # Get an orthogonal vector
    # PARAMETERS:
    # - vector: a numpy array, shape=[n, 3]
    # RETURN:
    # orthogonal: an numpy array, shape=[n, 3]
    vector = vector.reshape((-1, 3))
    size = vector.shape[0]
    norm = np.zeros((size, 1))
    while(norm == 0.).all():
        k = vector[:]/np.linalg.norm(vector, axis=1).reshape((size, 1))
        orthogonal = np.random.randn(size, 3)  # take a random vector
        orthogonal -= np.sum(orthogonal*k, axis=1).reshape((size, 1)) * k       # make it orthogonal to k
        norm = np.linalg.norm(orthogonal, axis=1).reshape((size, 1))
        orthogonal /= norm
    return orthogonal

def test_get_orthogonal():
    # Test "get_orthogonal"
    vector = np.array([1., 0., 0.])
    orthogonal = get_orthogonal(vector)
    assert np.isclose(np.dot(vector, orthogonal), 0)
    

def is_contact(contact_pts, max_dist=0):
    """Check if there is at leat one contact point between two bodies.
    
    Parameters
    ----------
    contact_pts: a list of contact points
    
    Returns
    -------
    contact:   True if there is at least one contact and the contact pts
               False otherwise
    """
    MAX_DIST = max_dist
    
    contact = False
    
    for id_pts, pts in enumerate(contact_pts):
        distance = pts[8]
        if distance <= MAX_DIST:
            contact = True
            break
            
    return contact

def create_vhacd(engine, file_in, file_out, file_log, resolution):
    """Create a vhacd mesh from visual mesh."""
    # Connection to a DIRECT server
    server_id = engine.connect(engine.DIRECT)
    # VHACD
    engine.vhacd(file_in, file_out, file_log, resolution=resolution, physicsClientId=server_id)
    # Disconnect
    engine.disconnect(server_id)

    
def load_vhacd_body(engine, parameters: dict, server_id=None):
    """Load a body with a vhacd mesh for collision."""
    # Load the collision mesh into Trimesh
    mesh = trimesh.load(parameters["collision_mesh"])
    # Visual mesh
    visualShapeId = engine.createVisualShape(shapeType=engine.GEOM_MESH,
                                             fileName=parameters["visual_mesh"],
                                             visualFramePosition=-parameters["cm"],
                                             meshScale=parameters["scale"],
                                             physicsClientId=server_id)
    # Collision mesh
    collisionShapeId = engine.createCollisionShape(shapeType=engine.GEOM_MESH,
                                                   fileName=parameters["collision_mesh"],
                                                   collisionFramePosition=-parameters["cm"],
                                                   meshScale=parameters["scale"],
                                                   physicsClientId=server_id)
    # Rigid body
    base_pos = parameters["base_pos"][0:2] + [parameters["base_pos"][2]+parameters["z"]]
    if "useFixedBase" in parameters.keys():
        baseMass = 0.
    else:
        baseMass = 1.
    obj_id = engine.createMultiBody(baseMass=baseMass,
                                    baseInertialFramePosition=[0.]*3,
                                    baseCollisionShapeIndex=collisionShapeId,
                                    baseVisualShapeIndex=visualShapeId,
                                    basePosition=base_pos,
                                    baseOrientation=parameters["base_ori"],
                                    useMaximalCoordinates=False,
                                    physicsClientId=server_id)
    
    # Change the mass, inertial values and friction coefficients
    engine.changeDynamics(obj_id, -1, rollingFriction=0.,
                          localInertiaDiagonal=mesh.principal_inertia_components)
    return obj_id

def load_pts3d(engine, pts3d, server_id, size=0.003, color=[1, 1, 1, 1]):
    # Load Point Cloud in the world with collision box
    # Create a collsion shape
    collisionShapeId = engine.createCollisionShape(shapeType=engine.GEOM_SPHERE,
                                                   radius=size,
                                                   physicsClientId=server_id)
    PYBULLET_SERVER = engine.getConnectionInfo(server_id)
    if PYBULLET_SERVER['connectionMethod'] == engine.GUI:  
        visualShapeId = engine.createVisualShape(shapeType=engine.GEOM_SPHERE,
                                                 rgbaColor=color,
                                                 radius=size,
                                                 physicsClientId=server_id) 
    else:
        visualShapeId = -1                                       
    
    # Create the point cloud
    pts_id = engine.createMultiBody(baseMass=0.,
                                    baseInertialFramePosition=[0.]*3,
                                    baseCollisionShapeIndex=collisionShapeId,
                                    baseVisualShapeIndex=visualShapeId,
                                    batchPositions=pts3d,
                                    useMaximalCoordinates=False,
                                    physicsClientId=server_id)
    
    return pts_id
        
    
    
def show_pose(p, T, Q, id_server, name="", scale=0.1, line_width=4):
    # Show the frame of a point in the world coordinates
    # p: pybullet simulator
    # T: position (x,y,z) in the world coordinates
    # Q: quaternions (a,b,c,d) in the world coordinates
    PYBULLET_SERVER = p.getConnectionInfo(0)
    if PYBULLET_SERVER['connectionMethod'] == p.GUI:
        Matrix = p.getMatrixFromQuaternion(Q)
        id1 = p.addUserDebugLine(T, [T[0]+scale*Matrix[0], T[1]+scale*Matrix[3], T[2]+scale*Matrix[6]], [1, 0, 0], lineWidth=line_width, physicsClientId=id_server)
        id2 = p.addUserDebugLine(T, [T[0]+scale*Matrix[1], T[1]+scale*Matrix[4], T[2]+scale*Matrix[7]], [0, 1, 0], lineWidth=line_width, physicsClientId=id_server)
        id3 = p.addUserDebugLine(T, [T[0]+scale*Matrix[2], T[1]+scale*Matrix[5], T[2]+scale*Matrix[8]], [0, 0, 1], lineWidth=line_width, physicsClientId=id_server)
        
        if name != "":
            p.addUserDebugText(name, T,textColorRGB=[1, 0, 0], textSize=1.5)
        
        return id1, id2, id3
    else:
        return 0, 0, 0


def show_friction_cone(p, T, normal, edges, id_server, name="", scale=0.1, line_width=4):
    # Show the frame of a point in the world coordinates
    # p: pybullet simulator
    # T: position (x,y,z) in the world coordinates
    # Q: quaternions (a,b,c,d) in the world coordinates
    PYBULLET_SERVER = p.getConnectionInfo(0)
    if PYBULLET_SERVER['connectionMethod'] == p.GUI:
        # Normal
        n = p.addUserDebugLine(T, [T[0]+scale*normal[0], T[1]+scale*normal[1], T[2]+scale*normal[2]], [0, 0, 1], lineWidth=line_width, physicsClientId=id_server)
        # Cone
        cone_id = []
        for edge in edges:
            cone_id.append(p.addUserDebugLine(T, [T[0]+scale*edge[0], T[1]+scale*edge[1], T[2]+scale*edge[2]], [0, 1, 0], lineWidth=line_width, physicsClientId=id_server))
        
        if name != "":
            p.addUserDebugText(name, T,textColorRGB=[1, 0, 0], textSize=1.5)


def show_pose_matrix(p, T, Matrix, name="", scale=0.1, line_width=2):
    # Show the frame of a point in the world coordinates
    # p: pybullet simulator
    # T: position (x,y,z) in the world coordinates
    # Q: quaternions (a,b,c,d) in the world coordinates
    PYBULLET_SERVER = p.getConnectionInfo(0)
    if PYBULLET_SERVER['connectionMethod'] == p.GUI:
        id1 = p.addUserDebugLine(T, [T[0]+scale*Matrix[0], T[1]+scale*Matrix[3], T[2]+scale*Matrix[6]], [1, 0, 0], lineWidth=line_width)
        id2 = p.addUserDebugLine(T, [T[0]+scale*Matrix[1], T[1]+scale*Matrix[4], T[2]+scale*Matrix[7]], [0, 1, 0], lineWidth=line_width)
        id3 = p.addUserDebugLine(T, [T[0]+scale*Matrix[2], T[1]+scale*Matrix[5], T[2]+scale*Matrix[8]], [0, 0, 1], lineWidth=line_width)
        
        if name != "":
            p.addUserDebugText(name, T,textColorRGB=[1, 0, 0], textSize=1.5)
        
        return id1, id2, id3
    else:
        return 0, 0, 0
    
def show_pts(p, T, physicsClientId, name="", size=0.003, color=[1, 1, 1, 1]):
    # Show a point in the world coordinates
    # p: the pybullet API
    # T: a list of float, (x,y,z) in the world coordinates
    PYBULLET_SERVER = p.getConnectionInfo(physicsClientId)
    if PYBULLET_SERVER['connectionMethod'] == p.GUI:  
        visualShapeId = p.createVisualShape(shapeType=p.GEOM_SPHERE, rgbaColor=color, radius=size)
        collisionShapeId = -1
        if type(T) == list:
            nb_particles = len(T)/3
            if nb_particles == 1:
                p.createMultiBody(baseMass=0,
                                  baseCollisionShapeIndex=collisionShapeId,
                                  baseVisualShapeIndex=visualShapeId,
                                  basePosition=T,
                                  useMaximalCoordinates=True,
                                  physicsClientId=physicsClientId)
            else:
                p.createMultiBody(baseMass=0,
                                  baseCollisionShapeIndex=collisionShapeId,
                                  baseVisualShapeIndex=visualShapeId,
                                  useMaximalCoordinates=True,
                                  batchPositions=T,
                                  physicsClientId=physicsClientId)
        else:
            nb_particles = T.shape[0]
            if nb_particles == 1:
                p.createMultiBody(baseMass=0,
                                  baseCollisionShapeIndex=collisionShapeId,
                                  baseVisualShapeIndex=visualShapeId,
                                  basePosition=T[0],
                                  useMaximalCoordinates=True,
                                  physicsClientId=physicsClientId)
            else:
                p.createMultiBody(baseMass=0,
                                  baseCollisionShapeIndex=collisionShapeId,
                                  baseVisualShapeIndex=visualShapeId,
                                  useMaximalCoordinates=True,
                                  batchPositions=T,
                                  physicsClientId=physicsClientId)
        print("nb particles", nb_particles)
        
            
                
    else:
        print("No GUI mode")

    
def show_trajectory(p, traj):
    # Show a trajectory (line)
    # p: the pybullet API
    # traj: a list of 3D position
    nb_pts = len(traj)
    gui_traj = []
    for i in range(nb_pts-1):
        gui_traj.append(p.addUserDebugLine(traj[i], traj[i+1], [0, 1, 0]))
        
    return gui_traj


def delete_gui_trajectory(p, gui_traj):
    # Delete a gui-trajectory
    # p: the pybullet API
    # gui_traj: the gui line trajectory
    for i, pts in enumerate(gui_traj):
        p.removeUserDebugItem(pts)


def rodrigue_formula(vector, axis, angle):
    """Rodrigues formula.

    Performs rigid body rotation
    """
    cos_angle = tf.cos(angle)
    axis_dot_point = tf.reduce_sum(tf.multiply(axis, vector), axis=-1, keepdims=True)
    return vector * cos_angle + tf.linalg.cross(axis, vector)*tf.sin(angle)  + axis * axis_dot_point * (tf.ones(axis_dot_point.shape) - cos_angle)

def get_links(engine, object_id, server_id=None):
    """Return the links number with name."""
    _link_name_to_index = {engine.getBodyInfo(object_id, physicsClientId=server_id)[0].decode('UTF-8'):-1,}
    
    for _id in range(engine.getNumJoints(object_id, physicsClientId=server_id)):
    	_name = engine.getJointInfo(object_id, _id, physicsClientId=server_id)[12].decode('UTF-8')
    	_link_name_to_index[_name] = _id
    print(_link_name_to_index)

def filter_true_img(depth_img, max_dist=1.1):
    """Remove NaN values and set to 0 useless pixels."""
    # Remove NaN
    depth_img = np.nan_to_num(depth_img)
    # Set to 0 distance > max_dist (useless background)
    depth_img = np.where(depth_img < max_dist, depth_img, np.zeros(shape=depth_img.shape))
    # Set to 0 pixels not include in the ROI
    depth_img[400:] = 0.
    
    return depth_img

    
def get_obj_pts_from_kinect_depth(depth, distance_threshold=0.015, ransac_n=4, num_iterations=400):
    """Extract point cloud associated to the object and convert it to a mesh."""
    # Convert numpy array to Open3D class
    kinect_o3d_img = o3d.geometry.Image(depth.astype(np.float32))
    # Intrinsic parameters of the kinect
    kinect_intrinsic = o3d.camera.PinholeCameraIntrinsic(width=640, height=480, fx=4.73/0.0078, fy=4.73/0.0078, cx=240, cy=320)
    angle = -np.pi/2
    rot = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([0., 0., 1.])*angle)
    T_align = np.eye(4)
    T_align[:3, :3] = rot
    # Not-oriented Point cloud from Open3D
    kinect_o3d_pts = o3d.geometry.PointCloud.create_from_depth_image(kinect_o3d_img,
                                                                     kinect_intrinsic,
                                                                     depth_scale=1.)
    # Get the plane
    plane, index = kinect_o3d_pts.segment_plane(distance_threshold, ransac_n, num_iterations)
    # z-axis : normal to the plane
    normal = plane[0:3]/np.linalg.norm(plane[0:3])
    # Angle between z-table and plane
    angle = np.arccos(np.dot(normal, [0., 0., 1.]))
    angle = np.pi-angle
    # Assumption: the two y-axis are parallel 
    rot = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([0., 1., 0.])*angle)
    T_final = np.eye(4)
    T_final[:3, :3] = rot
    kinect_extrinsic = T_align@T_final
    pts_3d = o3d.geometry.PointCloud.create_from_depth_image(kinect_o3d_img,
                                                             kinect_intrinsic,
                                                             kinect_extrinsic,
                                                             depth_scale=1.)
    
    return pts_3d.select_by_index(index, invert=True)


def get_pts_from_kinect_depth(depth):
    """Extract point cloud associated to the object and convert it to a mesh."""
    # Convert numpy array to Open3D class
    kinect_o3d_img = o3d.geometry.Image(depth.astype(np.float32))
    # Intrinsic parameters of the kinect
    kinect_intrinsic = o3d.camera.PinholeCameraIntrinsic(width=640, height=480, fx=4.73/0.0078, fy=4.73/0.0078, cx=240, cy=320)
    # Not-oriented Point cloud from Open3D
    kinect_o3d_pts = o3d.geometry.PointCloud.create_from_depth_image(kinect_o3d_img,
                                                                     kinect_intrinsic,
                                                                     depth_scale=1.)
    
    return kinect_o3d_pts

def get_mesh_from_point_cloud(pts3d, strategy="BPA"):
    """Create a mesh from a point cloud."""
    # Hyper parameters
    distances = pts3d.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 3*avg_dist
    # Estimate normals
    pts3d.estimate_normals()
    
    # Create the mesh
    bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pts3d, o3d.utility.DoubleVector([radius, radius * 2]))
    # Ensure consistency of the mesh
    bpa_mesh.remove_degenerate_triangles()
    bpa_mesh.remove_duplicated_triangles()
    bpa_mesh.remove_duplicated_vertices()
    bpa_mesh.remove_non_manifold_edges()
    
    return bpa_mesh
    
    

if __name__ == "__main__":
    from metrics import grasp_isotropy
    """
    import pybullet
    G, G_ = test_grasp_matrix()
    print(grasp_isotropy(G))
    print(grasp_isotropy(G_))
    print("Rank G:", np.linalg.matrix_rank(G))
    print("Rank G_:", np.linalg.matrix_rank(G_))
    """
    r = 0.5
    pts_1 = (0., 0., 0., 0, 0., [0., -r, 0.], 0., [0., 1., 0.], 0., 0.1, 0., [0., 0., 1.], 0., [1., 0., 0.])
    pts_2 = (0., 0., 0., 0, 0., [0., r, 0.], 0., [0., -1., 0.], 0., 0.4, 0., [1., 0., 0.], 0., [0., 0., 1.])
    contact_pts = (pts_1, pts_2)
    e = linearized_friction_cone(pts_2, ng=10, fric_coef=0.5, fric_moment=0.5*0.001, model="hard")
    print(e)
    #print(np.linalg.norm(e, axis=1))
    """
    server_id = pybullet.connect(pybullet.GUI)
    show_friction_cone(pybullet, pts_2[5], pts_2[7], e, server_id, name="", scale=0.1, line_width=4)
    """
    W = get_wrench_space(contact_pts, [0., 0., 0.], fric_coef=0.5, fric_moment=0.5*0.01, ng=5, model="soft")
    print("Wrench space:", W)
    print(np.linalg.matrix_rank(W))
    input("stop")
    
    