# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 15:21:25 2020

@author: norman marlier

"""
import numpy as np


class Simulator():
    """Abstract for simulation."""
    def __init__(self, output_dim, *args, **kwargs):
        
        # General parameters
        self.args = args
        self.kwargs = kwargs
        # Output dim
        self.output_dim = output_dim

    def forward(self, inputs, parameters, *args, **kwargs):
        """Forward pass. (Deterministic)"""
        raise NotImplementedError

    def sample(self, inputs, sim_noise, parameters_noise, *args, **kwargs):
        """Sample form the simulator. (Stochastic)"""
        raise NotImplementedError
    
    def get_scaling(self, engine, obj):
        """Get the scaling with respect to the geometry of the object and its
        height."""
        # Connect to the engine
        id_server = engine.connect(engine.DIRECT)
        # Load object
        id_obj = engine.loadURDF(obj["urdf"],
                                basePosition=[0., 0., 0.],
                                baseOrientation=obj["base_ori"],
                                physicsClientId=id_server)
        # Get mesh data
        if "rectangle" in obj["urdf"]:
            extends = engine.getCollisionShapeData(id_obj, -1,
                                                   physicsClientId=id_server)
            height = extends[0][3][2]/2.
        else:
            node, vertices = engine.getMeshData(id_obj, -1, physicsClientId=id_server)
            z_vertices = [pos[1] for pos in vertices]
            # Height
            height = -np.amin(z_vertices)
        # Close the server
        engine.disconnect(id_server)

        # Scale
        scale = obj["z"]/height
        # Return the min
        return scale

if __name__ == "__main__":
    import pybullet as pb
    
    sim = Simulator()
    # obj = {'urdf': 'D:/ShapeNetCore.v2/02880940/45603bffc6a2866b5d1ac0d5489f7d84/models/model_normalized_vhacd.urdf', 'scaling': 0.15, 'base_pos': [0.0, 0.0, 0.7], 'base_ori': [0.7071067811865476, 0.0, 0.0, 0.7071067811865476], 'z': 0.090818495}
    obj = {'urdf': 'D:/ShapeNetCore.v2/rectangle/rectangle_1.urdf', 'scaling': 0.15, 'base_pos': [0.0, 0.0, 0.7], 'base_ori': [0., 0.0, 0.0, 1.], 'z': 0.090818495}
    scale = sim.get_scaling(pb, obj)
    print("scale", scale)
    print("z", obj["z"])
    plane_dic = {"urdf": "../data/plane.urdf",
             "base_pos": [0., 0., 0.],
             "base_ori": [0., 0., 0., 1.]}
    id_server = pb.connect(pb.GUI)
    id_plane = pb.loadURDF(plane_dic["urdf"],
                         basePosition=[0., 0., 0.],
                         baseOrientation=plane_dic["base_ori"],
                         physicsClientId=id_server)
    id_obj = pb.loadURDF(obj["urdf"],
                         basePosition=[0., 0., obj["z"]],
                         baseOrientation=obj["base_ori"],
                         globalScaling=scale,
                         physicsClientId=id_server)

    print("final pos:", pb.getBasePositionAndOrientation(id_obj, physicsClientId=id_server))
    input("stop")
    pb.disconnect()
    