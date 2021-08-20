# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 18:14:48 2020

@author: norman marlier
"""
import tensorflow as tf
import tensorflow_probability as tfp
import json
import numpy as np
import sys
sys.path.append("../..")

from DexterousManipulation.model.object_model import ObjectDataset


class ObjectPrior():
    """Prior distribution for object, p(O).
    
    Discrete uniform distribution for object.
    An object O is defined as a dict with field:
        -"collision_mesh": the vhacd collision mesh file name
        -"visual_mesh": the visual mesh file name
        -"scaling": the size of the object (vec3)
        -"z": the height of the object
        -"base_pos": the initial position
        -"base_ori": the initial orientation
        -"volume": the volume of the collision mesh
        -"density": the density of the collision mesh
        -"mass": the "true" mass of the object
    """

    def __init__(self, json_config):
        """Constructor."""
        self.object_dataset = ObjectDataset(json_config)
        self.probs = [1/self.object_dataset.get_nb_obj()]\
            *self.object_dataset.get_nb_obj()
        # Categorical distribution for geometry
        self.geometry_distribution = tfp.distributions.Categorical(probs=self.probs)
        
    def get_obj(self, idx):
        """Get object dictionnary from idx."""
        if idx >= self.object_dataset.get_nb_obj():
            raise ValueError("idx is greater than the number of objects")
        return self.object_dataset.get_obj(idx)
    
    def get_idx(self, obj):
        return self.object_dataset.get_idx(obj)
        
    def sample(self, noise=True):
        idx = self.geometry_distribution.sample(1).numpy()[0]
        obj = self.object_dataset.get_obj(idx).copy()
        if noise:
            # Scaling factor
            scaling_factor = tfp.distributions.Uniform(0.9, 1.1).sample().numpy()*np.ones((3))
            #scaling_factor = np.ones((3))
            obj["scale"] = scaling_factor*obj["scale"][0]
            self.object_dataset.set_geometrical_attributs(obj)
            # Change density
            obj["density"] = tfp.distributions.Normal(obj["density"], 1.).sample().numpy()
        return obj

if __name__ == "__main__":
    p_obj = ObjectPrior("./egad.json")
    sample = p_obj.sample()
    print("Sample", sample)
    print("Idx", p_obj.get_idx(sample))
    #print(p_obj.get_obj(14))
