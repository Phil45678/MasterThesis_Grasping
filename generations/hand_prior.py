# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 18:25:37 2020

@author: norman marlier

Prior distribution
"""
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import sys
sys.path.append("../..")
from DexterousManipulation.simulations.utils_sim import rodrigue_formula


class ShellDistribution():
    """Shell distribution.
    
    p(X) = U(r) * Sphere
    """
    def __init__(self, rmin, rmax):
        self.rmin = rmin
        self.rmax = rmax
        self.rv_radius = tfp.distributions.Uniform([self.rmin], [self.rmax])
        self.rv_sphere = tfp.distributions.VonMisesFisher(mean_direction=[1., 0., 0.], concentration=0.)
    
    @tf.function
    def sample(self, nb_samples=None):
        if nb_samples == None:
            sphere = self.rv_sphere.sample()
            sphere = tf.where(sphere[2] < 0., sphere*[1., 1., -1.], sphere)
            sphere = tf.linalg.normalize(sphere)[0]
            radius = self.rv_radius.sample()
            return tf.math.multiply(radius, sphere)
        else:
            sphere = self.rv_sphere.sample(nb_samples)
            sphere = tf.where(sphere[:, -1:] < tf.zeros([nb_samples, 1]), tf.math.multiply(sphere, tf.constant([1., 1., -1.])), sphere)
            sphere = tf.linalg.normalize(sphere)[0]
            return tf.math.multiply(self.rv_radius.sample(nb_samples), sphere)
    
    @tf.function
    def log_prob(self, samples):
        sphere, radius = tf.linalg.normalize(samples)
        log_prob_sphere = self.rv_sphere.log_prob(sphere) + tf.math.log(2.)
        tf.print(log_prob_sphere)
        return  tf.add(log_prob_sphere, self.rv_radius.log_prob(radius))
    
    @tf.function
    def prob(self, samples):
        return tf.math.exp(self.log_prob(samples))


class HandPrior():
    """Prior distribution for the hand configuration, p(h).

    The prior distribution is a joint distribution of the hand configuration
    The hand configuration h = R3 x SO(3) x G

    R3 : Euclidean space
    It is the vector space of the position (x, y, z)
    Two possibilities:
        The position (x, y, z) is defined in a cube
        The position (x, y, z) is defined in shell

    SO(3) : Rienmanian manifold {Q in So(3): QQt = QtQ = I and det(Q)=1}
    It is a Lie group of the rotation.
    Here, the generation of the variables is done into R3 and R
    But a rotation matrix is given to the neural network

    G : Boolean space(3)
    basic/pinch/wide, the configuration of the fingers.
    """

    def __init__(self, pos_parameters, ori_parameters, finger_parameters):

        # R3 random variables x-y-z
        # Parameters of the distribution
        self.pos_distrib = pos_parameters["distrib"]
        if self.pos_distrib == "cube":
            self.xmin = pos_parameters["xmin"]
            self.xmax = pos_parameters["xmax"]
            self.ymin = pos_parameters["ymin"]
            self.ymax = pos_parameters["ymax"]
            self.zmin = pos_parameters["zmin"]
            self.zmax = pos_parameters["zmax"]
            # Distribution
            self.rv_pos = tfp.distributions.Uniform([self.xmin, self.ymin, self.zmin], [self.xmax, self.ymax, self.zmax])
        elif self.pos_distrib == "shell":
            self.rmin = pos_parameters["rmin"]
            self.rmax = pos_parameters["rmax"]
            if pos_parameters["rdistrib"] == "uniform":
                self.rv_radius = tfp.distributions.Uniform([self.rmin], [self.rmax])
            elif pos_parameters["rdistrib"] == "gamma":
                self.rv_radius = tfp.distributions.Gamma([self.rmin], [self.rmax])
            elif pos_parameters["rdistrib"] == "normal":
                self.rv_radius = tfp.distributions.Normal(loc=[self.rmin], scale=[self.rmax])
            elif pos_parameters["rdistrib"] == "truncnorm":
                low_r = pos_parameters["rlow"]
                high_r = pos_parameters["rhigh"]
                self.rv_radius = tfp.distributions.TruncatedNormal(loc=[self.rmin],
                                                                   scale=[self.rmax],
                                                                   low=[low_r],
                                                                   high=[high_r])
            self.rv_sphere = tfp.distributions.VonMisesFisher(mean_direction=[0., 0., 1.],
                                                              concentration=pos_parameters["shell"])


        # SO(3) random variables  x axis - angle
        # VonMisesFisher distribution for the x-axis and uniform distribution
        # for the angle
        # Orientation distribution is position dependent
        # Parameters of the distribution
        self.kappa = ori_parameters["kappa"]
        # Distribution
        # Number of DOF
        self.ori_dof = ori_parameters["dof"]
        self.rv_angle = tfp.distributions.Uniform(-np.pi, np.pi)
        if self.ori_dof == 3:
            if self.pos_distrib == "cube":
                self.rv_axis = lambda pos: tfp.distributions.VonMisesFisher(mean_direction=-tf.stop_gradient(tf.linalg.normalize(pos, axis=-1)[0]),
                                                                            concentration=tf.constant([self.kappa], dtype=tf.float32))
            elif self.pos_distrib == "shell":
                self.rv_axis = lambda pos: tfp.distributions.VonMisesFisher(mean_direction=-tf.stop_gradient(tf.where(tf.reshape(pos, [-1, 3])[:, -1:] < tf.zeros_like(pos), tf.math.multiply(pos, tf.constant([1., 1., -1.])), pos)),
                                                                            concentration=tf.constant([self.kappa], dtype=tf.float32))
        elif self.ori_dof == 1:
            self.rv_axis = lambda pos: tfp.distributions.VectorDeterministic([0., 0., -1]*tf.ones_like(tf.reshape(pos, [-1, 3])[:, 0:1]))
        
                

        # M random variables
        # Preshape
        # Parameters of the distribution
        self.preshape_probs = finger_parameters["preshape"]
        self.rv_preshape = tfp.distributions.OneHotCategorical(probs=self.preshape_probs, dtype=tf.float32)

        # Joint Distribuiton
        # Generative model:
        #     pos ~ Uniform()
        #     axis ~ VonMisesFisher(mean_direction=-pos, concentration=kappa)
        #     angle ~ Uniform(-pi, pi)
        #     m ~ OneHotCategorical([1/3, 1/3, 1/3])
        if self.pos_distrib == "cube":
            self.joint_distribution = tfp.distributions.JointDistributionSequential([
                tfp.distributions.Independent(self.rv_pos, reinterpreted_batch_ndims=1),
                self.rv_axis,
                self.rv_angle,
                self.rv_preshape])
        elif self.pos_distrib == "shell":
            self.joint_distribution = tfp.distributions.JointDistributionSequential([
                self.rv_radius,
                self.rv_sphere,
                self.rv_axis,
                self.rv_angle,
                self.rv_preshape])
        #self.joint_distribution.resolve_graph()
    
    @tf.function
    def sample_from_joint(self, n):
        return self.joint_distribution.sample(n)        

    def to_network(self, x):
        """Change the representation axis-angle to rotation matrix.

        Put every variables into a single vector.
        """
        # Split into variables
        if self.pos_distrib == "cube":
            pos = x[0]
            axis = x[1]
            angle = tf.reshape(x[2], [-1, 1])
            # Gradient Tape can give None gradient => convert to zeros tensor
            # TODO : shape can change if len(preshape_probs ) == 1
            if x[3] is None:
                preshape = tf.zeros([pos.shape[0], len(self.preshape_probs)])
            else:
                preshape = tf.reshape(x[3], [-1, len(self.preshape_probs)])
        elif self.pos_distrib == "shell":
            pos_halfsphere = tf.where(x[1][:, -1:] < tf.zeros_like(x[1]), tf.math.multiply(x[1], tf.constant([1., 1., -1.])), x[1])
            pos = tf.math.multiply(x[0], pos_halfsphere)
            axis = x[2]
            angle = tf.reshape(x[3], [-1, 1])
            # Gradient Tape can give None gradient => convert to zeros tensor
            # TODO : shape can change if len(preshape_probs ) == 1
            if x[4] is None:
                preshape = tf.zeros([pos.shape[0], len(self.preshape_probs)])
            else:
                preshape = tf.reshape(x[4], [-1, len(self.preshape_probs)])

        # Base frame
        x_axis = tf.tile(tf.constant([[1., 0., 0.]]), [axis.shape[0], 1])
        y_axis = tf.tile(tf.constant([[0., 1., 0.]]), [axis.shape[0], 1])
        z_axis = tf.tile(tf.constant([[0., 0., 1.]]), [axis.shape[0], 1])

        # Align with the wanted x axis
        cross_axis = tf.linalg.normalize(tf.linalg.cross(x_axis, axis), axis=-1)[0]
        # Where cross_axis is NaN, x and x' are // so replace with [0., 1., 0.] vector
        cross_axis = tf.where(tf.math.is_nan(cross_axis), tf.constant([0., 1., 0.]), cross_axis)
        dot_product = tf.reduce_sum(tf.multiply(x_axis, axis), axis=-1, keepdims=True)
        int_angle = tf.acos(dot_product)
        y_int_axis = tf.linalg.normalize(rodrigue_formula(y_axis, cross_axis, int_angle), axis=-1)[0]
        z_int_axis = tf.linalg.normalize(rodrigue_formula(z_axis, cross_axis, int_angle), axis=-1)[0]

        # Rotation by angle
        y_f_axis = tf.linalg.normalize(rodrigue_formula(y_int_axis, axis, angle), axis=-1)[0]
        z_f_axis = tf.linalg.normalize(rodrigue_formula(z_int_axis, axis, angle), axis=-1)[0]

        # Reshape
        axis = tf.reshape(axis, [-1, 3, 1])
        y_f_axis = tf.reshape(y_f_axis, [-1, 3, 1])
        z_f_axis = tf.reshape(z_f_axis, [-1, 3, 1])
        rot_mat = tf.concat((axis, y_f_axis, z_f_axis), axis=-1)
        rot_matrix = tf.reshape(rot_mat, [-1, 9])

        #print("Preshape: ", preshape)
        # Concat every variables
        return tf.concat((pos, rot_matrix, preshape), axis=-1)

    def from_network(self, x):
        """Change the representation rotation matrix to axis-angle.

        Split the x into pos, axis, angle, preshape list.
        """
        # Position variables
        pos = x[..., 0:3]
        if self.pos_distrib == "shell":
            sphere, radius = tf.linalg.normalize(pos, axis=1)

        # Preshape
        preshape = x[..., 12:]

        # Rotation matrices
        rot_mat = x[..., 3:12]
        axis = rot_mat[:, 0:9:3]
        ry = rot_mat[:, 1:9:3]
        rz = rot_mat[:, 2:9:3]

        # Base frame
        x_axis = tf.tile(tf.constant([[1., 0., 0.]]), [pos.shape[0], 1])
        y_axis = tf.tile(tf.constant([[0., 1., 0.]]), [pos.shape[0], 1])

        # Align with the wanted x axis
        cross_axis = tf.linalg.normalize(tf.linalg.cross(axis, x_axis), axis=-1)[0]
        cross_axis = tf.where(tf.math.is_nan(cross_axis), tf.constant([0., 1., 0.]), cross_axis)
        dot_product = tf.reduce_sum(tf.multiply(axis, x_axis), axis=-1, keepdims=True)
        int_angle = tf.acos(dot_product)
        ry_int_axis = tf.linalg.normalize(rodrigue_formula(ry, cross_axis, int_angle), axis=-1)[0]

        # Compute the angle
        dot_product = tf.reduce_sum(tf.multiply(y_axis, ry_int_axis), axis=-1, keepdims=True)
        y_angle = tf.acos(dot_product)
        angle = tf.reshape(tf.math.sign(ry_int_axis)[:, -1:]*y_angle, [-1])
        
        if self.pos_distrib == "cube":
            return [pos, axis, angle, preshape]
        elif self.pos_distrib == "shell":
            return [radius, sphere, axis, angle, preshape]


if __name__ == "__main__":
    """
    pos_parameters = {"distrib": "shell",
                      "rmin": 0.1,
                      "rmax": 0.2}
    """

    """
    pos_parameters = {"distrib": "shell",
                        "rdistrib": "truncnorm", 
                        "rlow": 0.1, 
                        "rhigh" : 0.2, 
                      "rmin": 0.1,
                      "rmax": 0.2,
                      "shell": 0.0}
    """
                      
    pos_parameters = {"distrib": "cube",
                            "shellk": 10., 
                          "xmin": -0.03,
                          "xmax": 0.03,
                          "ymin": -0.03,
                          "ymax": 0.03,
                          "zmin": 0.11,
                          "zmax": 0.14,
                          #"rdistrib": "truncnorm",
                          "rdistrib": "uniform",
                          "rmin": 0.1,
                          "rmax": 0.15}
    ori_parameters = {"dof": 1,
                        "kappa": 20}
    finger_parameters = {"preshape": [1./3.]*1}
    hand_prior = HandPrior(pos_parameters, ori_parameters, finger_parameters)
    h_sample = hand_prior.joint_distribution.sample(1)
    h_sim = hand_prior.to_network(h_sample).numpy()
    #hand_prior = HandPrior(pos_parameters, {"kappa": 20, "dof" : 1}, {"preshape": [1./3]*3})
    print(h_sim)
    