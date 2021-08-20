# -*- coding: utf-8 -*-
"""
Creates samples for the classification task 
Adapted from Norman Marlier's code. 
"""
import pybullet
import numpy as np
import sys
sys.path.append("../..")

def generate_task_sample(nb_samples, obj_prior, hand_prior, frame_prior,
                         robot_simulator, planner, parameters_noise, sim_noise,
                         engine_connection, verbosity, bullet_file="sim"):
    """Generate St from h, pO and obj."""
    # Data structure which retain all variables
    metric = np.zeros((nb_samples, robot_simulator.output_dim))
    hands = np.zeros((nb_samples, 12 + len(hand_prior.preshape_probs)))
    frames = np.zeros((nb_samples, 3))
    objs = np.zeros((nb_samples, 5)) # Obj(id, z, scale, density, volume)
    parameters_sim = np.zeros((nb_samples, 3)) # Lat coef, Spin coef, table_scale
    planner_sim = np.zeros((nb_samples, 1), dtype="<U6")
    planner_sim[:] = planner
    savefile_sim = np.zeros((nb_samples, 1), dtype="<U32")
    
    # Generate 'nb_samples'
    for i in range(nb_samples):
        # Sample h
        h_sample = hand_prior.joint_distribution.sample(1)
        h_sim = hand_prior.to_network(h_sample).numpy()
        hands[i] = h_sim[0]
        # Sample an object
        obj_sim = obj_prior.sample(noise=False)
        objs[i, 0] = obj_prior.get_idx(obj_sim)
        objs[i, 1] = obj_sim["z"]
        objs[i, 2] = obj_sim["scale"][0]
        objs[i, 3] = obj_sim["density"]
        objs[i, 4] = obj_sim["volume"]
        # Sample pO
        # pO_sample = frame_prior.joint_distribution.sample(1)
        # pO_sim = frame_prior.to_network(pO_sample).numpy()
        pO_sample = [0, 0, 0]
        frames[i] = pO_sample[0]
        # Inputs
        inputs = [obj_sim.copy(), h_sim.copy(), pO_sample.copy()]
        # Parameters
        if parameters_noise:
            lat_coef = np.random.default_rng().uniform(0.8, 2.5, 1)
            spin_coef = lat_coef*np.random.default_rng().normal(2*1e-3, 1e-4) # gamma = µ*radius, radius ~ N(0.002, 0.0001)
            # Scale of the table
            table_scale = np.random.default_rng().normal(1., 0.005, 1)
        else:
            lat_coef = 0.5
            spin_coef = lat_coef*2*1e-3 # gamma = µ*radius
            table_scale = 1.
        parameters = [lat_coef, spin_coef, planner, table_scale]
        parameters_sim[i] = [lat_coef, spin_coef, table_scale]
        # Bullet_file
        if bullet_file:
            save_file = bullet_file + "_" + str(i) + ".bullet"
            savefile_sim[i] = save_file
        else:
            savefile_sim[i] = ""
            save_file = None
        # Forward sim
        metric[i] = robot_simulator.forward(inputs, parameters,
                                            engine_connection=engine_connection,
                                            verbosity=verbosity,
                                            save_file=save_file)
        
    
    return metric, hands, frames, objs, parameters_sim, planner_sim, savefile_sim
        

if __name__ == "__main__":
    from DexterousManipulation.simulations.classification_simulator import RobotSimulator
    from DexterousManipulation.simulations.utils_sim import get_background, get_robot
    from DexterousManipulation.generations.object_prior import ObjectPrior
    from DexterousManipulation.generations.hand_prior import HandPrior
    from DexterousManipulation.generations.frame_prior import FramePrior
    from DexterousManipulation.controllers.gripper_controller import Robotiq3fControllerRegular, Robotiq3fControllerSimple, SawyerElectricController
    import time
    
    sim = "sawyer"
    # Background
    background = get_background(sim)
    robot_dict = get_robot(sim)
    # Gripper
    if sim =="sawyer":
        gripper_ctrl = SawyerElectricController(None, sim, None)
        # Prior distrib for h
        pos_parameters = {"distrib": "cube",
                            "shellk": 10., 
                          "xmin": 0,
                          "xmax": 0,
                          "ymin": 0.0,
                          "ymax": 0.0,
                          "zmin": 0.36,
                          "zmax": 0.36,
                          #"rdistrib": "truncnorm",
                          "rdistrib": "uniform",
                          "rmin": 0.1,
                          "rmax": 0.15}
        ori_parameters = {"dof": 1,
                          "kappa": 20}
        finger_parameters = {"preshape": [1./3.]*1}
        hand_prior = HandPrior(pos_parameters, ori_parameters, finger_parameters)
        # Prior distrib for pO
        """
        frame_prior = FramePrior(xmin=-0.2, xmax=0.2,
                                 ymin=-0.2, ymax=0.2)
        """
        frame_prior = FramePrior(xmin=0, xmax=0,
                                 ymin=0, ymax=0)
        # dataset_file = "../generations/BIG_BOX.json"
        # dataset_file = "../generations/MEDIUM_BOX.json"
        # dataset_file = "../generations/SMALL_BOX.json"
        # dataset_file = "../generations/CYLINDRE.json"
        # dataset_file = "../generations/ds_obj_config.json"
        dataset_file = "../generations/single_obj_YCB.json"

    # Simulation parameters
    sim_parameters = {"model": "soft",
                      "ng": 20,
                      "restitution": 0,
                      "threshold": 1e-3}
    # Robot simulator
    robot_sim = RobotSimulator(pybullet, robot_dict, background, gripper_ctrl,
                               sim_parameters)
    
    # engine_connection = "DIRECT"
    engine_connection = "GUI"
    verbosity = 2
    
    
    # Object
    obj_prior = ObjectPrior(dataset_file)
    # Samples
    nb_samples = 1
    ts = time.time()
    metric, hands, frames, objs, P1_sim, P2_sim, P3_sim = generate_task_sample(nb_samples,
                                                                               obj_prior,
                                                                               hand_prior,
                                                                               frame_prior,
                                                                               robot_sim,
                                                                               "birrt",
                                                                               True,
                                                                               True,
                                                                               engine_connection=engine_connection,
                                                                               verbosity=verbosity)
    print("Time:", time.time()-ts)
    np.savez_compressed("test_YCB/test_immovable_object.npz", metric=metric, hand=hands,
                        frame=frames, obj=objs, parameters=P1_sim,
                        planner=P2_sim, savefile=P3_sim)
    
    print("Percentage of reachable poses")
    print(np.mean(metric[:, 0]))

    
    print("Done")
