"""
Script implementing bayesian optimization in a large loop for statistical sampling 
"""

# Importing packages 
import pybullet
import numpy as np
import sys
sys.path.append("../..")

import trimesh
import tensorflow as tf

import gpflow

# from __future__ import annotations

import numpy as np

import trieste 
from trieste.space import Box

from DexterousManipulation.simulations.classification_simulator import RobotSimulator
from DexterousManipulation.simulations.utils_sim import get_background, get_robot
from DexterousManipulation.generations.object_prior import ObjectPrior
from DexterousManipulation.generations.hand_prior import HandPrior
from DexterousManipulation.generations.frame_prior import FramePrior
from DexterousManipulation.controllers.gripper_controller import SawyerElectricController

from DexterousManipulation.training.NatTrained import NatGradTrainedVGP

import time

from scipy.stats import entropy

from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.acquisition import (
    SingleModelAcquisitionBuilder, ExpectedImprovement, Product, lower_confidence_bound
)


class ProbabilityOfValidity(SingleModelAcquisitionBuilder):
    def prepare_acquisition_function(self, dataset, model):
        def acquisition(at):
            mean, _ = model.predict_y(tf.squeeze(at, -2))
            return mean
        return acquisition




def check_bayesian_opt_suggestion(bay_h): 
    # Checking the hand configuration on the 'real' simulation 
    current_grasp = bay_h
    h_pos = current_grasp[:3]
    h_pos = tf.reshape(h_pos, [1,3])

    angle = current_grasp[-1:]

    final_h = []
    final_h.append(h_pos)
    final_h.append(tf.constant([[0.,  0., -1. ]], dtype = tf.float32))
    final_h.append(angle)
    final_h.append(tf.constant([[1.]], dtype = tf.float32))
    real_simulation_result = run_simu(final_h, robot_sim_real, obj_prior_class_1)
    
    # Append to the sequence of grasps 
    full_h_conv = hand_prior.to_network(final_h).numpy() 
    tmp_grasp_list = full_h_conv.tolist()
    tmp_grasp_list[0].insert(0, real_simulation_result[1])
    previous_grasps_list.append(tmp_grasp_list[0])
    previous_grasps_tens = tf.constant(previous_grasps_list)
    
    # Make the prediction 
    embedding_grasp_seq = model_prediction_PHI.predict(previous_grasps_tens)
    seq_embedding_sum = embedding_grasp_seq.sum(0)
    input_tensor_seq = tf.constant(seq_embedding_sum)
    input_tensor_seq = tf.reshape(input_tensor_seq, [1,-1])
    seq_prediction_final = model_prediction_RHO.predict(input_tensor_seq)
    
    return seq_prediction_final

# generating task sample Function used (Credit-> Norman)
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


def run_simu(h_sample, robot_simulator, obj_prior):
    # Hard-coded parameters 
    engine_connection = "DIRECT"
    verbosity = 0
    save_file = None
    planner = "birrt"
    
    pO_sample = [0, 0, 0]
    
    obj_sim = obj_prior.sample(noise=False)
        
    
    h_sim = hand_prior.to_network(h_sample).numpy()    
    
    
    inputs = [obj_sim.copy(), h_sim.copy(), pO_sample.copy()]
    
    lat_coef = 0.5
    spin_coef = lat_coef*2*1e-3 # gamma = µ*radius
    table_scale = 1.
    
    parameters = [lat_coef, spin_coef, planner, table_scale] 
    
    
    metric = robot_simulator.forward(inputs, parameters,
                                            engine_connection=engine_connection,
                                            verbosity=verbosity,
                                            save_file=save_file)
    
    return metric

# Some auxiliary functions used in the objective function 
def convert_x_to_fullhand_pos(x, index): 
    current_grasp = tf.cast(x[index], tf.float32)
    h_pos = current_grasp[:3]
    h_pos = tf.reshape(h_pos, [1,3])
    
    angle = current_grasp[-1:]

    final_h = []
    final_h.append(h_pos)
    final_h.append(tf.constant([[0.,  0., -1. ]], dtype = tf.float32))
    final_h.append(angle)
    final_h.append(tf.constant([[1.]], dtype = tf.float32))
    
    return final_h

if __name__ == "__main__":
    start_time = time.time()
    # Hyper-paramters 
    arguments = sys.argv 
    data_index = arguments[1]

    nb_samples_per_h = 2
    nb_initial_points = 2
    final_seq_len = 3 
    size_init_data_set_bay = 30
    nb_bay_sugg  = 2
    final_nb_regr = final_seq_len - 1 

    dataset_file_class_1 = "../generations/BIG_BOX.json"
    dataset_file_class_0 = "../generations/MEDIUM_BOX.json"
    dataset_file_real = "../generations/MEDIUM_BOX.json"

    # Creating the data arrays
    initial_h = np.zeros(14)
    inital_data_set = np.zeros((size_init_data_set_bay, 4))
    sub_data_sets = np.zeros((nb_samples_per_h, nb_initial_points, 4))
    prediction_first_h = np.zeros(2)
    predictions_bay_opt = np.zeros((nb_samples_per_h, final_nb_regr, 2))
    predictions_rand_grasp = np.zeros((nb_samples_per_h, final_nb_regr, 2))
    bay_op_suggs = np.zeros((nb_samples_per_h, final_nb_regr, 4))
    bay_op_suggs_scores =  np.zeros((nb_samples_per_h, final_nb_regr))
    random_h_regrs = np.zeros((nb_samples_per_h, final_nb_regr, 13))
    first_grasp_metric_save = np.zeros(1)
    bay_grasps_metric = np.zeros((nb_samples_per_h, final_nb_regr, 1))
    rand_grasps_metric = np.zeros((nb_samples_per_h, final_nb_regr, 1))

    # Loading the prediction models for variable sequence length 
    load_str_model_prediction_PHI = r"C:\Users\phili\Documents\GitHub\DexterousManipulation\Figures\Training\500k_Data_Set\Big_Cube_and_Medium_Cube\var_seq_len_max_5\Model_PHI_1000_epochs_deep_sets_nb_75"
    model_prediction_PHI = tf.keras.models.load_model(load_str_model_prediction_PHI)

    load_str_model_prediction_RHO = r"C:\Users\phili\Documents\GitHub\DexterousManipulation\Figures\Training\500k_Data_Set\Big_Cube_and_Medium_Cube\var_seq_len_max_5\Model_RHO_1000_epochs_deep_sets_nb_75"
    model_prediction_RHO = tf.keras.models.load_model(load_str_model_prediction_RHO)


    # Loading the prediction models for fixed sequence length 
    # load_str_model_prediction_1 = r"C:\Users\phili\Documents\GitHub\DexterousManipulation\Figures\Training\500k_Data_Set\Big_Cube_and_Medium_Cube\seq_len_1\Model_1000_epochs__session_1"
    # model_prediction_1 = tf.keras.models.load_model(load_str_model_prediction_1)
    
    # load_str_model_prediction_2_PHI = r"C:\Users\phili\Documents\GitHub\DexterousManipulation\Figures\Training\500k_Data_Set\Big_Cube_and_Medium_Cube\seq_len_2_no_redraw\Model_PHI_1000_epochs_deep_sets_nb_30"
    # load_str_model_prediction_2_RHO =  r"C:\Users\phili\Documents\GitHub\DexterousManipulation\Figures\Training\500k_Data_Set\Big_Cube_and_Medium_Cube\seq_len_2_no_redraw\Model_RHO_1000_epochs_deep_sets_nb_30"

    # model_prediction_PHI_2 = tf.keras.models.load_model(load_str_model_prediction_2_PHI)
    # model_prediction_RHO_2 = tf.keras.models.load_model(load_str_model_prediction_2_RHO)

    # Generating a first valid grasp 
    sim = "sawyer" # For the Sawyer Robot
    # Background
    background = get_background(sim)
    robot_dict = get_robot(sim)
    gripper_ctrl_class_0 = SawyerElectricController(None, sim, None)
    gripper_ctrl_class_1 = SawyerElectricController(None, sim, None)
    gripper_ctrl_real = SawyerElectricController(None, sim, None)

    # Prior distrib for h
    pos_parameters = {"distrib": "cube",
                        "shellk": 10., 
                    "xmin": -0.02,
                    "xmax": 0.02,
                    "ymin": -0.02,
                    "ymax": 0.02,
                    "zmin": 0.11,
                    "zmax": 0.16,
                    #"rdistrib": "truncnorm",
                    "rdistrib": "uniform",
                    "rmin": 0.1,
                    "rmax": 0.15}
    ori_parameters = {"dof": 1,
                    "kappa": 20}
    finger_parameters = {"preshape": [1./3.]*1}
    hand_prior = HandPrior(pos_parameters, ori_parameters, finger_parameters)
    frame_prior = FramePrior(xmin=0, xmax=0, ymin=0, ymax=0) # Object does not move 

    sim_parameters = {"model": "soft",
                        "ng": 20,
                        "restitution": 0,
                        "threshold": 1e-3}
    robot_sim_class_0 = RobotSimulator(pybullet, robot_dict, background, gripper_ctrl_class_0,
                                sim_parameters)
    robot_sim_class_1 = RobotSimulator(pybullet, robot_dict, background, gripper_ctrl_class_1,
                                sim_parameters)
    robot_sim_real = RobotSimulator(pybullet, robot_dict, background, gripper_ctrl_real,
                                sim_parameters)
    engine_connection = "DIRECT"
    verbosity = 0 
    nb_samples = 1 

    # dataset_file = "../generations/BIG_BOX.json"

    # dataset_file = "../generations/SMALL_BOX.json"
    # dataset_file = "../generations/CYLINDRE.json"
    obj_prior_class_0 = ObjectPrior(dataset_file_class_0)
    obj_prior_class_1 = ObjectPrior(dataset_file_class_1)

    obj_prior_real = ObjectPrior(dataset_file_real)

    # Running the simulation a first time 
    metric, hands, frames, objs, P1_sim, P2_sim, P3_sim = generate_task_sample(nb_samples,
                                                                               obj_prior_real,
                                                                               hand_prior,
                                                                               frame_prior,
                                                                               robot_sim_real,
                                                                               "birrt",
                                                                               True,
                                                                               True,
                                                                               engine_connection=engine_connection,
                                                                               verbosity=verbosity)

    valid_grasp = True 
    if metric[0,0] == 0.0:
        # print("Unsuccessful grasp...")
        valid_grasp = False 
        pass 
    else: 
        # print("Successful grasp!")
        pass 

    # Retrying random grasps, until one is sucessful
    while not valid_grasp: 
        metric, hands, frames, objs, P1_sim, P2_sim, P3_sim = generate_task_sample(nb_samples,
                                                                                obj_prior_real,
                                                                                hand_prior,
                                                                                frame_prior,
                                                                                robot_sim_real,
                                                                                "birrt",
                                                                                True,
                                                                                True,
                                                                                engine_connection=engine_connection,
                                                                                verbosity=verbosity)
        
        if metric[0,0] == 0.0:
            # print("Unsuccessful grasp...")
            valid_grasp = False 
        else: 
            valid_grasp = True 
            

    first_grasp_metric = metric
    first_grasp_hand = hands

    # Saving the data 
    first_grasp_metric_save = first_grasp_metric[0,1]
    initial_h = hands 

    # Creating the input 
    first_grasp_list = []
    first_grasp_list.append(first_grasp_metric[0,1])
    for it in range(len(first_grasp_hand[0])): 
        first_grasp_list.append(first_grasp_hand[0,it])

    first_grasp_tensor = tf.constant(first_grasp_list)
    first_grasp_tensor = tf.reshape(first_grasp_tensor, [1,-1])

    embedding_first_grasp = model_prediction_PHI.predict(first_grasp_tensor)
    first_grasp_prediction = model_prediction_RHO.predict(embedding_first_grasp)

    # Saving the first prediction 
    prediction_first_h = first_grasp_prediction

    # print(first_grasp_prediction)
    # print(first_grasp_prediction.sum())

    # -------------------------- Starting the bayesian optimization part -------------------------- # 
    # Defining the search space 
    search_space = Box([-0.02, -0.02, 0.11, -np.pi], [0.02, 0.02, 0.16, np.pi])


    previous_grasps_tensor = first_grasp_tensor
    previous_grasps_list = [] 
    previous_grasps_list.append(first_grasp_list.copy())

    def obj_function(x):
        # In this function, the whole thing will be done, i.e.: 
        # - the simulation of the hand position for both possible object classes 
        # - the prediction on the whole new sequence of grasps for both cases 
        # - the expected entropy of these predictions 
        
        nb_points = x.shape[0]
        y_list = []
        for i in range(nb_points): 
        
            # Simulating both object class cases 
            full_h = convert_x_to_fullhand_pos(x, i)
            class_0_simulation_result = run_simu(full_h, robot_sim_class_0, obj_prior_class_0)
            class_1_simulation_result = run_simu(full_h, robot_sim_class_1, obj_prior_class_1)

            # Output will be nan, if at least one of the simulations fail 
            if class_0_simulation_result[0] == 1. and class_1_simulation_result[0] == 1.: 
                # Both simulations were successfull                
                # Build the new grasp lists
                full_h_conv = hand_prior.to_network(full_h).numpy() 
                
                input_list_new_grasp_class_0 = full_h_conv.tolist()
                input_list_new_grasp_class_0[0].insert(0, class_0_simulation_result[1])
                class_0_grasp_seq_list = previous_grasps_list.copy()
                class_0_grasp_seq_list.append(input_list_new_grasp_class_0[0])
                
                # print(class_0_grasp_seq_list)
                
                class_0_seq_tensor = tf.constant(class_0_grasp_seq_list)
                # print(class_0_seq_tensor)
                
                input_list_new_grasp_class_1 = full_h_conv.tolist()
                input_list_new_grasp_class_1[0].insert(0, class_1_simulation_result[1])
                class_1_grasp_seq_list =  previous_grasps_list.copy()
                class_1_grasp_seq_list.append(input_list_new_grasp_class_1[0])
                
                class_1_seq_tensor = tf.constant(class_1_grasp_seq_list)
                # print(class_1_seq_tensor)
                
                
                
                # Compute the predictions of the newly obtained sequences 
                embedding_class_0 = model_prediction_PHI.predict(class_0_seq_tensor)
                class_0_embedding_sum = embedding_class_0.sum(0)
                input_tensor_class_0 = tf.constant(class_0_embedding_sum)
                input_tensor_class_0 = tf.reshape(input_tensor_class_0, [1,-1])
                class_0_seq_prediction = model_prediction_RHO.predict(input_tensor_class_0)
                
                # print(class_0_seq_prediction)
                
                embedding_class_1 = model_prediction_PHI.predict(class_1_seq_tensor)
                class_1_embedding_sum = embedding_class_1.sum(0)
                input_tensor_class_1 = tf.constant(class_1_embedding_sum)
                input_tensor_class_1 = tf.reshape(input_tensor_class_1, [1,-1])
                class_1_seq_prediction = model_prediction_RHO.predict(input_tensor_class_1)
                
                # print(class_1_seq_prediction)
                
                # Computing the expected entropy of the predictions 
                class_0_pred_list = class_0_seq_prediction.tolist()
                class_0_entropy = entropy(class_0_pred_list[0], base = 2)
                
                class_1_pred_list = class_1_seq_prediction.tolist()
                class_1_entropy = entropy(class_1_pred_list[0], base = 2)
                
                # For now, simply computing the average, can be changed later 
                expected_entropy = (class_0_entropy + class_1_entropy)/2
                
                # Adding to the outputs 
                y_list.append(expected_entropy)
                # y_list.append(1 - expected_entropy) #trying the opposite, just for the heck of it 
            else: 
                y_list.append(np.nan)
                
        # Reshaping output for compatibility
        y_tens = tf.constant(y_list, dtype=tf.float64)
        y_tens = tf.reshape(y_tens, [-1,1])
        return y_tens

    # test = obj_function(search_space.sample(2))    
    # print(test)

    OBJECTIVE = "OBJECTIVE"
    FAILURE = "FAILURE"

    def observer(x):
        y = obj_function(x)
        mask = np.isfinite(y).reshape(-1)
        return {
            OBJECTIVE: trieste.data.Dataset(x[mask], y[mask]),
            FAILURE: trieste.data.Dataset(x, tf.cast(np.isfinite(y), tf.float64))
        }

    # Sampling the big data set once 
    num_init_points = size_init_data_set_bay
    initial_data = observer(search_space.sample(num_init_points))

    # Saving the data set 
    inital_data_set = initial_data[FAILURE].query_points.numpy()

    sucess_indicators = initial_data[FAILURE].observations.numpy()
    success_index = [] 
    failure_index = [] 
    success_index_y = [] 
    success_index_y_counter = 0 
    for index in range(len(sucess_indicators)): 
        if sucess_indicators[index] == 1.0: 
            success_index.append(index)
            success_index_y.append(success_index_y_counter)
            success_index_y_counter += 1 
        else: 
            failure_index.append(index)

    # First loop: iterating over different initial data sets for the bayesian optimization
    for sub_sample in range(nb_samples_per_h): 
        tmp_index_0 = success_index[sub_sample%len(success_index)]
        tmp_index_1 = failure_index[sub_sample%len(failure_index)]
        x_tmp_initial_data = np.concatenate(([initial_data[FAILURE].query_points[tmp_index_0].numpy()], 
                                        [initial_data[FAILURE].query_points[tmp_index_1].numpy()]), axis=0)
        
        # print(success_index[sub_sample%len(success_index)])
        # print(failure_index[sub_sample%len(failure_index)])
        y_tmp_initial_data = np.concatenate(([initial_data[OBJECTIVE].observations[sub_sample%len(success_index)].numpy()], [[np.nan]]), axis=0)
        # print(sub_sample%len(success_index))
        x_tmp_tens = tf.constant(x_tmp_initial_data)
        y_tmp_tens = tf.constant(y_tmp_initial_data)
        y_tmp_tens = tf.reshape(y_tmp_tens, [-1,1])
        mask = np.isfinite(y_tmp_tens).reshape(-1)
        tmp_data_set = {
            OBJECTIVE: trieste.data.Dataset(x_tmp_tens[mask], y_tmp_tens[mask]),
            FAILURE: trieste.data.Dataset(x_tmp_tens, tf.cast(np.isfinite(y_tmp_tens), tf.float64))
        }

        # print(tmp_data_set)

        # Saving the sub data set 
        sub_data_sets[sub_sample] = tmp_data_set[FAILURE].query_points.numpy()

        
        
        # print(tmp_data_set)
        import gpflow 
        def create_regression_model(data):
            variance = tf.math.reduce_variance(data.observations)
            # kernel = gpflow.kernels.Matern52(variance=variance, lengthscales=[0.2, 0.2, 0.2, 0.2])
            kernel = gpflow.kernels.Matern52(variance=0.1, lengthscales=[0.2, 0.2, 0.2, 0.2])
            gpr = gpflow.models.GPR(data.astuple(), kernel, noise_variance=1e-5)
            gpflow.set_trainable(gpr.likelihood, False)
            return gpr


        def create_classification_model(data):
            kernel = gpflow.kernels.SquaredExponential(
                variance=100.0, lengthscales=[0.2, 0.2, 0.2, 0.2]
            )
            likelihood = gpflow.likelihoods.Bernoulli()
            vgp = gpflow.models.VGP(data.astuple(), kernel, likelihood)
            gpflow.set_trainable(vgp.kernel.variance, False)
            return vgp

        regression_model = create_regression_model(tmp_data_set[OBJECTIVE])
        classification_model = create_classification_model(tmp_data_set[FAILURE])
        # print("classification model: ", classification_model)
        # print("classification model type : ", type(classification_model))

        import gpflow 

        OBJECTIVE = "OBJECTIVE"
        FAILURE = "FAILURE"

        models= {
            OBJECTIVE: {
                "model": regression_model,
                "optimizer": gpflow.optimizers.Scipy(),
                "optimizer_args": {
                    "minimize_args": {"options": dict(maxiter=100)},
                },
            },
            FAILURE: NatGradTrainedVGP(classification_model),
        }

        ei = ExpectedImprovement()
        pov = ProbabilityOfValidity()
        acq_fn = Product(ei.using(OBJECTIVE), pov.using(FAILURE))
        rule = EfficientGlobalOptimization(acq_fn)  # type: ignore

        bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)

        result = bo.optimize(1, tmp_data_set, models, rule).final_result.unwrap()

        arg_min_idx = tf.squeeze(tf.argmin(result.datasets[OBJECTIVE].observations, axis=0))
        # print(f"query point: {result.datasets[OBJECTIVE].query_points[arg_min_idx, :]}")

        # Saving suggestions 
        bay_op_suggs[sub_sample, 0] = result.datasets[OBJECTIVE].query_points[arg_min_idx, :]
        bay_op_suggs_scores[sub_sample, 0] = result.datasets[OBJECTIVE].observations[arg_min_idx]


        # Checking the hand configuration on the 'real' simulation 
        bay_h = tf.cast(result.datasets[OBJECTIVE].query_points[arg_min_idx, :], tf.float32)
        current_grasp = bay_h
        h_pos = current_grasp[:3]
        h_pos = tf.reshape(h_pos, [1,3])

        angle = current_grasp[-1:]

        final_h = []
        final_h.append(h_pos)
        final_h.append(tf.constant([[0.,  0., -1. ]], dtype = tf.float32))
        final_h.append(angle)
        final_h.append(tf.constant([[1.]], dtype = tf.float32))
        real_simulation_result = run_simu(final_h, robot_sim_real, obj_prior_class_1)

        # Save bay metric
        bay_grasps_metric[sub_sample, 0] = real_simulation_result[1]
        
        # Append to the sequence of grasps 
        full_h_conv = hand_prior.to_network(final_h).numpy() 
        tmp_grasp_list = full_h_conv.tolist()
        tmp_grasp_list[0].insert(0, real_simulation_result[1])
        previous_grasps_list.append(tmp_grasp_list[0])
        previous_grasps_tens = tf.constant(previous_grasps_list)
        
        # Make the prediction 
        embedding_grasp_seq = model_prediction_PHI.predict(previous_grasps_tens)
        seq_embedding_sum = embedding_grasp_seq.sum(0)
        input_tensor_seq = tf.constant(seq_embedding_sum)
        input_tensor_seq = tf.reshape(input_tensor_seq, [1,-1])
        seq_prediction_final = model_prediction_RHO.predict(input_tensor_seq)

        predictions_bay_opt[sub_sample, 0] = seq_prediction_final

        # Comparing this with a random second grasp (which is reachable)
        second_random_metric, second_random_hands, _, _, _, _, _ = generate_task_sample(nb_samples,
                                                                                    obj_prior_real,
                                                                                    hand_prior,
                                                                                    frame_prior,
                                                                                    robot_sim_real,
                                                                                    "birrt",
                                                                                    True,
                                                                                    True,
                                                                                    engine_connection=engine_connection,
                                                                                    verbosity=verbosity)
        
        second_valid_grasp = True 
        if second_random_metric[0,0] == 0.0:
            # print("Unsuccessful grasp...")
            second_valid_grasp = False 
            pass 
        else: 
            # print("Successful grasp!")
            pass 

        # Retrying random grasps, until one is sucessful
        while not second_valid_grasp: 
            second_random_metric, second_random_hands, _, _, _, _, _ = generate_task_sample(nb_samples,
                                                                                    obj_prior_real,
                                                                                    hand_prior,
                                                                                    frame_prior,
                                                                                    robot_sim_real,
                                                                                    "birrt",
                                                                                    True,
                                                                                    True,
                                                                                    engine_connection=engine_connection,
                                                                                    verbosity=verbosity)
            
            if second_random_metric[0,0] == 0.0:
                # print("Unsuccessful grasp...")
                second_valid_grasp = False 
            else: 
                second_valid_grasp = True   

        # Save second random grasp 
        random_h_regrs[sub_sample, 0] = second_random_hands
        rand_grasps_metric[sub_sample, 0] = second_random_metric[0,1]

        # Making the comparative prediction on the second random grasp 
        # Creating the input 
        second_random_grasp_list = []
        second_random_grasp_list.append(second_random_metric[0,1])
        for it in range(len(second_random_hands[0])): 
            second_random_grasp_list.append(second_random_hands[0,it])

        second_random_grasp_tensor = tf.constant(second_random_grasp_list)
        second_random_grasp_tensor = tf.reshape(second_random_grasp_tensor, [1,-1])
        # print(second_random_grasp_tensor)

        second_random_seq_grasp_list = []
        second_random_seq_grasp_list.append(first_grasp_list.copy())
        second_random_seq_grasp_list.append(second_random_grasp_list)
        sec_rand_seq_tens = tf.constant(second_random_seq_grasp_list)
        # print(sec_rand_seq_tens)

        # Make prediction on random second grasp 
        embedding_random_grasp_seq = model_prediction_PHI.predict(sec_rand_seq_tens)
        random_seq_embedding_sum = embedding_random_grasp_seq.sum(0)
        random_input_tensor_seq = tf.constant(random_seq_embedding_sum)
        random_input_tensor_seq = tf.reshape(random_input_tensor_seq, [1,-1])
        random_seq_prediction_final = model_prediction_RHO.predict(random_input_tensor_seq)
        # print(random_seq_prediction_final)
        predictions_rand_grasp[sub_sample, 0] = random_seq_prediction_final


        # Repeat this for x new grasps 
        it_counter = 1
        while it_counter < final_nb_regr: 
            # Redrawing from the bayesian optimizer model 

            result = bo.optimize(1, result.datasets, result.models, rule).final_result.unwrap()

            #  Saving suggestions 
            bay_op_suggs[sub_sample, it_counter] = result.datasets[OBJECTIVE].query_points[arg_min_idx, :]
            bay_op_suggs_scores[sub_sample, it_counter] = result.datasets[OBJECTIVE].observations[arg_min_idx]


            bay_h = tf.cast(result.datasets[OBJECTIVE].query_points[arg_min_idx, :], tf.float32)
            current_grasp = bay_h
            h_pos = current_grasp[:3]
            h_pos = tf.reshape(h_pos, [1,3])

            angle = current_grasp[-1:]

            final_h = []
            final_h.append(h_pos)
            final_h.append(tf.constant([[0.,  0., -1. ]], dtype = tf.float32))
            final_h.append(angle)
            final_h.append(tf.constant([[1.]], dtype = tf.float32))
            real_simulation_result = run_simu(final_h, robot_sim_real, obj_prior_class_1)

            # Save bay metric
            bay_grasps_metric[sub_sample, it_counter] = real_simulation_result[1]
            
            # Append to the sequence of grasps 
            full_h_conv = hand_prior.to_network(final_h).numpy() 
            tmp_grasp_list = full_h_conv.tolist()
            tmp_grasp_list[0].insert(0, real_simulation_result[1])
            previous_grasps_list.append(tmp_grasp_list[0])
            previous_grasps_tens = tf.constant(previous_grasps_list)
            
            # Make the prediction 
            embedding_grasp_seq = model_prediction_PHI.predict(previous_grasps_tens)
            seq_embedding_sum = embedding_grasp_seq.sum(0)
            input_tensor_seq = tf.constant(seq_embedding_sum)
            input_tensor_seq = tf.reshape(input_tensor_seq, [1,-1])
            seq_prediction_final = model_prediction_RHO.predict(input_tensor_seq)

            predictions_bay_opt[sub_sample, it_counter] = seq_prediction_final

            # Comparing this with a random second grasp (which is reachable)
            second_random_metric, second_random_hands, _, _, _, _, _ = generate_task_sample(nb_samples,
                                                                                        obj_prior_real,
                                                                                        hand_prior,
                                                                                        frame_prior,
                                                                                        robot_sim_real,
                                                                                        "birrt",
                                                                                        True,
                                                                                        True,
                                                                                        engine_connection=engine_connection,
                                                                                        verbosity=verbosity)
            
            second_valid_grasp = True 
            if second_random_metric[0,0] == 0.0:
                # print("Unsuccessful grasp...")
                second_valid_grasp = False 
                pass 
            else: 
                # print("Successful grasp!")
                pass 

            # Retrying random grasps, until one is sucessful
            while not second_valid_grasp: 
                second_random_metric, second_random_hands, _, _, _, _, _ = generate_task_sample(nb_samples,
                                                                                        obj_prior_real,
                                                                                        hand_prior,
                                                                                        frame_prior,
                                                                                        robot_sim_real,
                                                                                        "birrt",
                                                                                        True,
                                                                                        True,
                                                                                        engine_connection=engine_connection,
                                                                                        verbosity=verbosity)
                
                if second_random_metric[0,0] == 0.0:
                    # print("Unsuccessful grasp...")
                    second_valid_grasp = False 
                else: 
                    second_valid_grasp = True   

            # Save second random grasp 
            random_h_regrs[sub_sample, it_counter] = second_random_hands
            rand_grasps_metric[sub_sample, it_counter] = second_random_metric[0,1]

            # Making the comparative prediction on the second random grasp 
            # Creating the input 
            tmp_random_grasp_list = []
            tmp_random_grasp_list.append(second_random_metric[0,1])
            for it in range(len(second_random_hands[0])): 
                tmp_random_grasp_list.append(second_random_hands[0,it])

            tmp_random_grasp_tensor = tf.constant(tmp_random_grasp_list)
            tmp_random_grasp_tensor = tf.reshape(tmp_random_grasp_tensor, [1,-1])
            # print(second_random_grasp_tensor)

            second_random_seq_grasp_list.append(tmp_random_grasp_list.copy())
            sec_rand_seq_tens = tf.constant(second_random_seq_grasp_list)
            # print(sec_rand_seq_tens)

            # Make prediction on random second grasp 
            embedding_random_grasp_seq = model_prediction_PHI.predict(sec_rand_seq_tens)
            random_seq_embedding_sum = embedding_random_grasp_seq.sum(0)
            random_input_tensor_seq = tf.constant(random_seq_embedding_sum)
            random_input_tensor_seq = tf.reshape(random_input_tensor_seq, [1,-1])
            random_seq_prediction_final = model_prediction_RHO.predict(random_input_tensor_seq)
            # print(random_seq_prediction_final)
            predictions_rand_grasp[sub_sample, it_counter] = random_seq_prediction_final


            it_counter += 1  


                        
    # Saving all the data 

    np.savez_compressed("bay_opt_DATA_med_2/data_h_" +str(data_index) +  ".npz", initial_h=initial_h, 
                        inital_data_set=inital_data_set,
                        sub_data_sets = sub_data_sets,
                        prediction_first_h = prediction_first_h,
                        predictions_bay_opt = predictions_bay_opt,
                        predictions_rand_grasp = predictions_rand_grasp, 
                        bay_op_suggs = bay_op_suggs, 
                        bay_op_suggs_scores = bay_op_suggs_scores, 
                        random_h_regrs = random_h_regrs, 
                        first_grasp_metric_save = first_grasp_metric_save, 
                        bay_grasps_metric = bay_grasps_metric, 
                        rand_grasps_metric = rand_grasps_metric
                        )

    print("Time taken: ", time.time() - start_time)