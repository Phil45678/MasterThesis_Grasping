"""
Data laoder for YCB (small changes from the custom data loader script)
"""

import numpy as np
import os 
import tensorflow as tf 
import time 
import random 
from sklearn.model_selection import train_test_split


def load_non_zero_seq(data_set_loc_str, nb_samples, test_train_ratio, rand_seed, verbose, seq_len, nb_classes, lim):
    if verbose >= 1:
        print("Loading Data.")
    start_time = time.time()
    non_zero_metrics_list, non_zero_hands_list = load_non_zero_grasps(data_set_loc_str, nb_samples, nb_classes, lim)
    end_time = time.time()
    if verbose >= 1:
        print("Time taken to load data: ", end_time - start_time)
        
    inputs = [] 
    outputs = []
    
    # Parsing all buckets 
    for bucket_index in range(nb_classes):
        bucket_offset = 0 
        while (bucket_offset + seq_len - 1  < len(non_zero_metrics_list[bucket_index])):
            # Build the sequence 
            tmp_buffer = [] 
            for seq_index in range(seq_len): 
                # Fusing the data 
                tmp_list = [] 
                tmp_list.append(non_zero_metrics_list[bucket_index][bucket_offset + seq_index])
                for hand_index in range(13): 
                    tmp_list.append(non_zero_hands_list[bucket_index][bucket_offset + seq_index][hand_index])
                tmp_buffer.append(tmp_list.copy())
            
            inputs.append(tmp_buffer.copy())
            outputs.append(tf.one_hot(bucket_index, nb_classes, dtype=np.float64).numpy().tolist())
            bucket_offset += seq_len 
            
        
    # Shuffling the data 
    X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=test_train_ratio, random_state=rand_seed)
    
    if verbose >= 1:
        print("Size of Train set: ", len(X_train))
        print("Size of Test set: ", len(X_test))
        
    return X_train, X_test, y_train, y_test

def load_non_zero_grasps(data_set_loc_str, nb_samples, nb_classes, lim):

    x_min = -0.15
    x_max = 0.15
    y_min = -0.15
    y_max = 0.15 
    z_min = 0.13
    z_max = 0.35
    gripper_min = 0 
    gripper_max = 0.041664
     
    non_zero_metrics_list = [] 
    non_zero_hands_list = [] 
    for index in range(nb_classes):
        non_zero_metrics_list.append([])
        non_zero_hands_list.append([])
        
    for filename in os.listdir(data_set_loc_str): 
        file_data = np.load(data_set_loc_str + "/" + filename)
        
        file_metrics = file_data["metric"].astype(np.float64)
        file_hand_info = file_data["hand"].astype(np.float64)
        file_obj_classes = file_data["obj"].astype(np.float64)
        
        for metric_index in range(len(file_metrics)):
            if file_metrics[metric_index, 0] == 1.0 and file_metrics[metric_index,1] > lim:
                curr_class = int(file_obj_classes[metric_index,0])
                
                non_zero_metrics_list[curr_class].append((file_metrics[metric_index,1] - gripper_min) / (gripper_max - gripper_min))
                tmp_hand_info = file_hand_info[metric_index]
                tmp_hand_info[0] = (tmp_hand_info[0] - x_min) / (x_max - x_min)
                tmp_hand_info[1] = (tmp_hand_info[1] - y_min) / (y_max - y_min)
                tmp_hand_info[2] = (tmp_hand_info[2] - z_min) / (z_max - z_min)
                
                non_zero_hands_list[curr_class].append(tmp_hand_info)
                
    return non_zero_metrics_list, non_zero_hands_list


def load_non_zero_fused_grasps(data_set_loc_str, nb_classes, lim):

    x_min = -0.15
    x_max = 0.15
    y_min = -0.15
    y_max = 0.15 
    z_min = 0.13
    z_max = 0.35
    gripper_min = 0 
    gripper_max = 0.041664
     
    non_zero_grasps_list = []
    for index in range(nb_classes):
        non_zero_grasps_list.append([])
        
    for filename in os.listdir(data_set_loc_str): 
        file_data = np.load(data_set_loc_str + "/" + filename)
        
        file_metrics = file_data["metric"].astype(np.float64)
        file_hand_info = file_data["hand"].astype(np.float64)
        file_obj_classes = file_data["obj"].astype(np.float64)
        
        for metric_index in range(len(file_metrics)):
            if file_metrics[metric_index, 0] == 1.0 and file_metrics[metric_index,1] > lim:
                curr_class = int(file_obj_classes[metric_index,0])
                tmp_list = [] 
                
                tmp_list.append((file_metrics[metric_index,1] - gripper_min) / (gripper_max - gripper_min))
                tmp_hand_info = file_hand_info[metric_index]
                tmp_hand_info[0] = (tmp_hand_info[0] - x_min) / (x_max - x_min)
                tmp_hand_info[1] = (tmp_hand_info[1] - y_min) / (y_max - y_min)
                tmp_hand_info[2] = (tmp_hand_info[2] - z_min) / (z_max - z_min)
                
                for tmp_index in range(13): 
                    tmp_list.append(tmp_hand_info[tmp_index])
                
                non_zero_grasps_list[curr_class].append(tmp_list.copy())
                
    return non_zero_grasps_list

def load_zero_fused_grasps_buckets(data_set_loc_str, nb_classes, lim, nb_tot_samp_class): 
    x_min = -0.15
    x_max = 0.15
    y_min = -0.15
    y_max = 0.15 
    z_min = 0.13
    z_max = 0.35
    gripper_min = 0 
    gripper_max = 0.041664
     
    zero_grasps_list = []
    for index in range(nb_classes):
        zero_grasps_list.append([])
        
    for filename in os.listdir(data_set_loc_str): 
        file_data = np.load(data_set_loc_str + "/" + filename)
        
        file_metrics = file_data["metric"].astype(np.float64)
        file_hand_info = file_data["hand"].astype(np.float64)
        file_obj_classes = file_data["obj"].astype(np.float64)
        
        counters = [] 
        for tmp_index in range(nb_classes): 
            counters.append(len(zero_grasps_list[tmp_index]))
            
        enough_samples = True 
        for tmp_index in range(nb_classes): 
            if counters[tmp_index] < nb_tot_samp_class:
                # not enough samples 
                enough_samples = False 
        if enough_samples: 
            break 
        
        for metric_index in range(len(file_metrics)):
            if file_metrics[metric_index, 0] == 1.0 and file_metrics[metric_index,1] < lim:
                curr_class = int(file_obj_classes[metric_index,0])
                tmp_list = [] 
                
                tmp_list.append((file_metrics[metric_index,1] - gripper_min) / (gripper_max - gripper_min))
                tmp_hand_info = file_hand_info[metric_index]
                tmp_hand_info[0] = (tmp_hand_info[0] - x_min) / (x_max - x_min)
                tmp_hand_info[1] = (tmp_hand_info[1] - y_min) / (y_max - y_min)
                tmp_hand_info[2] = (tmp_hand_info[2] - z_min) / (z_max - z_min)
                
                for tmp_index in range(13): 
                    tmp_list.append(tmp_hand_info[tmp_index])
                
                zero_grasps_list[curr_class].append(tmp_list.copy())
                
    return zero_grasps_list

def equalize_class_buckets(non_zero_grasps_list, zero_grasps_list, nb_tot_samp_class, nb_classes):
    equalized_grasps = non_zero_grasps_list.copy()
    for class_index in range(nb_classes): 
        curr_len = len(non_zero_grasps_list[class_index])
        samp_to_complete = nb_tot_samp_class - curr_len
        for samp_index in range(samp_to_complete): 
            equalized_grasps[class_index].append(zero_grasps_list[class_index][samp_index])
    # Shuffle the data per class 
    for class_index in range(nb_classes): 
        random.shuffle(equalized_grasps[class_index])
    return equalized_grasps

def load_total_YCB_eq_data_seq(data_set_loc_str, nb_samples, nb_classes, joint_dist_lim, seq_len, nb_tot_samp_class, test_train_ratio, rand_seed, verbose): 
    if verbose >= 1:
        print("Loading Data.")
    start_time = time.time()
    
    non_zero_grasps_list = load_non_zero_fused_grasps(data_set_loc_str, nb_classes, joint_dist_lim)
    zero_grasps_list = load_zero_fused_grasps_buckets(data_set_loc_str, nb_classes, joint_dist_lim, nb_tot_samp_class)
    equalized_grasps = equalize_class_buckets(non_zero_grasps_list, zero_grasps_list, nb_tot_samp_class, nb_classes)
    
    end_time = time.time()
    if verbose >= 1:
        print("Time taken to load data: ", end_time - start_time)
        
    inputs = [] 
    outputs = []
    
    # Parsing all buckets 
    for bucket_index in range(nb_classes):
        bucket_offset = 0 
        while (bucket_offset + seq_len - 1  < len(equalized_grasps[bucket_index])):
            # Build the sequence 
            tmp_buffer = [] 
            for seq_index in range(seq_len): 
                # Fusing the data 
                tmp_buffer.append(equalized_grasps[bucket_index][bucket_offset + seq_index].copy())
            
            inputs.append(tmp_buffer.copy())
            outputs.append(tf.one_hot(bucket_index, nb_classes, dtype=np.float64).numpy().tolist())
            bucket_offset += seq_len 
            
        
    # Shuffling the data 
    X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=test_train_ratio, random_state=rand_seed)
    
    if verbose >= 1:
        print("Size of Train set: ", len(X_train))
        print("Size of Test set: ", len(X_test))
        
    return X_train, X_test, y_train, y_test

def load_norm_data_buckets(data_set_loc_str, nb_samples, nb_classes): 
    x_min = -0.15
    x_max = 0.15
    y_min = -0.15
    y_max = 0.15 
    z_min = 0.13
    z_max = 0.35
    gripper_min = 0 
    gripper_max = 0.041664
    
    nb_data_per_class = int((nb_samples * 1.5)/nb_classes)
    
    norm_metrics_buckets = np.zeros((nb_classes, nb_data_per_class, 1))
    norm_hand_infos_buckets  = np.zeros((nb_classes, nb_data_per_class, 13))
    bucket_counters = np.zeros(nb_classes, dtype=int)
    
    # Counter for the filtered data samples 
    nb_OK_samples = 0 
    nb_failure_samples = 0 
    
    # Loading the data from all files 
    for filename in os.listdir(data_set_loc_str): 
        file_data = np.load(data_set_loc_str + "/" + filename)
        
        file_metrics = file_data["metric"].astype(np.float64)
        file_hand_info = file_data["hand"].astype(np.float64)
        file_obj_classes = file_data["obj"].astype(np.float64)
        
        if nb_OK_samples == nb_samples: 
            break
        
        for metric_index in range(len(file_metrics)):
            if file_metrics[metric_index, 0] == 1.0 and file_metrics[metric_index,1] > 0.0:
                if nb_OK_samples == nb_samples: 
                    break
                curr_class = int(file_obj_classes[metric_index,0])
                
                # Normalizing data 
                # only taking joint distance 
                norm_metrics_buckets[curr_class, bucket_counters[curr_class]] = (file_metrics[metric_index,1] - gripper_min) / (gripper_max - gripper_min)
                tmp_hand_info = file_hand_info[metric_index]
                tmp_hand_info[0] = (tmp_hand_info[0] - x_min) / (x_max - x_min)
                tmp_hand_info[1] = (tmp_hand_info[1] - y_min) / (y_max - y_min)
                tmp_hand_info[2] = (tmp_hand_info[2] - z_min) / (z_max - z_min)
                
                norm_hand_infos_buckets[curr_class,bucket_counters[curr_class]] = np.array(tmp_hand_info)
                
                # Increasing the appropriate counter 
                bucket_counters[int(file_obj_classes[metric_index, 0])] += 1 
                
                nb_OK_samples += 1 
            else: 
                nb_failure_samples += 1 

        
    
    return norm_metrics_buckets, norm_hand_infos_buckets, bucket_counters, nb_OK_samples, nb_failure_samples


def load_YCB_dataset_grasping(data_set_loc_str, nb_samples, test_train_ratio, rand_seed, verbose, seq_len, nb_classes): 
    if verbose >= 1:
        print("Loading Data.")
    start_time = time.time()
    norm_metrics, norm_hand_infos, bucket_counters, _, _ = load_norm_data_buckets(data_set_loc_str, nb_samples, nb_classes)
    end_time = time.time()
    if verbose >= 1:
        print("Time taken to load data: ", end_time - start_time)
    
    nb_data_per_class = int((nb_samples * 1.5)/nb_classes)
    
    inputs = [] 
    outputs = [] 
    
    # Parsing all buckets 
    for bucket_index in range(nb_classes):
        bucket_offset = 0 
        while (bucket_offset < nb_data_per_class) and (norm_metrics[bucket_index, bucket_offset + 4] != 0.0): 
            
            # Build the sequence 
            tmp_buffer = [] 
            for seq_index in range(seq_len): 
                # Fusing the data 
                tmp_list = [] 
                tmp_list.append(norm_metrics[bucket_index, bucket_offset + seq_index, 0])
                for hand_index in range(13): 
                    tmp_list.append(norm_hand_infos[bucket_index, bucket_offset + seq_index, hand_index])
                tmp_buffer.append(tmp_list.copy())
            
            inputs.append(tmp_buffer.copy())
            outputs.append(tf.one_hot(bucket_index, nb_classes, dtype=np.float64).numpy().tolist())
            bucket_offset += 5 
    
    # Shuffling the data 
    X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=test_train_ratio, random_state=rand_seed)
    
    if verbose >= 1:
        print("Size of Train set: ", len(X_train))
        print("Size of Test set: ", len(X_test))

    return X_train, X_test, y_train, y_test

def load_norm_data(data_set_loc_str, nb_samples, one_hot, nb_classes): 
    x_min = -0.15
    x_max = 0.15
    y_min = -0.15
    y_max = 0.15 
    z_min = 0.13
    z_max = 0.35
    gripper_min = 0 
    gripper_max = 0.041664
    
    norm_metrics = []
    norm_hand_infos = []
    object_classes = [] 
   
    # Counter for the filtered data samples 
    nb_OK_samples = 0 
    nb_failure_samples = 0 
    
    # Loading the data from all files 
    for filename in os.listdir(data_set_loc_str): 
        file_data = np.load(data_set_loc_str + "/" + filename)
        
        file_metrics = file_data["metric"].astype(np.float64)
        file_hand_info = file_data["hand"].astype(np.float64)
        file_obj_classes = file_data["obj"].astype(np.float64)
        
        if nb_OK_samples == nb_samples: 
                    break
        
        for metric_index in range(len(file_metrics)):
            if file_metrics[metric_index, 0] == 1.0 and file_metrics[metric_index,1] > 0.0:
                if nb_OK_samples == nb_samples: 
                    break
                
                # Normalizing data 
                # only taking joint distance 
                norm_metrics.append((file_metrics[metric_index,1] - gripper_min) / (gripper_max - gripper_min)) 
                tmp_hand_info = file_hand_info[metric_index]
                tmp_hand_info[0] = (tmp_hand_info[0] - x_min) / (x_max - x_min)
                tmp_hand_info[1] = (tmp_hand_info[1] - y_min) / (y_max - y_min)
                tmp_hand_info[2] = (tmp_hand_info[2] - z_min) / (z_max - z_min)
                norm_hand_infos.append(tmp_hand_info)
                if one_hot: 
                    object_classes.append(tf.one_hot(int(file_obj_classes[metric_index, 0]), nb_classes, dtype=np.float64).numpy().tolist()) # only taking object class 
                else:     
                    object_classes.append(file_obj_classes[metric_index, 0]) # only taking object class 

                nb_OK_samples += 1 
            else: 
                nb_failure_samples += 1 

        
    
    return norm_metrics, norm_hand_infos, object_classes, nb_OK_samples, nb_failure_samples

def load_YCB_grasp_dataset(data_set_loc_str, nb_samples, test_train_ratio, rand_seed, verbose): 
    if verbose >= 1:
        print("Loading Data.")
    start_time = time.time()
    norm_metrics, norm_hand_infos, object_classes, nb_OK_samples, nb_failure_samples = load_norm_data(data_set_loc_str, nb_samples, True, 24)
    end_time = time.time()
    print("Time taken to load data: ", end_time - start_time)
    
    inputs = [] 
    outputs = object_classes
    
    # Fusing the informations 
    for index in range(len(norm_metrics)): 
        tmp_list = [] 
        tmp_list.append(np.float64(norm_metrics[index]))
        for hand_index in range(len(norm_hand_infos[-1])): 
            tmp_list.append(np.float64(norm_hand_infos[index][hand_index]))
        inputs.append(tmp_list.copy())
    # Shuffling the data 
    X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=test_train_ratio, random_state=rand_seed)
    
    if verbose >= 1:
        print("Size of Train set: ", len(X_train))
        print("Size of Test set: ", len(X_test))
        
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Hyper-parameters 
    # nb_samples = 3000000
    nb_samples = 30
    nb_classes = 24
    one_hot = True

    # Data set location 
    data_set_loc_str = r"C:\Users\phili\Documents\GitHub\DexterousManipulation\generations\DATA_SET_YCB"

    norm_metrics, norm_hand_infos, object_classes, nb_OK_samples, nb_failure_samples = load_noram_data(data_set_loc_str, nb_samples, one_hot, nb_classes)

    print("Number of useable samples: ", nb_OK_samples)
    print("Number of unuseable samples: ", nb_failure_samples)