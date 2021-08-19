"""

Extended data loader, with several functionlaities, to load the data from the npz files 

"""



# Importing packackages 

import numpy as np 

from sklearn.model_selection import train_test_split

import os 

from os import path

import time

import tensorflow as tf 

import random 

import matplotlib.pyplot as plt
from tensorflow.python.ops.gen_array_ops import one_hot 


from tensorflow.keras import layers
from tensorflow import keras

# Function that produces the random sequence length data for 2 classes and draws random samples to build the sequence 
def load_rand_seq_data_rand_redraw(class_0_folder_name, class_1_folder_name, max_seq_len, nb_raw_samples, nb_seq_samples, one_hot_encoding, shuffling_seed, test_train_ratio, verbose): 
    # Loading the data first 
    class_0_metrics , class_0_hand_infos, time_taken_0 = load_filtered_class_data(class_0_folder_name, nb_raw_samples)
    if verbose >= 1:
        print("Time needed to load class 0 data (in seconds): " , time_taken_0)
        print("Number of samples for class 0: ", len(class_0_metrics))
    class_1_metrics , class_1_hand_infos, time_taken_1 = load_filtered_class_data(class_1_folder_name, nb_raw_samples)
    if verbose >= 1:
        print("Time needed to load class 1 data (in seconds): " , time_taken_1)
        print("Number of samples for class 1: ", len(class_1_metrics)) 

    # Complete input and output lists 
    inputs = [] 
    outputs = []

    # Creating sequences of random length for class 0 
    for _ in range(nb_seq_samples): 
        # Determining the current sequence length at random 
        curr_seq_len = random.randint(1, max_seq_len)

        curr_sample = [] 

        # Building the data sequence 
        for _ in range(curr_seq_len):
            tmp_list = [] 
            # Draw random sample 
            rand_sample = random.randint(0, nb_raw_samples-1)
            tmp_list.append(np.float64(class_0_metrics[rand_sample]))
            for hand_index in range(len(class_0_hand_infos[-1])): 
                tmp_list.append(np.float64(class_0_hand_infos[rand_sample][hand_index]))
            curr_sample.append(tmp_list)
        
        inputs.append(curr_sample.copy())
        curr_sample.clear()
        
        # Adding class to ouputs 
        if one_hot_encoding: 
            outputs.append(tf.one_hot(int(0), 2, dtype=np.float64).numpy().tolist())
        else: 
            outputs.append([0])

    
    # Creating sequences of random length for class 1 
    for _ in range(nb_seq_samples): 
        # Determining the current sequence length at random 
        curr_seq_len = random.randint(1, max_seq_len)

        curr_sample = [] 

        # Building the data sequence 
        for _ in range(curr_seq_len):
            tmp_list = [] 
            # Draw random sample 
            rand_sample = random.randint(0, nb_raw_samples-1)
            tmp_list.append(np.float64(class_1_metrics[rand_sample]))
            for hand_index in range(len(class_1_hand_infos[-1])): 
                tmp_list.append(np.float64(class_1_hand_infos[rand_sample][hand_index]))
            curr_sample.append(tmp_list)
        
        inputs.append(curr_sample.copy())
        curr_sample.clear()
        
        # Adding class to ouputs 
        if one_hot_encoding: 
            outputs.append(tf.one_hot(int(1), 2, dtype=np.float64).numpy().tolist())
        else: 
            outputs.append([1])

    # Shuffling the data 
    X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=test_train_ratio, random_state=shuffling_seed)

    return X_train, X_test, y_train, y_test


# Function loading the  (raw) data for one class, up to a certain number of samples. 
# Filters the data 
# NOTE: the object class is determined later and thus the eventual one-hot encoding is applied later too 
def load_filtered_class_data(folder_name, nb_samples): 

    # Using this for measuring the total data loading time 
    start_time = time.time()

    basepath = path.dirname(__file__)
    # Assuming that the data folder is in the generations folder 
    filepath = path.abspath(path.join(basepath, "..", "generations", folder_name))

    # Will contain the filtered data points 
    filtered_metrics = [] 
    filtered_hand_infos = [] 

    # Counter for the filtered data samples 
    nb_curr_filt_samples = 0 

    # Will contain all data points 
    metrics =  [] 
    hand_infos = []

    # Loading the data per npz file
    for filename in os.listdir(filepath): 
        file_data = np.load(filepath + "/" + filename)

        file_metrics = file_data["metric"].astype(np.float64)
        file_hand_info = file_data["hand"].astype(np.float64)

        metrics.append(file_metrics)
        hand_infos.append(file_hand_info)

    
    # Filtering out the reachable data points 
    for index in range(len(metrics)): 
        curr_metrics = metrics[index]
        curr_hand_info = hand_infos[index]
        for metric_index in range(len(curr_metrics)):
            if curr_metrics[metric_index,0] == 1 and curr_metrics[metric_index,1] > 0:
                if nb_curr_filt_samples == nb_samples: 
                    break

                # Only taking reachable objects with a joint distance > 0: 
                # NOTE: could do: multiplying joint distance by 100 to get cm, it's more readable 
                filtered_metrics.append((curr_metrics[metric_index,1])) # only taking joint distance 
                # filtered_metrics.append(curr_metrics[metric_index,1]*100) # only taking joint distance 
                filtered_hand_infos.append(curr_hand_info[metric_index])

                nb_curr_filt_samples += 1 
                

    # End of time measuring 
    end_time = time.time()

    time_taken = end_time - start_time

    return filtered_metrics, filtered_hand_infos, time_taken

# Data loader function for sequences of fixed length, with no redraw 
# NB: number of sequence data thus immediately determined by nb_samp_class
def load_2_class_sequence_data_no_redraw(class_0_folder, class_1_folder, one_hot, nb_samp_class, seq_len, rand_seed, test_train_ratio, batch_size, verbose): 
    # Loading filtered raw data 
    if verbose >= 1:
        print("Loading class 0.")
    class_0_dist, class_0_hands, time_taken = load_filtered_class_data(folder_name = class_0_folder, nb_samples = nb_samp_class)
    if verbose >= 1: 
        print("Time taken to load class 0: ", time_taken)

    if verbose >= 1:
        print("Loading class 1.")
    class_1_dist, class_1_hands, time_taken = load_filtered_class_data(folder_name = class_1_folder, nb_samples = nb_samp_class)
    if verbose >= 1: 
        print("Time taken to load class 1: ", time_taken)

    # Fusing the informations 
    inputs = [] 
    outputs = [] 

    nb_seq_data_class = int(nb_samp_class / seq_len)


    # Class 0 
    for index in range(nb_seq_data_class): 
        tmp_buffer = [] 
        for seq_index in range(seq_len):
            curr_index = int(index * seq_len + seq_index)
            tmp_list = []
            tmp_list.append(np.float64(class_0_dist[curr_index]))
            for hand_index in range(len(class_0_hands[-1])): 
                tmp_list.append(np.float64(class_0_hands[curr_index][hand_index]))
            tmp_buffer.append(tmp_list.copy())
        
        inputs.append(tmp_buffer.copy())
        if one_hot: 
            outputs.append(tf.one_hot(int(0), 2, dtype=np.float64).numpy().tolist())
        else: 
            outputs.append([0])

    # Class 1 
    for index in range(nb_seq_data_class): 
        tmp_buffer = [] 
        for seq_index in range(seq_len):
            curr_index = int(index * seq_len + seq_index)
            tmp_list = []
            tmp_list.append(np.float64(class_1_dist[curr_index]))
            for hand_index in range(len(class_1_hands[-1])): 
                tmp_list.append(np.float64(class_1_hands[curr_index][hand_index]))
            tmp_buffer.append(tmp_list.copy())
        
        inputs.append(tmp_buffer.copy())
        if one_hot: 
            outputs.append(tf.one_hot(int(1), 2, dtype=np.float64).numpy().tolist())
        else: 
            outputs.append([1])


    # Shuffling the data 
    X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=test_train_ratio, random_state=rand_seed)

    if verbose >= 1:
        print("Size of Train set: ", len(X_train))
        print("Size of Test set: ", len(X_test))

    """
    # Creating the Tensors 
    input_tensor_training = tf.constant(X_train)
    output_tensor_training = tf.constant(y_train)
    input_tensor_test = tf.constant(X_test)
    output_tensor_test = tf.constant(y_test)

    # Creating the data set 
    dataset_training = tf.data.Dataset.from_tensor_slices((input_tensor_training, output_tensor_training))
    dataset_testing = tf.data.Dataset.from_tensor_slices((input_tensor_test, output_tensor_test))

    if verbose >= 1: 
        print("Finished preparing data")

    return dataset_training, dataset_testing
    """

    return X_train, X_test, y_train, y_test

# Data loader function for simple 1 Grasp classification 
def load_2_class_data(class_0_folder, class_1_folder, one_hot, nb_samp_class, rand_seed, test_train_ratio, batch_size, verbose):
    # Loading filtered raw data 
    if verbose >= 1:
        print("Loading class 0.")
    class_0_dist, class_0_hands, time_taken = load_filtered_class_data(folder_name = class_0_folder, nb_samples = nb_samp_class)
    if verbose >= 1: 
        print("Time taken to load class 0: ", time_taken)

    if verbose >= 1:
        print("Loading class 1.")
    class_1_dist, class_1_hands, time_taken = load_filtered_class_data(folder_name = class_1_folder, nb_samples = nb_samp_class)
    if verbose >= 1: 
        print("Time taken to load class 1: ", time_taken)

    # Fusing the informations 
    inputs = [] 
    outputs = [] 
    # Class 0 
    for index in range(len(class_0_dist)): 
        tmp_list = []
        tmp_list.append(np.float64(class_0_dist[index]))
        for hand_index in range(len(class_0_hands[-1])): 
            tmp_list.append(np.float64(class_0_hands[index][hand_index]))
        inputs.append(tmp_list)
        # First class = class 0 
        if one_hot: 
            outputs.append(tf.one_hot(int(0), 2, dtype=np.float64).numpy().tolist())
        else: 
            outputs.append([0])

    # Class 1 
    for index in range(len(class_1_dist)): 
        tmp_list = []
        tmp_list.append(np.float64(class_1_dist[index]))
        for hand_index in range(len(class_1_hands[-1])): 
            tmp_list.append(np.float64(class_1_hands[index][hand_index]))
        inputs.append(tmp_list)
        # Second class = class 1 
        if one_hot: 
            outputs.append(tf.one_hot(int(1), 2, dtype=np.float64).numpy().tolist())
        else: 
            outputs.append([1])
    
    # Shuffling the data 
    X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=test_train_ratio, random_state=rand_seed)

    # Creating the Tensors 
    input_tensor_training = tf.constant(X_train)
    output_tensor_training = tf.constant(y_train)
    input_tensor_test = tf.constant(X_test)
    output_tensor_test = tf.constant(y_test)

    # Creating the data set 
    # dataset_training = tf.data.Dataset.from_tensor_slices((input_tensor_training, output_tensor_training))
    #dataset_testing = tf.data.Dataset.from_tensor_slices((input_tensor_test, output_tensor_test))

    if verbose >= 1: 
        print("Finished preparing data")
    return input_tensor_training, output_tensor_training, input_tensor_test, output_tensor_test


# Function that load FIXED sequence lngth data for classification 
#for both classes at the same time 
def load_fixed_seq_rand(class_0_folder, class_1_folder, one_hot, nb_samp_class, nb_seq_samp_per_class, rand_seed, seq_len, test_train_ratio, batch_size): 
    # Loading filtered raw data 
    class_0_dist, class_0_hands, time_taken = load_filtered_class_data(folder_name = class_0_folder, nb_samples = nb_samp_class)
    class_1_dist, class_1_hands, time_taken = load_filtered_class_data(folder_name = class_1_folder, nb_samples = nb_samp_class)

    # Class 0 

    # Merge data 
    class_0_inputs = [] 
    for index in range(len(class_0_dist)): 
        tmp_list = [] 
        tmp_list.append(np.float64(class_0_dist[index]))
        for hand_index in range(len(class_0_hands[-1])): 
            tmp_list.append(np.float64(class_0_hands[index][hand_index]))
        
        class_0_inputs.append(tmp_list)

    class_0_outputs = [] 
    for index in range(len(class_0_dist)): 
        if one_hot: 
            class_0_outputs.append(tf.one_hot(int(0), 2, dtype=np.float64).numpy().tolist())
        else: 
            class_0_outputs.append([0])

    # Split and shuffle data into train and test split  
    class_0_X_train_pool, class_0_X_test_pool, class_0_y_train, class_0_y_test = train_test_split(class_0_inputs, class_0_outputs, test_size=test_train_ratio, random_state=rand_seed)


    # Now draw from these 2 pools of data. 
    # Constructing sequences of variable length 
    nb_test_data = int(test_train_ratio * nb_seq_samp_per_class)
    nb_train_data = int(nb_seq_samp_per_class - nb_test_data)

    class_0_train_sequences = [] 
    for _ in range(nb_train_data): 
        tmp_list = []

        for _ in range(seq_len):
            # Draw random sample from pool 
            rand_sample = random.randint(0, len(class_0_X_train_pool)-1)

            tmp_list.append(class_0_X_train_pool[rand_sample])
        class_0_train_sequences.append(tmp_list.copy())

    class_0_train_outputs = class_0_y_train[:nb_train_data]

    # test set 
    class_0_test_sequences = [] 
    for _ in range(nb_test_data): 
        tmp_list = []

        for _ in range(seq_len):
            # Draw random sample from pool 
            rand_sample = random.randint(0, len(class_0_X_test_pool)-1)

            tmp_list.append(class_0_X_test_pool[rand_sample])
        class_0_test_sequences.append(tmp_list.copy())

    class_0_test_outputs = class_0_y_test[:nb_test_data]

    # Class 1 
    
    # Merge data 
    class_1_inputs = [] 
    for index in range(len(class_1_dist)): 
        tmp_list = [] 
        tmp_list.append(np.float64(class_1_dist[index]))
        for hand_index in range(len(class_1_hands[-1])): 
            tmp_list.append(np.float64(class_1_hands[index][hand_index]))
        
        class_1_inputs.append(tmp_list)

    class_1_outputs = [] 
    for index in range(len(class_1_dist)): 
        if one_hot: 
            class_1_outputs.append(tf.one_hot(int(1), 2, dtype=np.float64).numpy().tolist())
        else: 
            class_1_outputs.append([1])

    # Split and shuffle data into train and test split  
    class_1_X_train_pool, class_1_X_test_pool, class_1_y_train, class_1_y_test = train_test_split(class_1_inputs, class_1_outputs, test_size=test_train_ratio, random_state=rand_seed)


    # Now draw from these 2 pools of data. 
    # Constructing sequences of variable length 
    nb_test_data = int(test_train_ratio * nb_seq_samp_per_class)
    nb_train_data = int(nb_seq_samp_per_class - nb_test_data)

    class_1_train_sequences = [] 
    for _ in range(nb_train_data): 
        tmp_list = []

        for _ in range(seq_len):
            # Draw random sample from pool 
            rand_sample = random.randint(0, len(class_1_X_train_pool)-1)

            tmp_list.append(class_1_X_train_pool[rand_sample])
        class_1_train_sequences.append(tmp_list.copy())

    class_1_train_outputs = class_1_y_train[:nb_train_data]

    # test set 
    class_1_test_sequences = [] 
    for _ in range(nb_test_data): 
        tmp_list = []

        for _ in range(seq_len):
            # Draw random sample from pool 
            rand_sample = random.randint(0, len(class_1_X_test_pool)-1)

            tmp_list.append(class_1_X_test_pool[rand_sample])
        class_1_test_sequences.append(tmp_list.copy())

    class_1_test_outputs = class_1_y_test[:nb_test_data]


    class_0_train_tensor = tf.constant(class_0_train_sequences)
    class_1_train_tensor = tf.constant(class_1_train_sequences)

    final_train_tensor = tf.concat([class_0_train_tensor, class_1_train_tensor], 0)

    c_0_train_outputs = tf.constant(class_0_train_outputs)
    c_1_train_outputs = tf.constant(class_1_train_outputs)

    final_train_tensor_out = tf.concat([c_0_train_outputs, c_1_train_outputs], 0)

    # print(final_train_tensor.shape)

    class_0_train_tensor = tf.constant(class_0_test_sequences)
    class_1_train_tensor = tf.constant(class_1_test_sequences)

    final_test_tensor = tf.concat([class_0_train_tensor, class_1_train_tensor], 0)

    c_0_test_outputs = tf.constant(class_0_y_test[:nb_test_data])
    c_1_test_outputs = tf.constant(class_1_y_test[:nb_test_data])

    final_test_tensor_out = tf.concat([c_0_test_outputs, c_1_test_outputs], 0)

    # print(ragged_tens_0_test)
    # print(ragged_tens_1_test)
    # print(final_test_tensor)

    # print(final_train_tensor)

    train_data_set = tf.data.Dataset.from_tensor_slices((final_train_tensor,final_train_tensor_out))
    train_data_set = train_data_set.shuffle(buffer_size=batch_size).batch(batch_size)

    # print(train_data_set)

    #  Same with test set 

    test_data_set = tf.data.Dataset.from_tensor_slices((final_test_tensor,final_test_tensor_out))
    test_data_set = test_data_set.shuffle(buffer_size=batch_size).batch(batch_size)

    #print(test_data_set)


    #TODO: RENAME EVERYTHING AND CLEAN IT UP ! 
    return train_data_set, test_data_set





def prep_seq_data(class_0_folder, class_1_folder, one_hot, nb_samp_class, nb_seq_samp, rand_seed, max_seq_len, test_train_ratio, batch_size):
    # Loading filtered raw data 
    class_0_dist, class_0_hands, time_taken = load_filtered_class_data(folder_name = class_0_folder, nb_samples = nb_samp_class)
    class_1_dist, class_1_hands, time_taken = load_filtered_class_data(folder_name = class_1_folder, nb_samples = nb_samp_class)

    # Treating data per class 

    # Class 0 
    
    # Merge data 
    class_0_inputs = [] 
    for index in range(len(class_0_dist)): 
        tmp_list = [] 
        tmp_list.append(np.float64(class_0_dist[index]))
        for hand_index in range(len(class_0_hands[-1])): 
            tmp_list.append(np.float64(class_0_hands[index][hand_index]))
        
        class_0_inputs.append(tmp_list)

    class_0_outputs = [] 
    for index in range(len(class_0_dist)): 
        if one_hot: 
            class_0_outputs.append(tf.one_hot(int(0), 2, dtype=np.float64).numpy().tolist())
        else: 
            class_0_outputs.append([0])

    # Split and shuffle data into train and test split  
    class_0_X_train, class_0_X_test, class_0_y_train, class_0_y_test = train_test_split(class_0_inputs, class_0_outputs, test_size=test_train_ratio, random_state=rand_seed)
    
    # Now draw from these 2 pools of data. 
    # Constructing sequences of variable length 
    nb_test_data = int(test_train_ratio * nb_seq_samp)
    nb_train_data = int(nb_seq_samp - nb_test_data)

    # Constructing random sequence lengths 
    seq_lenghts = [] 
    for index in range(nb_train_data):
        rand_sample = random.randint(1, max_seq_len)
        seq_lenghts.append(rand_sample)
      
    needed_length = sum(seq_lenghts)

    # Draw out of that pool of data enough times ... 
    class_0_train_data_pool = []
    for curr_sample_index in range(needed_length): 

        rand_pick = random.randint(0, len(class_0_X_train)-1)

        class_0_train_data_pool.append(class_0_X_train[rand_pick])

         
    

    ragged_tens_0_train = tf.RaggedTensor.from_row_lengths(
        values=class_0_train_data_pool,
        row_lengths=seq_lenghts)

    # TEST SET 
    # Constructing random sequence lengths 
    seq_lenghts = [] 
    for index in range(nb_test_data):
        rand_sample = random.randint(1, max_seq_len)
        seq_lenghts.append(rand_sample)
      
    needed_length = sum(seq_lenghts)

    # Draw out of that pool of data enough times ... 
    class_0_test_data_pool = []
    for curr_sample_index in range(needed_length): 

        rand_pick = random.randint(0, len(class_0_X_test)-1)

        class_0_test_data_pool.append(class_0_X_test[rand_pick])

         
    

    ragged_tens_0_test = tf.RaggedTensor.from_row_lengths(
        values=class_0_test_data_pool,
        row_lengths=seq_lenghts)

   
    
    # print(ragged_tens_0_train.shape)

    # print(ragged_tens_0_test.shape)


    # Now, Same thing for the other class 

    # Class 1
    
    # Merge data 
    class_1_inputs = [] 
    for index in range(len(class_1_dist)): 
        tmp_list = [] 
        tmp_list.append(np.float64(class_1_dist[index]))
        for hand_index in range(len(class_1_hands[-1])): 
            tmp_list.append(np.float64(class_1_hands[index][hand_index]))
        
        class_1_inputs.append(tmp_list)

    class_1_outputs = [] 
    for index in range(len(class_1_dist)): 
        if one_hot: 
            class_1_outputs.append(tf.one_hot(int(1), 2, dtype=np.float64).numpy().tolist())
        else: 
            class_1_outputs.append([1])

    # Split and shuffle data into train and test split  
    class_1_X_train, class_1_X_test, class_1_y_train, class_1_y_test = train_test_split(class_1_inputs, class_1_outputs, test_size=test_train_ratio, random_state=rand_seed)
    
    # Now draw from these 2 pools of data. 
    # Constructing sequences of variable length 
    nb_test_data = int(test_train_ratio * nb_seq_samp)
    nb_train_data = int(nb_seq_samp - nb_test_data)

    # Constructing random sequence lengths 
    seq_lenghts = [] 
    for index in range(nb_train_data):
        rand_sample = random.randint(1, max_seq_len)
        seq_lenghts.append(rand_sample)
      
    needed_length = sum(seq_lenghts)

    # Draw out of that pool of data enough times ... 
    class_1_train_data_pool = []
    for curr_sample_index in range(needed_length): 

        rand_pick = random.randint(0, len(class_1_X_train)-1)

        class_1_train_data_pool.append(class_1_X_train[rand_pick])

         
    

    ragged_tens_1_train = tf.RaggedTensor.from_row_lengths(
        values=class_1_train_data_pool,
        row_lengths=seq_lenghts)

    # TEST SET 
    # Constructing random sequence lengths 
    seq_lenghts = [] 
    for index in range(nb_test_data):
        rand_sample = random.randint(1, max_seq_len)
        seq_lenghts.append(rand_sample)
      
    needed_length = sum(seq_lenghts)

    # Draw out of that pool of data enough times ... 
    class_1_test_data_pool = []
    for curr_sample_index in range(needed_length): 

        rand_pick = random.randint(0, len(class_1_X_test)-1)

        class_1_test_data_pool.append(class_1_X_test[rand_pick])

         
    

    ragged_tens_1_test = tf.RaggedTensor.from_row_lengths(
        values=class_1_test_data_pool,
        row_lengths=seq_lenghts)

   
    
    # print(ragged_tens_1_train.shape)

    # print(ragged_tens_1_test.shape)
    

    # Now merging both test and train sets 

    final_train_tensor = tf.concat([ragged_tens_0_train, ragged_tens_1_train], 0)

    c_0_train_outputs = tf.constant(class_0_y_train[:nb_train_data])
    c_1_train_outputs = tf.constant(class_1_y_train[:nb_train_data])

    final_train_tensor_out = tf.concat([c_0_train_outputs, c_1_train_outputs], 0)

    # print(final_train_tensor.shape)

    final_test_tensor = tf.concat([ragged_tens_0_test, ragged_tens_1_test], 0)

    c_0_test_outputs = tf.constant(class_0_y_test[:nb_test_data])
    c_1_test_outputs = tf.constant(class_1_y_test[:nb_test_data])

    final_test_tensor_out = tf.concat([c_0_test_outputs, c_1_test_outputs], 0)

    # print(ragged_tens_0_test)
    # print(ragged_tens_1_test)
    # print(final_test_tensor)

    # print(final_train_tensor)

    train_data_set = tf.data.Dataset.from_tensor_slices((final_train_tensor,final_train_tensor_out))
    # Drop remainder is to have only full batches (easier to go over) 
    train_data_set = train_data_set.shuffle(buffer_size=batch_size).batch(batch_size, drop_remainder=True) 

    print(train_data_set)

    #  Same with test set 

    test_data_set = tf.data.Dataset.from_tensor_slices((final_test_tensor,final_test_tensor_out))
    test_data_set = test_data_set.shuffle(buffer_size=batch_size).batch(batch_size, drop_remainder=True)

    print(test_data_set)


    #TODO: RENAME EVERYTHING AND CLEAN IT UP ! 
    return train_data_set, test_data_set


class DeepSets(keras.Model): 
    def __init__(self, Phi, Rho, seq_len):
        super(DeepSets, self).__init__()
        self.Phi = Phi 
        self.Rho = Rho 
        self.seq_len = seq_len 

    def compile(self, phi_optimizer, rho_optimizer, loss_fn):
        super(DeepSets, self).compile()
        self.phi_optimizer = phi_optimizer
        self.rho_optimizer = rho_optimizer
        self.loss_fn = loss_fn
    
    def train_step(self, data): 
        X, Y = data 

        # Run through first network 
        with tf.GradientTape(persistent=True) as tape:
            predictions_list = [] 
            for index in range(self.seq_len): 
                index_pred = self.Phi(X[:,index], training=True)
                predictions_list.append(index_pred) 
            sums =  layers.add(predictions_list)
            class_predictions = self.Rho(sums, training=True)
            loss = self.loss_fn(Y, class_predictions)
            

        # For both networks 
        trainable_vars_phi  = self.Phi.trainable_weights 
        trainable_vars_rho = self.Rho.trainable_weights

        gradients_Phi = tape.gradient(loss, trainable_vars_phi)
        gradients_Rho = tape.gradient(loss, trainable_vars_rho)
        # Update weights
        self.phi_optimizer.apply_gradients(zip(gradients_Phi, trainable_vars_phi))
        self.rho_optimizer.apply_gradients(zip(gradients_Rho, trainable_vars_rho))

        # Update metrics (includes the metric that tracks the loss)
        # self.compiled_metrics.update_state(Y, class_predictions)
        # Return a dict mapping metric names to current value
        return {"loss": loss}
    
    def call(self, inputs):
        predictions_list = [] 
        for index in range(self.seq_len): 
            index_pred = self.Phi(inputs[:,index])
            predictions_list.append(index_pred) 
        sums = layers.add(predictions_list)
        class_predictions = self.Rho(sums)
        return class_predictions

    def getModels(self): 
        return self.Phi, self.Rho



if __name__ == "__main__":
    # Main to test the functions 

    # The line below activates eager execution !!
    # tf.config.run_functions_eagerly(True)


    nb_samples = 500000
    """
    filtered_metrics_big_box, filtered_hand_infos_big_box, time_taken = load_filtered_class_data(folder_name = "DATA_SET_CYLINDRE", nb_samples = nb_samples)

    print("Number of samples: ", len(filtered_metrics_big_box))

    print("Time taken: ", (time_taken))
    """
    input_size = 14
    nb_epochs = 100 
    seq_len = 5

    
    train_data_set, test_data_set = prep_seq_data(class_0_folder = "DATA_SET_SMALL_CUBE", 
                    class_1_folder = "DATA_SET_BIG_CUBE", 
                    one_hot = True, 
                    nb_samp_class = 100, 
                    nb_seq_samp = 5, 
                    rand_seed = 42, 
                    max_seq_len = 5,
                    test_train_ratio = 0.2,
                    batch_size = 5)
    

    """
    train_data_set, test_data_set = load_fixed_seq_rand(class_0_folder = "DATA_SET_SMALL_CUBE",
                    class_1_folder = "DATA_SET_BIG_CUBE", 
                    one_hot = True , 
                    nb_samp_class = 5000, 
                    nb_seq_samp_per_class = 3000, 
                    rand_seed = 42, 
                    seq_len = 4, 
                    test_train_ratio = 0.2, 
                    batch_size = 100)

    """

    
    """
    X_train, X_test, y_train, y_test = load_rand_seq_data_rand_redraw(class_0_folder_name = "DATA_SET_SMALL_BOX", class_1_folder_name = "DATA_SET_BIG_BOX", max_seq_len = 5, nb_raw_samples = 200000, nb_seq_samples = 100000, one_hot_encoding = False, shuffling_seed = 42, test_train_ratio = 0.2, verbose = 2)
    print("Size of training set: ", len(X_train))
    """ 

    

    Phi_network = keras.Sequential(
        [
            keras.Input(shape=(input_size,)), 
            layers.Dense(64, activation='relu'), 
            layers.Dense(64, activation='relu'), 
            layers.Dense(64, activation='relu')
        ], 
        name = "Phi"
    )
    


    Rho_network = keras.Sequential(
        [
            keras.Input(shape=(64,)), 
            layers.Dense(64, activation='relu'), 
            layers.Dense(64, activation='relu'), 
            layers.Dense(2, activation=tf.nn.softmax)
        ], 
        name = "Rho"
    )

    verbose = 2 

    # A first save 
    # Phi_network.save(folder_saving_str + "/Model_PHI_" +str(nb_epochs)+ "_epochs_deep_sets_nb_" + str(training_run_nb))
    # Rho_network.save(folder_saving_str + "/Model_RHO_" +str(nb_epochs)+ "_epochs_deep_sets_nb_" + str(training_run_nb))

    model = DeepSets(Phi_network, Rho_network, seq_len=seq_len)


    loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)
    phi_optimizer = tf.keras.optimizers.Adam()        
    rho_optimizer = tf.keras.optimizers.Adam()

    # Preparing the metrics
    train_acc_metric = keras.metrics.CategoricalAccuracy()
    val_acc_metric = keras.metrics.CategoricalAccuracy()
    train_loss_metric = keras.metrics.CategoricalCrossentropy(from_logits=True)
    val_loss_metric = keras.metrics.CategoricalCrossentropy(from_logits=True)


    @tf.function
    def train_step(x, y):
        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.
        with tf.GradientTape(persistent=True) as tape:
            predictions_list = [] 

            #print(x.shape)
            # x_as_arr = x.numpy()
            # exit()
            print(x[0,0])
            exit()

            for index in range(seq_len): 
                # index_pred = Phi_network(x[:,index], training=True)
                #input_tens = tf.concat(x_as_arr[:][index], axis=0 )
                index_pred = Phi_network(x[:][index], training=True)
               
                predictions_list.append(index_pred) 
            sums =  layers.add(predictions_list)
            # sums =  layers.Average()(predictions_list) #average instead of sum 
            logits = Rho_network(sums, training=True)
            loss_value = loss_fn(y, logits)
            

        # For both networks 
        trainable_vars_phi  = Phi_network.trainable_weights 
        trainable_vars_rho = Rho_network.trainable_weights

        gradients_Phi = tape.gradient(loss_value, trainable_vars_phi)
        gradients_Rho = tape.gradient(loss_value, trainable_vars_rho)
        # Update weights
        phi_optimizer.apply_gradients(zip(gradients_Phi, trainable_vars_phi))
        rho_optimizer.apply_gradients(zip(gradients_Rho, trainable_vars_rho))

        # Update training metric.
        train_acc_metric.update_state(y, logits)
        train_loss_metric.update_state(y, logits)

        return loss_value
    
    @tf.function
    def test_step(x, y):
        predictions_list = [] 
        for index in range(seq_len): 
            index_pred = Phi_network(x[:,index], training=False)
            predictions_list.append(index_pred) 
        sums =  layers.add(predictions_list)
        # sums =  layers.Average()(predictions_list) #average instead of sum 
        val_logits = Rho_network(sums, training=False)
        # Update val metrics
        val_acc_metric.update_state(y, val_logits)
        val_loss_metric.update_state(y, val_logits)

    # Training loop 
    best_val_acc = 0.5 # best val acc 
    for epoch in range(nb_epochs):
        if verbose >= 1:  
            print("Epoch nb: ", epoch)
        start_time = time.time()
        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_data_set):
            loss_value = train_step(x_batch_train, y_batch_train)
        
        # Display metrics at the end of each epoch.
        train_acc = train_acc_metric.result()
        if verbose >= 1: 
            print("Training acc over epoch: %.4f" % (float(train_acc),))
        train_loss = train_loss_metric.result()
        if verbose >= 1: 
            print("Training loss over epoch: %.4f" % (float(train_loss),))

        # Run a validation loop at the end of each epoch.
        for x_batch_val, y_batch_val in test_data_set:
            val_loss = test_step(x_batch_val, y_batch_val)

        val_acc = val_acc_metric.result()
        if val_acc > best_val_acc and epoch > 0: 
            best_val_acc = val_acc
            # Saving the best model so far NOTE: strictly better so as to avoid unnecessary model saving
            # Saving the models 
            

            if verbose >= 1: 
                print("New best model saved at epoch: ", epoch)
                print("New best accuracy on validation set: ", best_val_acc)
            

        val_loss = val_loss_metric.result()
        if verbose >= 1: 
            print("Validation acc: %.4f" % (float(val_acc),))
            print("Validation loss: %.4f" % (float(val_loss),))
            print("Time taken: %.2fs" % (time.time() - start_time))


        # Reset training metrics at the end of each epoch
        train_acc_metric.reset_states()
        train_loss_metric.reset_states()
        val_acc_metric.reset_states()
        val_loss_metric.reset_states()
