"""
Training Deep Sets on variable sequence lengths 

Author: Philippe Schneider 
"""

# Importing packages 

import time 

from os import path

import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import random
from tensorflow.python.ops.gen_batch_ops import batch

import xlsxwriter # csv log for losses and accuracies 

from sklearn.model_selection import train_test_split

import sys
sys.path.append("../..")
from DexterousManipulation.training.data_loader import load_filtered_sequences, load_data_two_classes_seq
from DexterousManipulation.training.custom_data_loader import load_rand_seq_data_rand_redraw, load_fixed_seq_rand, load_2_class_sequence_data_no_redraw, prep_seq_data
from DexterousManipulation.training.custom_data_loader import load_filtered_class_data

# NOTE: This can be sratched later (I think)
def get_batch_padding_info(x_batch):

    batch_list = [] 

    curr_batch_size = x_batch.shape[0]

    # Going over each sequence one by one 
    for index in range(curr_batch_size):

        curr_sequence = x_batch[index]
        # print(curr_sequence)
        
        
        # print(x[index].shape[0])
        # Detecting 0 tensors (padding)
        max_index = int(x_batch[index].shape[0])

        # Counting the non-padded grasps in sequence 
        counter = 0 
        for sequence_index in range(max_index): 
            tmp = tf.math.count_nonzero(x_batch[index, sequence_index], keepdims=True)
            # print(tmp[0])
            # print(tmp)
            # test = tmp.numpy()
            # print(test)

            
            total = tf.reduce_sum(tmp)
            is_all_zero = tf.equal(total, 0)
            # exit()
            
            
            # if tmp == tf.zeros(tmp.shape, tf.int64):
            if is_all_zero: 
                # if test == 0: 
                # Detected zero-tensor 
                # print("Zero tensor fouund!")
                pass 
                # Skip the rest of the sequence 
                break 
            else: 
                counter += 1 
                pass 
        
        
        batch_list.append(counter)
        
    
    # print(batch_list)
    return  batch_list


# NOTE: The below function might better be put into the custom data loader 
"""
Function that loads sequences of variable length from the data 

- Arguments: 
    - class_0_folder: folder containing the data for class 0 
    - class_1_folder: folder containing the data for class 1 
    - one_hot: boolean indicating whether the object classes are encoded in one-hot vectors or not
    - nb_samp_class: number of samples loaded per class 
    - nb_total_samples: number of total sequence samples generated (and returned) by the function 
    - seq_len: the maximum length a sequence can have 
    - rand_seed: TBD
    - test_train_ratio: TBD
    - batch_size: TBD
    - verbose: used to print more information during function execution (>=1)

- Returns: 
    - inputs: list containing the sequences of grasps variable length 
    - outputs: list containing the corresponding class 
"""
def load_var_seq_padded(class_0_folder, class_1_folder, one_hot, nb_samp_class, nb_total_samples, seq_len, rand_seed, test_train_ratio, batch_size, verbose): 
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

    # Will contain final data to be returned 
    inputs = [] 
    outputs = []

    # Draw samples 
    for _ in range(nb_total_samples):
        curr_sample = [] 

        # Draw class at random 
        curr_class = random.randint(0,1)

        # Draw random sequence length 
        curr_seq_len = random.randint(1, seq_len)

        # First element of the sequence contains the length (needed in the train_step function)
        seq_len_info = np.zeros((14,), dtype=np.float64) 
        seq_len_info[0] = curr_seq_len
        curr_sample.append(seq_len_info.copy())

        for _ in range(curr_seq_len): 
            # Filling the sequence with grasp data 
            if curr_class == 0: 
                # Class 0 
                # Drawing from the data at random 
                rand_class_index = random.randint(0,len(class_0_dist)-1)

                tmp_list = [] 
                tmp_list.append(np.float64(class_0_dist[rand_class_index]))
                for hand_index in range(len(class_0_hands[-1])): 
                    tmp_list.append(np.float64(class_0_hands[rand_class_index][hand_index]))
                curr_sample.append(tmp_list)
            
            else: 
                # Class 1 
                # Drawing from the data at random 
                rand_class_index = random.randint(0,len(class_1_dist)-1)

                tmp_list = [] 
                tmp_list.append(np.float64(class_1_dist[rand_class_index]))
                for hand_index in range(len(class_1_hands[-1])): 
                    tmp_list.append(np.float64(class_1_hands[rand_class_index][hand_index]))
                curr_sample.append(tmp_list)

        # Padding the rest of the sample to be able to convert the data to rectanguler Tensors later 
        padding_amount = seq_len - curr_seq_len
        z = np.zeros((14,), dtype=np.float64) 
        padding_list = z.tolist() 
        for _ in range(padding_amount): 
            curr_sample.append(padding_list.copy())

        # Adding the current class to outputs 
        if curr_class == 0: 
            # Class 0 
            if one_hot: 
                outputs.append(tf.one_hot(int(0), 2, dtype=np.float64).numpy().tolist())
            else: 
                outputs.append([0])
        else: 
            # Class 1  
            if one_hot: 
                outputs.append(tf.one_hot(int(1), 2, dtype=np.float64).numpy().tolist())
            else: 
                outputs.append([1])

        # Finaly, adding the current sample to inputs 
        inputs.append(curr_sample.copy())
        curr_sample.clear()

    # Returing the data according to the function description 
    return inputs, outputs


if __name__ == "__main__":

    tf.keras.backend.set_floatx('float64') # Just setting the float default to 64 in keras 


    # Hyper-parameters 
    nb_epochs = 20
    # batch_size = 100
    batch_size = 50

    input_size = 14 

    test_train_ratio = 0.2

    training_run_nb = 71 

    nb_data_files = 1000
    # nb_data_files = 500
    # nb_raw_samples = 20
    shuffling_seed = 42
    # nb_seq_samples = 10
    apply_one_hot_encoding = True 
    # data_set_folder = "Grasping_data_Set"
    data_set_folder = "Grasping_similar_boxes" # Date set for 2 similar boxes 
    file_prefix = "classification_data_grasping_" 
    seq_len = 5

    verbose = 2 

    nb_sample_per_class = 500000 # 500000
    nb_total_samples = 200000

    folder_class_0 = "DATA_SET_BIG_CUBE"


    # folder_class_1 = "DATA_SET_SMALL_CUBE"
    folder_class_1 = "DATA_SET_MEDIUM_CUBE"

    nb_seq_samp = int(2*nb_sample_per_class/seq_len)

    """
    train_data_set =  var_seq_fixed_batch_data(class_0_folder = folder_class_0, 
                                    class_1_folder = folder_class_1, 
                                    one_hot = apply_one_hot_encoding, 
                                    nb_samp_class = nb_sample_per_class, 
                                    nb_samples_per_seq_len = batch_size, 
                                    seq_len = seq_len, 
                                    rand_seed = shuffling_seed, 
                                    test_train_ratio = test_train_ratio,
                                    batch_size = batch_size, 
                      
                                    verbose = verbose)
    """

    # Main folder where the training data is stored
    basepath = path.dirname(__file__)
    # folder_saving_str = path.abspath(path.join(basepath, "..", "Figures/Training/2_Similar_Boxes_Custom_Loop_Tensors_Custom_Deep_Sets"))
    # folder_saving_str = path.abspath(path.join(basepath, "..", "Figures/Training/2_Similar_Boxes_Deep_Sets"))
    # folder_saving_str = path.abspath(path.join(basepath, "..", "Figures/Training/Final_Data_Set/Similar_Boxes_Classification_" + str(seq_len) + "_Grasps"))
    # folder_saving_str = path.abspath(path.join(basepath, "..", "Figures/Training/Final_Data_Set/Different_Boxes_Classification_" + str(seq_len) + "_Grasps"))
    # folder_saving_str = path.abspath(path.join(basepath, "..", "Figures/Training/Final_Data_Set/Box_and_Cyli_" + str(seq_len) + "_Grasps"))

    # folder_saving_str = path.abspath(path.join(basepath, "..", "Figures/Training/500k_Data_Set/Big_Cube_and_Cylinder/seq_len_" + str(seq_len) + "_no_redraw"))
    # folder_saving_str = path.abspath(path.join(basepath, "..", "Figures/Training/500k_Data_Set/Big_Cube_and_Small_Cube/seq_len_" + str(seq_len) + "_no_redraw"))
    folder_saving_str = path.abspath(path.join(basepath, "..", "Figures/Training/500k_Data_Set/Big_Cube_and_Medium_Cube/var_seq_len_max_" + str(5)))



    # Creating the logfile of the training session 
    training_log = open(folder_saving_str + "/Training_log_" +str(nb_epochs)+ "_epochs__session_" +str(training_run_nb)+ ".txt", "w+")

    # Testing on variable sequence length 
    # folder_saving_str = path.abspath(path.join(basepath, "..", "Figures/Training/500k_Data_Set/Variable_Seq_Len_MAX_" + str(seq_len)))

    
    input_size = 14 
    final_output_size = 2 

    verbose = 1 

    # csv log 
    Log_data = xlsxwriter.Workbook(folder_saving_str + "/Log_loss_acc_training_" + str(training_run_nb) + "_nb_epochs_" + str(nb_epochs) + ".xlsx") 
    log_sheet = Log_data.add_worksheet()
    log_sheet.set_column(0, 7, 25) # Column width for better readability 
    # Column names 
    log_sheet.write(0, 0, "Epoch number")
    log_sheet.write(0, 1, "Training loss")
    log_sheet.write(0, 2, "Validation loss")
    log_sheet.write(0, 3, "Training accuracy")
    log_sheet.write(0, 4, "Validation accuracy")

    inputs, outputs = load_var_seq_padded(class_0_folder = folder_class_0,
                                        class_1_folder = folder_class_1, 
                                        one_hot = apply_one_hot_encoding,
                                        nb_samp_class = nb_sample_per_class, 
                                        nb_total_samples = nb_total_samples, 
                                        seq_len = seq_len, 
                                        rand_seed = shuffling_seed, 
                                        test_train_ratio = test_train_ratio, 
                                        batch_size = batch_size, 
                                        verbose = verbose)
    
    X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=test_train_ratio, random_state=shuffling_seed)

    input_training_tensor = tf.constant(X_train)
    
    output_training_tensor = tf.constant(y_train)

    train_data_set = tf.data.Dataset.from_tensor_slices((input_training_tensor,output_training_tensor))

    train_data_set = train_data_set.shuffle(buffer_size=batch_size).batch(batch_size)

    print(train_data_set)


    input_test_tensor = tf.constant(X_test)

    output_test_tensor = tf.constant(y_test)

    test_data_set = tf.data.Dataset.from_tensor_slices((input_test_tensor,output_test_tensor))
    test_data_set = test_data_set.shuffle(buffer_size=batch_size).batch(batch_size)

    # NOTE: The below architectures for PHI and RHO were empirically determined to be the most stable ones. 
    model_PHI = keras.Sequential(
        [
            keras.layers.InputLayer(input_shape=(input_size,)), 
            layers.Dense(64, activation='relu'), 
            layers.Dense(32, activation='relu'), 
            # layers.Dense(input_size, activation='relu')
            layers.Dense(input_size, activation=tf.nn.softmax)
        ], 
        name = "Phi"
    )

    model_PHI.compile(optimizer=tf.keras.optimizers.Adam(), loss=keras.losses.CategoricalCrossentropy(from_logits=True))

    model_RHO = keras.Sequential(
        [
            keras.layers.InputLayer(input_shape=(input_size,)), 
            layers.Dense(64, activation='relu'), 
            layers.Dense(32, activation='relu'), 
            layers.Dense(2, activation=tf.nn.softmax)
        ], 
        name = "Rho"
    )
    model_RHO.compile(optimizer=tf.keras.optimizers.Adam(), loss=keras.losses.CategoricalCrossentropy(from_logits=True))

    loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)

    phi_optimizer = tf.keras.optimizers.Adam()        
    rho_optimizer = tf.keras.optimizers.Adam()

    # Preparing the metrics
    train_acc_metric = keras.metrics.CategoricalAccuracy()
    val_acc_metric = keras.metrics.CategoricalAccuracy()
    train_loss_metric = keras.metrics.CategoricalCrossentropy(from_logits=True)
    val_loss_metric = keras.metrics.CategoricalCrossentropy(from_logits=True)

    # tf.config.run_functions_eagerly(True)

    # Custom implementation of the train_step function for TensorFlow 
    @tf.function
    def train_step(x, y):
        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.
        with tf.GradientTape(persistent=True) as tape:
            # Fetching current batch size (last  batch can be of different size)
            curr_batch_size = x.shape[0]

            # First iteration before the loop to create the starting tensor (the rest will be concatenated to it) 
            curr_sequence = x[0]
            
            # The first list in the sequence is not a grasp but contains the length of said sequence (the rest is padding)
            seq_info = int(x[0,0,0])
            
            # Contains all element of the sequences that are not padded (non-zero)
            non_zero_seq = curr_sequence[2:(seq_info+1), :]

            # Passing these sequences through the PHI model 
            phi_pass = model_PHI(non_zero_seq, training=True)

            # Computing the sum of these outputs (as described in the DeepSet procedure)
            tmp_tens = tf.reduce_sum(phi_pass, 0)
            # Reshaping the sum, in order to be compatible with the next step 
            sums = tf.reshape(tmp_tens, [1,14])

            # Going over each sequence one by one (after the first one)
            for index in range(curr_batch_size):
                if index != 0: 
                    # Skip first it 
                    curr_sequence = x[index]
                    seq_info = int(x[index,0,0])    
                    
                    non_zero_seq = curr_sequence[2:(seq_info+1), :]

                    phi_pass = model_PHI(non_zero_seq, training=True)
                    
                    tmp_sum = tf.reduce_sum(phi_pass, 0)
                    tmp_tens = tf.reshape(tmp_sum, [1,14])
                    sums = tf.concat([sums, tmp_tens], 0)

    
            # Passing the sums to the rho model
            logits = model_RHO(sums, training=True)

            # Computing the loss using the defined loss function 
            loss_value = loss_fn(y, logits)
                

        # For both networks: Fetching trainable parameters 
        trainable_vars_phi  = model_PHI.trainable_weights 
        trainable_vars_rho = model_RHO.trainable_weights


        gradients_Phi = tape.gradient(loss_value, trainable_vars_phi)
        gradients_Rho = tape.gradient(loss_value, trainable_vars_rho)
        # Update weights
        phi_optimizer.apply_gradients(zip(gradients_Phi, trainable_vars_phi))
        rho_optimizer.apply_gradients(zip(gradients_Rho, trainable_vars_rho))

        # Update training metric (for the log)
        train_acc_metric.update_state(y, logits)
        train_loss_metric.update_state(y, logits)

        return loss_value
    
    # Custom implementation of the test_step function for TensorFlow 
    @tf.function
    def test_step(x, y):

        curr_batch_size = x.shape[0]

        curr_sequence = x[0]
        
        seq_info = int(x[0,0,0])
        
        non_zero_seq = curr_sequence[2:(seq_info+1), :]

        phi_pass = model_PHI(non_zero_seq, training=False)

        tmp_tens = tf.reduce_sum(phi_pass, 0)

        sums = tf.reshape(tmp_tens, [1,14])

        # Going over each sequence one by one 
        for index in range(curr_batch_size):
            if index != 0: 
                # Skip first it 
                curr_sequence = x[index]
                seq_info = int(x[index,0,0])
                
                
                # non_zero_seq = tf.slice(x, [index, 0, 0], [index, counter, 0])
                non_zero_seq = curr_sequence[2:(seq_info+1), :]

                phi_pass = model_PHI(non_zero_seq, training=False)
                
                tmp_sum = tf.reduce_sum(phi_pass, 0)
                tmp_tens = tf.reshape(tmp_sum, [1,14])
                sums = tf.concat([sums, tmp_tens], 0)


        # Passing the sums to the rho model
        val_logits = model_RHO(sums, training=False)
        
        val_acc_metric.update_state(y, val_logits)
        val_loss_metric.update_state(y, val_logits)


    # Training loop 
    best_val_acc = 0.5 # best validation accuracy should be better than a random guess 
    for epoch in range(nb_epochs):
        if verbose >= 1:  
            print("Epoch nb: ", epoch)
            doc_string = "Epoch nb: " + str(epoch)
            training_log.write(doc_string + "\n")
        start_time = time.time()
        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_data_set):
            # list_info = get_batch_padding_info(x_batch_train)
            # tensor_info = tf.constant(list_info)
            loss_value = train_step(x_batch_train, y_batch_train)
        
        # Display metrics at the end of each epoch.
        train_acc = train_acc_metric.result()
        if verbose >= 1: 
            print("Training acc over epoch: %.4f" % (float(train_acc),))
            doc_string = "Training acc over epoch: " + str(train_acc)
            training_log.write(doc_string + "\n")
        train_loss = train_loss_metric.result()
        if verbose >= 1: 
            print("Training loss over epoch: %.4f" % (float(train_loss),))
            doc_string = "Training loss over epoch: "  + str(train_loss)
            training_log.write(doc_string + "\n")

        # Run a validation loop at the end of each epoch.
        for x_batch_val, y_batch_val in test_data_set:
            val_loss = test_step(x_batch_val, y_batch_val)

        val_acc = val_acc_metric.result()
        if val_acc > best_val_acc and epoch > 0: 
            best_val_acc = val_acc
            # Saving the best model so far NOTE: strictly better so as to avoid unnecessary model saving
            # Saving the models 
            model_PHI.save(folder_saving_str + "/Model_PHI_" +str(nb_epochs)+ "_epochs_deep_sets_nb_" + str(training_run_nb))
            model_RHO.save(folder_saving_str + "/Model_RHO_" +str(nb_epochs)+ "_epochs_deep_sets_nb_" + str(training_run_nb))

            if verbose >= 1: 
                print("New best model saved at epoch: ", epoch)
                doc_string = "New best model saved at epoch: " + str(epoch)
                training_log.write(doc_string + "\n")
                print("New best accuracy on validation set: ", best_val_acc)
                doc_string = "New best accuracy on validation set: " + str(best_val_acc)
                training_log.write(doc_string + "\n")
            

        val_loss = val_loss_metric.result()
        if verbose >= 1: 
            print("Validation acc: %.4f" % (float(val_acc),))
            doc_string = "Validation acc: " + str(val_acc)
            training_log.write(doc_string + "\n")
            print("Validation loss: %.4f" % (float(val_loss),))
            doc_string = "Validation loss: " + str(val_loss)
            training_log.write(doc_string + "\n")
            print("Time taken: %.2fs" % (time.time() - start_time))
            time_diff = time.time() - start_time
            doc_string = "Time taken: "  + str(time_diff)
            training_log.write(doc_string + "\n")

        # Writing in the log 
        log_sheet.write(epoch + 1, 0, str(epoch))
        log_sheet.write(epoch + 1, 1, str(train_loss.numpy()))
        log_sheet.write(epoch + 1, 2, str(val_loss.numpy()))
        log_sheet.write(epoch + 1, 3, str(train_acc.numpy()))
        log_sheet.write(epoch + 1, 4, str(val_acc.numpy()))

        # Reset training metrics at the end of each epoch
        train_acc_metric.reset_states()
        train_loss_metric.reset_states()
        val_acc_metric.reset_states()
        val_loss_metric.reset_states()

    # Done training 

    # Closing logs
    Log_data.close()
    training_log.close()
