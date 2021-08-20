""" 
Deep Set learning: 
archtiecture is also saved as a plot 
custom architecture 

(https://papers.nips.cc/paper/2017/file/f22e4747da1aa27e363d86d40ff442fe-Paper.pdf)

custom learning loop, in order to compute all relevant metrics 

SCRIPT FOR THE YCB DATA SET 

"""
import time 

from os import path

import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import xlsxwriter # csv log for losses and accuracies 

from sklearn.model_selection import train_test_split

import sys
sys.path.append("../..")
from DexterousManipulation.training.data_loader_YCB import load_total_YCB_eq_data_seq, load_norm_data, load_YCB_grasp_dataset, load_YCB_dataset_grasping, load_non_zero_seq, load_non_zero_grasps

if __name__ == "__main__":
    tf.keras.backend.set_floatx('float64') # Just setting the float default to 64 in keras 

    # Hyper-parameters 
    nb_epochs = 1000
    batch_size = 25

    test_train_ratio = 0.2


    # Starting at 20
    training_run_nb = 96

    shuffling_seed = 42

    apply_one_hot_encoding = True 

    basepath = path.dirname(__file__)
    data_set_folder = path.abspath(path.join(basepath, "..", "generations/DATA_SET_YCB_filtered"))
    
    seq_len = 5

    nb_tot_samp_class = 10000

    verbose = 2 
    nb_classes = 16
    nb_tot_samples = 4500000 # NB irrelevant for non-zero grasp loading ... 
    joint_dist_lim = 0.005

    # Main folder where the training data is stored
    folder_saving_str = path.abspath(path.join(basepath, "..", "Figures/Training/custom_YCB_Data_Set/seq_len_" + str(seq_len)))


    # Creating the logfile of the training session 
    training_log = open(folder_saving_str + "/Training_log_" +str(nb_epochs)+ "_epochs__session_" +str(training_run_nb)+ ".txt", "w+")
    
    input_size = 14
    final_output_size = nb_classes

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

    # Loading the data 
    # X_train, X_test, y_train, y_test = load_total_YCB_eq_data_seq(data_set_folder, nb_tot_samples, nb_classes, joint_dist_lim, seq_len, nb_tot_samp_class, test_train_ratio, shuffling_seed, verbose)
    
    # Loading the already preprocessed data: 
    prefix_str = path.abspath(path.join(basepath, "..", "generations/Preprocessed_YCB"))
    file_nb_int = int(nb_tot_samp_class/1000)
    # data_load_str_train = prefix_str + "/" + str(file_nb_int) + "K_per_class/seq_len_" + str(seq_len) + "/YCB_train.npz"
    # data_load_str_test = prefix_str + "/" + str(file_nb_int) + "K_per_class/seq_len_" + str(seq_len) + "/YCB_test.npz"

    # Loading non-zero only data 
    data_load_str_train = prefix_str + "/non_zero_only/seq_len_" + str(seq_len) + "/YCB_train.npz"
    data_load_str_test = prefix_str + "/non_zero_only/seq_len_" + str(seq_len) + "/YCB_test.npz"

    file_data_train = np.load(data_load_str_train)
    X_train = file_data_train["inputs"].astype(np.float64)
    y_train = file_data_train["outputs"].astype(np.float64)

    file_data_test = np.load(data_load_str_test)
    X_test = file_data_test["inputs"].astype(np.float64)
    y_test = file_data_test["outputs"].astype(np.float64)

    input_training_tensor = tf.constant(X_train)
    
    output_training_tensor = tf.constant(y_train)

    train_data_set = tf.data.Dataset.from_tensor_slices((input_training_tensor,output_training_tensor))

    train_data_set = train_data_set.shuffle(buffer_size=1024).batch(batch_size)


    input_test_tensor = tf.constant(X_test)

    output_test_tensor = tf.constant(y_test)

    test_data_set = tf.data.Dataset.from_tensor_slices((input_test_tensor,output_test_tensor))
    test_data_set = test_data_set.batch(batch_size)

    print("Finished preparing data...")

    # A first save 
    # Phi_network.save(folder_saving_str + "/Model_PHI_" +str(nb_epochs)+ "_epochs_deep_sets_nb_" + str(training_run_nb))
    # Rho_network.save(folder_saving_str + "/Model_RHO_" +str(nb_epochs)+ "_epochs_deep_sets_nb_" + str(training_run_nb))

    # model = DeepSets(Phi_network, Rho_network, seq_len=seq_len)

    model_PHI = keras.Sequential(
        [
            keras.layers.InputLayer(input_shape=(input_size,)), 
            layers.Dense(64, activation='relu'), 
            # layers.Dense(32, activation='relu'), 
            layers.Dense(input_size, activation='relu')
        ], 
        name = "Phi"
    )

    model_PHI.compile(optimizer=tf.keras.optimizers.Adam(), loss=keras.losses.CategoricalCrossentropy(from_logits=True))

    model_RHO = keras.Sequential(
        [
            keras.layers.InputLayer(input_shape=(input_size,)), 
            layers.Dense(64, activation='relu'), 
            # layers.Dense(32, activation='relu'), 
            layers.Dense(final_output_size, activation=tf.nn.softmax)
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


    @tf.function
    def train_step(x, y):
        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.
        with tf.GradientTape(persistent=True) as tape:
            predictions_list = [] 

            for index in range(seq_len): 
                index_pred = model_PHI(x[:,index], training=True)
               
                predictions_list.append(index_pred) 
            sums =  layers.add(predictions_list)
            # sums =  layers.Average()(predictions_list) #average instead of sum 
            logits = model_RHO(sums, training=True)
            loss_value = loss_fn(y, logits)            

        # For both networks 
        trainable_vars_phi  = model_PHI.trainable_weights 
        trainable_vars_rho = model_RHO.trainable_weights

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
            index_pred = model_PHI(x[:,index], training=False)
            predictions_list.append(index_pred) 
        sums =  layers.add(predictions_list)
        # sums =  layers.Average()(predictions_list) #average instead of sum 
        val_logits = model_RHO(sums, training=False)
        # Update val metrics
        val_acc_metric.update_state(y, val_logits)
        val_loss_metric.update_state(y, val_logits)

    # Training loop 
    best_val_acc = 0.0001 # best val acc 
    for epoch in range(nb_epochs):
        if verbose >= 1:  
            print("Epoch nb: ", epoch)
            doc_string = "Epoch nb: " + str(epoch)
            training_log.write(doc_string + "\n")
        start_time = time.time()
        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_data_set):
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


    # Closing log 
    Log_data.close()

    # Closing log file 
    training_log.close()