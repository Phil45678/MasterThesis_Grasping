""" 
Deep Set learning: 
archtiecture is also saved as a plot 
custom architecture 

(https://papers.nips.cc/paper/2017/file/f22e4747da1aa27e363d86d40ff442fe-Paper.pdf)

custom learning loop, in order to compute all relevant metrics 

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
from DexterousManipulation.training.data_loader import load_filtered_sequences, load_data_two_classes_seq
from DexterousManipulation.training.custom_data_loader import load_rand_seq_data_rand_redraw, load_fixed_seq_rand, load_2_class_sequence_data_no_redraw


"""
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
"""            

if __name__ == "__main__":
    tf.keras.backend.set_floatx('float64') # Just setting the float default to 64 in keras 

    # Hyper-parameters 
    nb_epochs = 10
    # batch_size = 100
    batch_size = 50

    test_train_ratio = 0.2

    training_run_nb = 1

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

    nb_sample_per_class = 50 # 500000

    # nb_seq_samples_per_class = 50000

    # folder_class_0 = "DATA_BIG_BOX"
    # folder_class_1 = "DATA_SMALL_BOX"

    folder_class_0 = "DATA_SET_BIG_CUBE"

    # folder_class_1 = "DATA_SET_SMALL_BOX"

    # folder_class_1 = "DATA_SET_CYLINDER"
    folder_class_1 = "DATA_SET_SMALL_CUBE"
    # folder_class_1 = "DATA_SET_MEDIUM_CUBE"


    # Data loader for 500K sets 
    """
    train_data_set, test_data_set = load_fixed_seq_rand(class_0_folder = folder_class_0,
                    class_1_folder = folder_class_1, 
                    one_hot = True , 
                    nb_samp_class = nb_sample_per_class, 
                    nb_seq_samp_per_class = nb_seq_samples_per_class, 
                    rand_seed = 42, 
                    seq_len = seq_len, 
                    test_train_ratio = test_train_ratio, 
                    batch_size = batch_size)
    """ 
    X_train, X_test, y_train, y_test = load_2_class_sequence_data_no_redraw(class_0_folder = folder_class_0, 
                    class_1_folder = folder_class_1, 
                    one_hot = apply_one_hot_encoding, 
                    nb_samp_class = nb_sample_per_class, 
                    seq_len = seq_len, 
                    rand_seed = shuffling_seed, 
                    test_train_ratio = test_train_ratio, 
                    batch_size = batch_size, 
                    verbose = verbose)

    # Main folder where the training data is stored
    basepath = path.dirname(__file__)
    # folder_saving_str = path.abspath(path.join(basepath, "..", "Figures/Training/2_Similar_Boxes_Custom_Loop_Tensors_Custom_Deep_Sets"))
    # folder_saving_str = path.abspath(path.join(basepath, "..", "Figures/Training/2_Similar_Boxes_Deep_Sets"))
    # folder_saving_str = path.abspath(path.join(basepath, "..", "Figures/Training/Final_Data_Set/Similar_Boxes_Classification_" + str(seq_len) + "_Grasps"))
    # folder_saving_str = path.abspath(path.join(basepath, "..", "Figures/Training/Final_Data_Set/Different_Boxes_Classification_" + str(seq_len) + "_Grasps"))
    # folder_saving_str = path.abspath(path.join(basepath, "..", "Figures/Training/Final_Data_Set/Box_and_Cyli_" + str(seq_len) + "_Grasps"))

    # folder_saving_str = path.abspath(path.join(basepath, "..", "Figures/Training/500k_Data_Set/Big_Cube_and_Cylinder/seq_len_" + str(seq_len) + "_no_redraw"))
    folder_saving_str = path.abspath(path.join(basepath, "..", "Figures/Training/500k_Data_Set/Big_Cube_and_Small_Cube/seq_len_" + str(seq_len) + "_no_redraw"))


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

    # Loading the data 
    # print("Loading data")
    # Inputs_seq, Classes_seq = load_filtered_sequences(seq_len, apply_one_hot_encoding, data_set_folder, file_prefix, nb_data_files)

    # Splitting and shuffling the data
    # X_train_double, X_test_double, y_train, y_test = train_test_split(Inputs_seq, Classes_seq, test_size=test_train_ratio, random_state=42)


    """
    # Data loader for final data set (FIXED SEQUENCE LENGTH)
    X_train_double, X_test_double, y_train, y_test = load_data_two_classes_seq(folder_class_0, folder_class_1, seq_len, file_prefix, nb_data_files, True, 42, test_train_ratio, 2)

    # Data loader for final data set (VARIABLE SEQUENCE LENGTH)
    

    print("Size of train set: ", len(X_train_double))
    print("Size of test set: ", len(X_test_double))

    input_training_tensor = tf.constant(X_train_double)
    
    output_training_tensor = tf.constant(y_train)

    train_data_set = tf.data.Dataset.from_tensor_slices((input_training_tensor,output_training_tensor))

    train_data_set = train_data_set.shuffle(buffer_size=1024).batch(batch_size)


    input_test_tensor = tf.constant(X_test_double)

    output_test_tensor = tf.constant(y_test)

    test_data_set = tf.data.Dataset.from_tensor_slices((input_test_tensor,output_test_tensor))
    test_data_set = test_data_set.batch(batch_size)

    # The networks 
    """

    input_training_tensor = tf.constant(X_train)
    
    output_training_tensor = tf.constant(y_train)

    train_data_set = tf.data.Dataset.from_tensor_slices((input_training_tensor,output_training_tensor))

    train_data_set = train_data_set.shuffle(buffer_size=1024).batch(batch_size)


    input_test_tensor = tf.constant(X_test)

    output_test_tensor = tf.constant(y_test)

    test_data_set = tf.data.Dataset.from_tensor_slices((input_test_tensor,output_test_tensor))
    test_data_set = test_data_set.batch(batch_size)

    print("Finished preparing data...")
    

    """
    Phi_network = keras.Sequential(
        [
            keras.Input(shape=(input_size,)), 
            layers.Dense(64, activation='relu'), 
            layers.Dense(32, activation='relu'), 
            layers.Dense(14, activation=tf.nn.softmax)
        ], 
        name = "Phi"
    )
    """
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
    """
    
    """
    Rho_network = keras.Sequential(
        [
            keras.Input(shape=(input_size,)), 
            layers.Dense(64, activation='relu'), 
            layers.Dense(32, activation='relu'), 
            layers.Dense(2, activation=tf.nn.softmax)
        ], 
        name = "Rho"
    )
    """
    """
    Rho_network = keras.Sequential(
        [
            keras.Input(shape=(64,)), 
            layers.Dense(64, activation='relu'), 
            layers.Dense(64, activation='relu'), 
            layers.Dense(2, activation=tf.nn.softmax)
        ], 
        name = "Rho"
    )
    """

    # A first save 
    # Phi_network.save(folder_saving_str + "/Model_PHI_" +str(nb_epochs)+ "_epochs_deep_sets_nb_" + str(training_run_nb))
    # Rho_network.save(folder_saving_str + "/Model_RHO_" +str(nb_epochs)+ "_epochs_deep_sets_nb_" + str(training_run_nb))

    # model = DeepSets(Phi_network, Rho_network, seq_len=seq_len)

    model_PHI = keras.Sequential(
        [
            keras.layers.InputLayer(input_shape=(input_size,)), 
            layers.Dense(64, activation='relu'), 
            layers.Dense(32, activation='relu'), 
            layers.Dense(input_size, activation='relu')
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

    """
    model.compile(
        phi_optimizer = tf.keras.optimizers.Adam(),
        rho_optimizer = tf.keras.optimizers.Adam(),
        loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True),
    )
    """

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
            """
            print(x)
            exit()
            """

            for index in range(seq_len): 
                index_pred = model_PHI(x[:,index], training=True)
               
                predictions_list.append(index_pred) 
            sums =  layers.add(predictions_list)
            # sums =  layers.Average()(predictions_list) #average instead of sum 
            logits = model_RHO(sums, training=True)
            loss_value = loss_fn(y, logits)

            """
            
            # TEST PRINTS 
            print("Printing prediction list:")
            print(predictions_list)

            print("Printing sums:")
            print(sums)

            print("Printing logits:")
            print(logits)

            # exit()
            """
            

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
    best_val_acc = 0.5 # best val acc 
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