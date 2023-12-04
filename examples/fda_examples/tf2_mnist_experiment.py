from kungfu.python import current_cluster_size, current_rank
from kungfu.tensorflow.ops import group_all_reduce, set_tree, broadcast
from kungfu.tensorflow.initializer import broadcast_variables
from kungfu.tensorflow.optimizers import MySynchronousSGDOptimizer, SynchronousSGDOptimizer
from kungfu._utils import map_maybe

import tensorflow as tf
from math import sqrt
import os, argparse, time, copy, sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from models.lenet5 import create_lenet5
from models.adv_cnn import create_adv_cnn
from dataset.mnist import create_dataset
from dataframe.logs_dict import logs_dict
from dataframe.logs_df import logs_df
from fda_functions.naive_fda import approx_rtc_naive, rtc_check
from fda_functions.linear_fda import approx_rtc_linear
from fda_functions.sketch_fda import approx_rtc_sketch, AmsSketch
from fda_functions.tensor_list_functions import tensor_list_to_vector

parser = argparse.ArgumentParser(description='KungFu mnist example.')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to run experiment')
parser.add_argument('--model', type=str, default="lenet5",
                    help='available options: lenet5, adv_cnn')
parser.add_argument('--batch', type=int, default=64,
                    help='batch size')
parser.add_argument("-l", action="store_true", help="Enable logs")
parser.add_argument("--threshold", type=float, default= 1.1,
                     help="synchronization threshold")
parser.add_argument("--clients_distr", type=str, 
                    default='[]', help="Client and node configuration")
parser.add_argument("--optimizer", type=str, 
                    default='Adam', help='Optimizer used')
parser.add_argument("--algorithm", type=str, default= "synchronous",
                     help="type of experiment algorithmrithm")
parser.add_argument("--topology", type=str, default= "ring",
                     help="network topology")
args = parser.parse_args()

# Get number of inputs from input 
epochs = args.epochs
syncs = tf.Variable(0, dtype=tf.int32)
com_duration = tf.Variable(0, dtype=tf.float64)
calc_duration = tf.Variable(0, dtype=tf.float64)

# Load mnist dataset
train_dataset, test_dataset, epoch_steps, epoch_steps_float = \
    create_dataset(args.batch, current_cluster_size(), current_rank())

if args.algorithm == "synchronous": threshold = None
else: threshold = args.threshold

if current_rank() == 0 and args.l: 

    print("Steps per Epoch: "+ str(epoch_steps))
    print("Steps per Epoch in float: "+str(epoch_steps_float))

    # Create a dictionary for the logs
    logs_dict = logs_dict(args.algorithm, args.model, current_cluster_size(), args.clients_distr,threshold, args.batch, epoch_steps, epochs, args.topology)

# Create selected model
if args.model == "lenet5":
    train_model, loss_fun = create_lenet5(input_shape=(28,28,1), num_classes=10)
    w_t0, _ = create_lenet5(input_shape=(28,28,1), num_classes=10)   
    w_tminus1, _ = create_lenet5(input_shape=(28,28,1), num_classes=10)   
elif args.model == "adv_cnn":
    train_model, loss_fun = create_adv_cnn(input_shape=(28,28,1), num_classes=10)
    w_t0, _ = create_adv_cnn(input_shape=(28,28,1), num_classes=10)
    w_tminus1, _ = create_adv_cnn(input_shape=(28,28,1), num_classes=10)   

if args.algorithm == "sketch":
    sketch_width = 250
    sketch_depth = 5
    epsilon = 1. / sqrt(sketch_width)
    ams_sketch = AmsSketch(depth=5, width=250)

if args.topology == "star":
    # Set a star-topology with node 0 as the root
    tree = tf.constant(0, shape=(current_cluster_size(),), dtype=tf.int32)
    set_tree_op = set_tree(broadcast(tree))
elif args.topology == "binary_tree":
    topology = tf.range(current_cluster_size(), dtype=tf.int32)
    tree = tf.where(topology > 0, (topology - 1) // 2, 0)
    print(tree.numpy())
    set_tree_op = set_tree(broadcast(tree))

# Set Adam along with KungFu Synchronous SGD optimizer
opt = tf.keras.optimizers.Adam()

def training_step(images, labels):

    # Open a GradientTape to record the operations run
    # during the forward pass, which enables auto-differentiation    
    with tf.GradientTape() as tape:
        
        # Predicted probability values
        probs = train_model(images, training=True)
        
        # Compute the loss function for this minibatch
        batch_loss = loss_fun(labels, probs)
        
    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss.
    grads = tape.gradient(batch_loss, train_model.trainable_variables)

    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    opt.apply_gradients(zip(grads, train_model.trainable_variables))

    return batch_loss

#
# Function that performs one training step of one batch
# Baseline Synchronous
#
def training_step_synchronous(images, labels):

    batch_loss = training_step(images, labels)

    syncs.assign_add(1)

    start_time = tf.timestamp()
    summed_models = group_all_reduce(train_model.trainable_variables)
    end_time = tf.timestamp()
    com_duration.assign_add(end_time - start_time)

    num_of_nodes = tf.cast(current_cluster_size(), tf.float32)
    # Reduce the gradients of the current node based on the average of all nodes
    averaged_models = map_maybe(lambda g: g / num_of_nodes, summed_models)
    
    for variable, averaged_value in zip(train_model.trainable_variables, averaged_models):
        variable.assign(averaged_value)

    return batch_loss, com_duration

#
# Function that performs one training step of one batch
# Naive FDA
#
def training_step_naive(images, labels, first_step):
    
    batch_loss = training_step(images, labels)

    if not first_step:
        rtc_approx, com_duration_step, calc_duration_step = approx_rtc_naive \
            (w_t0, train_model.trainable_variables, current_cluster_size())
        com_duration.assign_add(com_duration_step)    
        calc_duration.assign_add(calc_duration_step)
    else:
        rtc_approx = 0

    if rtc_check(rtc_approx, threshold):
        syncs.assign_add(1)

        start_time = tf.timestamp()
        summed_models = group_all_reduce(train_model.trainable_variables)
        end_time = tf.timestamp()
        com_duration.assign_add(end_time - start_time)

        num_of_nodes = tf.cast(current_cluster_size(), tf.float32)
        # Reduce the gradients of the current node based on the average of all nodes
        averaged_models = map_maybe(lambda g: g / num_of_nodes, summed_models)
        for variable, averaged_value in zip(train_model.trainable_variables, averaged_models):
            variable.assign(averaged_value)
        for variable, averaged_value in zip(w_t0.trainable_variables, averaged_models):
            variable.assign(averaged_value)

    return batch_loss, com_duration, calc_duration

#
# Function that performs one training step of one batch
# Linear FDA
#
def training_step_linear(images, labels, first_step):
    
    batch_loss = training_step(images, labels)

    if not first_step:
        rtc_approx, com_duration_step, calc_duration_step = approx_rtc_linear\
            (w_tminus1, w_t0, train_model.trainable_variables, current_cluster_size())
        com_duration.assign_add(com_duration_step)
        calc_duration.assign_add(calc_duration_step)
    else:
        rtc_approx = 0

    if rtc_check(rtc_approx, threshold):
        syncs.assign_add(1)

        start_time = tf.timestamp()
        summed_models = group_all_reduce(train_model.trainable_variables)
        end_time = tf.timestamp()
        com_duration.assign_add(end_time - start_time)

        num_of_nodes = tf.cast(current_cluster_size(), tf.float32)
        # Reduce the gradients of the current node based on the average of all nodes
        averaged_models = map_maybe(lambda g: g / num_of_nodes, summed_models)
        for variable, averaged_value in zip(train_model.trainable_variables, averaged_models):
            variable.assign(averaged_value)
        for variable, averaged_value in zip(w_tminus1.trainable_variables, w_t0.trainable_variables):
            variable.assign(averaged_value)
        for variable, averaged_value in zip(w_t0.trainable_variables, averaged_models):
            variable.assign(averaged_value)

    return batch_loss, com_duration, calc_duration

#
# Function that performs one training step of one batch
# Sketch FDA
#
def training_step_sketch(images, labels, first_step):
    
    batch_loss = training_step(images, labels)

    if not first_step:
        rtc_approx, com_duration_step, calc_duration_step = approx_rtc_sketch \
            (w_t0, train_model.trainable_variables, ams_sketch, epsilon, current_cluster_size())
        com_duration.assign_add(com_duration_step)    
        calc_duration.assign_add(calc_duration_step)
        #if current_rank() == 0: print(rtc_approx)
    else:
        rtc_approx = 0

    if rtc_check(rtc_approx, threshold):
        syncs.assign_add(1)
        start_time = tf.timestamp()
        summed_models = group_all_reduce(train_model.trainable_variables)
        end_time = tf.timestamp()
        com_duration.assign_add(end_time - start_time)

        num_of_nodes = tf.cast(current_cluster_size(), tf.float32)
        # Reduce the gradients of the current node based on the average of all nodes
        averaged_models = map_maybe(lambda g: g / num_of_nodes, summed_models)
        for variable, averaged_value in zip(train_model.trainable_variables, averaged_models):
            variable.assign(averaged_value)
        for variable, averaged_value in zip(w_t0.trainable_variables, averaged_models):
            variable.assign(averaged_value)


    return batch_loss, com_duration, calc_duration

# Start timer
steps_remainder = 0
total_steps = 0
duration = 0

train_dataset_iter = iter(train_dataset)
broadcast_variables(train_model.variables)
broadcast_variables(opt.variables())

for epoch in range(1, epochs+1):
    
    if steps_remainder < 1 or epoch == 1:
        steps_remainder += epoch_steps_float - epoch_steps
        steps_next_epoch = epoch_steps
    else:
        steps_remainder = steps_remainder - 1
        steps_next_epoch = epoch_steps + 1

    # Take the batches needed for this epoch and take the steps needed
    for step in range(steps_next_epoch):
        images, labels = next(train_dataset_iter)

        total_steps += 1
        start_time = time.time() 
        
        # Take a training step
        if args.algorithm == "naive":
            batch_loss, com_duration_step, calc_duration_step = training_step_naive(images, labels, step == 0 and epoch == 1)
        elif args.algorithm == "linear":
            batch_loss, com_duration_step, calc_duration_step = training_step_linear(images, labels, step == 0 and epoch == 1)
        elif args.algorithm == "sketch":
            batch_loss, com_duration_step, calc_duration_step = training_step_sketch(images, labels, step == 0 and epoch == 1)
        elif(args.algorithm == "synchronous"):
            batch_loss, com_duration_step = training_step_synchronous(images, labels)
            calc_duration_step = tf.constant(0, dtype=tf.float64)
        end_time = time.time()
        duration += end_time - start_time

        # Log loss and syncs data at every step
        if args.l and current_rank() == 0:
            logs_dict.step_update(total_steps, epoch, syncs, batch_loss, duration, com_duration.numpy(), calc_duration.numpy())
    
    if args.l and current_rank() == 0:

        print("Epoch #%d\tSteps: %d\t Steps remainder: %.2f" % (epoch, step+1, steps_remainder))
        print("Total Steps: %d\tSyncs: %d" % (total_steps, syncs))
        if args.algorithm == "synchronous":
            epoch_loss, epoch_accuracy = train_model.evaluate(test_dataset)
        else:
            epoch_loss, epoch_accuracy = w_t0.evaluate(test_dataset)
        logs_dict.epoch_update(epoch, total_steps, syncs, epoch_accuracy, epoch_loss, duration, com_duration.numpy(), calc_duration.numpy())


if current_rank()==0:

    # Update training logs and export using pickle
    if args.l: 
        logs_dict.id_update()
        logs_df = logs_df(logs_dict)
        logs_df.append_in_csv()
