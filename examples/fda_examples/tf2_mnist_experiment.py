from kungfu.python import current_cluster_size, current_rank
from kungfu.tensorflow.ops import group_all_reduce
from kungfu.tensorflow.optimizers import MySynchronousSGDOptimizer
from kungfu._utils import map_maybe

import os, argparse, time, copy

import tensorflow as tf
if tf.config.list_physical_devices('GPU'):
        for gpu in tf.config.experimental.list_physical_devices('GPU'):
                tf.config.experimental.set_memory_growth(gpu,True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from models.lenet5 import create_lenet5
from models.adv_cnn import create_adv_cnn
from dataset.mnist import create_dataset
from dataframe.logs_dict import logs_dict
from dataframe.logs_df import logs_df
from fda_functions.naive_fda import compute_averaged_divergence, rtc_check

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
parser.add_argument("--clients_per_node", type=str, 
                    default='[]', help="Client and node configuration")
parser.add_argument("--optimizer", type=str, 
                    default='Adam', help='Optimizer used')
parser.add_argument("--exper_type", type=str, default= "synchronous",
                     help="type of experiment algorithm")
args = parser.parse_args()

# Get number of inputs from input 
epochs = args.epochs
syncs = tf.Variable(0, dtype=tf.int32)
agg_duration = tf.Variable(0, dtype=tf.float64)
norm_duration = tf.Variable(0, dtype=tf.float64)

# Load mnist dataset
train_dataset, test_dataset, steps_per_epoch, steps_per_epoch_float = \
    create_dataset(epochs, args.batch, current_cluster_size(), current_rank())

if current_rank() == 0 and args.l: 

    print("Steps per Epoch: "+ str(steps_per_epoch))
    print("Steps per Epoch in float: "+str(steps_per_epoch_float))

    if args.exper_type == "synchronous": logs_dict = logs_dict("Synchronous SGD", args.model, current_cluster_size(), args.clients_per_node,None, args.batch, steps_per_epoch, epochs)
    elif args.exper_type == "naive": logs_dict = logs_dict("Naive FDA", args.model, current_cluster_size(), args.clients_per_node, args.threshold, args.batch, steps_per_epoch, epochs) 


# Create selected model
if args.model == "lenet5":
    train_model, loss_fun = create_lenet5(input_shape=(28,28,1), num_classes=10)
    last_sync_model, loss_fun_last_sync = create_lenet5(input_shape=(28,28,1), num_classes=10)
    last_sync_model.build((None,28,28))    
elif args.model == "adv_cnn":
    train_model, loss_fun = create_adv_cnn(input_shape=(28,28,1), num_classes=10)
    last_sync_model, loss_fun_last_sync = create_adv_cnn(input_shape=(28,28,1), num_classes=10)
    last_sync_model.build((None,28,28))

# Set Adam along with KungFu Synchronous SGD optimizer
opt = tf.keras.optimizers.Adam()

@tf.function
def training_step(images, labels):

        
    #if current_rank() == 0: print("Retrace")
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

#
# Function that performs one training step of one batch
#
def training_step_synchronous(images, labels, first_step):

    training_step(images, labels)

    start_time = tf.timestamp()
    summed_models = group_all_reduce(train_model.trainable_variables)
    end_time = tf.timestamp()
    agg_duration.assign_add(end_time - start_time)

    num_of_nodes = tf.cast(current_cluster_size(), tf.float32)
    # Reduce the gradients of the current node based on the average of all nodes
    averaged_models = map_maybe(lambda g: g / num_of_nodes, summed_models)
    for variable, averaged_value in zip(train_model.trainable_variables, averaged_models):
        variable.assign(averaged_value)
    syncs.assign_add(1)

    # KungFu: broadcast is done after the first gradient step to ensure optimizer initialization.
    # This way all models across the network initialize with the same weights
    if first_step:
        from kungfu.tensorflow.initializer import broadcast_variables
        broadcast_variables(train_model.variables)
        broadcast_variables(opt.variables())

    return batch_loss, end_time - start_time

#
# Function that performs one training step of one batch
#
def training_step_naive(images, labels, first_step):
    
    training_step(images, labels)

    #if current_rank() == 0:
    #    tf.print("New 1 last sync model: ")
    #    tf.print(tf.norm(last_sync_model.trainable_variables[0]))

    if not first_step:
        averaged_divergence, agg_duration_step, norm_duration_step = compute_averaged_divergence \
            (last_sync_model, train_model.trainable_variables, current_cluster_size())
        agg_duration.assign_add(agg_duration_step)    
        norm_duration.assign_add(norm_duration_step)
    else:
        averaged_divergence = 0
    
    #if current_rank() == 0:         
    #    tf.print("Averaged divergence: ")
    #    tf.print(averaged_divergence)

    #tf.print(averaged_divergence)
    if rtc_check(averaged_divergence, args.threshold):
        #if current_rank() == 0: tf.print("Syncing!")
        #tf.print("SYNC!")
        syncs.assign_add(1)

        start_time = tf.timestamp()
        summed_models = group_all_reduce(train_model.trainable_variables)
        end_time = tf.timestamp()
        agg_duration.assign_add(end_time - start_time)

        num_of_nodes = tf.cast(current_cluster_size(), tf.float32)
        # Reduce the gradients of the current node based on the average of all nodes
        averaged_models = map_maybe(lambda g: g / num_of_nodes, summed_models)
        for variable, averaged_value in zip(train_model.trainable_variables, averaged_models):
            variable.assign(averaged_value)
        for variable, averaged_value in zip(last_sync_model.trainable_variables, averaged_models):
            variable.assign(averaged_value)

        #if current_rank() == 0:
        #    tf.print("New 2 last sync model: ")
        #    tf.print(tf.norm(last_sync_model.trainable_variables[0]))

    # KungFu: broadcast is done after the first gradient step to ensure optimizer initialization.
    # This way all models across the network initialize with the same weights
    if first_step:
        from kungfu.tensorflow.initializer import broadcast_variables
        broadcast_variables(train_model.variables)
        broadcast_variables(opt.variables())
        syncs.assign_add(1)

    return batch_loss, agg_duration, norm_duration

# Start timer
steps_remainder = 0
total_steps = 0
duration = 0

# Initialize last sync model

train_dataset_iter = iter(train_dataset)

for epoch in range(1, epochs+1):
    
    if steps_remainder < 1 or epoch == 1:
        steps_remainder += steps_per_epoch_float - steps_per_epoch
        steps_next_epoch = steps_per_epoch
    else:
        steps_remainder = steps_remainder - 1
        steps_next_epoch = steps_per_epoch + 1

    # Take the batches needed for this epoch and take the steps needed
    for step in range(steps_next_epoch):
        images, labels = next(train_dataset_iter)

        total_steps += 1
        start_time = time.time() 
        
        # Take a training step
        if args.exper_type == "naive":
            batch_loss, agg_duration_step, norm_duration_step = training_step_naive(images, labels, step == 0 and epoch == 1)
        elif(args.exper_type == "synchronous"):
            batch_loss, agg_duration_step = training_step_synchronous(images, labels, step == 0 and epoch == 1)
            norm_duration_step = tf.constant(0, dtype=tf.float64)
        end_time = time.time()
        duration += end_time - start_time

        # Log loss and syncs data at every step
        if args.l and current_rank() == 0:
            logs_dict.step_update(total_steps, epoch, syncs, batch_loss, duration, agg_duration.numpy(), norm_duration.numpy())
    
    if args.l and current_rank() == 0:

        print("Epoch #%d\tSteps: %d\t Steps remainder: %.2f" % (epoch, step+1, steps_remainder))
        print("Total Steps: %d\tSyncs: %d" % (total_steps, syncs))
        epoch_loss, epoch_accuracy = train_model.evaluate(test_dataset)
        logs_dict.epoch_update(epoch, total_steps, syncs, epoch_accuracy, epoch_loss, duration, agg_duration.numpy(), norm_duration.numpy())


if current_rank()==0:

    # Update training logs and export using pickle
    if args.l: 
        logs_dict.id_update()
        logs_df = logs_df(logs_dict)
        logs_df.append_in_csv()
