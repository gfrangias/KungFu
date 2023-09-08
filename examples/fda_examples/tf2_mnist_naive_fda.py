import os, argparse, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from models.lenet5 import create_lenet5
from models.adv_cnn import create_adv_cnn
from dataset.mnist import create_dataset
from fda_functions.naive_fda import compute_averaged_divergence, rtc_check
from pickle_data.pickle_functions import initialize_logs,\
                                    step_update_logs,\
                                    export_pickle

import tensorflow as tf
from kungfu.python import current_cluster_size, current_rank
from kungfu.tensorflow.ops import group_all_reduce
from kungfu._utils import map_maybe
from kungfu.tensorflow.optimizers import MySynchronousSGDOptimizer
                                          
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
args = parser.parse_args()

# Get number of inputs from input 
epochs = args.epochs
syncs = tf.Variable(0, dtype=tf.int32)

# Load mnist dataset
train_dataset, test_dataset, steps_per_epoch = \
    create_dataset(epochs, args.batch, current_cluster_size(), current_rank())

if current_rank() == 0 and args.l: 

    print("Steps per Epoch: "+ str(steps_per_epoch))

    # Initialize the training logs
    training_logs = initialize_logs("Naive FDA", args.model, current_cluster_size(),\
                                             epochs, args.batch)

# Create selected model
if args.model == "lenet5":
    train_model, loss_fun = create_lenet5(input_shape=(28,28,1), num_classes=10)
elif args.model == "adv_cnn":
    train_model, loss_fun = create_adv_cnn(input_shape=(28,28,1), num_classes=10)

# Set Adam along with KungFu Synchronous SGD optimizer
opt = tf.compat.v1.train.AdamOptimizer()
my_opt = MySynchronousSGDOptimizer(opt)

#
# Function that performs one training step of one batch
#
@tf.function
def training_step(images, labels, first_step, last_sync_model):
    
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

    my_opt.apply_gradients(zip(grads, train_model.trainable_variables))

    if not first_step:
        averaged_divergence = compute_averaged_divergence \
            (last_sync_model, train_model.trainable_variables, current_cluster_size())
        
    else:
        averaged_divergence = 0
    
    #if current_rank() == 0:         
    #    tf.print("Averaged divergence: ")
    #    tf.print(averaged_divergence)
    

    if rtc_check(averaged_divergence, args.threshold):
        if current_rank() == 0: tf.print("Syncing!")
        syncs.assign_add(1)
        summed_models = group_all_reduce(train_model.trainable_variables)

        num_of_nodes = tf.cast(current_cluster_size(), tf.float32)
        # Reduce the gradients of the current node based on the average of all nodes
        averaged_models = map_maybe(lambda g: g / num_of_nodes, summed_models)
        for variable, averaged_value in zip(train_model.trainable_variables, averaged_models):
            variable.assign(averaged_value)
        last_sync_model = train_model.trainable_variables
        #tf.print("New last sync model: ")
        #tf.print(tf.norm(last_sync_model[0]))

    # Update training metric.
    train_model.compiled_metrics.update_state(labels, probs)
    batch_accuracy = train_model.metrics[0].result()
    
    # Reset accuracy metric so that the next batch accuracy doesn't accumulate
    train_model.metrics[0].reset_states()

    # KungFu: broadcast is done after the first gradient step to ensure optimizer initialization.
    # This way all models across the network initialize with the same weights
    if first_step:
        from kungfu.tensorflow.initializer import broadcast_variables
        broadcast_variables(train_model.variables)
        broadcast_variables(opt.variables())
        syncs.assign_add(1)


    return batch_loss, batch_accuracy, last_sync_model

# Start timer
start_time = time.time()

# Initialize last sync model
last_sync_model = train_model.trainable_variables

# Take the batches needed for this epoch and take the steps needed
for step, (images, labels) in enumerate(train_dataset.take(steps_per_epoch*epochs)):

    # Take a training step
    batch_loss, batch_accuracy, last_sync_model = training_step(images, labels, step == 0, last_sync_model)

    # Log loss and accuracy data every 10 steps
    if (step % 10 == 0 or step == steps_per_epoch*epochs - 1) and args.l and current_rank() == 0:
        training_logs = step_update_logs(training_logs, step, steps_per_epoch, \
                                         syncs.numpy(), batch_loss, batch_accuracy)

    # Print data to terminal
    if (((step % steps_per_epoch) % 10 == 0) or (step % (steps_per_epoch - 1) == 0)) and current_rank() == 0:
        print('Epoch #%d\tStep #%d \tLoss: %.6f\tBatch Accuracy: %.6f\tSyncs: %d' % \
              (step / steps_per_epoch + 1, step % steps_per_epoch, batch_loss, batch_accuracy, syncs))
 
# Stop timer
end_time = time.time()

if current_rank()==0:

    # Evaluate the learning using test data
    print("Evaluating final model...")
    loss, accuracy = train_model.evaluate(test_dataset)

    # Update training logs and export using pickle
    if args.l: 
        export_pickle(training_logs, loss, accuracy, end_time - start_time)
        