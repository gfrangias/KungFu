import tensorflow as tf
if tf.config.list_physical_devices('GPU'):
        for gpu in tf.config.experimental.list_physical_devices('GPU'):
                tf.config.experimental.set_memory_growth(gpu,True)

from kungfu.python import current_cluster_size, current_rank
from kungfu.tensorflow.ops import group_all_reduce
from kungfu.tensorflow.optimizers import MySynchronousSGDOptimizer
from kungfu._utils import map_maybe

import os, argparse, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from models.lenet5 import create_lenet5
from models.adv_cnn import create_adv_cnn
from dataset.mnist import create_dataset
from dataframe.logs_dict import logs_dict
from dataframe.logs_df import logs_df
                      
parser = argparse.ArgumentParser(description='KungFu mnist example.')
parser.add_argument('--epochs',
                    type=int,
                    default=10,
                    help='number of epochs to run experiment')
parser.add_argument('--model',
                    type=str,
                    default="lenet5",
                    help='available options: lenet5, adv_cnn')
parser.add_argument('--batch',
                    type=int,
                    default=64,
                    help='batch size')
parser.add_argument("-l", action="store_true", help="Enable logs")
args = parser.parse_args()

# Get number of inputs from input 
epochs = args.epochs

# Load mnist dataset
train_dataset, test_dataset, epoch_steps, epoch_steps_float = \
    create_dataset(epochs, args.batch, current_cluster_size(), current_rank())

if current_rank() == 0 and args.l: 

    print("Steps per Epoch: "+ str(epoch_steps))
    print("Steps per Epoch in float: "+str(epoch_steps_float))

    # Initialize the training logs
    logs_dict = logs_dict("Synchronous SGD", args.model, current_cluster_size(), None, args.batch, epoch_steps)


# Create selected model
if args.model == "lenet5":
    train_model, loss_fun = create_lenet5(input_shape=(28,28,1), num_classes=10)
elif args.model == "adv_cnn":
    train_model, loss_fun = create_adv_cnn(input_shape=(28,28,1), num_classes=10)

# Set Adam along with KungFu Synchronous SGD optimizer
opt = tf.keras.optimizers.Adam()

#
# Function that performs one training step of one batch
#
@tf.function
def training_step(images, labels, first_step):
    
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
    summed_models = group_all_reduce(train_model.trainable_variables)

    num_of_nodes = tf.cast(current_cluster_size(), tf.float32)
    # Reduce the gradients of the current node based on the average of all nodes
    averaged_models = map_maybe(lambda g: g / num_of_nodes, summed_models)
    for variable, averaged_value in zip(train_model.trainable_variables, averaged_models):
        variable.assign(averaged_value)

    # KungFu: broadcast is done after the first gradient step to ensure optimizer initialization.
    # This way all models across the network initialize with the same weights
    if first_step:
        from kungfu.tensorflow.initializer import broadcast_variables
        broadcast_variables(train_model.variables)
        broadcast_variables(opt.variables())

    return batch_loss

# Start timer
steps_remainder = 0
epoch = 1
step_in_epoch = 0
duration = 0

# Take the batches needed for this epoch and take the steps needed
for step, (images, labels) in enumerate(train_dataset.take(epoch_steps*epochs)):
    
    start_time = time.time() 

    # Take a training step
    batch_loss = training_step(images, labels, step == 0)

    end_time = time.time()
    duration += end_time - start_time


    step_in_epoch+=1

    if step_in_epoch == epoch_steps and steps_remainder < 1:
        if args.l and current_rank() == 0:
            print("Epoch #%d\tSteps: %d\t Steps remainder: %.2f" % (epoch, step_in_epoch, steps_remainder))
            print("Total Steps: %d\tSyncs: %d" % (step+1, step+1))
            epoch_loss, epoch_accuracy = train_model.evaluate(test_dataset)
            logs_dict.epoch_update(epoch_accuracy, epoch_loss, duration)
        epoch += 1
        step_in_epoch = 0
        steps_remainder += epoch_steps_float - epoch_steps

    if step_in_epoch > epoch_steps and steps_remainder >= 1:
        if args.l and current_rank() == 0:
            print("Epoch #%d\tSteps: %d\t Steps remainder: %.2f" % (epoch, step_in_epoch, steps_remainder))
            print("Total Steps: %d\tSyncs: %d" % (step+1, step+1))
            epoch_loss, epoch_accuracy = train_model.evaluate(test_dataset)
            logs_dict.epoch_update(epoch_accuracy, epoch_loss, duration)
        epoch += 1
        step_in_epoch = 0
        steps_remainder = steps_remainder - 1
    
    # Log loss and accuracy data every 10 steps
    #if (step % 10 == 0 or step == epoch_steps*epochs - 1) and args.l and current_rank() == 0:
    #    logs_dict.step_update(step, step, batch_loss)


    # Print data to terminal
    #if (((step % epoch_steps) % 10 == 0) or (step % (epoch_steps - 1) == 0)) and current_rank() == 0:
    #    print('Epoch #%d\tStep #%d \tLoss: %.6f' % \
    #          (epoch, step_in_epoch, batch_loss))

if current_rank()==0:
    
    epoch_loss, epoch_accuracy = train_model.evaluate(test_dataset)
    logs_dict.epoch_update(epoch_accuracy, epoch_loss, duration) 

    # Update training logs and export using pickle
    if args.l: 
        logs_dict.id_update()
        logs_df = logs_df(logs_dict)
        logs_df.append_in_csv()        




