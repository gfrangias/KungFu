import os
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from models.lenet5 import create_lenet5
from models.adv_cnn import create_adv_cnn
from dataset.mnist import create_dataset
import tensorflow as tf
from kungfu.python import current_cluster_size, current_rank
from kungfu.tensorflow.optimizers import SynchronousSGDOptimizer
                                          
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
args = parser.parse_args()

# Get number of inputs from input 
epochs = args.epochs

# Load mnist dataset
train_dataset, test_dataset, steps_per_epoch = create_dataset(epochs, args.batch, current_cluster_size(), current_rank())

if current_rank() == 0: print("Steps per Epoch: "+ str(steps_per_epoch))

# Create selected model
if args.model == "lenet5":
    train_model, loss = create_lenet5(input_shape=(28,28,1), num_classes=10)
elif args.model == "adv_cnn":
    train_model, loss = create_adv_cnn(input_shape=(28,28,1), num_classes=10)

# Set Adam along with KungFu Synchronous SGD optimizer
opt = tf.compat.v1.train.AdamOptimizer()
opt = SynchronousSGDOptimizer(opt)

#
# Function that performs one training step of one batch
#

def training_step(images, labels, first_step):
    
    # Open a GradientTape to record the operations run
    # during the forward pass, which enables auto-differentiation    
    with tf.GradientTape() as tape:
        
        # Predicted probability values
        probs = train_model(images, training=True)
        
        # Compute the loss function for this minibatch
        loss_value = loss(labels, probs)
        
    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss.
    grads = tape.gradient(loss_value, train_model.trainable_variables)

    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    opt.apply_gradients(zip(grads, train_model.trainable_variables))

    # Update training metric.
    train_model.compiled_metrics.update_state(labels, probs)
    accuracy = train_model.metrics[0].result()
    
    # Reset accuracy metric so that the next batch accuracy doesn't accumulate
    train_model.metrics[0].reset_states()

    # KungFu: broadcast is done after the first gradient step to ensure optimizer initialization.
    if first_step:
        from kungfu.tensorflow.initializer import broadcast_variables
        broadcast_variables(train_model.variables)
        broadcast_variables(opt.variables())

    return loss_value, accuracy

# Take the batches needed for this epoch and take the steps needed
for step, (images, labels) in enumerate(train_dataset.take(steps_per_epoch*epochs)):

    # Take a training step
    loss_value, accuracy = training_step(images, labels, step == 0)

    # Log loss and accuracy data every 10 steps
    if (((step % steps_per_epoch) % 10 == 0) or (step % (steps_per_epoch - 1) == 0)) and current_rank() == 0:
        print('Epoch #%d\tStep #%d \tLoss: %.6f\tBatch Accuracy: %.6f' % (step / steps_per_epoch + 1, step % steps_per_epoch, loss_value, accuracy))
 
# Evaluate the learning using test data
if current_rank()==0: train_model.evaluate(test_dataset)