import argparse

import tensorflow as tf
from kungfu.python import current_cluster_size, current_rank
from kungfu.tensorflow.ops import (current_rank)
from kungfu.tensorflow.optimizers import (PairAveragingOptimizer,
                                          SynchronousAveragingOptimizer,
                                          SynchronousSGDOptimizer,
                                          MySynchronousSGDOptimizer)

parser = argparse.ArgumentParser(description='KungFu mnist example.')
parser.add_argument('--kf-optimizer',
                    type=str,
                    default='my-sync-sgd',
                    help='available options: sync-sgd, async-sgd, sma, my-sync-sgd')
parser.add_argument('--barrier',
                    type=str,
                    default='10')
args = parser.parse_args()


(mnist_images, mnist_labels), _ = \
    tf.keras.datasets.mnist.load_data(path='mnist-%d.npz' % current_rank())

dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(mnist_images[..., tf.newaxis] / 255.0,
             tf.float32), tf.cast(mnist_labels, tf.int64)))
dataset = dataset.repeat().shuffle(10000).batch(128)

mnist_model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # Flatten the input image
    tf.keras.layers.Dense(128, activation='relu'),  # Hidden layer with 128 neurons and ReLU activation
    tf.keras.layers.Dense(10, activation='softmax')  # Output layer with 10 neurons (one for each digit) and softmax activation
])

loss = tf.losses.SparseCategoricalCrossentropy()

# KungFu: adjust learning rate based on number of GPUs.
# opt = tf.keras.optimizers.SGD(0.001 * current_cluster_size())
opt = tf.compat.v1.train.AdamOptimizer(0.001 * current_cluster_size())

barrier = int(args.barrier)/100
barrier = tf.constant(barrier)

# KungFu: wrap tf.compat.v1.train.Optimizer.
if args.kf_optimizer == 'sync-sgd':
    opt = SynchronousSGDOptimizer(opt)
elif args.kf_optimizer == 'async-sgd':
    opt = PairAveragingOptimizer(opt)
elif args.kf_optimizer == 'sma':
    opt = SynchronousAveragingOptimizer(opt)
elif args.kf_optimizer == 'my-sync-sgd':
    opt = MySynchronousSGDOptimizer(opt, barrier=barrier)
else:
    raise RuntimeError('Unknown KungFu optimizer')

@tf.function
def training_step(images, labels, first_batch, prev_variables):
    global barrier
    mape = []

    # Perform TensorFlow GradientTape
    with tf.GradientTape() as tape:
        probs = mnist_model(images, training=True)
        loss_value = loss(labels, probs)

    updated_prev_variables = [0] * len(mnist_model.trainable_variables)

    if len(prev_variables) == 0:
        max_mape = tf.constant(float('inf'))
    else:
        # Calculate MAPE (Mean Absolute Percentage Error) for each layer's weights
        for i in range(len(mnist_model.trainable_variables)):
            denominator = tf.abs(prev_variables[i])
            numerator = tf.abs(mnist_model.trainable_variables[i]-prev_variables[i])
            mape.append(tf.reduce_mean(numerator/denominator))
        
        # Find the max MAPE between the layers
        max_mape = tf.reduce_max(mape)
        tf.print("MAPE: ",max_mape)

    # Find the local gradients 
    grads = tape.gradient(loss_value, mnist_model.trainable_variables)

    # If MAPE > barrier perform all-reduce with other nodes and update local gradients based on that
    if tf.math.greater(max_mape,barrier):
        opt.apply_gradients(zip(grads, mnist_model.trainable_variables),name=1)
        updated_prev_variables = mnist_model.trainable_variables
    # Else apply local gradients with no synchronization
    else:
        opt.apply_gradients(zip(grads, mnist_model.trainable_variables),name=0)
        updated_prev_variables = prev_variables

    # KungFu: broadcast is done after the first gradient step to ensure optimizer initialization.
    if first_batch:
        from kungfu.tensorflow.initializer import broadcast_variables
        broadcast_variables(mnist_model.variables)
        broadcast_variables(opt.variables())

    return loss_value, updated_prev_variables

prev_variables = [0.0] * len(mnist_model.trainable_variables)

# KungFu: adjust number of steps based on number of GPUs.
for batch, (images, labels) in enumerate(
        dataset.take(10000 // current_cluster_size())):
    
    print("Step #"+str(batch))
    loss_value, prev_variables = training_step(images, labels, batch == 0, prev_variables)

    if batch % 10 == 0 and current_rank() == 0:
        print('Step #%d\tLoss: %.6f' % (batch, loss_value))