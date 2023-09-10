import tensorflow as tf 
import numpy as np

def create_dataset(epochs, batch_size, N, i):
    # Load mnist dataset
    (train_images, train_labels), (test_images, test_labels) = \
        tf.keras.datasets.mnist.load_data()

    # Convert the dataset to TensorFlow dataset
    # The i-th node will get the i-th piece of the dataset
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(train_images[..., tf.newaxis] / 255.0,
            tf.float32), tf.cast(train_labels, tf.int64))).shard(N, i)
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(test_images[..., tf.newaxis] / 255.0,
            tf.float32), tf.cast(test_labels, tf.int64)))

    # Calculate the number of steps per epoch needed for each node
    steps_per_epoch = int(np.floor(len(train_images) / (N * batch_size)))

    # Duplicate the dataset multiple times
    # Shuffle the samples of every epoch
    # Set batch size
    train_dataset = train_dataset.shuffle(epochs*steps_per_epoch).repeat().batch(64)
    test_dataset = test_dataset.batch(64)

    return train_dataset, test_dataset, steps_per_epoch