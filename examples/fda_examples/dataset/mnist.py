import tensorflow as tf 
import numpy as np

def load_mnist_from_local_nz(file_path='examples/fda_examples/dataset/mnist.npz'):
    
    with np.load(file_path) as data:
        return (data['x_train'], data['y_train']), (data['x_test'], data['y_test'])

def create_dataset(batch_size, N, i):
    
    # Load mnist dataset
    (train_images, train_labels), (test_images, test_labels) = \
        load_mnist_from_local_nz()

    # Convert the dataset to TensorFlow dataset
    # The i-th node will get the i-th piece of the dataset
    train_images, test_images = tf.cast(train_images[..., tf.newaxis] / 255.0,
            tf.float32), tf.cast(test_images[..., tf.newaxis] / 255.0, tf.float32)
    train_labels, test_labels = tf.cast(train_labels, tf.int64), tf.cast(test_labels, tf.int64)

    train_images_list = np.array_split(train_images, N)
    train_labels_list = np.array_split(train_labels, N)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_images_list[i], train_labels_list[i]))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

    # Calculate the number of steps per epoch needed for each node
    epoch_steps = int(np.floor(len(train_images) / (N * batch_size)))
    epoch_steps_float = len(train_images) / (N * batch_size)
    
    # Duplicate the dataset multiple times
    # Shuffle the samples of every epoch
    # Set batch size
    shuffle_size = train_dataset.cardinality()
    train_dataset = train_dataset.shuffle(shuffle_size).repeat().batch(batch_size).prefetch(10)
    test_dataset = test_dataset.batch(1024)

    return train_dataset, test_dataset, epoch_steps, epoch_steps_float
