import tensorflow as tf
from kungfu.tensorflow.ops import group_all_reduce
from kungfu._utils import map_maybe
from .tensor_list_functions import tensor_list_to_vector, tensor_to_tensor_list
import sys

# Compute the divergence using the 2-norm for Naive FDA
def compute_averaged_divergence(last_sync_model, local_model, num_of_nodes):

    # Convert the two tensor list models to vectors
    last_sync_model_vector = tensor_list_to_vector(last_sync_model)
    local_model_vector = tensor_list_to_vector(local_model)

    # Local divergence is the norm of the local model drift in comparison to the 
    # last synced model
    local_divergence =  tf.norm(local_model_vector - last_sync_model_vector, 2)
    #tf.print("Local divergence: ")

    # Calculate the average divergence of the network using all-reduce
    local_divergence = tensor_to_tensor_list(local_divergence)
    summed_divergences = group_all_reduce(local_divergence)
    num_of_nodes = tf.cast(num_of_nodes, tf.float32)
    averaged_divergence = map_maybe(lambda d: d / num_of_nodes, summed_divergences)

    return tensor_to_tensor_list(averaged_divergence)

# Check if the divergence satisfies the RTC
def rtc_check(divergence, threshold):
    if tf.math.greater(tf.cast(divergence, tf.float32),tf.cast(threshold, tf.float32)):
        return True
    else:
        return False