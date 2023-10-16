import tensorflow as tf
from kungfu.tensorflow.ops import group_all_reduce, all_reduce
from kungfu._utils import map_maybe
from .tensor_list_functions import tensor_list_to_vector, tensor_to_tensor_list
import time

# Compute the divergence using the 2-norm for Naive FDA
def compute_averaged_divergence(last_sync_model, local_model, num_of_nodes):

    start_time_norm = tf.timestamp()
    # Convert the two tensor list models to vectors
    last_sync_model_vector = tensor_list_to_vector(last_sync_model.trainable_variables)
    local_model_vector = tensor_list_to_vector(local_model)

    # Local divergence is the norm of the local model drift in comparison to the 
    # last synced model
    delta_i = local_model_vector - last_sync_model_vector
    local_divergence =  tf.reduce_sum(tf.square(delta_i))
    
    #tf.print("Local divergence: ")
    #tf.print(local_divergence)
    end_time_norm = tf.timestamp()

    start_time = tf.timestamp()
    # Calculate the average divergence of the network using all-reduce
    summed_divergences = all_reduce(local_divergence)
    end_time = tf.timestamp()
    
    averaged_divergence = summed_divergences / num_of_nodes

    #tf.print("Averaged divergence: ")
    #tf.print(averaged_divergence)
    return averaged_divergence, end_time - start_time, end_time_norm - start_time_norm

# Check if the divergence satisfies the RTC
def rtc_check(divergence, threshold):
    if tf.math.greater(tf.cast(divergence, tf.float32),tf.cast(threshold, tf.float32)):
        return True
    else:
        return False