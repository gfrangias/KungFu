import tensorflow as tf
from kungfu.tensorflow.ops import all_reduce
from .tensor_list_functions import tensor_list_to_vector

### Calculate the local update norm
#
# ||Delta^(k)||^2 = ||w_local-w_t0||^2
#
def local_update_norm_comp(w_t0, w_local):

    local_update = w_local - w_t0
    local_update_norm = tf.reduce_sum(tf.square(local_update))

    return local_update_norm

### Approximate the RTC using Linear FDA
#
# RTC = 1/n * all_reduce(||Delta^(k)||^2)
#
def approx_rtc_naive(w_t0, w_local, num_of_nodes):
    w_t0_vector = tensor_list_to_vector(w_t0.trainable_variables)
    w_local_vector = tensor_list_to_vector(w_local)

    start_time_calc = tf.timestamp()
    local_update_norm = local_update_norm_comp(w_t0_vector, w_local_vector)
    end_time_calc = tf.timestamp()

    start_time_com = tf.timestamp()
    local_update_norm_sum = all_reduce(local_update_norm)
    end_time_com = tf.timestamp()

    local_update_norm_avg = local_update_norm_sum / num_of_nodes

    return local_update_norm_avg, end_time_com - start_time_com, end_time_calc - start_time_calc

# Check if the divergence satisfies the RTC
def rtc_check(average_update_norm, threshold):
    if tf.math.greater(tf.cast(average_update_norm, tf.float32),tf.cast(threshold, tf.float32)):
        return True
    else:
        return False