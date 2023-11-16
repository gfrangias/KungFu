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

### Calculate unit vector xi
#
# xi = ||w_t0-w_tminus1||
#
def xi_vector_comp(w_tminus1, w_t0):
    
    xi = w_t0 - w_tminus1
    xi = tf.divide(xi, tf.norm(xi))

    return xi

### Calculate the local state of the node
#
# Sk(t) = (||Delta^(k)||^2, xi*Delta^(k))^T
#
def local_state_comp(w_tminus1, w_t0, w_local):
    
    start_time_calc = tf.timestamp()
    local_update_norm = local_update_norm_comp(w_t0, w_local)
    xi_vector = xi_vector_comp(w_tminus1, w_t0)

    xi_dot_update = tf.reduce_sum(tf.multiply(xi_vector, w_local - w_t0))
    end_time_calc = tf.timestamp()

    return local_update_norm, xi_dot_update, end_time_calc - start_time_calc

### Approximate the RTC using Linear FDA
#
# RTC = 1/n * all_reduce(||Delta^(k)||^2) - (1/n * (all_reduce(xi*Delta^(k))))^2
#
def approx_rtc_linear(w_tminus1, w_t0, w_local, num_of_nodes):
    w_tminus1_vector = tensor_list_to_vector(w_tminus1.trainable_variables)
    w_t0_vector = tensor_list_to_vector(w_t0.trainable_variables)
    w_local_vector = tensor_list_to_vector(w_local)

    local_update_norm, xi_dot_update, calc_duration = local_state_comp(w_tminus1_vector, w_t0_vector, w_local_vector)

    start_time_com = tf.timestamp()
    local_update_norm_sum = all_reduce(local_update_norm)
    xi_dot_update_sum = all_reduce(xi_dot_update)
    end_time_com = tf.timestamp()

    local_update_norm_avg = local_update_norm_sum / num_of_nodes
    xi_dot_update_avg = xi_dot_update_sum / num_of_nodes

    return local_update_norm_avg - xi_dot_update_avg**2, end_time_com - start_time_com, calc_duration