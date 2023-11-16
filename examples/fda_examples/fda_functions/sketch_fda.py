import tensorflow as tf
import numpy as np
from kungfu.tensorflow.ops import all_reduce
from .tensor_list_functions import tensor_list_to_vector

### Adapted from Michail Theologitis' implementation 
### Source https://github.com/miketheologitis/FedL-Sync-FDA
class AmsSketch:
    """
    AMS Sketch class for approximate second moment estimation.
    """

    def __init__(self, depth=5, width=250):
        self.depth = tf.constant(depth)
        self.width = tf.constant(width)
        self.F = tf.random.uniform(shape=(6, depth), minval=0, maxval=(1 << 31) - 1, dtype=tf.int32)
        self.zeros_sketch = tf.zeros(shape=(self.depth, self.width), dtype=tf.float32)

        self.precomputed_dict = {}

    def precompute(self, d):
        pos_tensor = self.tensor_hash31(tf.range(d), self.F[0], self.F[1]) % self.width  # shape=(d, 5)

        self.precomputed_dict[('four', d)] = tf.cast(self.tensor_fourwise(tf.range(d)),
                                                     dtype=tf.float32)  # shape=(d, 5)

        range_tensor = tf.range(self.depth)  # shape=(5,)

        # Expand dimensions to create a 2D tensor with shape (1, `self.depth`)
        range_tensor_expanded = tf.expand_dims(range_tensor, 0)  # shape=(1, 5)

        # Use tf.tile to repeat the range `d` times
        repeated_range_tensor = tf.tile(range_tensor_expanded, [d, 1])  # shape=(d, 5)

        # shape=(`d`, `self.depth`, 2)
        self.precomputed_dict[('indices', d)] = tf.stack([repeated_range_tensor, pos_tensor],
                                                         axis=-1)  # shape=(d, 5, 2)

    @staticmethod
    def hash31(x, a, b):
        r = a * x + b
        fold = tf.bitwise.bitwise_xor(tf.bitwise.right_shift(r, 31), r)
        return tf.bitwise.bitwise_and(fold, 2147483647)

    @staticmethod
    def tensor_hash31(x, a, b):  # GOOD
        """ Assumed that x is tensor shaped (d,) , i.e., a vector (for example, indices, i.e., tf.range(d)) """

        # Reshape x to have an extra dimension, resulting in a shape of (k, 1)
        x_reshaped = tf.expand_dims(x, axis=-1)

        # shape=(`v_dim`, 7)
        r = tf.multiply(a, x_reshaped) + b

        fold = tf.bitwise.bitwise_xor(tf.bitwise.right_shift(r, 31), r)

        return tf.bitwise.bitwise_and(fold, 2147483647)

    def tensor_fourwise(self, x):
        """ Assumed that x is tensor shaped (d,) , i.e., a vector (for example, indices, i.e., tf.range(d)) """

        # 1st use the tensor hash31
        in1 = self.tensor_hash31(x, self.F[2], self.F[3])  # shape = (`x_dim`,  `self.depth`)

        # 2st use the tensor hash31
        in2 = self.tensor_hash31(x, in1, self.F[4])  # shape = (`x_dim`,  `self.depth`)

        # 3rd use the tensor hash31
        in3 = self.tensor_hash31(x, in2, self.F[5])  # shape = (`x_dim`,  `self.depth`)

        in4 = tf.bitwise.bitwise_and(in3, 32768)  # shape = (`x_dim`,  `self.depth`)

        return 2 * (tf.bitwise.right_shift(in4, 15)) - 1  # shape = (`x_dim`,  `self.depth`)

    def fourwise(self, x):
        result = 2 * (tf.bitwise.right_shift(tf.bitwise.bitwise_and(
            self.hash31(self.hash31(self.hash31(x, self.F[2], self.F[3]), x, self.F[4]), x, self.F[5]), 32768), 15)) - 1
        return result

    def sketch_for_vector(self, v):
        """ Extremely efficient computation of sketch with only using tensors.

        Args:
        - v (tf.Tensor): Vector to sketch. Shape=(d,).

        Returns:
        - tf.Tensor: An AMS - Sketch. Shape=(`depth`, `width`).
        """

        d = v.shape[0]
        
        if ('four', d) not in self.precomputed_dict:
            self.precompute(d)

        return self._sketch_for_vector(v, self.precomputed_dict[('four', d)], self.precomputed_dict[('indices', d)])

    @tf.function
    def _sketch_for_vector(self, v, four, indices):
        v_expand = tf.expand_dims(v, axis=-1)  # shape=(d, 1)

        # shape=(d, 5): +- for each value v_i , i = 1, ..., d
        deltas_tensor = tf.multiply(four, v_expand)

        sketch = tf.tensor_scatter_nd_add(self.zeros_sketch, indices, deltas_tensor)  # shape=(5, 250)

        return sketch

    @staticmethod
    def estimate_euc_norm_squared(sketch):
        """ Estimate the Euclidean norm squared of a vector using its AMS sketch.

        Args:
        - sketch (tf.Tensor): AMS sketch of a vector. Shape=(`depth`, `width`).

        Returns:
        - tf.Tensor: Estimated squared Euclidean norm.
        """

        norm_sq_rows = tf.reduce_sum(tf.square(sketch), axis=1)
        return np.median(norm_sq_rows)

### Calculate the local update norm
#
# ||Delta^(k)||^2 = ||w_local-w_t0||^2
#
def local_update_norm_comp(w_t0, w_local):

    local_update = w_local - w_t0
    local_update_norm = tf.reduce_sum(tf.square(local_update))

    return local_update_norm

### Calculate the local state of the node
#
# Sk(t) = (||Delta^(k)||^2, sk(Delta^(k)))^T
#
def local_state_comp(w_t0, w_local, ams_sketch):
    
    start_time_calc = tf.timestamp()
    local_update_norm = local_update_norm_comp(w_t0, w_local)
    sketch = ams_sketch.sketch_for_vector(w_local - w_t0)
    end_time_calc = tf.timestamp()

    return local_update_norm, sketch, end_time_calc - start_time_calc

### Approximate the RTC using Linear FDA
#
# RTC = 1/n * all_reduce(||Delta^(k)||^2) - 1/(1+epsilon) * M_2 (1/n * all_reduce(sk(Delta^(k))))
#
def approx_rtc_sketch(w_t0, w_local, ams_sketch, epsilon, num_of_nodes):
    w_t0_vector = tensor_list_to_vector(w_t0.trainable_variables)
    w_local_vector = tensor_list_to_vector(w_local)

    local_update_norm, sketch, calc_duration = local_state_comp(w_t0_vector, w_local_vector, ams_sketch)

    print(tf.reduce_sum(tf.square(sketch)))
    
    start_time_com = tf.timestamp()
    local_update_norm_sum = all_reduce(local_update_norm)
    sketch_sum = all_reduce(sketch)
    end_time_com = tf.timestamp()

    local_update_norm_avg = local_update_norm_sum / num_of_nodes
    sketch_avg = sketch_sum / num_of_nodes

    start_time_calc = tf.timestamp()
    sketch_m_2 = (1. / (1. + epsilon)) * AmsSketch.estimate_euc_norm_squared(sketch_avg)
    end_time_calc = tf.timestamp()

    return local_update_norm_avg - sketch_m_2, end_time_com - start_time_com, calc_duration + end_time_calc - start_time_calc
