import tensorflow as tf

def tensor_list_to_vector(tensor_list):
    return tf.concat([tf.reshape(var, [-1]) for var in tensor_list], axis=0)

def tensor_to_tensor_list(tensor):
    tensor_list = []
    tensor_list.append(tensor)
    return tensor_list