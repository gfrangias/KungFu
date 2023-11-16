import argparse
import tensorflow as tf
import csv, time, os
from kungfu._utils import map_maybe
from kungfu.python import current_cluster_size, current_rank
from kungfu.tensorflow.ops import group_subset_all_reduce, set_tree, broadcast, subset_all_reduce
from kungfu.tensorflow.optimizers import (PairAveragingOptimizer,
                                          SynchronousAveragingOptimizer,
                                          SynchronousSGDOptimizer,
                                          MySynchronousSGDOptimizer)

parser = argparse.ArgumentParser(description='KungFu mnist example.')
parser.add_argument('--kf-optimizer',
                    type=str,
                    default='my-sync-sgd',
                    help='available options: sync-sgd, async-sgd, sma, my-sync-sgd')
parser.add_argument('--fda',
                    type=str,
                    default='naive',
                    help='available options: naive, linear, sketch')
parser.add_argument('--threshold',
                    type=str,
                    default='0.8')
parser.add_argument('--batches',
                    type=str,
                    default='5000')
args = parser.parse_args()

(mnist_images, mnist_labels), _ = \
    tf.keras.datasets.mnist.load_data(path='mnist-%d.npz' % current_rank())

dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(mnist_images[..., tf.newaxis] / 255.0,
             tf.float32), tf.cast(mnist_labels, tf.int64)))

dataset = dataset.repeat().shuffle(10000).batch(64)

mnist_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, [3, 3], activation='relu'),
    tf.keras.layers.Conv2D(64, [3, 3], activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

loss = tf.losses.SparseCategoricalCrossentropy()

# KungFu: adjust learning rate based on number of GPUs.
# opt = tf.keras.optimizers.SGD(0.001 * current_cluster_size())
opt = tf.compat.v1.train.AdamOptimizer(0.001 * current_cluster_size())

steps_since_sync = tf.Variable(0, dtype=tf.int32)
total_syncs = tf.Variable(0, dtype=tf.int32)
threshold = float(args.threshold)
threshold_str = str(args.threshold)
threshold = tf.constant(threshold)
method = args.fda
batches = int(args.batches)
np = current_cluster_size()

# Set a star-topology with node 0 as the root
tree = tf.constant([0, 0, 2], dtype=tf.int32)
tree2 = tf.constant([0, 0, 2], dtype=tf.int32)
set_tree_op = set_tree(broadcast(tree))
syncs_hist = []
batch_hist = []
loss_hist = []

# KungFu: wrap tf.compat.v1.train.Optimizer.
if args.kf_optimizer == 'sync-sgd':
    opt = SynchronousSGDOptimizer(opt)
elif args.kf_optimizer == 'async-sgd':
    opt = PairAveragingOptimizer(opt)
elif args.kf_optimizer == 'sma':
    opt = SynchronousAveragingOptimizer(opt)
elif args.kf_optimizer == 'my-sync-sgd':
    opt = MySynchronousSGDOptimizer(opt)
else:
    raise RuntimeError('Unknown KungFu optimizer')

@tf.function
def tensor_list_to_vector(tensor_list):
    return tf.concat([tf.reshape(var, [-1]) for var in tensor_list], axis=0)

@tf.function
def tensor_to_tensor_list(tensor):
    tensor_list = []
    tensor_list.append(tensor)
    return tensor_list

# Compute the divergence using the 2-norm for Naive FDA
@tf.function
def compute_divergence_2_norm(w_t0, local_model):
    w_t0_vector = tensor_list_to_vector(w_t0)
    local_model_vector = tensor_list_to_vector(local_model)
    norm_2 =  tf.norm(local_model_vector - w_t0_vector, 2)
    return tensor_to_tensor_list(norm_2)

# Compute a random xi for the first synchronization
def random_xi(model):
    random_tensor = tf.random.normal(shape=(len(tensor_list_to_vector(model)),), dtype=tf.float32)
    xi_tensor = tf.divide(random_tensor, tf.norm(random_tensor))
    return tensor_to_tensor_list(xi_tensor)

# Compute the "xi" unit vector
def compute_xi(second_w_t0, w_t0):
    w_t0_tensor = tensor_list_to_vector(w_t0) 
    second_w_t0_tensor = tensor_list_to_vector(second_w_t0)
    sync_models_diff_tensor = tf.abs(w_t0_tensor - second_w_t0_tensor)
    xi = tf.divide(sync_models_diff_tensor, tf.norm(sync_models_diff_tensor))
    return tensor_to_tensor_list(xi)

# Compute the dot product of xi and the model difference
@tf.function
def compute_xi_dot_diff(w_t0, local_model, xi):
    w_t0_tensor = tensor_list_to_vector(w_t0) 
    local_model = tensor_list_to_vector(local_model)
    models_diff_tensor = local_model - w_t0_tensor
    xi_tensor = tensor_list_to_vector(xi)
    xi_dot_diff = tf.tensordot(xi_tensor, models_diff_tensor, axes=1)
    return tensor_to_tensor_list(xi_dot_diff)

# Check if the divergence satisfies the RTC
@tf.function
def rtc_check(divergence):
    global threshold
    if tf.math.greater(tf.cast(divergence, tf.float32),tf.cast(threshold, tf.float32)):
        return True
    else:
        return False
    
@tf.function
def increment_counter():
    steps_since_sync.assign_add(1)
    
@tf.function
def reset_counter():
    steps_since_sync.assign(0)

# Save time duration
def save_duration(type , duration, method, batches, nodes, threshold, description):
    # Specify the CSV file name
    csv_file_name = "./csv_output/"+type+"."+method+"."+batches+".np"+nodes+".thr"+threshold+"."+description+".csv"

    with open(csv_file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        if not os.path.exists(csv_file_name):
            writer.writerow(['Duration'])
        
        writer.writerow([duration])

# Save output data for synchronizations and loss per batch
def save_csv(type ,batch_list, data_list, method, batches, nodes, threshold, description):
    # Specify the CSV file name
    csv_file_name = "./csv_output/"+type+"."+method+"."+batches+".np"+nodes+".thr"+threshold+"."+description+".csv"

    # Open the CSV file in write mode and overwrite if it exists
    with open(csv_file_name, mode='w', newline='') as csv_file:
        # Create a CSV writer
        csv_writer = csv.writer(csv_file)
        
        # Write each element of data_list as a separate row
        for batch_val, data_val in zip(batch_list, data_list):
            csv_writer.writerow([batch_val, data_val])

# Training step for Naive FDA
@tf.function
def training_step_naive(images, labels, first_batch, w_t0):
    increment_counter()
    # Perform TensorFlow GradientTape
    with tf.GradientTape() as tape:
        probs = mnist_model(images, training=True)
        loss_value = loss(labels, probs)

    # Find the local gradients 
    grads = tape.gradient(loss_value, mnist_model.trainable_variables)

    # Apply the new gradients locally
    opt.apply_gradients(zip(grads, mnist_model.trainable_variables))
    updated_w_t0 = w_t0

    if not first_batch:
        local_divergence = compute_divergence_2_norm(w_t0, mnist_model.trainable_variables)
        tf.print("Local divergence: ", local_divergence)
        summed_divergences = group_subset_all_reduce(local_divergence, tree2)
        tf.print("Summed divergence: ", summed_divergences)
        np = tf.cast(current_cluster_size(), tf.float32)
        averaged_divergence = map_maybe(lambda d: d / np, summed_divergences)
    else:
        averaged_divergence = 0

    if rtc_check(averaged_divergence) or first_batch:
        #if current_rank() == 0:
        #    tf.print("Steps since last sync: ", steps_since_sync)
        #    tf.print("Average divergence: ", averaged_divergence)
        reset_counter()
        total_syncs.assign_add(1)
        summed_models = group_subset_all_reduce(mnist_model.trainable_variables, tree2)
        # Cast the number of workers to tf.float32
        np = tf.cast(current_cluster_size(), tf.float32)
        # Reduce the gradients of the current node based on the average of all nodes
        averaged_models = map_maybe(lambda g: g / np, summed_models)
        for i, variable in enumerate(mnist_model.trainable_variables):
            variable.assign(averaged_models[i])
        updated_w_t0 = mnist_model.trainable_variables
        
    # KungFu: broadcast is done after the first gradient step to ensure optimizer initialization.
    if first_batch:
        from kungfu.tensorflow.initializer import broadcast_variables
        broadcast_variables(mnist_model.variables)
        broadcast_variables(opt.variables())

    return loss_value, updated_w_t0

# Training step for Linear FDA
@tf.function
def training_step_linear(images, labels, first_batch, w_t0, xi):
    increment_counter()
    # Perform TensorFlow GradientTape
    with tf.GradientTape() as tape:
        probs = mnist_model(images, training=True)
        loss_value = loss(labels, probs)

    # Find the local gradients 
    grads = tape.gradient(loss_value, mnist_model.trainable_variables)

    # Apply the new gradients locally
    opt.apply_gradients(zip(grads, mnist_model.trainable_variables))
    updated_w_t0 = w_t0

    if not first_batch:
        local_divergence = compute_divergence_2_norm(w_t0, mnist_model.trainable_variables)
        local_xi_dot_diff = compute_xi_dot_diff(w_t0, mnist_model.trainable_variables, xi)

        summed_divergences = group_subset_all_reduce(local_divergence, tree2)
        summed_xi_dot_diff = group_subset_all_reduce(local_xi_dot_diff, tree2)
        np = tf.cast(current_cluster_size(), tf.float32)
        averaged_divergence = map_maybe(lambda d: d / np, summed_divergences)
        averaged_xi_dot_diff = map_maybe(lambda d: d / np, summed_xi_dot_diff)
        #tf.print(averaged_xi_dot_diff)
        rtc_expr = averaged_divergence - tf.square(averaged_xi_dot_diff)
    else:
        rtc_expr = 0

    #tf.print(rtc_expr)
    if rtc_check(rtc_expr) or first_batch:
        #if current_rank() == 0:
        #    tf.print("Steps since last sync: ", steps_since_sync)
        #    tf.print("Average divergence: ", rtc_expr)
        reset_counter()
        total_syncs.assign_add(1)
        summed_models = group_subset_all_reduce(mnist_model.trainable_variables, tree2)
        # Cast the number of workers to tf.float32
        np = tf.cast(current_cluster_size(), tf.float32)
        # Reduce the gradients of the current node based on the average of all nodes
        averaged_models = map_maybe(lambda g: g / np, summed_models)
        for i, variable in enumerate(mnist_model.trainable_variables):
            variable.assign(averaged_models[i])
        # Compute the difference between the latest updated model and the second latest updated model
        if first_batch:
            xi = random_xi(mnist_model.trainable_variables)
        else:
            xi = compute_xi(updated_w_t0, mnist_model.trainable_variables)
        updated_w_t0 = mnist_model.trainable_variables

    # KungFu: broadcast is done after the first gradient step to ensure optimizer initialization.
    if first_batch:
        from kungfu.tensorflow.initializer import broadcast_variables
        broadcast_variables(mnist_model.variables)
        broadcast_variables(opt.variables())

    return loss_value, updated_w_t0, xi

_ = mnist_model(tf.random.normal((1, 28, 28, 1), dtype=tf.float32))
#print(mnist_model.summary())
w_t0 = [0.0] * len(mnist_model.trainable_variables)
xi = [0.0]
# KungFu: adjust number of steps based on number of GPUs.
start_time = time.time()

for batch, (images, labels) in enumerate(
        dataset.take(batches // current_cluster_size())):
    
    #print("Step #"+str(batch))
    if args.fda == 'naive':
        loss_value, w_t0 = training_step_naive(images, labels, batch == 0, w_t0)
    elif args.fda == 'linear':
        loss_value, w_t0, xi = training_step_linear(images, labels, batch == 0, w_t0, xi)
    elif args.fda == 'sketch':
        print("Sketch not implemented yet!")
        exit()
    else:
        print("FDA method \""+args.fda+"\" isn't available!")
        exit()

    if batch % 10 == 0 and current_rank() == 0:
        batch_hist.append(batch)
        syncs_hist.append(total_syncs.numpy())
        loss_hist.append(loss_value.numpy())
        print('Step #%d\tLoss: %.6f and %d synchronizations occured.' % (batch, loss_value, syncs_hist[-1]))

end_time = time.time()
elapsed_time = end_time - start_time
if current_rank() == 0:
    save_duration("duration", elapsed_time, method, str(batches), str(current_cluster_size()), threshold_str, "4x1.okeanos_star")
#    save_csv("sync", batch_hist, syncs_hist, method, str(batches), str(current_cluster_size()), threshold_str, "4x2.okeanos")
#    save_csv("loss", batch_hist, loss_hist, method, str(batches), str(current_cluster_size()), threshold_str, "4x2.okeanos")
