import os, argparse, time

import tensorflow as tf
if tf.config.list_physical_devices('GPU'):
        for gpu in tf.config.experimental.list_physical_devices('GPU'):
                tf.config.experimental.set_memory_growth(gpu,True)
                
from kungfu.python import current_cluster_size, current_rank
from kungfu.tensorflow.ops import group_all_reduce, all_reduce
from kungfu._utils import map_maybe

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from models.lenet5 import create_lenet5
from models.adv_cnn import create_adv_cnn
from dataset.mnist import create_dataset

parser = argparse.ArgumentParser(description='KungFu mnist example.')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to run experiment')
parser.add_argument('--model', type=str, default="lenet5",
                    help='available options: lenet5, adv_cnn')
parser.add_argument('--batch', type=int, default=64,
                    help='batch size')
parser.add_argument("-l", action="store_true", help="Enable logs")
parser.add_argument("--threshold", type=float, default= 1.1,
                     help="synchronization threshold")
parser.add_argument("--clients_distr", type=str, 
                    default='[]', help="Client and node configuration")
parser.add_argument("--optimizer", type=str, 
                    default='Adam', help='Optimizer used')
parser.add_argument("--algorithm", type=str, default= "synchronous",
                     help="type of experiment algorithm")
args = parser.parse_args()

# Load mnist dataset
train_dataset, test_dataset, epoch_steps, epoch_steps_float = \
    create_dataset(args.epochs, args.batch, current_cluster_size(), current_rank())

# Create selected model
if args.model == "lenet5":
    train_model, loss_fun = create_lenet5(input_shape=(28,28,1), num_classes=10)
elif args.model == "adv_cnn":
    train_model, loss_fun = create_adv_cnn(input_shape=(28,28,1), num_classes=10)

com_duration = tf.constant(0, dtype=tf.float64)
start_timestamp = tf.timestamp()
start_time = time.time()
for _ in range(2000):
    average = all_reduce(com_duration)
end_time = time.time()
end_timestamp = tf.timestamp()

print("Time")
print(end_time-start_time)
print("Timestamp")
print((end_timestamp-start_timestamp).numpy())