import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import re

sys.argv = ["visualize_result.py", "./csv_output/loss.linear.20000.np4.thr2.4x1.okeanos.csv"]
# Check if the correct number of command-line arguments are provided
if len(sys.argv) != 2:
    print("Usage: python visualize_result.py <file_name>")
    sys.exit(1)

parts = sys.argv[1].split('.')
filename_skeleton = '.'.join(parts[3:7])
print(filename_skeleton)

# Read the CSV files from command-line arguments
sync_file_naive = "./csv_output/sync.naive."+filename_skeleton+".csv"
sync_file_linear = "./csv_output/sync.linear."+filename_skeleton+".csv"

loss_file_naive = "./csv_output/loss.naive."+filename_skeleton+".csv"
loss_file_linear = "./csv_output/loss.linear."+filename_skeleton+".csv"

# Extract filenames without extensions
sync_image_file = "./image_output/sync."+filename_skeleton+".png"
loss_image_file = "./image_output/loss."+filename_skeleton+".png"

sync_data_naive = pd.read_csv(sync_file_naive, header=None)
sync_data_linear = pd.read_csv(sync_file_linear, header=None)

loss_data_naive = pd.read_csv(loss_file_naive, header=None)
loss_data_linear = pd.read_csv(loss_file_linear, header=None)

np = int(re.search(r'np(\d+)', filename_skeleton).group(1))

batches = sync_data_naive[0]*np
sync_data_naive = sync_data_naive[1]
sync_data_linear = sync_data_linear[1]
loss_data_naive = loss_data_naive[1]
loss_data_linear = loss_data_linear[1]

plt.figure()

plt.plot(batches, sync_data_naive, label='Naive FDA')
plt.plot(batches, sync_data_linear, label='Linear FDA')

plt.xlabel('Steps / Batch Count')
plt.ylabel('Synchronizations')
plt.title('Synchronizations per Batch Count')

plt.legend()

plt.savefig(sync_image_file)

plt.show()
