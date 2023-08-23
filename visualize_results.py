import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

# Check if the correct number of command-line arguments are provided
if len(sys.argv) != 3:
    print("Usage: python script_name.py <file1.csv> <file2.csv>")
    sys.exit(1)

# Read the CSV files from command-line arguments
file1 = sys.argv[1]
file2 = sys.argv[2]

# Extract filenames without extensions
filename1 = os.path.splitext(os.path.basename(file1))[0]
filename2 = os.path.splitext(os.path.basename(file2))[0]

data1 = pd.read_csv(file1, header=None)
data2 = pd.read_csv(file2, header=None)

x1 = data1[0]*8
y1 = data1[1]

x2 = data2[0]*8
y2 = data2[1]

plt.figure()

plt.plot(x1, y1, label='Naive FDA')
plt.plot(x2, y2, label='Linear FDA')

plt.xlabel('Steps / Batch Count')
plt.ylabel('Synchronizations')
plt.title('Synchronizations per Batch Count')

plt.legend()

output_filename = f'{filename1}_plot.png'
plt.savefig(output_filename)

plt.show()
