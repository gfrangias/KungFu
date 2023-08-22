import matplotlib.pyplot as plt
import pandas as pd

# Read the CSV files
file1 = 'naive.csv'
file2 = 'linear.csv'

data1 = pd.read_csv(file1, header=None)
data2 = pd.read_csv(file2, header=None)

# Extract the data from the CSV files
x1 = data1[0]
y1 = data1[1]

x2 = data2[0]
y2 = data2[1]

# Create a new figure
plt.figure()

# Plot the data
plt.plot(x1, y1, label='Naive FDA')
plt.plot(x2, y2, label='Linear FDA')

# Add labels and title
plt.xlabel('Steps / Batch Count')
plt.ylabel('Synchronizations')
plt.title('Synchronizations per Batch Count')

# Add legend
plt.legend()

# Save the plot as an image file (e.g., PNG)
plt.savefig('plot.png')

# Show the plot
plt.show()
