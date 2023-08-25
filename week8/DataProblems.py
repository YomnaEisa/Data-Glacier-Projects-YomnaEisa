import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('Bank Marketing Datasets/bank-additional/bank-additional-full.csv', sep=';')
print(df.head(5))

# Check if null values exist
print(df.isnull())

# Create a dictionary to store the column names and their respective sums of "unknown" values
column_sums = {}

# Loop through each column in the DataFrame
for column in df.columns:
    if "unknown" in df[column].values:
        sum_unknown = (df[column] == "unknown").sum()
        column_sums[column] = sum_unknown

# Print the total for each column
for column, sum_unknown in column_sums.items():
    print(f"Column '{column}' has {sum_unknown} 'unknown' values.")


# Use describe to get summary statistics
summary = df.describe()
print(summary)

# Through the graph, we can see that the column 'duration' has outliners 
df.boxplot()
plt.show()

# Select the column for which you want to detect outliers to calculate the sum of outliners
column = df['duration']

# Calculate the first quartile (Q1)
Q1 = np.percentile(column, 25)

# Calculate the third quartile (Q3)
Q3 = np.percentile(column, 75)

# Calculate the interquartile range (IQR)
IQR = Q3 - Q1

# Define a threshold for identifying outliers (e.g., 1.5 times the IQR)
threshold = 2.5

# Calculate the lower bound for outliers
lower_bound = Q1 - threshold * IQR

# Calculate the upper bound for outliers
upper_bound = Q3 + threshold * IQR

# Use boolean indexing to identify outliers in the column
outliers = (column < lower_bound) | (column > upper_bound)

# Count the number of outliers
num_outliers = outliers.sum()

print("Number of outliers:", num_outliers)

# Check if the data is inbalance
target = df['y']
class_proportions = target.value_counts(normalize=True)
print(class_proportions)


