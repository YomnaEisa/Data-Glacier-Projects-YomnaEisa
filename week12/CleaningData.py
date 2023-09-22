import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

df = pd.read_csv('Bank Marketing Datasets/bank-additional/bank-additional-full.csv', sep=';')
print(df.head(5))


  # Removing rows with 'unknown' values in the two columns 'marital' and 'job'

drop_unknown_rows_marital = df.drop(df[(df['marital'] == 'unknown')].index, inplace=True)
drop_unknown_rows_job = df.drop(df[(df['job'] == 'unknown')].index, inplace=True)
print(df.head(5))
 
 # Replacing all the 'unknown' values in coulmns 'housing' and 'loan' with mode for each coulmn
columns_to_replace_unknown = ['housing', 'loan']

for column in columns_to_replace_unknown:
    mode_value = df[column].mode()[0]
    df[column] = df[column].replace('unknown', mode_value)


print(df.head(5))



''' For removing the outliners, we’ll use the Z-Score Method and the IQR '''
Q1 = df['duration'].quantile(0.25)
Q3 = df['duration'].quantile(0.75)

IQR = Q3 - Q1

# Calculate the upper bound
upper_bound = Q3 + 1.5 * IQR

# Remove outliers 
df = df[df['duration'] <= upper_bound]

# Removing outliners using Z-score method
z_scores = stats.zscore(df['duration'])

# Define a threshold for Z-Scores
z_score_threshold = 3  # You can adjust this threshold as needed

# Filter the DataFrame to keep rows with Z-Scores within the threshold
df = df[(z_scores <= z_score_threshold)]




''' Balancing the dataset using undersampling '''
# Separate majority and minority classes based on 'y'
df_majority = df[df['y'] == 0]
df_minority = df[df['y'] == 1]

# Determine the number of samples to draw (use the size of the minority class)
n_samples = len(df_minority)

# Check if there are samples to downsample (n_samples should be greater than 0)
if n_samples > 0:
    # Downsample the majority class
    df_majority_downsampled = resample(df_majority, replace=False, n_samples=n_samples, random_state=42)

    # Combine the minority class with the downsampled majority class
    df_balanced = pd.concat([df_majority_downsampled, df_minority])
else:
    # If there are no minority samples, the DataFrame remains unchanged
    df_balanced = df.copy()

# Display the updated shape of the DataFrame
print(df.shape)



# Using Logistic Regression to remove 'unknown' values in the two columns 'education' and 'default'
# Replace 'unknown' with NaN for 'education' and 'default' columns
df.loc[df['education'] == 'unknown', 'education'] = np.nan
df.loc[df['default'] == 'unknown', 'default'] = np.nan

# Separate data into known and unknown values
df_known = df.dropna(subset=['education', 'default'])
df_unknown = df[df['education'].isnull() | df['default'].isnull()]

# Prepare features and labels for Logistic Regression
X_known = df_known[['age']]
y_known_education = df_known['education']
y_known_default = df_known['default']

# Scale the input features
scaler = StandardScaler()
X_known_scaled = scaler.fit_transform(X_known)

# Create and fit Logistic Regression models
model_education = LogisticRegression(max_iter=1000)
model_default = LogisticRegression(max_iter=1000)

model_education.fit(X_known_scaled, y_known_education)
model_default.fit(X_known_scaled, y_known_default)

# Predict 'education' and 'default' for unknown values
X_unknown = df_unknown[['age']]
X_unknown_scaled = scaler.transform(X_unknown)

df_unknown['predicted_education'] = model_education.predict(X_unknown_scaled)
df_unknown['predicted_default'] = model_default.predict(X_unknown_scaled)

# Replace 'unknown' values in the original DataFrame with predictions
df.loc[df['education'].isnull(), 'education'] = df_unknown['predicted_education']
df.loc[df['default'].isnull(), 'default'] = df_unknown['predicted_default']


# Display the updated DataFrame
print(df.head(5))


 # Checking if all the rows containing the value 'unknown' are droped
column_sums = {}
for column in df.columns:
   if "unknown" in df[column].values:
       sum_unknown = (df[column] == "unknown").sum()
       column_sums[column] = sum_unknown

# Print the total for each column
if len(column_sums) == 0:
  print("No unknown values")
else:
  for column, sum_unknown in column_sums.items():
    print(f"Column '{column}' has {sum_unknown} 'unknown' values.") 

# Save the DataFrame to a new CSV file
df.to_csv('cleaned_bank_data.csv', index=False)