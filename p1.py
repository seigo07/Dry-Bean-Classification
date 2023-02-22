import pandas as pd
from sklearn.preprocessing import LabelEncoder

from p1_functions import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# This file includes codes which were used in the process and comment out
# or not used in the end to show the process for this assignment.

# Part1: Data Processing

# (1-a)

CSV_FILE = "drybeans.csv"
RANDOM_STATE = 42

# Read csv
df = pd.read_csv(CSV_FILE)
# print(df)

# plot correlation between features
# plot_correlation(df)

# Store the original Class values for plot
target_names = df.loc[:, 'Class'].unique()

# Check outliers
# df['Area'] = remove_outliers(df, 'Area')
# df = remove_outliers(df)
# print(df)

# Check missing values
# check_missing_values(df)

# Check and delete duplicated rows
df = check_and_delete_duplicated_rows(df)

# (1-c)

# Splitting data into x and y
x = df.loc[:, 'Area':'ShapeFactor4']
# Encoding categorical values to numbers.
df.loc[:, 'Class'] = LabelEncoder().fit_transform(df.loc[:, 'Class'])
# Check the existence of negative numbers (No need to remove or replace them because there is no such values)
print("Count negative number = ", str(x.agg(lambda z: sum(z < 0)).sum()))
y = df.loc[:, 'Class']
#
# (1-b)

# 0.75 for train 0.25 for test and every time random
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=RANDOM_STATE)

# Select features
x_train, y_train, columns, std = select_features(x_train, y_train, None, None)

# (1-c, 1-d)

# Standardization
# x_train, x_test = standardization(x_train, x_test)

# Part2: Training

# (2-a, 2-b)

model2a = LogisticRegression(penalty="none", class_weight=None, max_iter=1000, tol=1e-1)
model2a.fit(x_train, y_train)
evaluate_model(model2a, x_train, y_train, target_names)

# (2-c)

model2c = LogisticRegression(penalty="none", class_weight='balanced', max_iter=1000, tol=1e-1)
model2c.fit(x_train, y_train)
evaluate_model(model2c, x_train, y_train, target_names)

# Part3: Evaluation

# (3-a, 3-b, 3-c)

x_test, y_test, columns, std = select_features(x_test, y_test, columns, std)
# evaluate_model(model2a, x_test, y_test, target_names)
# evaluate_model(model2c, x_test, y_test, target_names)

# Part4: Advanced Tasks

# Part4-a
# model3 = LogisticRegression(penalty="l2", class_weight='balanced', max_iter=1000, tol=1e-1)
# model3.fit(x_train, y_train)
# pred2 = model3.predict(x_test)
# output_result(y_test, pred2)
