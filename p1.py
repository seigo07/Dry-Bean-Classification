import pandas as pd
from p1_functions import *
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Part1

# (1-a)

CSV_FILE = "drybeans.csv"
RANDOM_STATE = 42

# Read csv
df = pd.read_csv(CSV_FILE)

print(df)

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
# pd = pd.get_dummies(df.loc[:, 'Class'])
# df.loc[:, 'Class'] = LabelEncoder().fit_transform(df.loc[:, 'Class'])
# Check the existence of negative numbers (No need to remove or replace them because there is no such values)
print("Count negative number = ", str(x.agg(lambda z: sum(z < 0)).sum()))
y = df.loc[:, 'Class']

# (1-b)

# 0.75 for train 0.25 for test and every time random
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=RANDOM_STATE)

# (1-c, 1-d)

# Standardization
x_train, x_test = standardization(x_train, x_test)

# Part2

# (2-a, 2-b, 2-c)

model = LogisticRegression(penalty="none", class_weight=None)
# model = LogisticRegression(penalty="none", class_weight=None, max_iter=10000, tol=1e-1)
# model = LogisticRegression(penalty="none", class_weight='balanced', max_iter=10000, tol=1e-1)
model.fit(x_train, y_train)
pred = model.predict(x_test)

# (2-d)

# print('pred = ', pred)
# print('decision_function = ', model.decision_function(x_test))
# print('predict_proba = ', model.predict_proba(x_test))

# Part3

output_result(y_test, pred)
# plot_matrix(target_names, pred, y_test)

# model2 = LogisticRegression(penalty="none", class_weight='balanced', max_iter=10000, tol=1e-1)
# model2.fit(x_train, y_train)
# pred2 = model2.predict(x_test)
# output_result(y_test, pred2)
# plot_matrix(target_names, pred2, y_test)

# Part4-a
# model3 = LogisticRegression(penalty="l2", class_weight='balanced', max_iter=10000, tol=1e-1)
# model3.fit(x_train, y_train)
# pred2 = model3.predict(x_test)
# output_result(y_test, pred2)
