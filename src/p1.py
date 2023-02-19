import pandas as pd
from src.p1_functions import *
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

csv_file = "../drybeans.csv"

# Part1

# Read csv
df = pd.read_csv(csv_file)
# print(df)

# Check outliers

# df['Area'] = remove_outliers(df, 'Area')
# df = remove_outliers(df)
# print(df)

# Check missing values
# check_missing_values(df)

# Check and delete duplicated rows
df = check_and_delete_duplicated_rows(df)
# print("df length after removing = ", len(df))

x = df.loc[:, 'Area':'ShapeFactor4']
df.loc[:, 'Class'] = LabelEncoder().fit_transform(df.loc[:, 'Class'])
print("Count negative number = ", str(df.agg(lambda x: sum(x < 0)).sum()))
y = df.loc[:, 'Class']
# print(x)
# print(y)

# Standardization
x_std = StandardScaler().fit_transform(x)
# Normalization
# x_nor = MinMaxScaler().fit_transform(x)
# print(x_nor)

# 0.75 for train 0.25 for test and every time random
x_train, x_test, y_train, y_test = train_test_split(x_std, y, stratify=y)
# print(x_train)
# print(y_train)

# Part2

model = LogisticRegression(penalty="none", class_weight=None)
# model = LogisticRegression(penalty="none", class_weight='balanced')
model.fit(x_train, y_train)
pred = model.predict(x_test)
# print('pred = ', pred)
# print('decision_function = ', model.decision_function(x_test))
# print('predict_proba = ', model.predict_proba(x_test))

# Part3

output_result(y_test, pred)

#%%

