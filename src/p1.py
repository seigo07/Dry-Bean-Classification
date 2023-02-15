import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_score, recall_score, \
    f1_score, classification_report

# Part1

df = pd.read_csv("../drybeans.csv")

# print(df)
# print("Count non-null values per row and column")
# print("df.isnull().all().sum() = " + str(df.isnull().all().sum()))
# print("Count non-nan values per row and column")
# print("df.isna().all().sum() = " + str(df.isna().all().sum()))
# print("Check the values whether contains at least one missing value per row and column and count the number of them")
# print("df.isnull().any().sum() = " + str(df.isnull().any().sum()))
# print("Count values which are not null and nan per row and column")
# print("df.count() = ")
# print(df.count())

x = df.loc[:, 'Area':'ShapeFactor4']
df.loc[:, 'Class'] = LabelEncoder().fit_transform(df.loc[:, 'Class'])
y = df.loc[:, 'Class']
# print(x)
# print(y)

# Standardization
x_std = StandardScaler().fit_transform(x)
# Normalization
# x_std = MinMaxScaler().fit_transform(x)
# 0.75 for train 0.25 for test and every time random
x_train, x_test, y_train, y_test = train_test_split(x_std, y, stratify=y)
# print(x_train)
# print(y_train)

# Part2

model = LogisticRegression(penalty="none", class_weight=None).fit(x_train, y_train)
pred = model.predict(x_test)
print('pred = ', pred)
print('decision_function = ', model.decision_function(x_test))
print('predict_proba = ', model.predict_proba(x_test))

print('classification accuracy = ', accuracy_score(y_test, pred))
# print('accuracy = ', accuracy_score(y_test, pred.round(), normalize=True))
print('balanced accuracy = ', balanced_accuracy_score(y_test, pred))
print('confusion matrix = \n', confusion_matrix(y_test, pred))
print('precision = ', precision_score(y_test, pred, average=None))
print('precision micro = ', precision_score(y_test, pred, average='micro'))
print('precision macro = ', precision_score(y_test, pred, average='macro'))
print('recall = ', recall_score(y_test, pred, average=None))
print('recall micro = ', recall_score(y_test, pred, average='micro'))
print('recall macro = ', recall_score(y_test, pred, average='macro'))
# print('f1 score = ', f1_score(y_test, pred, average=None))
# print('classification report = ', classification_report(y_test, pred))

# sum of diagonal elements of confusion matrix / sum of all elements of the confusion matrix
c = confusion_matrix(y_test, pred)
accuracy = c.trace() / c.sum()
print('classification accuracy = ', accuracy)

# sum of recall_score / size of recall_score
r = recall_score(y_test, pred, average=None)
accuracy = r.sum() / r.size
print('balanced accuracy = ', accuracy)

#%%

