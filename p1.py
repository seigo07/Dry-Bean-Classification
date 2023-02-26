from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.svm import LinearSVC
from p1_functions import *
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

# This file includes codes which were used in the process and comment out
# or not used in the end to show the process for this assignment.

# Part1: Data Processing

# (1-a)

CSV_FILE = "drybeans.csv"
RANDOM_STATE = 42
THE_LIMITED_NUMBER_OF_EPOCHS = 1000
STOP_TIMES = 1e-1

# Read csv
df = pd.read_csv(CSV_FILE)

# Store the original Class values for plot
target_names = df.loc[:, 'Class'].unique()

# Check and remove outliers
# df = remove_outliers(df)

# Check missing values
check_and_delete_missing_values(df)

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
x_train, y_train, columns, std = feature_selection(x_train, y_train, None, None)

# (1-c, 1-d)

# Part2: Training

# (2-a, 2-b)

model2a = LogisticRegression(penalty="none", class_weight=None, max_iter=THE_LIMITED_NUMBER_OF_EPOCHS, tol=STOP_TIMES)
model2a.fit(x_train, y_train)
evaluate_model(model2a, x_train, y_train, target_names)

# (2-c)

model2c = LogisticRegression(penalty="none", class_weight='balanced', max_iter=THE_LIMITED_NUMBER_OF_EPOCHS,
                             tol=STOP_TIMES)
model2c.fit(x_train, y_train)
evaluate_model(model2c, x_train, y_train, target_names)

# Part3: Evaluation

# (3-a, 3-b, 3-c)

x_test, y_test, columns, std = feature_selection(x_test, y_test, columns, std)
evaluate_model(model2a, x_test, y_test, target_names)
evaluate_model(model2c, x_test, y_test, target_names)

# Part4: Advanced Tasks

# (4-a)
# Prepare regularised and unregularised models for (4-c)
model4a = LogisticRegression(penalty="l2", class_weight=None, max_iter=THE_LIMITED_NUMBER_OF_EPOCHS, tol=STOP_TIMES)
model4c = LogisticRegression(penalty='none', class_weight=None, max_iter=THE_LIMITED_NUMBER_OF_EPOCHS, tol=STOP_TIMES)

# (4-b)
# Apply 2nd degree polynomial expansion
pf = PolynomialFeatures(degree=2, interaction_only=True)
x_train_pf = pd.DataFrame(pf.fit_transform(x_train))
x_test_pf = pd.DataFrame(pf.transform(x_test))

print("The number of features before applying a 2nd degree polynomial expansion = " + str(len(x_train.columns)))
print(
    "The number of features after applying a 2nd degree polynomial expansion = " + str(len(x_train_pf.columns)) + "\n")

# Standardization for evaluation
x_train_pf, std_pf = standardization(x_train_pf, None)
x_test_pf, std_pf = standardization(x_test_pf, std_pf)

# (4-c)
# Evalulate regularised and unregularised expansion models
model4a.fit(x_train_pf, y_train)
model4c.fit(x_train_pf, y_train)
evaluate_model(model4a, x_test_pf, y_test, target_names)
evaluate_model(model4c, x_test_pf, y_test, target_names)

# (4-d)

# Apply label_binarize to original output datasets to be multi-label
y_4d = label_binarize(df.loc[:, 'Class'], classes=df.loc[:, 'Class'].unique())
n_classes = y_4d.shape[1]

# Split original datasets into training and test
x_train_4d, x_test_4d, y_train_4d, y_test_4d = train_test_split(x, y_4d, random_state=RANDOM_STATE)

classifier = OneVsRestClassifier(
    make_pipeline(StandardScaler(), LinearSVC(max_iter=10000, tol=STOP_TIMES, random_state=RANDOM_STATE))
)
classifier.fit(x_train_4d, y_train_4d)
y_score = classifier.decision_function(x_test_4d)

# Define precision, recall, and average_precision
precision, recall, average_precision = setup_precision_recall_average_precision(n_classes, y_test_4d, y_score)

# Plot details
plot_precision_recall(precision, recall, average_precision, n_classes, target_names)