from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_score, recall_score, \
    classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# This file includes codes which were used in the process and comment out
# or not used in the end to show the process for this assignment.

# Function for remove outliers
def remove_outliers(df):
    for col in df.columns:
        # Exclude categorical variables
        if col == "Class":
            break
        if (df[col].mean() + (df[col].std()) * 3) > df[col].max():
            break
        # Specify the quantile
        # Find the value at 50% of the second quartile of the median
        q_50 = df[col].quantile(0.5)
        # Get 95% quantile values
        q_95 = df[col].quantile(0.95)
        # Extract data with values less than 95% order
        new_df = df.query(f'{col} < @q_95')
        df[col] = new_df[col]
    return df


# Function for plotting correlation between features
def plot_correlation(df):
    corr = df.corr()
    plt.figure(figsize=(16, 16))
    sns.heatmap(corr, annot=True, square=True)
    plt.tight_layout()
    plt.show()

# Function for checking and deleting missing values
def check_and_delete_missing_values(df):
    print("Count non-null values per row and column")
    print("df.isnull().all().sum() = " + str(df.isnull().all().sum()))
    print("Count non-nan values per row and column")
    print("df.isna().all().sum() = " + str(df.isna().all().sum()))
    print(
        "Check the values whether contains at least one missing value per row and column and count the number of them")
    print("df.isnull().any().sum() = " + str(df.isnull().any().sum()))
    print("Count values which are not null and nan per row and column")
    print("df.count() = ")
    print(df.count())
    if df.isnull().all().sum() > 0 or df.isna().all().sum() > 0:
        # Remove rows with null or NaN on all columns
        return df.dropna()


# Function for checking and deleting duplicated rows
def check_and_delete_duplicated_rows(df):
    print("Check rows with duplicates on all columns")
    print(df[df.duplicated()])
    print("df length = ", len(df))
    print("Deplicate value_counts = ")
    print(df.duplicated().value_counts())
    if df.duplicated().value_counts().get(True, 0) > 0:
        # Remove rows with duplicates on all columns
        return df.drop_duplicates()


# Function for scaling (standardization)
def standardization(x_train, x_test):
    scaler = StandardScaler()
    # fit on the training dataset
    scaler.fit(x_train)
    # scale the training dataset
    x_train = scaler.transform(x_train)
    # scale the test dataset
    x_test = scaler.transform(x_test)
    return x_train, x_test


# Function for scaling (normalization)
# def normalization(x_train, x_test):
#     scaler = MinMaxScaler()
#     # fit on the training dataset
#     scaler.fit(x_train)
#     # scale the training dataset
#     x_train = scaler.transform(x_train)
#     # scale the test dataset
#     x_test = scaler.transform(x_test)
#     return x_train, x_test


# Function for output solutions
def evaluate_model(model, x_train, y_train, x_test, y_test, target_names):

    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    # (2-d)
    print('pred = ', pred)
    print('decision_function = ', model.decision_function(x_test))
    print('predict_proba = ', model.predict_proba(x_test))

    # (3-a)
    print('classification accuracy = ', accuracy_score(y_test, pred))
    # (3-b)
    print('balanced accuracy = ', balanced_accuracy_score(y_test, pred))
    # (3-c)
    print('confusion matrix = \n', confusion_matrix(y_test, pred))
    # (3-d)
    print('precision micro = ', precision_score(y_test, pred, average='micro'))
    print('precision macro = ', precision_score(y_test, pred, average='macro'))
    print('recall micro = ', recall_score(y_test, pred, average='micro'))
    print('recall macro = ', recall_score(y_test, pred, average='macro'))
    print('classification report: \n', classification_report(y_test, pred))

    # (3-a)
    # sum of diagonal elements of confusion matrix / sum of all elements of the confusion matrix
    c = confusion_matrix(y_test, pred)
    accuracy = c.trace() / c.sum()
    # print('classification accuracy = ', accuracy)

    # (3-b)
    # sum of recall_score / size of recall_score
    r = recall_score(y_test, pred, average=None)
    accuracy = r.sum() / r.size
    # print('balanced accuracy = ', accuracy)
    print("\n")

    # plot_matrix(target_names, pred, y_test)


# Function for plot confusion matrix
def plot_matrix(target_names, y_pred, y_test):
    cm = confusion_matrix(y_pred, y_test)
    cmp = ConfusionMatrixDisplay(cm, display_labels=target_names)
    cmp.plot(cmap=plt.cm.Blues)
    plt.show()
