import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_score, recall_score, \
    classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

# This file includes codes which were used in the process and comment out
# or not used in the end to show the process for this assignment.

CORRELATION_MIN = 0.3
CORRELATION_MAX = 0.9


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


# Function for feature selection
def feature_selection(x, y, columns, std):
    selected_features, columns = select_features(x, y, columns)
    df, std = standardization(selected_features, std)
    return df, y, columns, std


# Function for select features
def select_features(x, y, columns):
    # In the case of train
    if columns is None:
        # Get select features
        selected_features = get_features(x, y, y.name)
        columns = list(selected_features.columns)
    # In the case of test
    else:
        # Get select features from stored select_features
        selected_features = x[x.columns.intersection(columns)]
    return selected_features, columns


# Function for scaling (standardization)
def standardization(x, std):
    # In the case of train
    if std is None:
        std = StandardScaler().fit(x)
    # scale the input dataset
    df = pd.DataFrame(std.transform(x), columns=x.columns)
    return df, std


# Function for get possible features
def get_features(x, y, target_names):
    # Check correlation between features and remove them whose correlation is weak to the output.
    correlations, df = select_features_based_on_min(x, y)
    # Check correlation between features and remove them whose correlation is strong for linear independence.
    df = select_features_based_on_max(df, correlations, target_names)
    return df


# Function for select features based on CORRELATION_MIN
def select_features_based_on_min(x, y):
    # Calculate the correlation plot of the data set.
    df = pd.concat([x, y], axis=1)
    correlations = pd.concat([x, y], axis=1).corr()
    # Get correlations with y.
    target = abs(correlations[y.name])
    # Pick up features which are correlated strongly based on CORRELATION_MIN.
    features = target[target >= CORRELATION_MIN]
    # Get a dataFrame of features.
    df = df.loc[:, features.index.values.tolist()]
    df.drop(y.name, axis=1, inplace=True)
    return correlations, df


# Function for select features based on CORRELATION_MAX
def select_features_based_on_max(df, correlations, target_names):
    # Remove linearly dependant features.
    feature_corr = abs(df.corr())
    for i in range(feature_corr.shape[0]):
        for j in range(i + 1, feature_corr.shape[0]):
            # Remove the feature whose correlation is weakest with the output
            # in case of having strong correlation with each other
            if feature_corr.iloc[i, j] >= CORRELATION_MAX:
                feature_col_1 = feature_corr.columns[i]
                y_corr_1 = correlations[feature_col_1][target_names]
                feature_col_2 = feature_corr.columns[j]
                y_corr_2 = correlations[feature_col_2][target_names]
                # Drop the lesser correlated feature with the output.
                if y_corr_1 > y_corr_2:
                    if feature_col_2 in df.columns:
                        df.drop(feature_col_2, axis=1, inplace=True)
                else:
                    if feature_col_1 in df.columns:
                        df.drop(feature_col_1, axis=1, inplace=True)
                    break
    return df


# Function for output solutions
def evaluate_model(model, x, y, target_names):
    pred = model.predict(x)

    # (2-d)
    print('pred = ', pred)
    print('decision_function = ', model.decision_function(x))
    print('predict_proba = ', model.predict_proba(x))

    # (3-a)
    print('classification accuracy = ', accuracy_score(y, pred))
    # (3-b)
    print('balanced accuracy = ', balanced_accuracy_score(y, pred))
    # (3-c)
    print('confusion matrix = \n', confusion_matrix(y, pred))
    # (3-d)
    print('precision micro = ', precision_score(y, pred, average='micro'))
    print('precision macro = ', precision_score(y, pred, average='macro'))
    print('recall micro = ', recall_score(y, pred, average='micro'))
    print('recall macro = ', recall_score(y, pred, average='macro'))
    print('classification report: \n', classification_report(y, pred))

    # (3-a)
    # sum of diagonal elements of confusion matrix / sum of all elements of the confusion matrix
    c = confusion_matrix(y, pred)
    accuracy = c.trace() / c.sum()
    # print('classification accuracy by confusion_matrix = ', accuracy)

    # (3-b)
    # sum of recall_score / size of recall_score
    r = recall_score(y, pred, average=None)
    accuracy = r.sum() / r.size
    # print('balanced accuracy by recall_score = ', accuracy)
    print("\n")

    plot_matrix(target_names, pred, y)


# Function for plot confusion matrix
def plot_matrix(target_names, y_pred, y_test):
    cm = confusion_matrix(y_pred, y_test)
    cmp = ConfusionMatrixDisplay(cm, display_labels=target_names)
    cmp.plot(cmap=plt.cm.Blues)
    plt.show()
