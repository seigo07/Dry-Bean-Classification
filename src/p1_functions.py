from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_score, recall_score, \
    f1_score, classification_report

# def remove_outliers(df):
#     for col in df.columns:
#         # Exclude categorical variables
#         if col == "Class":
#             break
#         if (df[col].mean() + (df[col].std()) * 3) > df[col].max():
#             break
#         # Specify the quantile
#         # Find the value at 50% of the second quartile of the median
#         q_50 = df[col].quantile(0.5)
#         # Get 95% quantile values
#         q_95 = df[col].quantile(0.95)
#         # Extract data with values less than 95% order
#         new_df = df.query(f'{col} < @q_95')
#         df[col] = new_df[col]
#     return df


def check_missing_values(df):
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


def check_and_delete_duplicated_rows(df):
    # print("Check rows with duplicates on all columns")
    # print(df[df.duplicated()])
    # print("df length = ", len(df))
    # print("Deplicate value_counts = ")
    # print(df.duplicated().value_counts())
    # Remove rows with duplicates on all columns
    return df.drop_duplicates()
    # print("df length after removing = ", len(df))


def output_result(y_test, pred):
    print('accuracy = ', accuracy_score(y_test, pred.round(), normalize=True))
    print('classification accuracy = ', accuracy_score(y_test, pred))
    print('balanced accuracy = ', balanced_accuracy_score(y_test, pred))
    print('confusion matrix = \n', confusion_matrix(y_test, pred))
    print('precision = ', precision_score(y_test, pred, average=None))
    print('precision micro = ', precision_score(y_test, pred, average='micro'))
    print('precision macro = ', precision_score(y_test, pred, average='macro'))
    print('recall = ', recall_score(y_test, pred, average=None))
    print('recall micro = ', recall_score(y_test, pred, average='micro'))
    print('recall macro = ', recall_score(y_test, pred, average='macro'))
    print('f1 score = ', f1_score(y_test, pred, average=None))
    print('classification report = ', classification_report(y_test, pred))

    # sum of diagonal elements of confusion matrix / sum of all elements of the confusion matrix
    c = confusion_matrix(y_test, pred)
    accuracy = c.trace() / c.sum()
    # print('classification accuracy = ', accuracy)

    # sum of recall_score / size of recall_score
    r = recall_score(y_test, pred, average=None)
    accuracy = r.sum() / r.size
    # print('balanced accuracy = ', accuracy)
