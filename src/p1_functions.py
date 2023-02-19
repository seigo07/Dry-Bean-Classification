# def remove_outliers(df):
#     for col in df.columns:
#         # Exclude categorical variables
#         if col == "Class":
#             break
#
#         # Review descriptive statistics
#         # print(df[col].describe())
#
#         # Show graph
#         # sns.distplot(df[col])
#         # sns.boxplot(df['Area'])
#         # sns.scatterplot(x="Area", data=df)
#
#         # Specify the quantile
#         # Find the value at 50% of the second quartile of the median
#         q_50 = df[col].quantile(0.5)
#         # print("The value at 50% of the second quartile of the median = ", q_50)
#
#         # Get 95% quantile values and remove outliers
#         q_95 = df[col].quantile(0.95)
#         # print("The value at 95% of the second quartile of the median = ", q_95)
#
#         # Extract data with values less than 95% order
#         new_df = df.query(f'{col} < @q_95')
#         df[col] = new_df[col]
#
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
