import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# One hot encode - turning categorical data into numerical
def encode(train_data, test_data, columns):
    cat_columns = columns

    df_processed = pd.get_dummies(train_data, prefix_sep="__", columns=cat_columns)
    cat_dummies = [col for col in df_processed
                   if "__" in col
                   and col.split("__")[0] in cat_columns]
    processed_columns = list(df_processed.columns[:])

    df_test_processed = pd.get_dummies(test_data, prefix_sep="__", columns=cat_columns)

    # Remove additional columns
    for col in df_test_processed.columns:
        if ("__" in col) and (col.split("__")[0] in cat_columns) and col not in cat_dummies:
            print("Removing additional feature {}".format(col))
            df_test_processed.drop(col, axis=1, inplace=True)

    for col in cat_dummies:
        if col not in df_test_processed.columns:
            print("Adding missing feature {}".format(col))
            df_test_processed[col] = 0

    return df_processed, df_test_processed, processed_columns


# filePath = filedialog.askopenfilename()  # Pop-up to choose csv location

# Creates pandas dataframe from chosen csv.
X_train = pd.read_csv('C:/Users/Ethan/Documents/School/Uni/AI/dataset_diabetes/diabetic_data_train_x.csv')
Y_train = pd.read_csv('C:/Users/Ethan/Documents/School/Uni/AI/dataset_diabetes/diabetic_data_train_y.csv')
print("\nShape of train set:", X_train.shape, Y_train.shape)

X_test = pd.read_csv('C:/Users/Ethan/Documents/School/Uni/AI/dataset_diabetes/diabetic_data_test_x.csv')
Y_test = pd.read_csv('C:/Users/Ethan/Documents/School/Uni/AI/dataset_diabetes/diabetic_data_test_y.csv')
print("\nShape of test set:", X_test.shape, Y_test.shape)

# Categorising all non float/non int data in dataset.
categories_x = ["age", "diabetesMed", "insulin", "race", "gender"]
X_train_processed, X_test_processed, columns_x = encode(X_train, X_test, categories_x)

# Normalizing data
x = X_train_processed[columns_x].values.astype(float)
min_max_scaler = preprocessing.MinMaxScaler()
X_train_scaled = min_max_scaler.fit_transform(x)
X_train_normalised = pd.DataFrame(X_train_scaled)

y = X_test_processed[columns_x].values.astype(float)
X_test_scaled = min_max_scaler.fit_transform(y)
X_test_normalised = pd.DataFrame(X_test_scaled)

# During normalisation, column names are changed to numbers. This changes column names used by k-NN model.
columns_z = []
for i in range(31):
    columns_z.append(i)

# Initializing and Fitting a k-NN model
knn = KNeighborsClassifier(n_neighbors=101)
knn.fit(X_train_normalised[columns_z], Y_train.values.ravel())

# Checking the performance of model on the testing data set
print("\nAccuracy score on test set :", accuracy_score(Y_test, knn.predict(X_test_normalised[columns_z])))

print("\nDistribution of readmitted in train set :")
print(Y_train.readmitted.value_counts() / Y_train.readmitted.count())

print("\nDistribution of readmitted in predictions on the test set :")
print(Y_test.readmitted.value_counts() / Y_test.readmitted.count())
