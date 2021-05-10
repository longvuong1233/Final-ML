
import pandas as pd
import numpy as np


def load_data():
    filename = 'fashion-mnist_train.csv'

    df = pd.read_csv(filename, header=None)

    # Tách dữ liệu dự đoán
    X = df.values[:, 1:]

    y = df.values[:, 0]

    # Normalize data
    from sklearn.preprocessing import MinMaxScaler
    minmax = MinMaxScaler()
    X = minmax.fit_transform(X)

    # Tách dữ liệu train, validate
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=1)

    print(X_train.shape, 'Kich thuoc trainning data')
    print(X_val.shape, "Kich thuoc validation data")

    # X_test = X_train.copy()
    # y_test = y_train.copy()

    y_train = [[1 if y_train[i] == j else 0 for j in range(
        10)] for i in range(len(y_train))]

    y_train = [np.array([y_train[i]]).T for i in range(len(y_train))]

    # đọc dữ liệu test

    filename_test = 'fashion-mnist_test.csv'

    df_test = pd.read_csv(filename_test, header=None)
    df_test = df_test[0:1000]

    X_test = df_test.values[:, 1:]
    X_test = minmax.fit_transform(X_test)
    print(X_test.shape, "Kich thước Test")
    y_test = df_test.values[:, 0]

    training_inputs = [np.reshape(x, (len(x), 1)) for x in X_train]
    training_data = zip(training_inputs, y_train)

    validation_inputs = [np.reshape(x, (len(x), 1)) for x in X_val]
    validation_data = zip(validation_inputs, y_val)

    # print(y_test)

    test_inputs = [np.reshape(x, (len(x), 1)) for x in X_test]

    test_data = zip(test_inputs, y_test)

    return (training_data, validation_data, test_data)
