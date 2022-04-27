import numpy as np
import pandas as pd
import sys
from scipy.stats import zscore
from numpy.linalg import inv
from sklearn.linear_model import Lasso


def target_encoding(df, target, xi):
    # df = pd.read_csv("data/train.csv", sep=",", index_col=False)
    stats = df[target].groupby(df[xi]).agg(['count', 'mean'])
    smoothing_factor = 1.0
    min_samples_leaf = 1
    prior = df[target].mean()
    smoove = 1 / (1 + np.exp(-(stats['count'] - min_samples_leaf) / smoothing_factor))
    smoothing = prior * (1 - smoove) + stats['mean'] * smoove
    encoded = pd.Series(smoothing)
    return encoded


def load_set(path, is_test):
    df = pd.read_csv(path, sep=",", index_col=False)
    pd.set_option('display.max_columns', None)

    encoded_gender = target_encoding(df, 'plata', 'pol')
    df['pol'].replace(['Female', 'Male'], encoded_gender, inplace=True)

    encoded_field = target_encoding(df, 'plata', 'oblast')
    df['oblast'].replace(['A', 'B'], encoded_field, inplace=True)

    if not is_test:
        df['anomaly'] = detect_outliers(df.iloc[:, :6])
        df = df[df.anomaly != -1]
        df = df.drop(['anomaly'], axis=1)

    encoded_title = target_encoding(df, 'plata', 'zvanje')
    df['zvanje'].replace(['AssocProf', 'AsstProf', 'Prof'], encoded_title, inplace=True)

    x = df.iloc[:, :5].values
    y = df.iloc[:, 5].values

    # if is_test:
    #     print("TEST")
    # print(df)

    return np.array(x, dtype=np.float64), np.array(y, dtype=np.float64)


def detect_outliers(df):
    """
    Function for detecting outliers in dataframe
    :param df: Dataframe with all data
    :return: Array of values [-1, 1], -1 meaning observation is outlier 1 meaning it is not
    """
    arr = []
    for index, row in df.iterrows():
        if row['zvanje'] == 'Prof':
            if row['plata'] > 190000 or row['plata'] < 70000:
                arr.append(-1)
            else:
                arr.append(1)
        elif row['zvanje'] == 'AssocProf':
            if row['plata'] > 125000:
                arr.append(-1)
            else:
                arr.append(1)
        else:
            arr.append(1)
    return arr


def ridge_regression(x, y, l2_penalty=0.0001):
    """
    Closed form ridge regression solution
    :param x: Dependent variables
    :param y: Target variable
    :param l2_penalty:
    :return: Array of parameters
    """
    diag_matrix = []
    for i in range(x.shape[1]):
        diag_row = []
        for j in range(x.shape[1]):
            if i > 0 and i == j:
                diag_row.append(1)
            else:
                diag_row.append(0)
        diag_matrix.append(diag_row)
    diag_matrix = np.array(diag_matrix, dtype=np.float64)
    theta = inv((x.T.dot(x) + l2_penalty * diag_matrix)).dot(x.T.dot(y))
    return theta


def lasso_regression(x, y, x_test):
    model = Lasso(alpha=.0001)
    model.fit(x, y)
    print(model.coef_)
    return model.predict(x_test)


def fit(x, y):
    return ridge_regression(x, y)  #lasso_regression


def predict(x, theta):
    return np.dot(x, theta)  # np.power(np.dot(x, theta), 2)


def calculate_rmse(predicted, true):
    return np.sqrt(((predicted - true) ** 2).mean())


def main(train_path, test_path):
    x, y = load_set(train_path, False)

    x_b = np.c_[np.ones((x.shape[0], 1)), x]

    theta = fit(x_b, y)
    print(theta)
    x_test, y_true = load_set(test_path, True)

    x_test_b = np.c_[np.ones((x_test.shape[0], 1)), x_test]

    y_predicted = predict(x_test_b, theta)

    error = calculate_rmse(y_predicted, y_true)

    y_predicted_lasso = lasso_regression(x_b, y, x_test_b)

    error_lasso = calculate_rmse(y_predicted_lasso, y_true)

    print(error)
    print(f"Lasso err: {error_lasso}")


if __name__ == '__main__':
    main("data/train.csv", "data/testiramo.csv")
    # main(sys.argv[1], sys.argv[2])
