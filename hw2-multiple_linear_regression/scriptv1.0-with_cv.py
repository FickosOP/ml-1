from scipy.stats import zscore
from numpy.linalg import inv
# from category_encoders import TargetEncoder
# from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys


def load_train_set(path):
    df = pd.read_csv(path, sep=",", index_col=False)

    pd.set_option('display.max_columns', None)

    df.insert(len(df.columns), 'y_sqrt', np.sqrt(df.iloc[:, -1]))  # try box_cox

    df['pol_bin'] = df['pol'].replace(['Female', 'Male'], [0, 1])
    df['oblast_bin'] = df['oblast'].replace(['A', 'B'], [0, 1])

    stats = df['y_sqrt'].groupby(df['zvanje']).agg(['count', 'mean'])
    smoothing_factor = 1.0
    min_samples_leaf = 1
    prior = df['y_sqrt'].mean()
    smoove = 1 / (1 + np.exp(-(stats['count'] - min_samples_leaf) / smoothing_factor))
    smoothing = prior * (1 - smoove) + stats['mean'] * smoove
    encoded = pd.Series(smoothing, name='zvanje_te')

    df['zvanje_te'] = df['zvanje'].replace(['AssocProf', 'AsstProf', 'Prof'], encoded)

    df['zvanje_z'] = zscore(df['zvanje_te'])
    df = df.drop(['zvanje_te'], axis=1)

    df['godina_doktor_z'] = zscore(df['godina_doktor'])
    df['godina_iskustva_z'] = zscore(df['godina_iskustva'])

    x = df.iloc[:, 7:].values
    y = df.iloc[:, 6].values

    # df.hist(grid=False, figsize=(10, 6), bins=30)
    # plt.show()

    # remove outliers
    # if_sk = IsolationForest(n_estimators=50, max_samples='auto', contamination=float(0.01))
    # if_sk.fit(x)
    df['anomaly'] = detect_outliers(df.iloc[:, :6])  # if_sk.predict(x)

    # print(df[df.anomaly == -1])

    df = df[df.anomaly != -1]  # UKLANJANJE AUTLAJERA IZ DATAFREJMA
    df = df.drop(['anomaly'], axis=1)

    # print(df)
    x = df.iloc[:, 7:].values
    y = df.iloc[:, 6].values
    # print(df)
    # print(f"NAKON BRISANJA AUTLAJERA OSTALO JE: {len(df)}")

    # print(df.agg(['skew', 'kurtosis']).transpose())
    # print(df)
    # wait = input()
    return np.array(x, dtype=np.float64), np.array(y, dtype=np.float64), encoded


def detect_outliers(df):
    arr = []
    for index, row in df.iterrows():
        if row['zvanje'] == 'Prof':
            if row['plata'] > 200000 or row['plata'] < 75000:
                arr.append(-1)
            else:
                arr.append(1)
        elif row['zvanje'] == 'AssocProf':
            if row['plata'] > 120000:
                arr.append(-1)
            else:
                arr.append(1)
        else:
            arr.append(1)
    return arr


def load_test_set(path, encoded):
    df = pd.read_csv(path, sep=",", index_col=False)

    df['pol_bin'] = df['pol'].replace(['Female', 'Male'], [0, 1])
    df['oblast_bin'] = df['oblast'].replace(['A', 'B'], [0, 1])

    df['zvanje_te'] = df['zvanje'].replace(['AssocProf', 'AsstProf', 'Prof'], encoded)

    df['zvanje_z'] = zscore(df['zvanje_te'])
    df = df.drop(['zvanje_te'], axis=1)

    df['godina_doktor_z'] = zscore(df['godina_doktor'])
    df['godina_iskustva_z'] = zscore(df['godina_iskustva'])

    x = df.iloc[:, 6:].values
    y = df.iloc[:, 5].values

    return np.array(x, dtype=np.float64), np.array(y, dtype=np.float64)


def h(x, theta, b):
    return x.dot(theta) + b


def ridge_regression(x, y, learning_rate=0.1, max_iterations=1000, stopping_threshold=1e-9, l2_penalty=0.5):
    n = len(y)

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
    # Closed form
    theta = inv((x.T.dot(x) + l2_penalty * diag_matrix)).dot(x.T.dot(y))
    # print("THETA")
    # print(theta)
    return theta


def fit(x, y):
    theta = ridge_regression(x, y)
    return theta


def predict(x, theta):
    # print("PREDICTED\n\n")
    # print(np.power(np.dot(x, theta), 2))
    return np.power(np.dot(x, theta), 2)  # np.dot(x, theta)


def calculate_rmse(predicted, true):
    return np.sqrt(((predicted - true) ** 2).mean())


def create_new_sets(x_parts, y_parts, i):
    new_x, new_y, new_x_val, new_y_true = [], [], [], []
    for j in range(len(x_parts)):
        if j == i:
            for row in x_parts[j]:
                new_x_val.append(row.tolist())
            for row in y_parts[j]:
                new_y_true.append(row.tolist())

        else:
            for row in x_parts[j]:
                new_x.append(row.tolist())
            for row in y_parts[j]:
                new_y.append(row.tolist())

    return np.array(new_x, dtype=np.float64),\
           np.array(new_y, dtype=np.float64),\
           np.array(new_x_val, dtype=np.float64),\
           np.array(np.power(new_y_true, 2), dtype=np.float64)  # take it from df


def cross_val(x_b, y, k):
    x_parts = np.array_split(x_b, k)
    y_parts = np.array_split(y, k)

    best_theta = []
    err_list = []
    lowest_err = None
    for i in range(k):
        new_x, new_y, new_x_val, new_y_true = create_new_sets(x_parts, y_parts, i)
        theta = ridge_regression(new_x, new_y)
        y_predicted = predict(new_x_val, theta)
        # print("\nPredicted - True\n")
        # for _ in range(len(y_predicted)):
        #     print(f"P: {y_predicted[_]} -- T: {new_y_true[_]}\t Diff: {y_predicted[_] - new_y_true[_]}")
        # wait = input()
        # print("\n\nTRUE\n\n")
        # print(new_y_true)
        err = calculate_rmse(y_predicted, new_y_true)
        if not lowest_err or err < lowest_err:
            lowest_err = err
            best_theta = theta
        # if len(err_list) == 0 or err < err_list[-1]:
        #     best_theta = theta

        err_list.append(err)
        # wait = input()
    print(f"Error list: {err_list}")
    print(f"Avg error: {sum(err_list)/len(err_list)}")
    return best_theta


def main(train_path, test_path):
    x, y, encoded = load_train_set(train_path)

    x_b = np.c_[np.ones((x.shape[0], 1)), x]

    best_theta = cross_val(x_b, y, 10)
    print(best_theta)
    print(f"Formula\nf = {best_theta[0]} + {best_theta[1]} * pol + {best_theta[2]} * oblast + {best_theta[3]} * zvanje"
          f" + {best_theta[4]} * god_doktor + {best_theta[5]} * god_iskustva")

    x_test, y_true = load_test_set(test_path, encoded)
    x_test_b = np.c_[np.ones((x_test.shape[0], 1)), x_test]
    y_predicted = predict(x_test_b, best_theta)
    print(y_predicted, y_true)
    error = calculate_rmse(y_predicted, y_true)

    print(f"RMSE:{error}")


if __name__ == '__main__':
    main("data/train.csv", "data/test_preview.csv")
    # main(sys.argv[1], sys.argv[2])


