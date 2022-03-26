import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy import stats
from scipy import special


def load_set(path):
    df = pd.read_csv(path, sep=",", index_col=False)
    df.columns = ["x", "y"]
    data = np.array(df, dtype=float)

    return np.array(data[:, 0]), np.array(data[:, 1])


def log_likelihood(lmbda, x, y):
    n, p = x.shape
    lnjacobi = (lmbda - 1) * np.sum(np.log(y))
    trans_y = stats.boxcox(y, lmbda=lmbda)
    xtxinv = np.linalg.inv(np.matmul(np.transpose(x), x))
    imxxtxinvxt = np.subtract(np.identity(n), np.matmul(np.matmul(x, xtxinv), np.transpose(x)))
    rss = np.matmul(np.matmul(np.transpose(trans_y), imxxtxinvxt), trans_y)
    return - n / 2.0 * (np.log(rss)) + lnjacobi


if __name__ == '__main__':
    x, y = load_set("data/train.csv")
    x = np.reshape(x, (x.shape[0], 1))
    y = np.squeeze(y)
    regressor = LinearRegression()
    val = regressor.fit(x, y)
    print(f"{val.coef_} + {val.intercept_}")
    plt.plot(x, regressor.predict(x), color='red', linewidth=0.5)
    plt.scatter(x, y, color='blue', s=2)
    plt.show()

    potential_lmbdas = np.linspace(-1, 1, 1000)
    likelihoods = np.array([log_likelihood(lmbda, x, y) for lmbda in potential_lmbdas])
    print('Estimation for lambda:', potential_lmbdas[np.argmax(likelihoods)])
    plt.plot(potential_lmbdas, likelihoods)
    plt.show()
    print('Estimation for lambda:', potential_lmbdas[np.argmax(likelihoods)])

    regressor = LinearRegression()
    regressor.fit(x, stats.boxcox(y, lmbda=0.5))
    val = regressor.fit(x, y)
    print(f"{val.coef_} + {val.intercept_}")
    plt.plot(x, special.inv_boxcox(regressor.predict(x), 1.1), color='red', linewidth=0.5)
    plt.scatter(x, y, color='blue', s=2)
    plt.show()