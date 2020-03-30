setup = '''
import numpy as np
import pandas as pd

def make_df(N, beta0, beta1, beta2, beta3):
    x1 = np.random.normal(size=N)
    x2 = np.random.normal(size=N)
    x3 = np.random.normal(size=N)
    eps = np.random.normal(scale=0.1, size=N)
    df = pd.DataFrame(np.array([x1, x2, x3, beta0 + beta1 * x1 + beta2 * x2 + beta3 * x3 + eps]).T,
                      columns=['X1', 'X2', 'X3', 'Y'])
    return df

df = make_df({}, 124.2, 325.1, 0.1, 0.124)
X = df[['X1', 'X2', 'X3']]
Y = df[['Y']]
'''


setup1 = '''
def OLS(X, y):
    beta = np.linalg.inv(np.dot(X.T, X)).dot(X.T).dot(y)
    return beta
'''


setup2 = '''
def cost(theta, X, y):
    m = len(y)
    pred = X.dot(theta)
    cost = (1 / 2 * m) * np.sum(np.square(pred - y))
    return cost


def gradient_descent(X, y, theta, learning_rate=0.01, iterations=100):
    m = len(y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations, theta.shape[0]))
    for i in range(iterations):
        pred = np.dot(X, theta)
        theta = theta - (1 / m) * learning_rate * (X.T.dot((pred - y)))
        theta_history[i, :] = theta.T
        cost_history[i] = cost(theta, X, y)
    return theta, cost_history, theta_history

theta = np.random.randn(4, 1)
X_b = np.c_[np.ones((X.shape[0], 1)), X]
'''

setup3 = 'from sklearn.linear_model import LinearRegression '


code1 = 'OLS(X, Y)'
code2 = 'gradient_descent(X_b, Y, theta)'
code3 = 'LinearRegression().fit(X, Y)'


def check_time(N):
    print(timeit.timeit(setup=setup.format(N) + setup1, stmt=code1, number=10))
    print(timeit.timeit(setup=setup.format(N) + setup2, stmt=code2, number=10))
    print(timeit.timeit(setup=setup.format(N) + setup3, stmt=code3, number=10))


if __name__ == '__main__':
    import timeit
    # df = make_df(100, 124.2, 325.1, 0.1, 0.124)
    # X = df[['X1', 'X2', 'X3']]
    # Y = df[['Y']]
    # print(f'МНК за формулою: {OLS(X, Y)}')
    # theta = np.random.randn(4, 1)
    # X_b = np.c_[np.ones((X.shape[0], 1)), X]
    # th, c, th_h = gradient_descent(X_b, Y, theta)
    # print(f'Градієнтний спуск: {th}')
    # reg = LinearRegression().fit(X, Y)
    # print(f'Бібліотечна функція з scikit-learn: {reg.intercept_} {reg.coef_}')
    print('N = 10')
    check_time(10)
    print('N = 100')
    check_time(100)
    print('N = 1000')
    check_time(1000)
    print('N = 10000')
    check_time(10000)

