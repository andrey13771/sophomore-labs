from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('HealthExpend.csv')
X = df[['AGE', 'famsize', 'COUNTIP', 'COUNTOP', 'EXPENDIP']]
y = df[['EXPENDOP']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# mpl.use('Agg')
# sns.pairplot(X, kind='reg')
# plt.savefig('pairplot.png')
#
linear = LinearRegression()
linear.fit(X_train, y_train)
print(f'Оцінка МНК для коефіцієнтів регресії: {linear.coef_}')
print(f'Оцінка МНК для вільного члена: {linear.intercept_}')
y_pred = linear.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f'R^2: {r2}')
print(f'MSE: {mse}')
residuals = y_test - y_pred
var = np.var(residuals**2)
print(f'Дисперсія: {var}')
sns.jointplot(y_pred, residuals)
plt.savefig('residuals.png')

ridgecv = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100, 1000], store_cv_values=True)
results = ridgecv.fit(X_train, y_train)
y_pred = ridgecv.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f'R^2: {r2}')
residuals = y_test - y_pred
var = np.var(residuals**2)
print(f'Дисперсія: {var}')
print(f'Підібране значення гіперпараметру: {results.alpha_}')

pca = PCA()
pca.fit(X)
# print('Головні компоненти:', pca.components_)
print('Пояснена дисперсія:', pca.explained_variance_ratio_)
pca = PCA(n_components=3)
X_2D = pca.fit_transform(X)
X_train, X_test = train_test_split(X_2D, test_size=0.2, random_state=42)
linear = LinearRegression()
print(linear.fit(X_train, y_train))
print('R^2:', linear.score(X_test, y_test))

fisher = f_regression(X, y)
print(fisher)
#
feature_selection = SelectKBest(score_func=f_regression, k=3)
results = feature_selection.fit(X, y.values.ravel())
print('Тест Фішера:', results.scores_)
print('p-value:', results.pvalues_)
