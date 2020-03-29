from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import pandas as pd
df = pd.read_csv('julia_lab/HealthExpend.csv')
X = df[['AGE', 'famsize', 'COUNTIP', 'COUNTOP', 'EXPENDIP']]
y = df[['EXPENDOP']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
linear = LinearRegression()
print(linear.fit(X_train, y_train))
print(linear.score(X_test , y_test))
ridge = Ridge()
print(ridge.fit(X_train, y_train))
print(ridge.score(X_test, y_test))
ridgecv = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100, 1000], store_cv_values=True)
print(ridgecv.fit(X_train, y_train))
print(ridgecv.score(X_test, y_test))
pca = PCA(n_components=2)
pca.fit(X)
X_2D = pca.transform(X)