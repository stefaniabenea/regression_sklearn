from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from utils import train_and_plot

# syntetic data: y = 2*x + 1 + noise
X = np.arange(10).reshape(-1,1)
y = 2*X.flatten() + 1 + np.random.randn(10)

# split data into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# models
models = {'Linear Regression':LinearRegression(), "KNN (k=3)":KNeighborsRegressor(n_neighbors=3),
          "Random Forest Regressor":RandomForestRegressor(n_estimators=100, random_state=42)}

for key, value in models.items():
    train_and_plot(value, key, X_train, X_test, y_train, y_test, X, y)