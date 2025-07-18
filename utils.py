from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def train_and_plot(model, model_name, X_train, X_test, y_train, y_test, X, y):
    
    # train
    model.fit(X_train, y_train)
    
    # predict
    y_pred = model.predict(X_test)

    # evaluation (loss)
    mse = mean_squared_error(y_test, y_pred)
    print(f"{model_name} MSE: {mse:.3f}")

    # show coeff: y=coef*x+intercept
    if hasattr(model, "coef_"):
        print(f'Model {model_name} coefficients: {model.coef_}, Intercept: {model.intercept_}')
    else:
        print(f'Model {model_name} does not have explicit coefficients (non-parametric)')
    
    # predict all and plot
    y_all_pred = model.predict(X)
    plt.figure(figsize=(6,4))
    plt.scatter(X, y, color='black', label='Real data')
    plt.plot(X, y_all_pred, label=model_name)
    plt.title(f"{model_name}")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()