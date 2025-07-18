# Linear Regression with scikit-learn

This project demonstrates and compares several regression models using scikit-learn on simple synthetic data.

## Models Included

- **Linear Regression** – A basic parametric regression model with coefficients and intercept.
- **K-Nearest Neighbors Regressor (KNN)** – A non-parametric model based on closest training examples.
- **Random Forest Regressor** – An ensemble method using multiple decision trees to improve prediction accuracy.
- **Support Vector Regressor (SVR)** – A kernel-based regression model that works well on small and non-linear datasets.

## Project Structure

- `main.py`: Runs data generation, training, evaluation, and visualization for all models.
- `utils.py`: Contains the `train_and_plot` helper function.
- `requirements.txt`: Lists all required Python packages.
- `.gitignore`: Ignores Python caches, VS Code settings and Matplotlib cache directory.

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the script:
   ```bash
   python main.py
   ```

## Output

- MSE (Mean Squared Error) for each model
- Coefficients (if available)
- Scatter plot of real data and prediction line from each model