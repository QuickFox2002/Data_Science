import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ----------------------------
# Load and preprocess the data
# ----------------------------
df = pd.read_csv("boston.csv")

X = df.drop("medv", axis=1)
y = df["medv"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------
# Linear Regression from scratch
# ----------------------------
class LinearRegressionCustom:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        X = np.c_[np.ones(X.shape[0]), X]
        y = y.values.reshape(-1, 1)
        self.theta = np.zeros((X.shape[1], 1))
        for _ in range(self.epochs):
            gradients = 2 / X.shape[0] * X.T @ (X @ self.theta - y)
            self.theta -= self.lr * gradients

    def predict(self, X):
        X = np.c_[np.ones(X.shape[0]), X]
        return X @ self.theta

lr_model = LinearRegressionCustom()
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled).flatten()

# ----------------------------
# Random Forest (simple version using sklearn tree)
# ----------------------------
class RandomForestCustom:
    def __init__(self, n_estimators=10, max_depth=5):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_estimators):
            idxs = np.random.choice(len(X), len(X), replace=True)
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X[idxs], y[idxs])
            self.trees.append(tree)

    def predict(self, X):
        return np.mean([tree.predict(X) for tree in self.trees], axis=0)

rf_model = RandomForestCustom(n_estimators=10, max_depth=5)
rf_model.fit(X_train_scaled, y_train.values)
y_pred_rf = rf_model.predict(X_test_scaled)

# ----------------------------
# XGBoost-style boosting (simplified)
# ----------------------------
class XGBoostCustom:
    def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.models = []

    def fit(self, X, y):
        y_pred = np.zeros(len(y))
        self.models = []
        for _ in range(self.n_estimators):
            residual = y - y_pred
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residual)
            update = tree.predict(X)
            y_pred += self.learning_rate * update
            self.models.append(tree)

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for tree in self.models:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred

xgb_model = XGBoostCustom()
xgb_model.fit(X_train_scaled, y_train.values)
y_pred_xgb = xgb_model.predict(X_test_scaled)

# ----------------------------
# Performance Comparison
# ----------------------------
def evaluate(name, y_true, y_pred):
    print(f"\n{name}")
    print("RMSE:", round(np.sqrt(mean_squared_error(y_true, y_pred)), 4))
    print("R²:", round(r2_score(y_true, y_pred), 4))

evaluate("Linear Regression", y_test, y_pred_lr)
evaluate("Random Forest", y_test, y_pred_rf)
evaluate("XGBoost", y_test, y_pred_xgb)

# ----------------------------
# Feature Importance (from first tree of RF and XGB)
# ----------------------------
def plot_importance(model, feature_names, title):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.title(title)
    plt.tight_layout()
    plt.show()

plot_importance(rf_model.trees[0], X.columns, "Random Forest - Feature Importance")
plot_importance(xgb_model.models[0], X.columns, "XGBoost - Feature Importance")
