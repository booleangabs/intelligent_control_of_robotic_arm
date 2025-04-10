import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load CSV
df = pd.read_csv("robot_arm_cam640x480.csv")

# Feature columns and target columns
X = df[['motor_0', 'motor_1', 'motor_2', 'motor_3']]
y = df[['x_img', 'y_img']]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features and targets
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# Define and train
dtr = DecisionTreeRegressor(random_state=42)
dtr.fit(X_train_scaled, y_train_scaled)

# Predict and inverse-transform
y_pred_scaled = dtr.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Root Mean Squared Error: {(mse)**0.5:.6f}")
print(f"R² Score: {r2:.4f}")

# Plot predicted vs actual with lines connecting them
plt.figure(figsize=(8, 6))

# Scatter actual points
plt.scatter(y_test['x_img'], y_test['y_img'], color='blue', label='Actual', alpha=0.6)
# Scatter predicted points
plt.scatter(y_pred[:, 0], y_pred[:, 1], color='red', label='Predicted', alpha=0.6)

# Draw lines from prediction to ground truth
for i in range(len(y_pred)):
    plt.plot(
        [y_test.iloc[i, 0], y_pred[i, 0]],  # x: actual to predicted
        [y_test.iloc[i, 1], y_pred[i, 1]],  # y: actual to predicted
        color='g', linestyle='--', linewidth=3, alpha=0.8
    )

plt.xlabel("x_img")
plt.ylabel("y_img")
plt.title("Actual vs Predicted Screen Coordinates\n(with error lines)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
