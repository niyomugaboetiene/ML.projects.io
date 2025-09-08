import numpy as np
import matplotlib.pyplot as plt

# Generate datasets
X = np.arange(0, 10, 1) # from 0 up to 9
y = 2 * X + 5 + np.random.randn(len(X)) # len(X) is noise

# convert to columns for matrix multiplication
X = X.reshape(-1, 1) # shape (10, 1)
y = y.reshape(-1, 1) # shape (10, 1)

# normalize feature
X_max = np.max(X)
X_min = np.min(X)

y_max = np.max(y)
y_min = np.max(y)

y_normalized = (y - y_min) / (y_max - y_min)
X_normalzied = (X - X_min) / (X_max - X_min)

# initialized parameters
w = np.array([[0.0]]) # weight
b = 0.0
learning_rate = 0.1
epochs = 1000
m = X_normalzied.shape[0] # number of samples

# train model gradient descent

for i in range(epochs):
    y_pred = X_normalzied @ w + b

    # compute gradients
    dw = (2/m) * (X_normalzied.T @ (y_pred - y_normalized))
    db = (2/m) * np.sum(y_pred - y_normalized)

    # update parameters
    w -= learning_rate * dw
    b = learning_rate * db

# Make smooth predictions for ploting
x_plot = np.linspace(0, 10, 100).reshape(-1, 1)
x_plot_norm = (x_plot - X_min) / (X_max - X_min)
y_plot_norm = (x_plot_norm @ w + b)

# denormalize predictions
y_plot = y_plot_norm * (y_max - y_min) + y_min

plt.scatter(X, y, color='blue', label='Data')
plt.plot(x_plot, y_plot, color='red', label='Prediction')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.title('Linear Regression from scratch')
plt.legend()
plt.show()


# predict for new value
hours = 7
hours_norm = (hours - X_min) / (X_max - X_min)
predicted_score_norm = hours_norm * w + b
predicted_score = predicted_score_norm * (y_max - y_min) + y_min
print(f"Predicred score from {hours} hours: {predicted_score[0][0]:2.f}")