import numpy as np

def gradient_descent(X, y, lr=1e-7, epochs=1000):
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    m, n = X.shape
    theta = np.zeros(n)

    for epoch in range(epochs):
        y_pred = X @ theta
        error = y_pred - y
        grad = (2/m) * X.T @ error
        theta -= lr * grad

        if epoch % 100 == 0:
            loss = np.mean(error ** 2)
            print(f"Epoch {epoch}: MSE = {loss:.2f}")

    return theta

def predict(X, theta):
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    return X @ theta