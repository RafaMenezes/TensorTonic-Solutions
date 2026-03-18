import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    N = X.shape[0]
    
    w = np.zeros(X.shape[1])
    b = 0.0
    loss = 0
    
    for i in range(steps):
        y_pred = _sigmoid(X @ w + b)
    
        loss -= np.sum(y*np.log(y_pred) + (1 - y)*np.log(1-y_pred)) / N

        # derived by hand. Took me a while, rusty
        dw = (X.T @ (y_pred - y)) / N
        db = np.sum(y_pred - y) / N

        w -= lr * dw
        b -= lr * db

    return w,b
