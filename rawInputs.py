import pandas as pd
import numpy as np

#everything is the same except i removed scaling, so the comments are unfortunately not included on this one
#the purpose is so you can run this file instead if you want to see unscaled learning with my gradient descent

DISPLAY_N = 20 

#helpers
def predict(x, betas):
    return sum(b * xi for b, xi in zip(betas, x))

def compute_mse(y_true, y_pred):
    return sum((yt - yp)**2 for yt, yp in zip(y_true, y_pred)) / len(y_true)

def compute_r2(y_true, y_pred):
    y_bar = sum(y_true) / len(y_true)
    ss_tot = sum((yt - y_bar)**2 for yt in y_true)
    ss_res = sum((yt - yp)**2 for yt, yp in zip(y_true, y_pred))
    return 1 - ss_res / ss_tot

def gradient_descent(X, Y, lr=0.0000001, epochs=5000):
    """
    Note: when training on raw features you may need a smaller lr
    to keep the steps stable. Tweak lr and epochs as needed.
    """
    n, m = len(Y), len(X[0])
    betas = [0.0] * m
    for _ in range(epochs):
        Y_pred = [predict(xi, betas) for xi in X]
        grads = [
            (2/n) * sum((yp - yt) * xi[j] for xi, yp, yt in zip(X, Y_pred, Y))
            for j in range(m)
        ]
        for j in range(m):
            betas[j] -= lr * grads[j]
    return betas


def main():

    df = pd.read_csv("2023_MLB_Player_Stats_clean.csv", encoding="latin1")

  
    target = "R"
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for drop in ["Rk", target]:
        if drop in num_cols:
            num_cols.remove(drop)

  
    corrs = df[num_cols].corrwith(df[target]).abs()
    top10 = corrs.sort_values(ascending=False).head(10).index.tolist()
    print("Top 10 predictors for Runs scored (R):", top10)

   
    X_cols = top10

    
    mid = len(df) // 2
    train_df = df.iloc[:mid].reset_index(drop=True)
    test_df  = df.iloc[mid:].reset_index(drop=True)

   
    X_train = [[1.0] + row.tolist() for row in train_df[X_cols].values]
    Y_train = train_df[target].tolist()
    X_test  = [[1.0] + row.tolist() for row in test_df[X_cols].values]
    Y_test  = test_df[target].tolist()

  
    lr, epochs = 1e-7, 5000
    betas = gradient_descent(X_train, Y_train, lr, epochs)

  
    eq = f"R = {betas[0]:.4f}"
    for j in range(1, len(betas)):
        eq += f" + {betas[j]:.4f}*{X_cols[j-1]}"
    print("\nLearned regression equation (raw features):")
    print(eq)

   
    Y_train_pred = [predict(xi, betas) for xi in X_train]
    Y_test_pred  = [predict(xi, betas) for xi in X_test]

    mse_train = compute_mse(Y_train, Y_train_pred)
    r2_train  = compute_r2(Y_train, Y_train_pred)
    mse_test  = compute_mse(Y_test, Y_test_pred)
    r2_test   = compute_r2(Y_test, Y_test_pred)

    print(f"\nTraining   MSE = {mse_train:.4f}   R² = {r2_train:.4f}")
    print(f"Test       MSE = {mse_test:.4f}   R² = {r2_test:.4f}")

    
    n = min(DISPLAY_N, len(Y_test))
    print(f"\nTest set: actual R → predicted R (first {n} samples)")
    for actual, pred in zip(Y_test[:n], Y_test_pred[:n]):
        print(f"{actual} → {pred:.2f}")

if __name__ == "__main__":
    main()
