# regressionFromScratch
Multivariate linear regression from scratch using custom gradient descent

# Project Summary

I build two versions of a linear regression model **from scratch** on MLB player statistics to predict key outcomes, employing my own implementation of **gradient descent**:

1. **Scaled-Feature Version**  
   - **Why scale?** Features like Plate Appearances (PA) can be in the hundreds while Home Runs (HR) are in the tens. Feeding raw features into gradient descent creates a “stretched” loss surface, where some directions require tiny steps and others huge steps, slowing convergence and making the algorithm sensitive to the learning rate.  
   - **What is scaling?** I transform each feature \(x\) into \((x - \mu)/\sigma\) (zero mean, unit variance). This puts all predictors on the same relative scale, smoothing out the loss surface and allowing a single learning rate to work well across all parameters.
2. **Raw-Feature Version**  
   - No transformations: the model trains directly on original statistics. This makes deployment simpler (no need to store or apply scaling parameters), but typically requires a much smaller learning rate and careful tuning to converge.

**Data Source**: 2023 MLB Player Stats (cleaned CSV).  
*(https://www.kaggle.com/datasets/vivovinco/2023-mlb-player-stats/data)*

**Feature Selection**:  
- Automated ranking by absolute Pearson correlation with the target.  
- Removal of multicollinear predictors (|pairwise corr| > 0.9).  
- Optional pairwise interaction terms, filtered by minimal signal.  

**Gradient Descent Methodology**:  
I use **plain-vanilla batch gradient descent** to minimize the Mean Squared Error (MSE) of our linear model.

1. **Loss function (MSE)**  

MSE(β) = (1 / n) * Σᵢ (yᵢ - xᵢ · β)²

2. **Computing the gradient**  
- The partial derivative with respect to each parameter βⱼ is:  
  ```
  ∂MSE/∂βⱼ = -(2 / n) * Σᵢ (yᵢ - xᵢ · β) * xᵢⱼ
  ```  
- In code I flip the sign inside and compute:  
  ```python
  grads[j] = (2/n) * sum((yp - yt) * xi[j]
                         for xi, yp, yt in zip(X, Y_pred, Y))
  ```

3. **Parameter update rule**  
I move opposite the gradient by a factor of the learning rate `lr`:  
```python
betas[j] -= lr * grads[j]

which implements: βⱼ = βⱼ - lr * (∂MSE/∂βⱼ)
```


4. **Learning rate** (lr = 0.01)

Chosen by experimentation:

Too large (e.g. 0.1) → updates overshoot and diverge

Too small (e.g. 0.0001) → very slow convergence

0.01 reliably reduces loss in a few thousand epochs without instability

I used a smaller learning rate for the raw features model since the loss surface is larger so movements need to be more varied


5. **Batch Updates**

I recompute the gradient on the entire training set each epoch (batch gradient descent).

Over epochs = 2000, this steadily moves β toward the MSE minimum.



**Final Task**: Predict **Runs scored (R)** using the top 10 most correlated numeric features.

**Outstanding Performance**:  
- **Scaled-Feature Version** achieved **R² ≈ 0.95** on the held-out test set.  
- **Raw-Feature Version** achieves R^2 of 0.94, and converges stably with a carefully chosen (much smaller) learning rate.

**Configurable Output**: Easily adjust the number of test-sample comparisons printed (first *x* predictions vs. actuals).

---

# Iteration History - used chat gpt to help make this by plugging in my version history 

1. **Age Prediction Prototype**  
   - Initial goal: predict player Age using all numeric features.  
   - Automated correlation-based feature ranking and multicollinearity checks.  
   - Added interaction terms; R² remained low.

2. **Visualization Attempts**  
   - Generated scatterplots for each predictor vs. Age to inspect linear relationships.  
   - Encountered environment issues with `matplotlib`; switched to correlation metrics.

3. **Metric-Based Selection**  
   - Computed correlation metrics for each predictor and ranked them.  
   - Concluded Age was not strongly linearly correlated with available stats.

4. **Pivot to Runs Scored (R)**  
   - Changed target variable from Age to Runs scored (R).  
   - Printed absolute correlations of all ~600 numeric features with R.  
   - Selected **top 10 predictors** based on correlation.

5. **Gradient Descent Regression**  
   - Implemented linear regression via gradient descent on the top predictors.  
   - Trained on the first half of the dataset; tested on the second half.  
   - Printed learned regression equation and performance metrics (MSE and R²).

6. **Configurable Predictions Display**  
   - Extended code to print the first *x* actual vs. predicted values for test samples.  
   - Made *x* adjustable via a single variable at the top of the script.

---

*This README encapsulates the development journey of building custom linear regression models—with and without feature scaling—using gradient descent, highlighting iterative refinements and the final high-performing runs prediction.*  
