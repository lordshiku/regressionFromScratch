# regressionFromScratch
multinomial linear regression from scratch using custom gradient descent

# Project Summary

I build a linear regression model **from scratch** using **gradient descent** I made myself on MLB player statistics to predict key outcomes. Key highlights:

**NO SCIKIT LEARN OR OTHER ML LIBRARIES
* **Data Source**: 2023 MLB Player Stats (cleaned CSV).
* **Feature Selection**: Automated ranking by absolute correlation, removal of multicollinear predictors, optional interaction terms.
* **Regression Methodology**:

  * Implemented gradient descent to optimize parameters for mean squared error.
  * Feature scaling (zero mean, unit variance) applied before training.
  * Custom `predict`, `compute_mse`, and `compute_r2` helper functions.
* **Final Task**: Predict **Runs scored (R)** using the top 10 most correlated numeric features.
* **Outstanding Performance**: Achieved **extremely high R²** on the test set, demonstrating a strong linear fit.
* **Configurable Output**: Easily adjust the number of test-sample comparisons printed (first *x* predictions vs. actuals).

---

# Iteration History

1. **Age Prediction Prototype**

   * Initial goal: predict player Age using all numeric features.
   * Automated correlation-based feature ranking and multicollinearity checks.
   * Tried adding interaction terms with minimal performance gain—R² remained low.

2. **Visualization Attempts**

   * Generated scatterplots for each predictor vs. Age to inspect linear relationships.
   * Encountered environment issues with `matplotlib` and moved to correlation metrics instead.

3. **Metric-based Selection**

   * Switched to computing correlation metrics for each predictor and ranking them.
   * Concluded Age was not strongly linearly correlated with available stats.

4. **Pivot to Runs Scored (R)**

   * Changed target variable from Age to Runs scored (R).
   * Printed absolute correlations of all \~600 numeric features with R.
   * Selected **top 10 predictors** based on correlation.

5. **Gradient Descent Regression**

   * Implemented linear regression via gradient descent on the top predictors.
   * Trained on the first half of the dataset; tested on the second half.
   * Printed learned regression equation and performance metrics (MSE and R²).

6. **Configurable Predictions Display**

   * Extended code to print the first *x* actual vs. predicted values for test samples.
   * Made `_x_` adjustable via a single variable at the top of the script.

---

*This README encapsulates the development journey of building a custom linear regression model using gradient descent, highlighting iterative refinements and the final high-performing runs prediction.*

