# JBI030 — Assignment 1

This folder contains the completed notebook for the JBI030 Assignment 1 group submission.

## Run instructions

Open `assignment_1.ipynb` in JupyterLab from this folder. Keep the `data/` and `images/` folders next to the notebook because the notebook refers to them through relative paths.

## Multiple-choice answers

| Question | Answer | Short reason |
|---|---|---|
| 1a — Total variance | **False** | Total variance is `tr(C)`, which equals the **sum** of all eigenvalues, not only the largest eigenvalue. |
| 1b — Covariance | **True** | The dot product of two centered feature columns, divided by `n - 1`, is their sample covariance. |
| 1c — Covariance under a linear transformation | **True** | For `Y = XW`, the covariance becomes `C_Y = W^T C_X W`. |
| 1d — Matrix norms and trace | **True** | Both sides sum every entry of `A` squared, only grouped by rows or by columns. |
| 6d — Standardized versus raw gradient descent | **A** | Both can reach the same minimum loss; the raw-feature run requires far more steps because its safe learning rate must be much smaller. |

## Important implementation note

In 7b, cross-validation statistics must be computed from the feature values in the current fitting folds, not from the Boolean fold mask:

```python
mean_raw = np.mean(x_train_fold, axis=0)
std_raw = np.std(x_train_fold, axis=0)
```

