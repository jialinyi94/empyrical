import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS


def ornstein_uhlenbeck(X: pd.Series, delta_t: float = 1/252):
    X_prev = X.shift(1).dropna().values.reshape(-1,1)
    delta_X = X.diff().dropna().values
    model = OLS(delta_X, X_prev)
    results = model.fit()
    b = results.params[0]
    residuals = results.resid
    k: float = -np.log(b) / delta_t
    m = X.mean()
    sigma: float = np.sqrt(np.var(residuals) / (2*k))
    return k, m, sigma
