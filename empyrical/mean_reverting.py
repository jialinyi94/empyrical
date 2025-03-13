import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS


def ornstein_uhlenbeck(X: pd.Series, delta_t: float = 1/252):
    """ Ornstein-Uhlenbeck process

    dX = k*(m-X)*dt + sigma*dW

    Parameters
    ----------
    X : pd.Series
        Time series of the process
    delta_t : float, optional
        the time bar measured in years , by default 1/252

    Returns
    -------
    k : float
        mean reversion speed
    m : float
        mean level
    sigma : float
        volatility of the process
    """
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
