## no plots, no side effects just robust univariate summaries and diagnostic flags for modeling risks

import numpy as np
import pandas as pd

np.random.seed(42)

# ## Big idea
# def robust_summary(df, trim_frac = 0.1, outlier_method = "iqr", mad_scale=True,
#   dropna=True):
#     ## no function yet





n = 25

df = pd.DataFrame({
    "clean_numeric": np.random.normal(loc=0, scale=1, size=n),

    "numeric_with_nan": np.where(
        np.random.rand(n) < 0.25,
        np.nan,
        np.random.normal(loc=5, scale=2, size=n)
    ),

    "mostly_nan": np.where(
        np.random.rand(n) < 0.85,
        np.nan,
        np.random.normal(loc=10, scale=1, size=n)
    ),

    "constant_numeric": np.full(n, 3.14),

    "numeric_with_inf": np.where(
        np.random.rand(n) < 0.15,
        np.inf,
        np.random.normal(loc=0, scale=1, size=n)
    ),

    "categorical_like": np.random.choice(
        ["A", "B", "C"],
        size=n
    ),

    "boolean_flag": np.random.choice(
        [True, False],
        size=n
    ),

    "integer_id": np.arange(1, n + 1)
})


