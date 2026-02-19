## no plots, no side effects just robust univariate summaries and diagnostic flags for modeling risks

## in case R
# library(reticulate)
# 
# py_install(
#   packages = c("numpy", "pandas", "scipy", "matplotlib", "statsmodels", "kagglehub"),
#   pip = TRUE
# )

## Download latest version of dataset
# print("Path to dataset files:", path)
# print(os.listdir(path))

import kagglehub
import os
import numpy as np
import pandas as pd
import statistics
from scipy import stats
np.random.seed(42)


path = kagglehub.dataset_download("yasserh/wine-quality-dataset")
csv_path = os.path.join(path, "WineQT.csv")
wine_df = pd.read_csv(csv_path)



## df size
n = 50
## Test Pandas DataFrame
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


## Main Robust Summary Function

"""
For eligibility we require:
  (i) integer or float type
  (ii) at least one non missing value
  (iii) more than 1 unique non-missing value
"""

## R for robust
def rsummary(df, trim = 0.1):
  if (not df.empty):
    #counting Nans
    nan_count = pd.DataFrame(np.zeros((18, len(df.columns))))
    nan_count.columns = df.columns
    nan_count.index = ["n_total", "n_nonmissing", "nonmissing_prop", "n_unique",
    "unique_prop", "dtype", "is_numeric", "eligible_numeric", "median", "n_clean",
    "trimmed_mean", "n_infinite", "inf_prop", "median_absolute_deviation", "iqr",
    "robust scale", "max_abs_robust_z", "outlier_frac_robust"]
    l = df.shape[0]
    for col in df.columns:
      s = df[col].count()
      u = df[col].nunique()
      typ = df[col].dtype
      is_num = pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col])
      eli = is_num and s > 0 and u > 1
      
      if (eli):
        n_inf = np.isinf(df[col]).sum()
        inf_prop = n_inf / l
      else:
        n_inf = np.nan
        inf_prop = np.nan
      clean_vals = df[col].replace([np.inf, -np.inf], np.nan).dropna().values
      n_clean = len(clean_vals)
      if (eli):
        ## define iqr
        iqr = stats.iqr(clean_vals)
        ## define median
        median = statistics.median(clean_vals)
        ## trimmed mean
        t_mean = stats.trim_mean(clean_vals, trim) if len(clean_vals) > 0 else np.nan
        ## MAD 
        mad = np.nanmedian(np.abs(clean_vals - median)) if len(clean_vals) > 0 else np.nan
        
        robust_scale = 1.4826 * mad if mad > 0 else np.nan
        robust_z = (clean_vals - median) / robust_scale
        max_abs_robust_z = np.max(np.abs(robust_z)) if robust_scale > 0 else np.nan
        outlier_frac = np.mean(np.abs(robust_z) > 3) if robust_scale > 0 else np.nan
      else:
        iqr = np.nan
        median = np.nan
        t_mean = np.nan
        mad = np.nan
      
      nan_count[col] = [l, s, s/l, u, u/l, typ, is_num, eli, median, n_clean, t_mean,
      n_inf, inf_prop, mad, iqr, robust_scale, max_abs_robust_z, outlier_frac]
    return(nan_count)
  else:
    print("Error: data frame is empty")
    
## Allows printing of entire df    
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

## printing df
out = rsummary(df)
print(out)
# print(out.shape) ## number of rows and columns
# print(out.loc[["median", "trimmed_mean", "median_absolute_deviation"]]) ##outputs rows





n = 200
## A second dataframe to test rsummary
test_df = pd.DataFrame({
    "clean_numeric": np.random.normal(0, 1, n),
    "numeric_with_nan": np.where(np.random.rand(n) < 0.3, np.nan, np.random.normal(5, 2, n)),
    "mostly_nan": np.where(np.random.rand(n) < 0.85, np.nan, np.random.normal(10, 1, n)),
    "constant_numeric": np.ones(n) * 7,
    "numeric_with_inf": np.where(np.random.rand(n) < 0.1, np.inf, np.random.normal(0, 1, n)),
    "categorical_like": np.random.choice(["A", "B", "C"], n),
    "boolean_flag": np.random.choice([True, False], n),
    "integer_id": np.arange(1, n + 1),
    "skewed_numeric": np.random.exponential(scale=2.0, size=n),
    "outlier_contaminated": np.concatenate([
        np.random.normal(0, 1, int(0.9 * n)),
        np.random.normal(15, 1, int(0.1 * n))
    ])
})

# rsummary(test_df)

# rsummary(wine_df)







