# pldfa

## contents
This package enables executing the [MFDFA](https://arxiv.org/abs/physics/0202070) method in a vectorized manner on Polars dataframes 
(although it also supports other data types for simpler use cases). The point is that multiple time series from the same dataframe can be analyzed "in parallel" 
making this greatly faster than for-looping around multiple time series. The function returns all of the usual results, along with R^2 (how strong is the correlation used to estimate generalized H's). 
See `pldfa.py` for implementation and details.

## installation
Currently there is no PyPI package, since only a single `.py` file is needed. Recommended use is
```
from pldfa import mfdfa
```

## todo
The code currently supports only first-order regression. Unlike other packages, this one does support non-integer q's.

## bugs?
This package does not produce the same results as the older [MATLAB](https://www.mathworks.com/matlabcentral/fileexchange/38262-multifractal-detrended-fluctuation-analyses) code that was also 
used for the [R](https://cran.r-project.org/web/packages/MFDFA/index.html) package. It would seem this code is closer to the truth, as the alternatives have problems with negative q's. This one produced
expected results in a few tests, though it was not otherwise reviewed at this point in time.
