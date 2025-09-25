import polars as pl
import functools as ft
import numpy as np
import sys

def mfdfa(df,
          target='x',
          idcol='id',
          sarr=np.logspace(2.7, 4.7, base=np.e),
          qarr=(a:=np.linspace(-3, 3, 15))[a != 0]
          
    ):
    """
    Perform MFDFA on Polars dataframe (multiple time series in parallel)
    currently supports only 1-st order detrending
    
    df:     Polars data frame to perform MFDFA on; also accepts Pandas df or numpy array, but both will be converted
    target: the response variable
    idcol:  name of column identifying time series (e.g. id of participant in study)
    sarr:   vector of scale parameters (iterable)
    qarr:   vector of exponents q (iterable)

    Returns: Polars data frame with
                hq   (generalized Hurst exponents, for all q, name is f'h{q}')
                R2q  (R2q in linear regression which approximates hq)
                tq   (tau; see Kantelhardt)
                aq   (alpha)
                fq   (f(alpha))
    """

    # check df
    if type(df) is np.ndarray:
        if df.ndim == 1:
            df = pl.from_numpy(df).with_columns(id = pl.lit(1))
            df.rename({'column_0': target})
        elif df.ndim == 2 and df.shape[1] == 2:
            df = pl.from_numpy(df)
            df = df.rename({'column_0': 'id', 'column_1': target})
        else:
            raise Exception('Could not understand passed Numpy array: should be 1dim array (single series) or nx2 where first column is id and second are series.')
    elif 'pandas' in sys.modules and type(df) is not pl.dataframe.frame.DataFrame:
        try:
            df = pl.from_pandas(df)
        except TypeError:
            raise TypeError('df must be Polars df, Numpy array, or Pandas df')

    if type(sarr) is not np.ndarray:
        try:
            sarr = np.array(list(sarr))
        except TypeError:
            raise TypeError('Could not convert sarr to Numpy array.')

    if type(qarr) is not np.ndarray:
        try:
            qarr = np.array(list(qarr))
        except TypeError:
            raise TypeError('Could not convert qarr to Numpy array.')

    qarr = qarr[qarr != 0]
    np.sort(sarr)
    np.sort(qarr)

    # reset event_id (aka t in X_t)
    if 'event_id' in df.columns:
        df.drop_in_place('event_id')
    
    q = df.lazy().with_columns(
        (pl.col(target).is_not_nan()).cum_sum().over('id').alias('event_id')
    )
    
    frame = q.collect()    

    q = frame.lazy().select(
        pl.col('id'),
        pl.col('event_id'),
        pl.col(target).cum_sum().over('id'),
    )

    # starting frame
    base_frame = q.collect()
    source = 'event_id'
    s_frames = {}

    # for each s, make brackets and calculate fluctuations
    for s in sarr:
        q = base_frame.lazy().with_columns(
            (pl.col('event_id') / s).over('id').ceil().alias('fw_bracket'),
            ((pl.col('event_id').max() + 1 - pl.col('event_id')) / s).over('id').ceil().alias('bk_bracket')
        )
        
        new_frame = q.collect()
        
        q = new_frame.lazy().with_columns(
            (pl.col(source) - (pl.col(source).mean().over('id', 'fw_bracket'))).alias('fw_dt'),
            (pl.col(target) - (pl.col(target).mean().over('id', 'fw_bracket'))).alias('fw_dx'),
            (pl.col(source) - (pl.col(source).mean().over('id', 'bk_bracket'))).alias('bk_dt'),
            (pl.col(target) - (pl.col(target).mean().over('id', 'bk_bracket'))).alias('bk_dx')
        )
        
        new_frame = q.collect()
        
        q = new_frame.lazy().with_columns(
            ((pl.col('fw_dt') * pl.col('fw_dx')).sum().over('id', 'fw_bracket') / (pl.col('fw_dt') ** 2).sum().over('id', 'fw_bracket')).alias('fw_beta'),
            ((pl.col('bk_dt') * pl.col('bk_dx')).sum().over('id', 'bk_bracket') / (pl.col('bk_dt') ** 2).sum().over('id', 'bk_bracket')).alias('bk_beta'),
        )
    
        new_frame = q.collect()
        
        q = new_frame.lazy().with_columns(
            (pl.col('fw_dx') - pl.col('fw_beta') * pl.col('fw_dt')).alias('fw_delta'),
            (pl.col('bk_dx') - pl.col('bk_beta') * pl.col('bk_dt')).alias('bk_delta'),
        )
    
        new_frame = q.collect()
        
        q = new_frame.lazy().with_columns(
            (pl.col('fw_delta') ** 2).mean().over('id', 'fw_bracket').alias('fw_F2'),
            (pl.col('bk_delta') ** 2).mean().over('id', 'bk_bracket').alias('bk_F2'),
        )
    
        new_frame = q.collect()

        # remove null-fluctuations in brackets that are too small
        new_frame = new_frame.filter(
            (pl.col('fw_F2') > 0) & (pl.col('bk_F2') > 0)
        )
        
        fw_frame = new_frame.group_by('id', 'fw_bracket').agg(pl.col('fw_F2').drop_nans().first())
        bk_frame = new_frame.group_by('id', 'bk_bracket').agg(pl.col('bk_F2').drop_nans().first())
        
        s_frames[s] = (fw_frame, bk_frame, )

    # start frame with fluctuation means
    F_frame = base_frame.select(pl.col('id')).group_by('id').agg(pl.all().first())

    # for each q and s, calculate F_2^q(s)
    for q in qarr:
        for s in sarr:
            fw_frame, bk_frame = s_frames.get(s)
            
            fw_frame = fw_frame.group_by('id').agg(
                (pl.col('fw_F2') ** (q/2)).mean()
            )
            
            bk_frame = bk_frame.group_by('id').agg(
                (pl.col('bk_F2') ** (q/2)).mean()
            )
            
            frame = fw_frame.join(
                bk_frame,
                on='id'
            )
                    
            frame = frame.select(
                pl.col('id'),
                (((pl.col('fw_F2') + pl.col('bk_F2')) / 2) ** (1/q)).alias(f'F_{q}({s})')
            )
        
            F_frame = F_frame.join(
                frame,
                on='id'
            )    

    # using F_frame, calculate Hs and R2s for each q
    ans_frame = base_frame.select(pl.col('id')).group_by('id').agg(pl.all().first())
    for q in qarr:
        F_np = F_frame.select(pl.col(f'^F_{q}(.*).*$')).to_numpy()
        
        line_fit = np.polyfit(
            np.log(sarr),
            np.log(F_np).T,
            deg=1
        ).T
        
        hq = line_fit[:, 0]
        R2q = R2(
            np.log(F_np),
            (hq * np.log(sarr).reshape(-1, 1) + line_fit[:, 1]).T
        )

        hq_series = pl.Series(f'h{q}', hq)
        R2q_series = pl.Series(f'R2{q}', R2q)
        
        new = ans_frame.with_columns(hq_series, R2q_series)
        
        ans_frame = ans_frame.join(
            new.select(
                pl.col('id'),
                pl.col(f'h{q}'),
                pl.col(f'R2{q}')
            ),
            on='id'
        )

    # now tau, alpha and f(alpha)
    h = ans_frame.select(pl.col('^h.*$')).to_numpy()
    tau = h * qarr.reshape(1,-1) - 1
    alpha = np.diff(tau, axis=1) / (qarr[1] - qarr[0])
    falpha = qarr[:-1] * alpha - tau[:, :-1]

    for (name, arr) in zip(['tau', 'alpha', 'falpha'], [tau, alpha, falpha]):
        ans_frame = ans_frame.with_columns(
            *[pl.Series(f'{name}{q}', arr[:, i]) for (i, q) in enumerate(qarr[:arr.shape[1]])]
        )

        

    return ans_frame

    
def R2(y, y_hat):
    """ Vectorized R^2 (coefficient of determination) function """
    if (y.ndim < 1 or y.shape != y_hat.shape):
        raise IndexError('`y` and `y_hat` must have the same shape.')
    
    y_mean = y.mean(axis=y.ndim - 1, keepdims=True)
    N = ((y_hat - y_mean) ** 2).sum(axis=y.ndim - 1)
    D = ((y - y_mean) ** 2).sum(axis=y.ndim - 1)
    
    result = N / D
    nan_mask = D == 0
    result[nan_mask] = 1
    
    return result
