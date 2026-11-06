"""Object-paired descriptive statistics and BCa intervals for a mean effect."""
from statistics import NormalDist
import numpy as np


def describe(values):
    finite=np.asarray([v for v in values if v is not None and np.isfinite(v)],dtype=np.float64)
    result={'expected':len(values),'count':len(finite),'missing_or_nonfinite':len(values)-len(finite)}
    if not len(finite):return result
    result.update(mean=float(finite.mean()),median=float(np.median(finite)),
        population_std=float(finite.std(ddof=0)),sample_std=float(finite.std(ddof=1)) if len(finite)>1 else None,
        min=float(finite.min()),max=float(finite.max()),
        **{f'q{q:02d}':float(np.quantile(finite,q/100)) for q in (5,25,75,95)})
    return result


def bca_mean(differences, *, seed=20261103, resamples=10000, confidence=.95):
    """Analytic leave-one-object-out jackknife; bounded-memory paired resampling.

    Input is one effect per independent object, already averaged over model
    seeds where applicable. A fresh identical RNG gives common resampling
    indices across all complete comparisons with the same ordered object set.
    """
    x=np.asarray(differences,dtype=np.float64)
    result={'method':'BCa','resamples':resamples,'seed':seed,'confidence':confidence,
        'n_objects':len(x),'interval':None,
        'interpretation':'Object uncertainty conditional on these trained models; no training-population or equivalence claim'}
    if x.ndim!=1 or resamples<1 or not 0<confidence<1:raise ValueError('Invalid bootstrap arguments')
    if not np.isfinite(x).all():return {**result,'status':'missing_or_nonfinite_pairs'}
    if len(x)<2:return {**result,'status':'insufficient_objects'}
    estimate=float(x.mean());result['estimate']=estimate
    if np.all(x==x[0]):
        return {**result,'status':'degenerate_constant_effect','interval':[estimate,estimate],
            'note':'Empirical distribution is constant; not evidence of equivalence'}
    rng=np.random.default_rng(seed);boot=np.empty(resamples)
    for first in range(0,resamples,128):
        count=min(128,resamples-first)
        indices=rng.integers(0,len(x),size=(count,len(x)))
        boot[first:first+count]=x[indices].mean(axis=1)
    percentile=(np.count_nonzero(boot<estimate)+.5*np.count_nonzero(boot==estimate))/resamples
    if not 0<percentile<1:return {**result,'status':'degenerate_bias_correction'}
    normal=NormalDist();z0=normal.inv_cdf(float(percentile))
    # Common jackknife scale cancels from acceleration. Scale again to avoid
    # squaring tiny metric differences into floating point underflow.
    centered=x-estimate;centered=centered/np.max(np.abs(centered))
    acceleration=float(np.sum(centered**3)/(6*np.sum(centered**2)**1.5))
    probabilities=[]
    for tail in ((1-confidence)/2,(1+confidence)/2):
        z=normal.inv_cdf(tail);denominator=1-acceleration*(z0+z)
        if denominator<=0:return {**result,'status':'singular_BCa_adjustment'}
        probabilities.append(normal.cdf(z0+(z0+z)/denominator))
    if probabilities[0]>=probabilities[1]:return {**result,'status':'invalid_BCa_quantiles'}
    return {**result,'status':'ok','interval':np.quantile(boot,probabilities).tolist(),
        'bias_correction':z0,'acceleration':acceleration,'adjusted_quantiles':probabilities}


def paired_effect(left,right,ids,*,seed=20261103,resamples=10000,higher_is_better=False):
    """Mappings contain object -> one scalar; never silently drop missing pairs."""
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate expected object IDs')
    missing=[i for i in ids if i not in left or i not in right or left[i] is None or right[i] is None
        or not np.isfinite(left[i]) or not np.isfinite(right[i])]
    values=[left[i]-right[i] for i in ids if i not in set(missing)]
    worse=[v<0 if higher_is_better else v>0 for v in values]
    return {'direction':'left_minus_right','summary':describe(values),'missing_pair_ids':missing,
        'complete':not missing,'worsened_fraction':float(np.mean(worse)) if worse else None,
        'improved_fraction':float(np.mean([v>0 if higher_is_better else v<0 for v in values])) if values else None,
        'tied_fraction':float(np.mean(np.asarray(values)==0)) if values else None,
        'bca':bca_mean(values,seed=seed,resamples=resamples) if not missing else
            {'status':'missing_pairs','interval':None,'n_expected':len(ids),'n_available':len(values)}}
