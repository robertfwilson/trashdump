import sys
from os import path, getcwd

import numpy as np
from scipy.signal import fftconvolve, resample, medfilt, find_peaks, resample_poly, convolve
from scipy.stats import trim_mean, sigmaclip
import pandas as pd
from astropy.io import fits
import glob

from statsmodels.tsa.ar_model import AutoReg
from tqdm import tqdm

import batman
from wotan import flatten

import warnings

from collections import deque
from bisect import insort, bisect_left
from itertools import islice




def get_lc_files(kic, directory='data/lightcurves'):

    lc_files = '{1}/{0}/kplr*_llc.fits'.format(kic, directory)

    return sorted(glob.glob( lc_files ) )

  
#############################################################################
#
## SAVING/LOADING h5 data with Pandas following guide here:
## https://stackoverflow.com/questions/29129095/save-additional-attributes-in-pandas-dataframe/29130146#29130146
#
############################################################################
def h5store(filename, df, **kwargs):
    
    store = pd.HDFStore(filename, )
    store.put('data', df, format='f')
    store.get_storer('data').attrs.metadata = kwargs
    store.close()

#def h5load(store):
#    data = store['data']
#    metadata = store.get_storer('data').attrs.metadata
#    return data, metadata

def loadh5file(filename):

    if path.exists(filename):
    
        with pd.HDFStore(filename) as store:
            data = store['data']
            metadata = store.get_storer('data').attrs.metadata
            store.close()

        data.astype(np.float64)
        detrend_keys = []
        for col in data.columns:
            if 'detrend_mask' in col:
                detrend_keys.append(col)
        data[detrend_keys].astype(np.bool)
            
        return data, metadata
    else:
        print('NO FILE {} FOUND!!'.format(filename) )
        return None

def pandas_read_hdf(fname):

   return  pd.read_hdf(fname)


def create_df_from_lc(lc):

    df_dic = {}

    all_keys = np.concatenate( [lc.detrend_mask_keys, lc.detrend_flux_keys, lc.ses_keys, lc.num_keys, lc.den_keys])
    all_detrend_masks = np.concatenate(lc.detrend_masks)
    all_detrend_flux = np.concatenate(lc.detrend_flux)
    
    all_ses =  np.concatenate(lc.ses)
    all_num =  np.concatenate(lc.num)
    all_den =  np.concatenate(lc.den)

    all_arr = np.concatenate([all_detrend_masks, all_detrend_flux, all_ses, all_num, all_den])

    reshaped_arr = all_arr.reshape( (-1, len(all_keys)), order='F'  )

    print(np.shape(lc.detrend_masks ) )
    
    ses_df = pd.DataFrame( reshaped_arr, columns = all_keys,  )
        

    lc_dic = { 'time':lc.time,'pdcsap':lc.pdcsap, 'mom_centr1':lc.mom_centr1, 'mom_centr1_err':lc.mom_centr1_err, 'mom_centr2':lc.mom_centr2, 'mom_centr2_err':lc.mom_centr2_err}

    lc_df = pd.DataFrame( data=lc_dic )
    df = pd.concat([lc_df, ses_df], axis=1)    

    return df


def sigma_clip(x, y, upper_sig=4., lower_sig=np.inf, use_mad=False):

    with warnings.catch_warnings():
        # Ignore RuntimeWarning: invalid value encountered in less_equal."
        warnings.simplefilter("ignore", RuntimeWarning)
        
        if use_mad:
            cut_upper = y - np.nanmedian(y) < upper_sig*mad(y)
            cut_lower = np.nanmedian(y) - y < lower_sig*mad(y)
        else:
            y_nonans = y[~np.isnan(y)]
            clipped, _, _ = sigmaclip(y_nonans, low=lower_sig, high=upper_sig)
            cut_upper = y <= np.max(clipped)
            cut_lower = y >= np.min(clipped)

    cut = np.logical_and(cut_upper, cut_lower)
    
    return x[cut], y[cut], cut



def make_binned_flux(t, f, texp, P=None):

    if P is None:
        P = max(t)
        fold_t = t
    else:
        fold_t = np.mod(t,P)

    t_bins = np.arange(0, P+texp, texp)
    f_bin, _ = np.histogram(fold_t, bins=t_bins, weights=f)
    tr_num, t_binedge = np.histogram(fold_t,bins=t_bins, )
    
    f_bin_norm = f_bin/tr_num
    t_bin_mid = 0.5*(t_binedge[1:] + t_binedge[:-1] )
    
    return t_bin_mid, f_bin_norm




def moving_average(a, n) :
    
    ret = np.cumsum(a, dtype=float, )
    ret[n:] = ret[n:] - ret[:-n]
    #print(len(a), n )
    return np.pad(ret[n - 1:] / n, int(n/2), mode='reflect')




def rolling_trim_mean(x, n, percentile=0.1):
    
    lo, hi = int(percentile*n),int( percentile*n)
    
    x_sorted = np.sort(x[:n])
    res = np.repeat(np.nan, len(x))
    
    for i in range(n, len(x)):
        res[i-1] = x_sorted[lo:hi].mean()
        if i != len(x):
            idx_old = np.searchsorted(x_sorted, x[i-50])
            x_sorted[idx_old] = x[i]
            x_sorted.sort()
    return res



    


def running_median_insort(x, window_size):
    """Contributed by Peter Otten"""
    x = np.pad(x, (window_size//2, 0), mode='reflect') # Pad beginning/end of array to avoid edge effects
    seq = iter(x)
    d = deque()
    s = []
    result = []
    for item in islice(seq, window_size):
        d.append(item)
        insort(s, item)
        result.append(s[len(d)//2])
    m = window_size // 2
    for item in seq:
        old = d.popleft()
        d.append(item)
        try:
            del s[bisect_left(s, old)]
        except IndexError:
            del s[bisect_left(s, old)-1]
        insort(s, item)
        result.append(s[m])
        
    return np.array(result[window_size//2:])


def running_median_edge_fix(x, window_size):

    res1 = running_median_insort(x, window_size)
    res2 = running_median_insort(x[::-1], window_size)

    window = min(len(x)//4, window_size)

    return np.concatenate([res1[window//2:],res2[-2*window//2:-window//2]])
    

def mad(arr, scale=1.4826):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation 
    """
    med = np.nanmedian(arr)
    return scale*np.nanmedian(np.abs(arr - med))



def median_detrend(flux, t_dur, exp_time = 0.020417):
    
    kernel = 2*int(10.*t_dur/exp_time)+1
    smoothed_flux = medfilt(flux, kernel)
    
    return flux/smoothed_flux



def flag_gap_edges(x, y, min_dif=0.3, sig=2.5, gap_size=1., npoints=15):

    dx = x[1:] - x[:-1]
    gap_idx = np.array(range(len(dx)))[dx>min_dif]

    dy_std = mad(y)
    cut = x==x

    y_median = np.median(y)
    
    for j in gap_idx:

        # cut left edge
        if np.nanmean((y[j-npoints:j]-y_median)**2./dy_std**2. ) > sig:
             cut &=  np.abs(x-x[j])>gap_size
         
        # cut right edge
        if np.nanmean((y[j+1:j+npoints+1]-y_median)**2. )  > sig:
             cut &= np.abs(x-x[j+1])>gap_size


    # check edge of data:
    if np.nanmean((y[-npoints:]-y_median)**2./dy_std**2. ) > sig:
        cut &= x[-1]-x > gap_size
    
        
    return x[cut], y[cut], cut





def fill_gap_AR(y, n_add, gap):
    
    nlag2 = min([5*n_add, len(y)-gap])
    nlag1 = min([5*n_add, gap])

    yreg = y[gap - nlag1:gap]
    yreg2 = y[gap+1:gap+nlag2][::-1]    

    
    #if np.sum((yreg-np.mean(yreg))**2.)>np.sum((yreg2-np.mean(yreg2))**2.):
    fill2 = AutoReg(yreg2, lags=int(0.25*nlag2) ).fit().predict( start=len(yreg2), end=len(yreg2)+n_add-1, )
    #else:
    fill = AutoReg(yreg, lags=int(0.25*nlag1) ).fit().predict( start=len(yreg), end=len(yreg)+n_add-1, )
    
    weights = np.linspace(0,1,len(fill))
    fillgap = fill * weights + fill2[::-1] * weights[::-1]

    med_fillgap =  np.median(fillgap)
    fillgap -= med_fillgap
    
    return med_fillgap + fillgap / np.max(np.array([weights, weights[::-1]]), axis=0)
    


def fill_all_gaps_AR(x, y, cadence,  gap_size=6):
    
    dx = x[1:]-x[:-1]
        
    gap_inds = np.arange(len(dx), dtype=int)[dx>1.01*cadence]    
    
    y_tofill = []
    x_tofill = []
    fill_ind = []


    for gap in gap_inds:
        
        n_add = int(np.round(dx[gap]/cadence)-1)
        fill_ind.extend(np.zeros(n_add,dtype=int)+gap)
        
        x_tofill.extend(np.linspace(x[gap]+cadence, x[gap+1]-cadence, n_add) )

        if n_add+1>gap_size:
            gap_fill = fill_gap_AR(y, n_add, gap )
            y_tofill.extend(gap_fill )

              
        else:
            if gap<n_add:
                gap_fill = y[gap+2:gap+n_add+2][::-1]
            else:
                gap_fill =  y[gap-n_add:gap][::-1]

            y_tofill.extend(gap_fill)  
    
    y_filled = np.insert(y, obj=fill_ind, values=y_tofill, axis=0)
    x_filled = np.insert(x, obj=fill_ind, values=x_tofill, axis=0)
    boo_filled = np.insert(np.ones_like(x),obj=fill_ind,
                           values=np.zeros_like(x_tofill), axis=0)
            
    return x_filled, y_filled, boo_filled





def pad_time_series(x,y,cadence, in_mode='reflect', pad_end=False, fill_gaps=True, constant_val=0., len_pad=None, small_gap=True):
    
    padded_y = y
    padded_x = x

    truth_array = np.ones_like(x, dtype=bool)

    min_dif = 1.1*cadence
    

    if fill_gaps:
        
        dx = x[1:] - x[:-1]
        #dy = y[1:] - y[:-1]
        bad_idx = np.arange(len(dx))[dx>min_dif]
        x_add = []
        y_add = []

        idx2add = []
        
        biggap_in_mode=in_mode
        lilgap_in_mode='line'
        

        for j in bad_idx:

            x_add.extend( list(np.arange(x[j]+cadence, x[j+1]-cadence/2, cadence) ) )
            n_add = int(np.round(dx[j]/cadence))-1


            idx2add.extend( [j+1]*n_add )
            
            if n_add<3:
                in_mode=lilgap_in_mode
            else:
                in_mode=biggap_in_mode


            if in_mode=='line':
                y1 = y[j]
                y2 = y[j+1]
                
                y2add = list(y1 + ((y2-y1)/dx[j]) * (np.arange(x[j]+cadence, x[j+1]-cadence/2, cadence) - x[j] ) )
                
                
                y_add.extend( y2add )

            elif in_mode=='mean':
                y_add.extend( [np.mean(y)]*n_add )

            elif in_mode=='constant':
                y_add.extend( [constant_val]*n_add )

                
            elif in_mode=='reflect':
                n_left = int(np.ceil(n_add/2.))
                n_right = int(np.floor(n_add/2.))

                y_negative = -1*y + 2.*np.median(y)

                if j-n_left<0:
                    #print(j, n_left, n_add, len(y))
                    #ntimes = int((j-n_left)/(2*len(y)) )+1
                    y_repeat_begin = np.concatenate( [y,y_negative[::-1]] )
                    
                    y_add.extend(  y_repeat_begin[range(j+len(y)-n_left,j+len(y))][::-1] )
                else:
                    y_add.extend( y_negative[range(j-n_left,j)][::-1]  )

                if j+n_right > len(y)-1:
                    y_repeat_end = np.concatenate( [ y,y_negative[::-1], y ]  )
                    y_add.extend(  y_repeat_end[range(j,j+n_right)][::-1] )
                else:
                    y_add.extend(  y_negative[range(j,j+n_right)][::-1] )



        if in_mode=='AR':
            padded_x, padded_y, padded_boo = fill_all_gaps_AR(x,y,cadence=cadence)
            
        else:
            padded_x = np.insert(x, idx2add, x_add    )
            padded_y = np.insert(y, idx2add, y_add  )

        truth_array = np.insert(truth_array, idx2add, np.zeros_like(x_add)  )
    
    if pad_end:

        #print(len_pad is None)

        if len_pad is None:
            padded_len = int(2**np.ceil(np.log2(len(padded_x)) )  )
            #if padded_len-len(padded_x)<10:
            #    padded_len *=2
        else:
            padded_len = len_pad

        total2add = padded_len-len(padded_x)
        add_right =  int(np.floor(total2add/2))
        add_left = int(np.ceil(total2add/2))

        if total2add==0:
            return padded_x, padded_y, truth_array

        #if total2add%2==1:
        #    add_left = int(total2add/2) + 1
        #else:
        #    add_left = int(total2add/2)

        x_add_right = padded_x[-1] + np.arange(1, add_right+1)*cadence
        x_add_left =  padded_x[0] - np.arange(1, add_left+1)[::-1]*cadence
        padded_x = np.concatenate([x_add_left, padded_x, x_add_right ])
        
        y_add_left = padded_y[:add_left]
        y_add_right = padded_y[-add_right:]
        padded_y = np.concatenate([y_add_left[::-1], padded_y, y_add_right[::-1] ])

        truth_array = np.concatenate([np.zeros(add_left),truth_array,np.zeros(add_right)])

    truth_array = np.array(truth_array, dtype=bool)

    return padded_x, padded_y, truth_array



def get_transit_depth_ppm(rp, rs):

    return 84. * rp**2. / rs**2.


def get_p_tdur_t0(tce):
    return tce[1], tce[4], tce[3]

#detrending with savgol from wotan but Juliefied
def detrend_lc_savgol(time, flux, sector_dates, tce, tce_index, n_window=10, sigma_clip=3, num_iter=15):

    #copy arrays so original isn’t touched
    flux = flux.copy()
    unmasked_flux = flux.copy()

    #setup window in cadences
    window_length = n_window * tce['tdur'][tce_index]  # in days
    dt = 1800 / 86400.0   # cadence in days
    window_cadences = int(round(window_length / dt))

    #mask transits from all planets
    for _, row in tce.iterrows():
        t0, period, duration = row['t0'], row['period'], row['tdur']
        n = np.round((time - t0) / period).astype(int)
        nearest_transit = t0 + n * period
        in_transit = np.abs(time - nearest_transit) < duration * (3/4)
        flux[in_transit] = np.nan

    #prepare output arrays
    trend = np.full_like(time, np.nan, dtype=float)

    #define sector boundaries
    sector_dates = sorted(sector_dates)
    sector_edges = list(zip(sector_dates, sector_dates[1:] + [np.nanmax(time)]))

    #loop through sectors
    for (start, end) in sector_edges:
        sector_mask = (time >= start) & (time < end)
        time_sec = time[sector_mask]
        flux_sec = np.asarray(flux[sector_mask], dtype=np.float64)

        #False is when in transit, and True is when out of transit
        transit_mask = ~np.isnan(flux_sec)

        # initialize clipping mask (start with all good)
        clip_mask = transit_mask.copy()

        #iterative sigma clipping with running median
        for _ in range(num_iter):
            flux_series = pd.Series(np.where(clip_mask, flux_sec, np.nan))
            running_med = flux_series.rolling(window_cadences, center=True, min_periods=1).median().to_numpy()

            residuals = flux_sec - running_med
            residuals_masked = residuals[clip_mask]
            std = 1.4826 * np.median(np.abs(residuals_masked - np.nanmedian(residuals_masked)))
            new_mask = np.abs(residuals) < sigma_clip * std

            if np.all(new_mask == clip_mask):
                break
            clip_mask = clip_mask & new_mask

        #remove bad data points
        clipped_flux = flux_sec[clip_mask]
        clipped_time = time_sec[clip_mask]

        #pad the gaps with numbers
        padded_time, padded_flux, pad_mask = pad_time_series(clipped_time,clipped_flux,dt,in_mode='line')
       
        #detrending the padded flux using wotans savgol
        flat_sec, trend_sec = flatten(
            padded_time, padded_flux,
            window_length=window_cadences,
            method='savgol',
            edge_cutoff=1.0,
            break_tolerance=0.5,
            cval=3,
            return_trend=True
        )

        #removing padded values
        trend_copy = trend_sec[pad_mask]
        
        #initialize a full-length trend array with NaNs
        full_trend_sec = np.full_like(flux_sec, np.nan, dtype=float)

        #fill the trend where transit_mask is True 
        full_trend_sec[clip_mask] = trend_copy 

        #interpolates over the "in transit" points so there aren't NaNs there
        full_trend_sec[~transit_mask] = np.interp(time_sec[~transit_mask],time_sec[clip_mask],trend_copy)

        #store into global trend array
        trend[sector_mask] = full_trend_sec

    #detrend full light curve
    flux_detrended = unmasked_flux / trend

    return flux_detrended, trend