
'''
Removal of Exoplanet Candidates Yearning to be Condoned as Lookalikes and Eclipsing BINaries
'''


import sys
from os import path, getcwd

import numpy as np
from scipy.signal import fftconvolve, resample, medfilt, find_peaks, resample_poly, convolve
from scipy.stats import trim_mean, sigmaclip
import pandas as pd
from astropy.io import fits
import glob

import matplotlib.pyplot as plt

from tqdm import tqdm

import batman
import wotan

from scipy.special import  erfcinv
from scipy.optimize import curve_fit
from lmfit import minimize, Parameters

from .dump import make_transit_mask, calc_mes, calc_var_stat
from .utils import *
from .fitfuncs import *
from .ses import get_transit_signal, ocwt, calc_var_stat


class RecycleBin(object):

    def __init__(self, Dump, TCEs):


        self.dump = Dump
        self.tces = TCEs

        self.refined_tces = TCEs.copy()

        self.exptime = Dump.lc.exptime
        self.limb_coeff = Dump.limb_darkening[0]

        self.time = Dump.lc.time[Dump.lc.mask]
        self.flux = Dump.lc.flux[Dump.lc.mask]
        self.trend = Dump.lc.trend[Dump.lc.mask]
        self.raw_flux = self.trend*self.flux
        self.flux_err = Dump.lc.flux_err[Dump.lc.mask]

        self.batman_model, self.batman_params = self._get_batman()


    def _get_tce_mask(self, tce_num, time=None):

        P,width,t0 = get_p_tdur_t0(self.tces[tce_num])
        return make_transit_mask(self.time, P, t0, width)

    def _get_previous_tce_masks(self, tce_num):

        i=0
        previous_tce_mask = np.ones_like(self.dump.lc.time, dtype=bool)
        while i<tce_num:
            P,width,t0 = get_p_tdur_t0(self.tces[i])
            previous_tce_mask &= make_transit_mask(self.dump.lc.time, P, t0, width)
            i+=1
        return previous_tce_mask

    def _get_other_tce_masks(self, tce_num):

        i=0
        other_tce_mask = np.ones_like(self.time, dtype=bool)
        while i<len(self.tces):
            if i==tce_num:
                i+=1
                continue
            P,width,t0 = get_p_tdur_t0(self.tces[i])
            other_tce_mask &= make_transit_mask(self.time, P, t0, width)
            i+=1
        return other_tce_mask



    def _get_batman(self, limb_law='nonlinear'):

        params = batman.TransitParams()       #object to store transit parameters
        params.t0 = self.time[0]              #time of inferior conjunction
        params.per = 1.                       #orbital period
        params.rp = 0.1                       #planet radius (in units of stellar radii)
        params.a = 15.                        #semi-major axis (in units of stellar radii)
        params.inc = 90.                      #orbital inclination (in degrees)
        params.ecc = 0.                       #eccentricity
        params.w = 90.                        #longitude of periastron (in degrees)
        params.limb_dark = limb_law           #limb darkening model
        params.u = self.limb_coeff

        m = batman.TransitModel(params, self.time, supersample_factor=5, exp_time=self.exptime, max_err=0.1)

        return m, params

    def _get_lightcurve(self, params=None):
        if params is None:
            return self.batman_model.light_curve(self.batman_params)
        else:
            return self.batman_model.light_curve(params)


    def _transit_resids(self, par, robust=False):

        try:
            vals = par.valuesdict()
        except:
            vals=par
    
        b = vals['b']
        tdur = vals['tdur']
        t0 = vals['t0']
        per = vals['per']
        logdepth = vals['logdepth']

        rp_rs = np.sqrt( 10.**logdepth )
        
        a_rs = np.sqrt( (1. + rp_rs)**2. - b**2.)/np.sin((np.pi*tdur)/per )
        

        if b/a_rs>1.:
            #print('b>a_rs!!! NO!!!!!')
            return np.ones_like(self.flux)*1e20
        
        inc = np.rad2deg( np.arccos(b/a_rs) )
    
        self.batman_params.t0 = t0                     
        self.batman_params.per = per
        self.batman_params.rp = rp_rs    
        self.batman_params.a =  a_rs
        self.batman_params.inc = inc

        lc = self._get_lightcurve()

        if robust:
            resids = (self.flux-lc) / self.flux_err**2.
        else:
            resids = (self.flux-lc) / self.flux_err**2.
            
        if any(np.isnan(resids)):
            print(inc, a_rs, rp_rs)
            print('NaN in Transit Modelling')

        return resids

        
    

    def _update_tce(self, tce_num, fit_method='leastsq', return_fit=False, robust=False):


        P,width,t0 = get_p_tdur_t0(self.tces[tce_num])        
        
        params = Parameters()

        params.add('logdepth', value=-3., vary=True, max=0, min=-6)

        if P>max(self.time)-min(self.time):
            params.add('per', value=max(self.time)-min(self.time), vary=False,)
        else:
            params.add('per', value=P, vary=True, min=.99*P, max=1.01*P)
            
        params.add('t0', value=t0, vary=True, min=t0-2*width, max=t0+2*width)
        params.add('tdur', value=width, min=0.25 * width, max=min(2*width,P/2), vary=True)
        params.add('b', value=0.1, min=0., max=1., vary=True)
        params.add('rp_rs', expr='10**(0.5 * logdepth)')
        params.add('a_rs', expr='sqrt( (1 + rp_rs)**2 - b**2)/sin(pi*tdur/per)', min=1.)
        

        #in_guess = minimize(self._transit_resids, params,method='LBFGS', nan_policy='omit')

        if fit_method=='emcee':
            emcee_kws = dict(steps=2000, burn=500,progress=True,nan_policy='omit')
            tce_out = minimize(self._transit_resids, params, method=fit_method , **emcee_kws)
        else:
            tce_out = minimize(self._transit_resids, params, method=fit_method)
            
        out = tce_out.params.valuesdict()

        if P>max(self.time)-min(self.time):
            self.refined_tces[tce_num,1] = P
        else:
            self.refined_tces[tce_num,1] = out['per']

        self.refined_tces[tce_num,2] = 10.**out['logdepth']
        self.refined_tces[tce_num,3] = out['t0']
        self.refined_tces[tce_num,4] = out['tdur']

        if return_fit:
            return out


        
    def _calc_mes(self, tce_num, mask_tce=False, calc_ses=False, use_mask=None,
                  width=None):

        if width is None:
            P,width,t0 = get_p_tdur_t0(self.tces[tce_num])
        else:
            P,_,t0 = get_p_tdur_t0(self.tces[tce_num])

       
        mask = self.dump.lc.mask.copy()
        if not(use_mask is None):
            mask&=use_mask

        if mask_tce:
            mask &= make_transit_mask(self.dump.lc.time, P, t0, width)

        if calc_ses:
            self.dump.Calculate_SES_by_Segment(mask=mask, tdurs=[width])

        num = self.dump.num[0]
        den = self.dump.den[0]
        
        foldtime = (self.dump.lc.time.copy() - t0 + P/4.)%P 
        
        mestime, mes = calc_mes(foldtime[mask],num,den,P,texp=self.dump.lc.exptime,n_trans=1., return_nans=True)


        return mestime-P/4., (mestime/P) , mes



    
    def _get_odd_even_mes(self, tce_num, ):

        P,width,t0 = get_p_tdur_t0(self.tces[tce_num])
        P_twice =P*2.
        

        foldtime = (self.time - t0 + P_twice/4.)%P_twice 

        width_i = np.argmin(np.abs(width-self.dump.tdurs) )
        num = self.dump.num[width_i]
        den = self.dump.den[width_i]

        mestime, mes = calc_mes(foldtime,num,den,P_twice,texp=self.dump.lc.exptime,n_trans=1)

        mestime =  mestime/(P/2.) - 0.5

        odd_mes = mes[mestime<0.5]
        even_mes = mes[mestime>=0.5]

        odd_time= mestime[mestime<0.5]
        even_time= mestime[mestime>=0.5]


        return odd_time, odd_mes, even_time, even_mes, mad(odd_mes), mad(even_mes)

    def robust_depth_test(self, tce_num, mask=None):


        

        

        return 1. 


    def weak_secondary_test(self, tce_num, mask=None):

        mestime, mesphase, mes = self._calc_mes(tce_num,calc_ses=True,mask_tce=True, use_mask=mask)

        mes[mes==0] = np.nan
        
        max_secondary_mes = np.nanmax(mes)
        max_secondary_phase = mesphase[np.nanargmax(mes)]-0.25
        min_secondary_mes = np.nanmin(mes)
        min_secondary_phase = mesphase[np.nanargmin(mes)]-0.25
        mad_secondary_mes = mad(mes)
        

        output = {'max_sec_mes': max_secondary_mes,
                  'max_sec_mes_phase': max_secondary_phase,
                  'min_sec_mes': min_secondary_mes,
                  'min_sec_mes_phase':min_secondary_phase,
                  'mad_sec_mes':mad_secondary_mes
                  }
        
        return output

    def same_period_test(self, tce_num):
        
        same_period_stats = same_period_test(self.tces)

        return same_period_stats[tce_num]


    def min_transits_test(self, tce_num, n_tdur=1, min_frac=0.5):


        P,width,t0 = get_p_tdur_t0(self.refined_tces[tce_num])
        cadence = self.dump.lc.exptime
        min_transits=self.dump.min_transits

        num_good_transits = check_number_good_transits(self.time, self.flux, P, t0, width,
                                                       cadence=cadence,min_frac=min_frac,
                                                       min_transits=min_transits)

        return num_good_transits>=min_transits
        
        
        
    
    def cosine_vs_transit_local(self, tce_num, showplot=False, nwidth=4,local_plots=False):

        P,width,t0 = get_p_tdur_t0(self.tces[tce_num])
        depth = self.refined_tces[tce_num,2]



        chit, chic, nf = compare_cosine_and_transit_model(time=self.time,
                                                          flux=self.flux*self.trend,
                                                          t0=t0, width=width, P=P, 
                                                          exptime=self.exptime,
                                                          depth=depth**0.5,
                                                          limb_dark_coeffs=self.limb_coeff,
                                                          flux_err=self.flux_err,
                                                          plot=showplot, nwidth=nwidth,
                                                          local_plots=local_plots)

        return chit, chic, nf

    def cosine_vs_transit_global(self, tce_num, showplot=False, nwidth=4, depth=None, return_plot_values=False):

        P,width,t0 = get_p_tdur_t0(self.refined_tces[tce_num])
        depth = self.refined_tces[tce_num,2]

        
        if not(return_plot_values):
            chic, chit, nf = compare_cosine_and_transit_model(time=self.time,
                                                          flux=self.flux,
                                                          t0=t0, width=width, P=P,
                                                          depth=depth**0.5,
                                                          exptime=self.exptime,
                                                          limb_dark_coeffs=self.limb_coeff,
                                                          flux_err=self.flux_err,
                                                          plot=showplot, nwidth=nwidth,
                                                          local_plots=False,global_fit=True)

            output = {'global_tran_red_chi2':chit, 'global_sin_red_chi2':chic, 'global_tran_chi2':chit*nf,  'global_sin_chi2':chic*nf, 'global_diff_red_chi2':chit-chic, 'global_diff_chi2':(chit-chic)*nf,}

            
        
            return output

        else:

            chic, chit, nf, plot_vals = compare_cosine_and_transit_model(time=self.time,
                                                          flux=self.flux,
                                                          t0=t0, width=width, P=P,
                                                          depth=depth**0.5,
                                                          exptime=self.exptime,
                                                          limb_dark_coeffs=self.limb_coeff,
                                                          flux_err=self.flux_err,
                                                          plot=showplot, nwidth=nwidth,
                                                          local_plots=False,
                                                             global_fit=True,
                                                              return_plot=True)

            output = {'global_tran_red_chi2':chit, 'global_sin_red_chi2':chic, 'global_diff_red_chi2':chit-chic, 'global_tran_chi2':chit*nf,  'global_sin_chi2':chic*nf, 'global_diff_chi2':(chit-chic)*nf,}

            return output, plot_vals




    def _get_mes_metrics(self, tce_num , use_mask=None):

        P,width,t0 = get_p_tdur_t0(self.refined_tces[tce_num])

        try:
            mestime, mesphase, mes = self._calc_mes(tce_num, use_mask=use_mask, calc_ses=True)
        except RuntimeError:
            self.dump.Calculate_SES()
            mestime, mesphase, mes = self._calc_mes(tce_num, use_mask=use_mask)
            

        phase = mestime/P - 0.5

        max_mes = np.nanmax(mes)
        #min_mes = np.nanmin(mes) 
        mad_mes = mad(mes)
        mes_over_mad = max_mes/mad_mes

        mes_secondary = np.max(mes[~np.logical_and(mestime>P/4.-width,mestime>P/4.-width)])
        
        
        out_dict = {'max_mes':max_mes, #'min_mes':min_mes,
                    'mad_mes':mad_mes,
                    'mes_over_mad':mes_over_mad, }#'max_secondary_mes':mes_secondary}
        return out_dict



    def odd_even_depth_test(self, tce_num, return_dict=False,fit_method='leastsq',):

        P,width,t0 = get_p_tdur_t0(self.refined_tces[tce_num])

        _, both, even, odd = odd_even_transit_depths(t=self.time,f=self.flux,ferr=self.flux_err,
                                             P=P,t0=t0,width=width,fit_method='leastsq',
                                                    initial_fit_method='LBFGS')


        if not(even.params['a'].stderr is None) and  not(odd.params['a'].stderr is None):
        #if even.errorbars and odd.errorbars:
            stat = np.abs(odd.params['a']-even.params['a'])/np.sqrt(even.params['a'].stderr**2. + odd.params['a'].stderr**2.)
        else:
            print('Odd-Even Test Failed')
            stat=np.nan

        if return_dict:
            return {'odd_even_depth_stat':stat,
                    'odd_depth':odd.params['a'].value,
                    'even_depth':even.params['a'].value,
                    'odd_depth_err':odd.params['a'].stderr,
                    'even_depth_err':even.params['a'].stderr,
                    }
        else:
            return both, even, odd, stat


    def odd_even_mes_test(self, tce_num):

        odd_phase, odd_mes, even_phase, even_mes, odd_mad, even_mad = self._get_odd_even_mes(tce_num)

        odd_peak_idx = np.argmin(np.abs(odd_phase))
        odd_peak = odd_mes[odd_peak_idx-2:odd_peak_idx+2]
        
        even_peak_idx = np.argmin(np.abs(even_phase-1.))
        even_peak = even_mes[even_peak_idx-2:even_peak_idx+2]

        odd_max = np.max(odd_peak)
        even_max = np.max(even_peak)


        out_dict = {'odd_even_mes_ratio':min(odd_max/even_max, even_max/odd_max),
                    'odd_mes': odd_max, 'even_mes':even_max}
        
        return out_dict



    def local_morphology_test(self, tce_num, **test_kw):
        
        P,width,t0 = get_p_tdur_t0(self.refined_tces[tce_num])
        depth = self.refined_tces[tce_num,2]

        t = self.time
        f = self.flux * self.trend
        ferr = self.flux_err
        texp=self.exptime

        if len(self.time) * texp / P > 100:
            return {}

        morph_test = bic_morphology_test(t,f,ferr,P,t0,width,depth,texp=texp,**test_kw)

        
        return morph_test


    def chi2_tests(self, tce_num, calc_ses=False, use_mask=None, mask_other_tces=False):

        P,width,t0 = get_p_tdur_t0(self.refined_tces[tce_num])

        texp=self.exptime


        if not(use_mask is None):
            #use_mask &= self.dump.lc.mask
            time = self.time[use_mask]
            flux = self.flux[use_mask]
        else:
            time = self.dump.lc.time[self.dump.lc.mask]
            flux = self.dump.lc.flux[self.dump.lc.mask]

        #print(texp)
            
                
        channel_red_chi2, channel_chi2_stat = channel_chi2_statistic(time, flux, t0, P, width, cadence=texp,
                                                                     sector_dates=self.dump.sector_dates)
        temporal_red_chi2, temporal_chi2_stat = temporal_chi2_statistic(time, flux, t0, P, width ,
                                                                        cadence=texp,sector_dates=self.dump.sector_dates)    
        
        return {'channel_red_chi2':np.round(channel_red_chi2, 3),'temporal_red_chi2': np.round(temporal_red_chi2,3),'channel_chi2_stat':np.round(channel_chi2_stat,3),'temporal_chi2_stat':np.round(temporal_chi2_stat,3)}


    def get_preliminary_vetting_metrics(self, tce_num):
        
        bestfit = self._update_tce(tce_num, return_fit=True)

        return 1. 


        

    def get_all_vetting_metrics(self, tce_num, local_test=True):

        bestfit = self._update_tce(tce_num, return_fit=True)

        depth = self.refined_tces[tce_num,2]
        P,width,t0 = get_p_tdur_t0(self.refined_tces[tce_num])
        b = bestfit['b']
        
        tce_id= float(self.tces[tce_num,0] + (tce_num+1)*1e-2 )

            
        other_tce_mask = self._get_other_tce_masks(tce_num)
        previous_tce_mask = self._get_previous_tce_masks(tce_num)

        if local_test:
            result_list =[{'tce_id':tce_id,'period':P, 'depth_ppt':depth*1e3,'tdur':width,'t0':t0, 'b':b} ,
                     self._get_mes_metrics(tce_num, use_mask=previous_tce_mask),
                     self.chi2_tests(tce_num, use_mask=None),
                     self.weak_secondary_test(tce_num, mask=previous_tce_mask),
                     #self.odd_even_mes_test(tce_num),
                     self.odd_even_depth_test(tce_num, True),
                     self.cosine_vs_transit_global(tce_num,depth=depth),
                     self.local_morphology_test(tce_num)]
        else:
            result_list =[{'tce_id':tce_id,'period':P, 'depth_ppt':depth*1e3,'tdur':width,'t0':t0, 'b':b} ,
                     self._get_mes_metrics(tce_num, use_mask=previous_tce_mask),
                     self.chi2_tests(tce_num),
                     self.weak_secondary_test(tce_num, mask=previous_tce_mask),
                     #self.odd_even_mes_test(tce_num),
                     self.odd_even_depth_test(tce_num, True),
                     self.cosine_vs_transit_global(tce_num,depth=depth)]
            
                     
        results = {k: v for d in result_list for k, v in d.items()}

        return results




    


def fit_sinewave(x, y, yerr, p0=None):
    
    sine_func = lambda x,c,a,p,x0: c+a*np.cos((2.*np.pi) * (x-x0) / p)

    result = curve_fit(sine_func, xdata=x, ydata=y, p0=p0, sigma=yerr, 
                       bounds=[[-np.inf,-np.inf, p0[2]/3.,-np.inf],[np.inf,np.inf, p0[2]*4.,np.inf]], 
              check_finite=True, method='trf', jac=None, )
    
    resids = y - sine_func(x, *result[0])
    
    chi2 = np.sum(resids**2./yerr**2.)

    plot_x = np.linspace(min(x), max(x), 100)
    
    return chi2, resids, plot_x, sine_func(plot_x, *result[0])
    
    
    
def fit_transit(x,y,yerr,period,limb_darkening,exptime,p0,t0):
    
    params = batman.TransitParams()
    params.t0 = t0                       #time of inferior conjunction
    params.per = np.pi*(p0[1])/np.arcsin(1./50.)                      #orbital period
    params.rp = 1e-3                     #planet radius (in units of stellar radii)
    params.a = 50.                       #semi-major axis (in units of stellar radii)
    params.inc = 90.                     #orbital inclination (in degrees)
    params.ecc = 0.                      #eccentricity
    params.w = 90.                       #longitude of periastron (in degrees)
    params.u = limb_darkening            #limb darkening coefficients [u1, u2]
    params.limb_dark = "nonlinear"       #limb darkening model
              
    m = batman.TransitModel(params, x, exp_time=exptime, fac=0.001)

    def lc_func(t, *par):
        
        rprs, width, t0, c = par
        params.per = np.pi*(width)/np.arcsin(1./50.)                      #orbital period
        params.rp = 10.**rprs   
        params.t0 = t0
        return m.light_curve(params)-1.+c
    

    p0 = np.append(p0,[0])
    result = curve_fit(lc_func, xdata=x, ydata=y, p0=p0, sigma=yerr, absolute_sigma=True, 
                       bounds=[[-8, p0[1]/5., -p0[1],-1],[0., p0[1]*5., p0[1],1]],
                       check_finite=True, jac=None, max_nfev=1000*len(x))
    
    
    resids = y - lc_func(x, *result[0])
    chi2 = np.sum(resids**2./yerr**2.)
    
    x_fold =(x+period/2.)%period-period/2.

    plot_x = np.linspace(min(x_fold), max(x_fold), 100)

    
    m = batman.TransitModel(params, plot_x, exp_time=exptime, fac=0.001, supersample_factor=5, max_err=0.1)
    
    return chi2, resids, plot_x, lc_func(plot_x,  *result[0] )
    
    
    


def compare_cosine_and_transit_model(time,flux,t0,width,P,limb_dark_coeffs,exptime,
                                     flux_err=None,nwidth=4,plot=True,local_plots=False,
                                     global_fit=False, depth=None, return_plot=False):

    if depth is None:
        depth=1e-3

    if global_fit:
        foldtime = (time - t0 + P/2.)%P - P/2.
        mask = ~make_transit_mask(time, P, t0, width*nwidth/2)

        transit_time = foldtime[mask]
        cosine_time = foldtime[mask]

        transit_flux = flux[mask]-1.
        cosine_flux = flux[mask]-1.

        transit_fluxerr=flux_err[mask]
        cosine_fluxerr=flux_err[mask]


    else:
        transit_time,cosine_time,transit_flux,cosine_flux,fluxerr,_ = get_local_cosine_and_transit_fits(time, flux, t0, width, P, limb_dark_coeffs, flux_err=flux_err, nwidth=nwidth, show_plots=local_plots)

    
        trim_to_width = np.abs(transit_time)< nwidth*width/2.
        transit_time = transit_time[trim_to_width]
        transit_flux = transit_flux[trim_to_width]
    
        trim_to_width_cos = np.abs(cosine_time)< nwidth*width/2.
        cosine_time = cosine_time[trim_to_width_cos]
        cosine_flux = cosine_flux[trim_to_width_cos]

        transit_fluxerr = fluxerr[trim_to_width]        
        cosine_fluxerr = fluxerr[trim_to_width_cos]

        
    sine_chi2s = []
    all_sine_results = []
    for n in [0.5,1,2]:
        results = fit_sinewave(cosine_time, cosine_flux, yerr=cosine_fluxerr,p0=[0., depth, n*width,0.])  
        sine_chi2s.append(results[0])
        all_sine_results.append(results)
        
    
    sine_chi2, sine_resids, sine_x, sine_y = all_sine_results[np.argmin(sine_chi2s)]


    tran_chi2, tran_resids, tran_x, tran_y = fit_transit(transit_time, transit_flux, yerr=transit_fluxerr, 
                                                         p0=[np.log10(depth),width,0.], t0=0.,
                                                         period=P,limb_darkening=limb_dark_coeffs,
                                                         exptime=exptime)
    

    tran_chi2 = tran_chi2/len(tran_resids)
    sine_chi2 = sine_chi2/len(sine_resids)
    

    if plot:
        
        t_bins = np.arange(min(transit_time), max(transit_time)+width/3, width/3.)
    
        cosine_hist, t_hist = np.histogram(cosine_time, bins=t_bins, weights=cosine_flux) 
        transit_hist, _ = np.histogram(transit_time, bins=t_bins, weights=transit_flux)
        t_transit_count, _ = np.histogram(transit_time, bins=t_bins,)     
        t_cosine_count, _ = np.histogram(cosine_time, bins=t_bins,)     

    
        sine_resid_hist,_ = np.histogram(cosine_time, bins=t_bins, weights=sine_resids)
        tran_resid_hist,_ = np.histogram(transit_time, bins=t_bins, weights=tran_resids)
        
        fig = plt.figure(figsize=(6,4))
        gs = fig.add_gridspec(3, 4, hspace=0, wspace=0)
        
        ax3 = fig.add_subplot(gs[:2, 2:], )
        ax4 = fig.add_subplot(gs[2, 2:], sharex=ax3,)

        ax1 = fig.add_subplot(gs[:2, :2], sharex=ax3,sharey=ax3)
        ax2 = fig.add_subplot(gs[2, :2], sharex=ax3, sharey=ax4)
        
        ax1.plot(cosine_time, cosine_flux, '.', color='0.7',markersize=2)
        ax1.plot((t_hist[1:]+t_hist[:-1])/2., cosine_hist/t_cosine_count, 'o', color='dodgerblue',markeredgecolor='k')
        
        ax3.plot(transit_time, transit_flux, '.', color='0.7',markersize=2)
        ax3.plot((t_hist[1:]+t_hist[:-1])/2., transit_hist/t_transit_count, 'o', color='dodgerblue',markeredgecolor='k')
        
        
        ax1.plot(sine_x, sine_y, '-', color='tomato', lw=2, 
                 label='$\mathregular{\chi^2_{\\nu, sine}}$'+'\n={:.3f}'.format(sine_chi2))
        ax1.legend(framealpha=0., loc='lower right')
        
        ax3.plot(tran_x, tran_y, '-', color='tomato', lw=2, 
                 label='$\mathregular{\chi^2_{\\nu, tran}}$'+'\n={:.3f}'.format(tran_chi2))
        ax3.legend(framealpha=0., loc='lower right')
        
        
        ax2.plot(cosine_time, sine_resids, '.', color='0.7',markersize=2)
        ax2.plot((t_hist[1:]+t_hist[:-1])/2., sine_resid_hist/t_cosine_count, 'o', color='dodgerblue',
                 markeredgecolor='k')
        ax2.plot(sine_x, np.zeros_like(sine_y), '-', color='tomato', lw=2)

        ax4.plot(transit_time, tran_resids, '.', color='0.7',markersize=2)
        ax4.plot((t_hist[1:]+t_hist[:-1])/2., tran_resid_hist/t_transit_count, 'o', color='dodgerblue',
                 markeredgecolor='k')
        ax4.plot(tran_x, np.zeros_like(tran_y), '-', color='tomato', lw=2)
        
        ax4.set_xlabel('$\mathregular{\Delta t_0}$ [days]')
        ax2.set_xlabel('$\mathregular{\Delta t_0}$ [days]')
        
        ax1.set_ylabel('$\mathregular{\delta F/F }$')
        ax2.set_ylabel('Resids')
        
        #ax1.set_xticklabels([])
        #ax3.set_xticklabels([])


        plt.tight_layout()
#        plt.show()
        
        
    nfreedom = len(transit_time) - 3.

    if return_plot:
        return sine_chi2, tran_chi2, nfreedom, ((cosine_time, cosine_flux, sine_resids), (sine_x, sine_y), (transit_time, transit_flux, tran_resids), (tran_x, tran_y) )
        
    return sine_chi2, tran_chi2, nfreedom




def get_local_cosine_and_transit_fits(time, flux, t0, width, P, limb_dark_coeffs,
                                      flux_err=None, nwidth=1,show_plots=False):
    
    
    if flux_err is None:
        flux_err = np.ones(len(flux))*np.nanstd(flux)
        
    i=0
    
    exptime = np.median(time[1:]-time[:-1])

    dt_cosine = np.array([])
    dt_transit = np.array([])
    df_transit = np.array([])
    df_cosine = np.array([])    
    dferr_transit = np.array([])
    dferr_cosine = np.array([])
    
    while t0 < max(time):

        t0 += i*P
        i+=1

        t_mask = np.abs(t0-time) <= nwidth*width
        time_seg = time[t_mask]
        flux_seg = flux[t_mask]
        fluxerr_seg = flux_err[t_mask]

        if len(flux_seg)> 0.05 * (width/exptime):
            c_results=[]
            t_results=[]
            
            try:
                c_result = fit_cosine_local_poly(x=time_seg, y=flux_seg, yerr=fluxerr_seg,
                                               p0=[1.,0,0,0.,width,t0])

                t_result = fit_transit_local_poly(x=time_seg, y=flux_seg,yerr=fluxerr_seg,
                                                  t0=t0,width=width,p0=[1.,0,0,-3,t0],
                                                  limb_darkening=limb_dark_coeffs)


                if show_plots:
                    plt.plot(time_seg, flux_seg, 'ko')
                    plt.plot(time_seg, t_result[1], 'r-', label='Transit')
                    plt.plot(time_seg, c_result[1], 'b-', label='Cosine')

                    plt.axvline(t0, ls='--',color='0.7')
                    plt.show()

                cosine_par = c_result[2]
                transit_par = t_result[2]

                t0_fit_tr = transit_par[-1]
                t0_fit_cos = cosine_par[-1]

                dt_transit = np.append(dt_transit, time_seg-t0_fit_tr)
                dt_cosine = np.append(dt_cosine, time_seg-t0_fit_cos)
                df_transit = np.append(df_transit, flux_seg/np.poly1d(transit_par[:3][::-1])(time_seg))
                df_cosine = np.append(df_cosine, flux_seg/np.poly1d(cosine_par[:3][::-1])(time_seg))
                dferr_transit = np.append(dferr_transit, fluxerr_seg)
                dferr_cosine = np.append(df_cosine, fluxerr_seg)

            
            except RuntimeError:
                pass
    
    return dt_transit, dt_cosine, df_transit-1., df_cosine-1., dferr_transit, dferr_cosine


def fit_cosine_local_poly(x,y,yerr,p0):
    
    all_chi2 = []
    all_fitpar = []
    for n in [0.5,1]:
        fitfunc = lambda x,a0,a1,a2,amp,per,t0: (a0 + a1*x +a2*x**2.) + amp*np.cos(2*np.pi*(x-t0)/per)
        a0,a1,a2,amp,per,t0 = p0
        p0_n = [a0,a1,a2,amp,per*n,t0]
        fit_par, fit_var = curve_fit(fitfunc, xdata=x, ydata=y, sigma=yerr, p0=p0_n,
                                    bounds=[[-np.inf,-np.inf,-np.inf,-np.inf, n*per/2.,
                                             t0-per/10.],
                                            [ np.inf, np.inf, np.inf, 0., n*per*2.,
                                              t0+per/10.]])
        
        all_chi2.append( sum(( (y[i] - fitfunc(x[i], *fit_par))**2./yerr[i]**2. for i in range(len(y)) )  ) )
        all_fitpar.append(fit_par)
        
    best_fitpar = all_fitpar[np.argmin(all_chi2)]
    best_chi2 = np.min(all_chi2)
    
    yfit = fitfunc(x, *best_fitpar)
            
    return best_chi2, yfit, fit_par


def fit_transit_local_poly(x,y,yerr,p0,t0,width,limb_darkening,exptime=0.0204):
        
    params = batman.TransitParams()
    params.t0 = t0                       #time of inferior conjunction
    params.per = np.pi*(width)/np.arcsin(1./50.)                      #orbital period
    params.rp = 0.1                      #planet radius (in units of stellar radii)
    params.a = 50.                       #semi-major axis (in units of stellar radii)
    params.inc = 90.                     #orbital inclination (in degrees)
    params.ecc = 0.                      #eccentricity
    params.w = 90.                       #longitude of periastron (in degrees)
    params.u = limb_darkening            #limb darkening coefficients [u1, u2]
    params.limb_dark = "nonlinear"       #limb darkening model
         
    m = batman.TransitModel(params, x, exp_time=exptime, max_err=0.1)

    def lc_func(t, *par):
        a0,a1,a2,rprs,t_0 = par
        params.rp = 10.**rprs  
        params.t0 = t_0
        return  (a0 +a1*t + a2*t**2.) *  m.light_curve(params)      
            
    fit_par, fit_var = curve_fit(lc_func, xdata=x, ydata=y, sigma=yerr, p0=p0,
                                bounds=[[-np.inf, -np.inf,-np.inf,-8,t0-width/10.],
                                        [np.inf, np.inf,np.inf, 0,t0+width/10.]])
    chi2 = sum( ((y[i]-lc_func(x[i],*fit_par))**2./yerr[i]**2. for i in range(len(y))) )
    
    yfit = lc_func(x, *fit_par)
            
    return chi2, yfit, fit_par






def odd_even_transit_depths(t,f,ferr,P,t0,width,initial_fit_method='LBFGS',
                            fit_method='leastsq'):

    phase = (t-t0 + P/4)%(2*P) 

    odd = phase<P
    even = phase>=P

    odd_flux = f[odd]
    odd_phase = phase[odd]
    
    even_flux = f[even]
    even_phase = phase[even]-P
    
    
    bothparams = Parameters()
    bothparams.add('t0', value=0.25*P, vary=True,min=0.2*P,max=0.3*P)
    bothparams.add('a', value=1e-3, vary=True, min=1e-6, max=.99)
    bothparams.add('b', value=1e3, vary=True, min=100, max=1e4)
    bothparams.add('tdur', value=width, min=0.2 * width, max=5*width, vary=True)
    bothparams.add('c0', value=1., vary=True,)
    bothparams.add('c1', value=0., vary=False)
    bothparams.add('c2', value=0., vary=False)

    

    initresult = minimize(trap_residual, bothparams,
                          args=(phase%P, f, ferr), method=initial_fit_method, reduce_fcn='neglogcauchy')

    bothresult =  minimize(trap_residual, initresult.params,
                           args=(phase%P, f, ferr),method=fit_method, reduce_fcn='neglogcauchy')

    bothresult.params['b'].vary=False
    
    evenresult = minimize(trap_residual, bothresult.params,
                          args=(even_phase, even_flux, ferr[even]),method=fit_method, reduce_fcn='neglogcauchy')

    oddresult = minimize(trap_residual, bothresult.params,
                          args=(odd_phase, odd_flux, ferr[odd]),method=fit_method, reduce_fcn='neglogcauchy')


    return phase, bothresult, evenresult, oddresult



def check_number_good_transits(t, f, P, t0, tdur, n_tdur=1, cadence=0.0204, min_frac=0.5, min_transits=4):

    tn=t0
    n_good_transits=0

    while tn<max(t) and n_good_transits <= min_transits:

        t_min, t_max = tn - tdur, tn + tdur

        min_points = min_frac * (t_max-t_min)/cadence

        if sum(np.logical_and(t>t_min, t<t_max))>min_points:
            n_good_transits+=1

        tn+=P

    return n_good_transits


def robust_stat():


    return 1. 




def bic_morphology_test(t, f, ferr, P, t0, tdur, depth=1e-4, fit_method='LBFGS',
                        ntdur=3,texp=0.0204,show_plot=False,min_frac=0.8,
                        show_progress=True,mask_detrend=False):



    if P<0:
        P = (t[-1]-t[0])
    
    boxparams = Parameters()
    boxparams.add('a', value=depth, min=0.)
    boxparams.add('t0', value=0., vary=True, min=-2*tdur, max=2*tdur )
    boxparams.add('tdur', value=tdur*0.75, vary=True, min=.5*tdur, max=2*tdur)
    boxparams.add('c0', value=np.median(f), vary=True)



    rampparams = Parameters()
    rampparams.add('a', value=depth*2., max=1., min=0. )
    rampparams.add('d', value=2, min=1, max=100)
    rampparams.add('t0', value=-tdur/4., vary=True, min=-tdur, max=tdur  )
    rampparams.add('tdur', value=tdur, vary=True, min=0, max=4*tdur  )
    rampparams.add('c0', value=np.median(f), vary=True)



    spsdparams = Parameters()
    spsdparams.add('a', value=depth*2., max=1., min=0. )
    spsdparams.add('d', value=2, min=1, max=100)
    spsdparams.add('t0', value=-tdur/4., vary=True, min=-tdur, max=tdur  )
    spsdparams.add('tdur', value=tdur, vary=True, min=0, max=2*tdur  )
    spsdparams.add('c0', value=np.median(f), vary=True)




    sineparams = Parameters()
    sineparams.add('a', value= depth*2., max=1., min=0. )
    sineparams.add('t0', value=0., vary=True, min=-tdur, max=tdur  )
    sineparams.add('tdur', value=tdur/4., vary=True, min=tdur/8., max=2*tdur  )
    sineparams.add('c0', value=np.median(f), vary=True)
    


    if mask_detrend:
        tmask = make_transit_mask(t, P, t0, tdur)
        f = wotan.flatten(t,f,method='lowess',window_length=4*tdur,return_trend=False,
                          robust=True,mask=tmask)
        
        sineparams.add('c1', value=0, vary=False)
        sineparams.add('c2', value=0, vary=False)

        spsdparams.add('c1', value=0, vary=False)
        spsdparams.add('c2', value=0, vary=False)

        rampparams.add('c1', value=0, vary=False)
        rampparams.add('c2', value=0, vary=False)

        boxparams.add('c1', value=0, vary=False)
        boxparams.add('c2', value=0, vary=False)

    else:
        sineparams.add('c1', value=0, vary=True)
        sineparams.add('c2', value=0, vary=True)

        spsdparams.add('c1', value=0, vary=True)
        spsdparams.add('c2', value=0, vary=True)

        rampparams.add('c1', value=0, vary=True)
        rampparams.add('c2', value=0, vary=True)

        boxparams.add('c1', value=0, vary=True)
        boxparams.add('c2', value=0, vary=True)
        


    ramp_bic_probs = []
    spsd_bic_probs = []
    sine_bic_probs = []

    num_good_transits = 0
    frac_good_transit_points=0


    n_transits=0
    tn=t0


    if show_plot:
        transit_labels=[]
        tce_times = []
        tce_fluxes = []
        tce_errs=[]
        
        plot_times=[]
        spsd_fits = []
        box_fits = []
        ramp_fits = []
        sine_fits = []
    
    while tn < max(t):

        t_min, t_max = tn - ntdur*tdur, tn + ntdur*tdur
        
        tce_cut = np.logical_and(t>t_min, t<t_max)
    
        tce_time = t[tce_cut] - tn
        tce_flux = f[tce_cut]
        tce_err = ferr[tce_cut]

        n_transits+=1
        tn+=P
    
        if sum(tce_cut)<min_frac*(2*ntdur*tdur/texp):
            #frac_good_transit_points += sum(tce_cut)/(2*ntdur*tdur/texp)
            continue
        
        num_good_transits+=1
        
        frac_good_transit_points += sum(np.abs(tce_time)<tdur)/(2*tdur/texp)
    

        box_out = minimize(box_residual, boxparams, args=(tce_time, tce_flux, tce_err),
                           method=fit_method,)
        ramp_out = minimize(spsd_residual, rampparams, args=(-1*tce_time, tce_flux, tce_err),
                            method=fit_method,)
        spsd_out = minimize(spsd_residual, spsdparams, args=(tce_time, tce_flux, tce_err),
                            method=fit_method,)
        sine_out = minimize(sine_residual, sineparams, args=(tce_time, tce_flux, tce_err),
                            method=fit_method,)


            
        ramp_bic_probs.append( (box_out.bic-ramp_out.bic)/2.) 
        #jump_aic_probs.append(  (box_out.aic-jump_out.aic)/2.) 
    
        spsd_bic_probs.append( (box_out.bic-spsd_out.bic)/2.) 
        #spsd_aic_probs.append(  (box_out.aic-spsd_out.aic)/2.) 
    
        sine_bic_probs.append( (box_out.bic-sine_out.bic)/2.) 
        #sine_aic_probs.append(  (box_out.aic-sine_out.aic)/2.) 


        if show_plot:
            plot_time = np.linspace(min(tce_time), max(tce_time), 500)
            plot_times.append(plot_time)
            tce_times.append(list(tce_time))
            tce_fluxes.append(tce_flux)
            tce_errs.append(tce_err)

            box_fits.append( box_residual(box_out.params, plot_time) )
            spsd_fits.append( spsd_residual(spsd_out.params, plot_time) )
            ramp_fits.append( spsd_residual(ramp_out.params, -1*plot_time) )
            sine_fits.append( sine_residual(sine_out.params, plot_time) )

            transit_labels.append(n_transits)

    if show_plot:

        def get_ppt(x):
            return (x-np.median(x))/np.median(x)*1e3


        for i in range(len(transit_labels)):

            snum = int(np.sqrt(len(transit_labels))+1)


            if i==0:
                plt.figure(figsize=(snum*2.5,snum*2.) )
                
            
            ax = plt.subplot(snum,snum,i+1)

            if i%snum==0:
                ax.set_ylabel(r'$\mathregular{\delta F/F }$ [ppt]')

            ax.set_xlabel(r'$\mathregular{\Delta t_0}$ [day]')

            plot_time=plot_times[i]
            ax.errorbar(tce_times[i], get_ppt(tce_fluxes[i]), yerr=1e3*tce_errs[i], fmt='.', color='k', ecolor='0.5')

            l1, = ax.plot(plot_time, get_ppt(box_fits[i]), '-' , zorder=2,
                    label='box')
            l2, = ax.plot(plot_time, get_ppt(ramp_fits[i]), '-' , zorder=2,
                    label='ramp')
            l3, = ax.plot(plot_time, get_ppt(spsd_fits[i]), '-' , zorder=2,
                    label='spsd')
            l4, = ax.plot(plot_time, get_ppt(sine_fits[i]), '-' , zorder=2,
                    label='sine')

            
            ax.text(0.15,0.1,'{}'.format(transit_labels[i]),ha='center',va='bottom',transform=ax.transAxes)

        ax = plt.subplot(snum,snum,i+2)
        ax.legend([l1,l2,l3,l4], ['box','ramp','spsd','sine'], ncol=1, loc='upper left')
        ax.axis('off')
    
            #plt.xlabel('$\mathregular{\Delta t_0}$')

            #plt.ylabel('flux')
        plt.tight_layout()
        #plt.show()

            

    if num_good_transits!=0:
        
        output_dict = {'num_transits': n_transits,
                   'num_good_transits':num_good_transits,
                   'frac_points_in_tran': frac_good_transit_points/num_good_transits,
                   #'jump_mean_bic_stat':np.mean(jump_bic_probs),
                   #'jump_mean_aic_stat':np.mean(jump_aic_probs),
                   'ramp_median_bic_stat':np.median(ramp_bic_probs),
                   'ramp_min_bic_stat':np.min(ramp_bic_probs),
                   'ramp_bic_stat':np.sum(ramp_bic_probs),
                   #'jump_aic_stat':np.sum(jump_aic_probs),
                   #'spsd_mean_bic_stat':np.mean(spsd_bic_probs),
                   #'spsd_mean_aic_stat':np.mean(spsd_aic_probs),
                   'spsd_median_bic_stat':np.median(spsd_bic_probs),
                   'spsd_min_bic_stat':np.min(spsd_bic_probs),
                   'spsd_bic_stat':np.sum(spsd_bic_probs),
                   #'spsd_aic_stat':np.sum(spsd_aic_probs),
                   #'sine_mean_bic_stat':np.mean(sine_bic_probs),
                   #'sine_mean_aic_stat':np.mean(sine_aic_probs),
                   'sine_median_bic_stat':np.median(sine_bic_probs),
                   'sine_min_bic_stat':np.min(sine_bic_probs),
                   'sine_bic_stat':np.sum(sine_bic_probs),
                   #'sine_aic_stat':np.sum(sine_aic_probs)
                   }

        return output_dict

    else:
        output_dict = {'num_transits': n_transits,
                   'num_good_transits':num_good_transits,
                   'frac_points_in_tran': 0.,
                   #'jump_mean_bic_stat':np.mean(jump_bic_probs),
                   #'jump_mean_aic_stat':np.mean(jump_aic_probs),
                   'ramp_median_bic_stat':np.nan,
                   'ramp_min_bic_stat':np.nan,
                   'ramp_bic_stat':np.nan,
                   #'jump_aic_stat':np.sum(jump_aic_probs),
                   #'spsd_mean_bic_stat':np.mean(spsd_bic_probs),
                   #'spsd_mean_aic_stat':np.mean(spsd_aic_probs),
                   'spsd_median_bic_stat':np.nan,
                   'spsd_min_bic_stat':np.nan,
                   'spsd_bic_stat':np.nan,
                   #'spsd_aic_stat':np.sum(spsd_aic_probs),
                   #'sine_mean_bic_stat':np.mean(sine_bic_probs),
                   #'sine_mean_aic_stat':np.mean(sine_aic_probs),
                   'sine_median_bic_stat':np.nan,
                   'sine_min_bic_stat':np.nan,
                   'sine_bic_stat':np.nan,
                   #'sine_aic_stat':np.sum(sine_aic_probs)
                   }
        return output_dict
        





    



def tce_masked_num_den(time, flux, t0, P, width, cadence, fill_mode='reflect', nwindow=7., n_channels=16):

        
    pad_time, pad_flux, pad_boo = pad_time_series(time, flux,in_mode=fill_mode,
                                                      pad_end=True,
                                                      fill_gaps=True,
                                                      cadence=cadence)


    tce_mask = make_transit_mask(pad_time, P, t0, width)

    if sum(tce_mask)==len(tce_mask):
        pad_time_tce = pad_time.copy()
        pad_flux_tce = pad_flux.copy()
        pad_boo_tce = pad_boo.copy()
        
    else:

        pad_flux_tce = np.interp(pad_time, pad_time[tce_mask], pad_flux[tce_mask] )

        
        #pad_time_tce, pad_flux_tce, pad_boo_tce = pad_time_series(pad_time[tce_mask], pad_flux[tce_mask], in_mode=fill_mode,pad_end=True, fill_gaps=True, cadence=cadence, len_pad=len(pad_time))

    #print(len(pad_time), len(pad_flux_tce), sum(tce_mask), len(tce_mask), len(time))



    #print(len(pad_flux_tce), len(pad_time))
    
    #pad_time_tce, pad_flux_tce, pad_boo_tce = pad_time_series(fill_time_tce, fill_flux_tce, in_mode=fill_mode,pad_end=True,fill_gaps=False,cadence=cadence)
    
        
    
    template = get_transit_signal(width=width, depth=100., pad=len(pad_flux), b=0.2)
    
    flux_transform = ocwt(pad_flux - np.median(pad_flux), max_level=n_channels-1)
    tce_masked_flux_transform = ocwt(pad_flux_tce - np.median(pad_flux_tce),  max_level=n_channels-1)
    
    template_transform = ocwt(template-1.,  max_level=n_channels-1)
    
    
    window_size=nwindow*width
    sig2 = [1./calc_var_stat(x, window_size*2**i, method='mad', exp_time=cadence) for i,x in enumerate( tce_masked_flux_transform)]
    sig2 = np.array(sig2, )
        
    n_levels = flux_transform.shape[0]
    levels = np.concatenate([np.arange(1,n_levels), [n_levels-1] ] )[:,np.newaxis]

    N_i=[]
    D_i=[]
    
    for i, l in enumerate(levels):
        
        N_channel = 2.** (-l) * convolve(flux_transform[i]*sig2[i], template_transform[i,::-1], mode='same')
        D_channel = 2.** (-l) * convolve(sig2[i], template_transform[i,::-1]**2., mode='same')  
        
        N_i.append(N_channel)
        D_i.append(D_channel)
        
    
    N_i=np.array(N_i)[:,pad_boo]
    D_i=np.array(D_i)[:,pad_boo]

    return N_i, D_i



def tce_masked_num_den_sectors(time, flux, t0, P, width, cadence, fill_mode='reflect', nwindow=7, sector_dates=None):


    if sector_dates is None:
        return tce_masked_num_den(time, flux, t0, P, width, cadence, fill_mode=fill_mode, nwindow=nwindow, )


    segment_inds = [np.argmin(np.abs(sd-time) ) for sd in sector_dates]

    if time[segment_inds[-1]] != time[-1]:
        segment_inds =  segment_inds + [None]


    segment_lengths = [int(np.log2((time[segment_inds[i+1]]-time[segment_inds[i]])/cadence)) for i in range(len(segment_inds)-2) ]

    max_n_channels = min(segment_lengths)

    #print(max_n_channels)

    N_i = np.zeros(shape=(max_n_channels, len(time)))
    D_i = np.zeros(shape=(max_n_channels, len(time)))
    
    for i , seg_init in enumerate(segment_inds[:-1]):
        
        seg_end = segment_inds[i+1]

        seg_time = time[seg_init:seg_end]
        seg_flux = flux[seg_init:seg_end]

        N_i_segment, D_i_segment = tce_masked_num_den(seg_time, seg_flux, t0, P, width, cadence, fill_mode=fill_mode, nwindow=nwindow, n_channels=max_n_channels)

        #print(N_i.shape, N_i_segment.shape)

        N_i[:,seg_init:seg_end] = N_i_segment
        D_i[:,seg_init:seg_end] = D_i_segment

            
    return N_i, D_i

    
    




def temporal_chi2_statistic(t, flux, t0, P, width , cadence, fill_mode='reflect', nwindow=7., sector_dates=None):    
    
    foldtime = (t-t0 + P/2.)%P - P/2.
    t0_mask = np.abs(foldtime)<=cadence
    n_transits = np.sum(t0_mask)

    try:
        N_i, D_i = tce_masked_num_den_sectors(t, flux, t0, P, width, cadence, fill_mode, nwindow, sector_dates=sector_dates)
    except IndexError:
        return np.nan, np.nan
    
    N = np.sum(N_i, axis=0)
    D = np.sum(D_i, axis=0)
    
    t0_num, t0_den = N[t0_mask], D[t0_mask]
    
    Z = np.sum(t0_num)/np.sqrt(np.sum(t0_den))
    Zj = t0_num/np.sqrt(np.sum(t0_den))
    Qj = t0_den/np.sum(t0_den)

    delta_Zj = (Zj - Qj*Z)**2.
    
    chi2 = np.sum( delta_Zj / Qj)
    red_chi2 = chi2/(n_transits-1)
    
    chi2_stat = Z /np.sqrt(red_chi2)
        
    return red_chi2, chi2_stat

    

def channel_chi2_statistic(time, flux, t0, P, width, cadence, fill_mode='reflect', nwindow=7,sector_dates=None):


    try:
        N_i, D_i = tce_masked_num_den_sectors(time, flux, t0, P, width, cadence, fill_mode, nwindow, sector_dates=sector_dates)
    except IndexError:
        return np.nan, np.nan

    n_levels = N_i.shape[0]
    
    
    N = np.sum(N_i, axis=0)
    D = np.sum(D_i, axis=0)


    z = N/np.sqrt(D)
    zi = N_i/np.sqrt(D)
    qi = D_i/D
        
    delta_zi = (zi - qi*z)
    chi2_n = np.sum(delta_zi**2./qi, axis=0)
    
    
    foldtime = (time-t0 + P/2.)%P - P/2.
    t0_mask = np.abs(foldtime)<cadence
    n_transits = np.sum(t0_mask)
    
    Z = np.sum(N[t0_mask])/np.sqrt(np.sum(D[t0_mask]) )
        
    chi2 = np.sum(chi2_n[t0_mask])
    red_chi2 = chi2 / (n_transits*(n_levels-1))
        
    chi2_stat = Z / np.sqrt(red_chi2)
        
    return red_chi2, chi2_stat




def check_for_min_good_transits(min_transits=3., required_fraction=0.5):

    

    return 1. 


def same_period_test(tces,):
    

    if isinstance(tces, np.ndarray):
        tces = pd.DataFrame(tces, columns=['star_id','period','mes','t0','tdur'])
    if len(tces)<=1:
        return np.array([0])

    mes_sort_order = np.argsort(-1*tces['mes'].to_numpy())
    periods_sorted = tces.period.to_numpy()[mes_sort_order]
    same_period_stats = np.zeros_like(mes_sort_order,dtype=float)

    for i,P in enumerate(periods_sorted[1:]):
                        
        P_A = periods_sorted[:i+1]
        P_B = periods_sorted[i+1]
        
        delta_P = np.min([(P_A-P_B)/P_A,(P_B-P_A)/P_B], axis=0)
                
        delta_P_int = np.abs(delta_P - np.round(delta_P,0))
        sigma_P = np.sqrt(2.) * erfcinv(delta_P_int)        
        same_period_stats[i+1] = np.max(sigma_P)

            
    return same_period_stats[np.argsort(mes_sort_order)]

    


    

def remove_secondary_tces(tces, threshold=3.):

    results = same_period_test(tces)
    
    return tces[results<threshold]

    
    



def make_data_validation_report(tce_num, recbin, color1='C0', color2='C3', savefig=True, save_directory='.', zoom_on_binned=True, fit_method='leastsq', local_test=True, ):
    
    def bin_flux(t, f, dt):
        t_bin_edges = np.arange(min(t), max(t), dt)    
        binned_f = np.histogram(t, t_bin_edges, weights=f)[0] / np.histogram(t, t_bin_edges)[0]
        return t_bin_edges[:-1]+dt/2., binned_f


    # Calculate Things here
    P,width,t0 = get_p_tdur_t0(recbin.tces[tce_num])

    fit = recbin._update_tce(tce_num, fit_method=fit_method, return_fit=True)

    vet_stats = recbin.get_all_vetting_metrics(tce_num, local_test=local_test, )

    P_best, width_best, t0_best = vet_stats['period'], vet_stats['tdur'], vet_stats['t0']

    other_tce_mask = recbin._get_previous_tce_masks(tce_num)
    
    mestime, mesphase, mes = recbin._calc_mes(tce_num, calc_ses=True, mask_tce=False,
                                              use_mask=other_tce_mask)
    mestime_sec, mesphase_sec, mes_sec = recbin._calc_mes(tce_num, calc_ses=True, mask_tce=True,
                                                          use_mask=other_tce_mask)

    min_width_searched = min(recbin.dump.tdurs)
    max_width_searched = max(recbin.dump.tdurs)

    _, _, mes_min_width = recbin._calc_mes(tce_num, calc_ses=True, mask_tce=False, width=min_width_searched)
    _, _, mes_max_width = recbin._calc_mes(tce_num, calc_ses=True, mask_tce=False,width=max_width_searched)


    full_mask = other_tce_mask
    full_mask &= recbin.dump.lc.mask

    folded_time = (recbin.dump.lc.time[full_mask]-t0_best + P_best/4.)%P_best - P_best/4.
    folded_flux = recbin.dump.lc.flux[full_mask]

    folded_time_zoom = (recbin.dump.lc.time[full_mask]-t0_best + P_best/2.)%P_best - P_best/2.

    t_bin, f_bin = bin_flux(t=folded_time, f=folded_flux - np.median(folded_flux), dt=width_best/4.)
    t_bin_zoom, f_bin_zoom = bin_flux(t=folded_time_zoom, f=folded_flux - np.median(folded_flux), dt=width_best/4.)


    transit_model_time = np.arange(min(folded_time), max(folded_time), recbin.dump.lc.exptime/8. )
    transit_model = plot_folded_transit(transit_model_time, period=P_best, t0=0., width=width_best,
                                        depth=vet_stats['depth_ppt']/1e3, b=vet_stats['b'],
                                        limb_dark=recbin.limb_coeff,exptime=recbin.dump.lc.exptime)-1.



    # Set up the plots
    fig = plt.figure(constrained_layout=True, figsize=(12,14))
    gs = fig.add_gridspec(7, 4)

    # Full Light Cuve Plot
    full_plot_ax = fig.add_subplot(gs[0,:])
    
    full_plot_ax.plot( recbin.dump.lc.time[full_mask], recbin.dump.lc.flux[full_mask]-np.nanmedian(recbin.dump.lc.flux[full_mask]), '.' , color='0.7', markersize=2,  rasterized=True)


    tce_mask = ~make_transit_mask( time=recbin.dump.lc.time[full_mask], P=P_best, t0=t0_best, dur=width_best )

    full_plot_ax.plot( recbin.dump.lc.time[full_mask][tce_mask], recbin.dump.lc.flux[full_mask][tce_mask]-np.nanmedian(recbin.dump.lc.flux[full_mask]), '.' , color=color1, markersize=4,  rasterized=True, label='In-Transit Points')
    full_plot_ax.set_xlabel('Time')
    full_plot_ax.set_ylabel('$\\mathregular{\\delta F/F}$')
    full_plot_ax.legend(loc='upper left', labelcolor='k')


    #folded plot
    folded_plot_ax = fig.add_subplot(gs[1,:-1])
    folded_plot_ax.plot( folded_time, folded_flux - np.median(folded_flux), '.' , color='0.7', markersize=3,  rasterized=True)
    folded_plot_ax.plot(t_bin, f_bin, 'o', markerfacecolor=color1, markeredgewidth=1, c='k')

    folded_plot_ax.plot(transit_model_time, transit_model, c=color2, lw=2)


    # Folded plot Zoomed
    folded_plot_ax_zoom = fig.add_subplot(gs[1,-1], sharey=folded_plot_ax)

    folded_plot_ax_zoom.plot( folded_time_zoom, folded_flux - np.median(folded_flux), '.' , color='0.7', markersize=3,  rasterized=True)
    folded_plot_ax_zoom.plot(t_bin_zoom, f_bin_zoom, 'o', markerfacecolor=color1, markeredgewidth=1, c='k')
    folded_plot_ax_zoom.plot(transit_model_time, transit_model, c=color2, lw=2)

    folded_plot_ax_zoom.set_xlim(-width*2, width*2)


    folded_plot_ax.set_ylabel('$\mathregular{\delta F / F}$')
    #folded_plot_ax.set_xlabel('$\mathregular{\Delta t_0}$')
    #folded_plot_ax_zoom.set_xlabel('$\mathregular{\Delta t_0}$')



    folded_plot_ax_crop =fig.add_subplot(gs[2,:-1])

    folded_plot_ax_crop.plot( folded_time, folded_flux - np.median(folded_flux), '.' , color='0.7', markersize=3,  rasterized=True, zorder=-9)
    folded_plot_ax_crop.plot(t_bin, f_bin, 'o', markerfacecolor=color1, markeredgewidth=1, c='k')

    folded_plot_ax_crop.plot(transit_model_time, transit_model, c=color2, lw=2)


    # Folded plot Zoomed
    folded_plot_ax_crop_zoom = fig.add_subplot(gs[2,-1], sharey=folded_plot_ax_crop, sharex=folded_plot_ax_zoom)

    folded_plot_ax_crop_zoom.plot( folded_time_zoom, folded_flux - np.median(folded_flux), '.' , color='0.7', markersize=3, rasterized=True, zorder=-9)
    folded_plot_ax_crop_zoom.plot(t_bin_zoom, f_bin_zoom, 'o', markerfacecolor=color1, markeredgewidth=1, c='k')
    folded_plot_ax_crop_zoom.plot(transit_model_time, transit_model, c=color2, lw=2)

    flux_label='$\mathregular{\delta F / F}$'
    t0_label = '$\mathregular{\Delta t_0}$'

    folded_plot_ax_crop.set_ylabel(flux_label)
    folded_plot_ax_crop_zoom.set_ylabel(flux_label)
    folded_plot_ax_zoom.set_ylabel(flux_label)
    
    folded_plot_ax_crop.set_ylim(np.nanmin(f_bin)-2*np.nanstd(f_bin), np.nanmax(f_bin)+np.nanstd(f_bin))


    # Mes plot
    mes_plot_ax = fig.add_subplot(gs[3,:-1], )

    mes_plot_ax.plot(mestime, mes, lw=2, color='k', label='tdur={:.2f}'.format(width), zorder=99)
    mes_plot_ax.plot(mestime, mes_min_width, lw=1, color=color1, label='tdur={:.2f}'.format(min_width_searched))
    mes_plot_ax.plot(mestime, mes_max_width, lw=1, color=color2, label='tdur={:.2f}'.format(max_width_searched))



    mes_plot_ax.axhline(0, color='0.7', zorder=-99, )
    mes_plot_ax.axhline(7., color='0.7', ls='--',)
    mes_plot_ax.legend(ncol=3)




    # Zoomed MES plot
    mes_plot_ax_zoom = fig.add_subplot(gs[3,-1], sharey=mes_plot_ax, sharex=folded_plot_ax_zoom)


    mes_plot_ax_zoom.plot(mestime, mes, lw=2, color='k', zorder=99)

    mes_plot_ax_zoom.plot(mestime, mes_min_width, lw=0.5, color=color1)
    mes_plot_ax_zoom.plot(mestime, mes_max_width, lw=0.5, color=color2)


    mes_plot_ax_zoom.axhline(0, color='0.7', zorder=-99)
    mes_plot_ax_zoom.axhline(7., color='0.7', ls='--', zorder=-99)

    mes_plot_ax_zoom.set_xlabel(t0_label)
    mes_plot_ax.set_xlabel(t0_label)
    mes_plot_ax.set_ylabel('MES')
    mes_plot_ax_zoom.set_ylabel('MES')




    # Odd-Even Plots

    both, even, odd, stat = recbin.odd_even_depth_test(tce_num)


    both_plot_ax = fig.add_subplot(gs[4,2], )
    odd_plot_ax = fig.add_subplot(gs[5,2], sharey=both_plot_ax, sharex=both_plot_ax)
    even_plot_ax = fig.add_subplot(gs[5,3], sharey=both_plot_ax, sharex=both_plot_ax)

    both_plot_ax.set_xlim(-width_best*2, width_best*2)

    both_plot_ax.set_title('Odd+Even')
    odd_plot_ax.set_title('Odd Only')
    even_plot_ax.set_title('Even Only')

    both_plot_ax.set_ylabel(flux_label)
    odd_plot_ax.set_ylabel(flux_label)
    even_plot_ax.set_ylabel(flux_label)
    

    #both_plot_ax.set_ylabel('$\mathregular{\delta F / F}$')

    folded_time_oddeven_cuts = (recbin.dump.lc.time[full_mask]-t0_best + P_best/2.)%(2*P_best) 

    odd_cut = folded_time_oddeven_cuts<=P_best
    even_cut = folded_time_oddeven_cuts>P_best


    folded_time_half_phase = (recbin.dump.lc.time[full_mask]-t0_best + P_best/2.)%P_best - P_best/2.

    both_phase = np.sort(folded_time_half_phase)
    odd_phase = both_phase[odd_cut]
    even_phase = both_phase[even_cut]

    both_offset = both.params['c0']
    even_offset = even.params['c0']
    odd_offset  = odd.params['c0']


    both_plot_ax.plot(folded_time_half_phase,  folded_flux-both_offset,'.', color='0.7', markersize=3,rasterized=True,zorder=-9)
    odd_plot_ax.plot(folded_time_half_phase[odd_cut], folded_flux[odd_cut]-odd_offset, '.',color='0.7',markersize=3,rasterized=True, zorder=-9)
    even_plot_ax.plot(folded_time_half_phase[even_cut], folded_flux[even_cut]-even_offset, '.', color='0.7',markersize=3, rasterized=True, zorder=-9)

    both_plot_ax.plot(both_phase, trap_residual(both.params, both_phase+P_best/4.)-both_offset, color='k')
    odd_plot_ax.plot(both_phase, trap_residual(odd.params, both_phase+P_best/4.)-odd_offset, color=color2)
    even_plot_ax.plot(both_phase, trap_residual(even.params, both_phase+P_best/4.)-even_offset, color=color1)


    binned_even_t, binned_even_f =  bin_flux(t=folded_time_half_phase[even_cut], f=folded_flux[even_cut]-even_offset, dt=width_best/6.)
    binned_odd_t, binned_odd_f =  bin_flux(t=folded_time_half_phase[odd_cut], f=folded_flux[odd_cut]-odd_offset, dt=width_best/6.)

    binned_both_t, binned_both_f =  bin_flux(t=folded_time_half_phase, f=folded_flux-both_offset, dt=width_best/6.)

    concat_odd_even_bin_flux = np.concatenate([binned_odd_f,binned_even_f])
    odd_even_plots_ymax = np.nanmax(concat_odd_even_bin_flux)+2*np.nanstd(concat_odd_even_bin_flux)
    odd_even_plots_ymin = np.nanmin(concat_odd_even_bin_flux)-2*np.nanstd(concat_odd_even_bin_flux)


    both_plot_ax.plot(binned_both_t, binned_both_f,  'o', color=color1, markeredgecolor='k')
    even_plot_ax.plot(binned_even_t, binned_even_f,  'o', color=color1, markeredgecolor='k')
    odd_plot_ax.plot(binned_odd_t, binned_odd_f,  'o', color=color1, markeredgecolor='k')

    both_plot_ax.set_ylim(odd_even_plots_ymin, odd_even_plots_ymax)

    for ax in [both_plot_ax,even_plot_ax,odd_plot_ax]:

        ax.set_xlabel(t0_label)

        if even.errorbars:

            even_depth = even.params['a']
            even_depth_err = even.params['a'].stderr
            even_min_depth =  -even_depth - even_depth_err
            even_max_depth =  -even_depth + even_depth_err

            ax.axhspan(even_min_depth, even_max_depth, color=color1, alpha=0.2 )
            
        if odd.errorbars:

            odd_depth = odd.params['a']
            odd_depth_err = odd.params['a'].stderr
            odd_min_depth = -odd_depth - odd_depth_err
            odd_max_depth = -odd_depth + odd_depth_err

            ax.axhspan(odd_min_depth, odd_max_depth,color=color2, alpha=0.2 )

        ax.axhline(-both.params['a'], color='k', ls='--')

    


    # Sine Test    
    sine_test_ax = fig.add_subplot(gs[4,1])
    tran_test_ax = fig.add_subplot(gs[4,0], sharey=sine_test_ax)

    sine_test_resid_ax = fig.add_subplot(gs[5,1], )
    tran_test_resid_ax = fig.add_subplot(gs[5,0], sharey=sine_test_resid_ax)


    test_results, sine_test_plotvals = recbin.cosine_vs_transit_global(tce_num, return_plot_values=True)

    (sin_t, sin_f, sin_resid), (sin_modx, sin_mody), (tra_t, tra_f, tra_resid), (tra_modx, tra_mody) =  sine_test_plotvals


    # Plot Sine/Transit points
    sine_test_ax.plot(sin_t, sin_f, '.', color='0.7', markersize=3, zorder=-9, rasterized=True)
    tran_test_ax.plot(tra_t, tra_f, '.', color='0.7', markersize=3, zorder=-9, rasterized=True)

    # Plot binned Values
    sin_t_bin, sin_f_bin = bin_flux(sin_t, sin_f, width_best/6.)
    sine_test_ax.plot(sin_t_bin, sin_f_bin, 'o', color=color1, markeredgecolor='k')

    tran_t_bin, tran_f_bin = bin_flux(tra_t, tra_f, width_best/6.)
    tran_test_ax.plot(tran_t_bin, tran_f_bin, 'o', color=color1, markeredgecolor='k')

    

    sine_test_ax.set_ylim(np.nanmin(tran_f_bin)-2*np.nanstd(tran_f_bin),np.nanmax(tran_f_bin)+2*np.nanstd(tran_f_bin) )


    # plot models
    sine_test_ax.plot(sin_modx, sin_mody, color=color2)
    tran_test_ax.plot(tra_modx, tra_mody, color=color2)

    tran_test_ax.set_title('Transit Fit')
    sine_test_ax.set_title('Sine Fit')

    tran_test_ax.set_ylabel(flux_label)
    sine_test_ax.set_ylabel(flux_label)
    tran_test_ax.set_xlabel('$\mathregular{\Delta t_0}$')
    sine_test_ax.set_xlabel('$\mathregular{\Delta t_0}$')


    #Plot Residuals
    sine_test_resid_ax.plot(sin_t, sin_resid, '.', color='0.7', markersize=3, zorder=-9, rasterized=True)
    tran_test_resid_ax.plot(tra_t, tra_resid, '.', color='0.7', markersize=3, zorder=-9, rasterized=True)


    #plot binned residuals
    sin_t_bin, sin_resid_bin = bin_flux(sin_t, sin_resid, width_best/6.)
    sine_test_resid_ax.plot(sin_t_bin, sin_resid_bin, 'o', color=color1, markeredgecolor='k')

    tran_t_bin, tran_resid_bin = bin_flux(tra_t, tra_resid, width_best/6.)
    tran_test_resid_ax.plot(tran_t_bin, tran_resid_bin, 'o', color=color1, markeredgecolor='k')

    sine_test_resid_ax.plot(sin_modx, sin_mody*0., color=color2)
    tran_test_resid_ax.plot(tra_modx, tra_mody*0., color=color2)

    sine_test_resid_ax.set_ylim(np.nanmin(sin_resid_bin)-2*np.nanstd(sin_resid_bin), np.nanmax(sin_resid_bin)+2*np.nanstd(sin_resid_bin))


    tran_test_resid_ax.set_title('Transit Residual')
    sine_test_resid_ax.set_title('Sine Residual')


    tran_test_resid_ax.set_ylabel('obs-mod')
    sine_test_resid_ax.set_ylabel('obs-mod')

    tran_test_resid_ax.set_xlabel('$\mathregular{\Delta t_0}$')
    sine_test_resid_ax.set_xlabel('$\mathregular{\Delta t_0}$')




    # Weak Secondary plot
    sec_test_ax = fig.add_subplot(gs[4,3])

    #mesphase = (mesphase-0.25)%1
    #mesphase_sec = (mesphase_sec-0.25)%1
    

    sec_test_ax.plot(mesphase-0.25, mes, color='0.7')
    sec_test_ax.plot(mesphase_sec-0.25, mes_sec, color=color1)
    sec_test_ax.axhline(7., ls='--',)
    sec_test_ax.axhline(0., ls='--', color='0.7')
    sec_test_ax.axvline(0.5, ls='--', color='0.7')


    sec_test_ax.set_title('Weak Secondary Test')
    sec_test_ax.set_xlabel('Phase')
    sec_test_ax.set_ylabel('MES')


    sec_test_ax.plot(vet_stats['min_sec_mes_phase'], vet_stats['min_sec_mes'], 'x', c=color2, markeredgewidth=2)
    sec_test_ax.plot(vet_stats['max_sec_mes_phase'], vet_stats['max_sec_mes'], 'x', c=color2, markeredgewidth=2)


    sec_test_ymax = np.nanmax(mes_sec) + np.nanstd(mes_sec)
    sec_test_ymin = np.nanmin(mes_sec) - np.nanstd(mes_sec)

    sec_test_ax.set_ylim(sec_test_ymin, max(sec_test_ymax,10. ) )

    
    # Write out Test Results:

    #print(vet_stats)

    write_axis = fig.add_subplot(gs[6,:])

    write_axis.axis('off')


    keys, values = vet_stats.keys(), vet_stats.values()

    results_string = ''

    #print(len(keys))

    n_per_col = 11
    n_col = 4


    thresholds={'b':0.9, 'channel_chi2_stat':10., 'temporal_chi2_stat':6., 'max_sec_mes':6., 'mes_over_mad':5.,
               'mad_sec_mes': 3., 'mad_mes':3, 'odd_even_depth_stat':3., 'global_diff_chi2':0., 
               'global_diff_red_chi2': 0., 'num_good_transits':recbin.dump.min_transits, 'ramp_median_bic_stat':0, 
               'spsd_median_bic_stat':0, 'sine_median_bic_stat':0, 'sine_bic_stat':0, 'ramp_bic_stat':0,
               'spsd_bic_stat':0, 'spsd_min_bic_stat':0, 'sine_min_bic_stat':0, 'ramp_min_bic_stat':0}

    for i,k in enumerate(keys):


        if k=='period':
            nsig=7
        elif k=='tce_id':
            nsig=2
        elif k[-4:] in ['stat','tran'] or k=='global_diff_chi2':
            nsig=2
        else:
            nsig=5

        y_pos = i%n_per_col

        if vet_stats[k] is None:
            results_string = '\n'* int(y_pos) + k + ': {}'.format(np.nan) 
            
        else:
            results_string = '\n'* int(y_pos) + k + ': {}'.format(np.round(vet_stats[k], nsig)) 

        x_text=np.floor(i/n_per_col) / n_col
        y_text=0.95

        try:
            thresh=thresholds[k]
            stat=vet_stats[k]

            if k in ['channel_chi2_stat','temporal_chi2_stat' ,'num_good_transits','mes_over_mad']:
                thresh*=-1
                stat*=-1

            if thresh<stat or np.isnan(stat):
                textcolor='C3'
            else:
                textcolor='C0'
        except:
            textcolor='k'


        #if i%n_per_col==n_per_col-1:
        write_axis.text( x_text, 0.95, results_string, transform=write_axis.transAxes, 
                            ha='left', va='top', fontsize=8, color=textcolor)
            #results_string=''        

    #x_lineup = np.floor(i/n_per_col) / n_col
    #write_axis.text( x_lineup, 0.9, results_string, transform=write_axis.transAxes, ha='left', va='top', fontsize=8)

    if recbin.dump.lc.mission=='KEPLER':
        cat_name='KIC'
    elif recbin.dump.lc.mission=='TESS':
        cat_name='TIC'
    else:
        cat_name='CANDIDATE'
    
    plt.suptitle('RECYCLEBin Validation Report for '+cat_name+' {}'.format(vet_stats['tce_id']), fontsize=15)

    #plt.tight_layout()
    
    if savefig:
        plt.savefig(save_directory+'/vetting_report_'+cat_name+ str(np.round(vet_stats['tce_id'],2) )+'.pdf', dpi=200, pad_inches=0.5, bbox_inches='tight' )

    


