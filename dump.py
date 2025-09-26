#import sys
from os import path #, getcwd

import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
from scipy.interpolate import griddata
#from scipy.ndimage import median_filter
from astroquery.mast import Catalogs
from fast_histogram import histogram1d

from astroquery.simbad import Simbad

import warnings

from .ffa import *
from .lightcurves import *
from .utils import *
from .ses import *

from tqdm import tqdm


filepath = path.abspath(__file__)
dir_path = path.dirname(filepath)

TRASH_DATA_DIR = dir_path+'/'


class Dump(object):

    def __init__(self, LightCurve, tdurs=None, star_logg=None, star_teff=None,
                 star_density=None, min_transits=2, P_range=None,
                 sector_size=13., gap_size=0.5, sector_dates=None):
        
        self.lc = LightCurve
        
        
        if star_logg is None or star_teff is None or star_density is None:
            star_mass, star_radius, star_teff, star_logg, star_density = self.get_stellar_parameters()

        if np.isnan(star_logg):
            star_logg=4.4
        self.star_logg = star_logg

        if np.isnan(star_teff):
            star_teff=5800.
        self.star_teff = star_teff

        if np.isnan(star_density):
            star_density=1.
        self.star_density = star_density

        
        if tdurs is None:
            self.tdurs = self._calc_best_tdurs(n_trans=min_transits)
        else:
            self.tdurs = tdurs
            
        self.tdur_sigma = [[]]*len(self.tdurs)
        self.limb_darkening = self._get_limb_darkening()
        
        self.ses = None
        self.num = None
        self.den = None

        self.TCEs = None
        self.tce_mask = self.lc.mask.copy()
        self.tdur_cdpp = []
        
        self.search_periods = kep_period_sampling(time=self.lc.time[self.lc.mask],
                                                  P_min=None, n_tr=min_transits,
                                                  rho_star=star_density)

        if not(P_range is None):
            self.search_periods = self.search_periods[np.logical_and( self.search_periods<max(P_range), self.search_periods>min(P_range))]
            
        self.min_transits = min_transits
        self.gap_size=gap_size
        self.sector_size=sector_size
        self.sector_dates = sector_dates





    def get_stellar_parameters(self, **kwargs):
    
        """
        NOTE: This function written by Ethan Kruse, modified for use here. 

        Find the stellar parameters (mass, radius, temperature) of this target.
        Will accept user supplied parameters if given, otherwise will use the
        Tess Input Catalog values. This requires an internet connection.
        Parameters
        ----------
        star_mass
            Stellar mass in units of Solar masses.
        star_radius
            Stellar radius in units of Solar radii.
        star_teff
            Stellar effective temperature in units of K.
        Returns
        
        -------
        """

        mission = self.lc.mission.upper()
        starid = self.lc.ID


        if mission.upper() not in ['KEPLER', 'K2', 'TESS']:
            warnings.warn(f'Cannot determine stellar parameters for star in '
                          f'mission {self.lc.mission}. Using solar values.')
            self.star_mass = 1.
            self.star_radius = 1.
            self.star_teff = 5772.
            self.star_density=1.
            self.star_logg=4.44
            return

        if mission == 'TESS':
            # query the TIC by TIC
            cat = Catalogs.query_criteria(catalog='tic', ID=starid)
            assert len(cat) == 1 and int(cat['ID'][0]) == starid
        elif mission == 'KEPLER':
            # query the TIC by KIC
            cat = Catalogs.query_criteria(catalog='tic', KIC=starid)
            assert len(cat) == 1 and int(cat['KIC'][0]) == starid
        else:
            # XXX: change this if we ever get a faster EPIC/TIC match
            print('Looking up stellar parameters for K2 targets is very slow '
                  'and could take up to a minute.')
            pref = 'EPIC'
            names = Simbad.query_objectids(f"{pref} {self.id}")
            dr2 = 0
            for iname in names:
                if iname[0][:8] == 'Gaia DR2':
                    dr2 = int(iname[0][8:])
                    break
            assert dr2 > 0
            # query the TIC by Gaia DR2 ID since EPIC is not a field and there's
            # really no easy way to query the EPIC right now.
            cat = Catalogs.query_criteria(catalog='tic', GAIA=dr2)
            assert len(cat) == 1 and int(cat['GAIA'][0]) == dr2

        star_mass = cat['mass'][0]
        star_radius = cat['rad'][0]
        star_teff = cat['Teff'][0]
        star_logg = cat['logg'][0]
        star_rho = cat['rho'][0]
        
        return star_mass, star_radius, star_teff, star_logg, star_rho


    


    def _add_tce_to_mask(self, TCEs):

        for tce in TCEs:
            self.tce_mask &= make_transit_mask(self.lc.time, P=tce[1], t0=tce[3], dur=tce[4])
        self.tce_mask &= self.lc.mask.copy()

    def _calc_best_tdurs(self, n_trans=2., maxtdur = 1.5):

        if self.star_density is None:
            rho_star = 1.
        else:
            rho_star = self.star_density

        exptime = self.lc.exptime

        baseline = max(self.lc.time[self.lc.mask]) - min(self.lc.time[self.lc.mask])
        max_period = baseline/n_trans
        max_dur = min( (13./24.) * (max_period/365.)**(1./3.) * rho_star**(-1./3.) * 1.5, maxtdur)
        
        tdurs = np.unique( exptime  * (1.1)**np.arange(2,100) // exptime ) * exptime
        #tdurs = (2.5*exptime)  * (1.1)**np.arange(2,100) 
        cut = tdurs<max_dur
        cut &= tdurs>max(0.04, exptime*2)
    
        return tdurs[cut]

        

    def _get_limb_darkening(self, interpfile=TRASH_DATA_DIR+'data/meta/limb_coeffs.h5'):
        
        mission = self.lc.mission

        if mission=='KEPLER':
            limb_grid = KEPLER_LIMB_DARKENING_GRID
        elif mission=='TESS':
            limb_grid = TESS_LIMB_DARKENING_GRID
        else:
            limb_grid = get_limb_darkening_grid(mission, grid_file=interpfile)
            
        limb_dark = griddata(points=limb_grid[['teff','logg']].to_numpy(), 
                             values=limb_grid[['a1','a2','a3','a4']].to_numpy(),
                             xi=[self.star_teff, self.star_logg], rescale=True,
                             method='nearest')
        
        return limb_dark


    def get_transit_model(self, depth, width, pad=None, b=0.25):

        if pad is None:
            pad = len(self.lc.flux)*2

        tr_model = get_transit_signal(width, depth, limb_dark=self.limb_darkening[0],exp_time=self.lc.exptime, pad=pad, b=b)
        
        return tr_model-1.





    def Calculate_SES_by_Segment(self, gap_dates=None, gap_size=None, sector_size=None, mask=None,  **calc_ses_kw):

        t = self.lc.time

        if not(self.sector_dates is None):
            gap_dates=self.sector_dates
        else:
            if not(self.lc.sector_dates is None):
                gap_dates=self.lc.sector_dates
            
            
        if gap_dates is None:
            
            if gap_size is None:
                gap_size = self.gap_size
            if sector_size is None:
                sector_size = self.sector_size
            
            dt = t[1:]-t[:-1]

            if mask is None:
                mask = self.tce_mask

            # identify gaps in data larger than gap_size
            gaps = np.arange(len(dt), dtype=int)[dt>gap_size]
            gaps = np.append(gaps, [len(t)-1])

        
            # remove multiple gaps in a sector
            #segment_inds = [g for i,g in enumerate(gaps[:-1]) if t[gaps[i+1]] - t[gaps[i]] > sector_size ]

            segment_inds = [0]
            for i,g in enumerate(gaps):
                if t[g] - t[segment_inds[-1]] > sector_size:
                     segment_inds.append(g)

        else:
            segment_inds = [np.argmin(np.abs(gd-t) ) for gd in gap_dates]

        if t[segment_inds[-1]] != t[-1]:
                segment_inds =  segment_inds + [None]

        print('\nCalculating Wavelet Transform ({} Segments) ... '.format(len(segment_inds)-1))
        
        for i,seg in enumerate(segment_inds[:-1]):


            seg_init = seg
            seg_end = segment_inds[i+1]

            seg_mask = mask[seg_init:seg_end]
            seg_time = t[seg_init:seg_end][seg_mask]
            seg_flux = self.lc.flux[seg_init:seg_end][seg_mask]

            #print('    Segment {}: {:.2f}-{:.2f} ({:.2f})'.format(i+1, seg_time[0], seg_time[-1], seg_time[-1]-seg_time[0]) )


            if len(seg_time)==0:
                continue
            
            seg_ses, seg_num, seg_den = self.Calculate_SES(time=seg_time, flux=seg_flux, mask=seg_mask, **calc_ses_kw)


            if i==0:
                ses = seg_ses
                num = seg_num
                den = seg_den
            else:
                ses = np.append(ses,seg_ses, axis=1)
                num = np.append(num,seg_num, axis=1)
                den = np.append(den,seg_den, axis=1)
        
        self.ses = ses
        self.num = num
        self.den = den

        
        return ses, num, den
    

    def Calculate_SES(self, time=None, flux=None, mask=None, var_calc='mad',fill_mode='reflect',tdurs=None, ses_weights=None, window_width=7., tdepth=100., multi_sector=False):

        if mask is None:
            mask = self.lc.mask.copy()
        if time is None:
            time = self.lc.time[mask]
        if flux is None:
            flux = self.lc.flux[mask]

            
        pad_time, pad_flux, pad_boo = pad_time_series(time, flux,
                                                      in_mode=fill_mode,
                                                      pad_end=True,
                                                      fill_gaps=True,
                                                      cadence=self.lc.exptime)

        #print(len(pad_flux), len(pad_time), len(pad_boo) )
        
        FluxTransform = ocwt( pad_flux-1., )

        if tdurs is None:
            dur_iter = self.tdurs
        else:
            dur_iter = tdurs

        ses_alldur = [[]]*len(dur_iter)
        num_alldur = [[]]*len(dur_iter)
        den_alldur = [[]]*len(dur_iter)
        

        for i,dur in enumerate(dur_iter):
            
            Signal = self.get_transit_model(depth=tdepth,width=dur, pad=len(pad_flux)  )
            SignalTransform = ocwt(Signal, )


            ses, num, den, sig2 = calculate_SES(FluxTransform,
                                                SignalTransform,
                                                texp=self.lc.exptime,
                                                window_size=window_width*dur,
                                                var_calc=var_calc )
            
            ses_alldur[i] = ses[pad_boo]
            num_alldur[i] = num[pad_boo]
            den_alldur[i] = den[pad_boo]

            self.tdur_cdpp.append( np.median(sig2 ) )

        if ses_weights is None:
            ses_alldur = np.array(ses_alldur)
            num_alldur = np.array(num_alldur)
            
        else:
            ses_alldur = np.array(ses_alldur)/ses_weights
            num_alldur = np.array(num_alldur)/ses_weights
            
        den_alldur = np.array(den_alldur)
        
        self.ses = ses_alldur
        self.num = num_alldur
        self.den = den_alldur
        
        return ses_alldur, num_alldur, den_alldur



    def Search_for_TCEs(self, ses_cut=20., threshold=7., remove_hises=True, calc_ses=True, return_df=False, use_tce_mask=False,fill_mode='reflect', **tce_search_kw):

            
        mask = self.lc.mask.copy()
        if use_tce_mask:
            mask &= self.tce_mask

            
        if self.ses is None or calc_ses:
            ses, num, den = self.Calculate_SES_by_Segment(mask=mask,fill_mode=fill_mode)
        else:
            ses, num, den = self.ses.copy(), self.num.copy(), self.den.copy()

        periods = self.search_periods

        
        time = self.lc.time.copy()
        mask_time = time[mask]

        ses_mask = np.ones(len(ses[0]), dtype=bool )

        
        #for s in ses:
        #    ses_mask = np.logical_and(ses_mask, s<ses_cut)

        #if np.sum(~ses_mask)>150:
        #    print('Not removing {0} points with SES>{1}'.format(np.sum(~ses_mask), int(ses_cut) ) )
        #    ses_mask =  np.ones(len(ses[0]), dtype=bool )
        #else:
        #    print('Removing {0} points with SES>{1}'.format(np.sum(~ses_mask), int(ses_cut) ) )

        single_TCE = None
        
        if remove_hises:
            # Cut highest SES:
            hises_ind = np.argmax(ses)%len(ses[0])
            hises_tdur_ind =  np.argmax(ses)//len(ses[0])

            hises_t0 = mask_time[hises_ind]
            hises_tdur = self.tdurs[hises_tdur_ind]
            
            hises_mask = make_transit_mask(time=mask_time, P=1e10, t0=hises_t0,
                                           dur=hises_tdur)
            max_mes = np.max(ses[hises_tdur_ind][hises_mask])


            if np.max(ses[hises_tdur_ind][hises_mask])<0.5*np.max(ses):
                print('\nSingle Transit Candidate at t0={:.2f}\n'.format(hises_t0))
                ses_mask &= hises_mask

                single_TCE = pd.DataFrame.from_dict({'star_id':[self.lc.ID],'period':[-1],'mes':[np.max(ses)],'t0':[hises_t0], 'tdur':[hises_tdur]})

            
            
        # Use for weird multi-dimensional python indexing reasons... 
        nd_ses_mask = np.ix_(np.ones(len(self.tdurs), dtype=bool), ses_mask)
        
        TCEs = Search_for_TCEs_in_all_Tdur_models(time=mask_time[ses_mask],
                                                  num=num[nd_ses_mask],
                                                  den=den[nd_ses_mask],
                                                  ses=ses[nd_ses_mask],
                                                  period_sampling=periods,
                                                  kicid=self.lc.ID,
                                                  t_durs=self.tdurs,
                                                  rho_star=self.star_density,
                                                  outputfile='tce_detection_test.h5',
                                                  texp=self.lc.exptime,
                                                  num_transits=self.min_transits,
                                                  return_df=False,
                                                  threshold=threshold,
                                                  **tce_search_kw)

        
        TCEs_checked = mask_highest_TCE_and_check_others(TCEs=TCEs,time=mask_time.copy(),
                                                         num=self.num.copy(), den=self.den.copy(), 
                                                         tdurs=self.tdurs, threshold=threshold)

        if not(single_TCE is None):
            if np.isnan(TCEs_checked).any():
                TCEs_checked = single_TCE.to_numpy()
            else:
                TCEs_checked = np.append(single_TCE.to_numpy(), TCEs_checked).reshape(-1,5)

        if return_df:
            return pd.DataFrame(TCEs_checked,
                                columns=['star_id','period','mes','t0','tdur'] ).dropna()

        return TCEs_checked





    def Search_TCEs_FFA(self,super_sample=5, dur_range=[0.25,2],use_tce_mask=False, remove_hises=True, P_range=None, **tce_search_kw,):

        
        all_tces = pd.DataFrame({'star_id':[],'period':[],'mes':[],'t0':[],'tdur':[]})


        if use_tce_mask:
            mask = self.tce_mask.copy()
        else:
            mask = self.lc.mask.copy()
            
        ses, num, den = self.Calculate_SES_by_Segment(mask=mask, fill_mode='reflect',
                                           tdurs=self.tdurs)

        masktime = self.lc.time.copy()[mask]

        ses_mask = np.ones(len(ses[0]), dtype=bool )

        single_TCE=None

        if remove_hises:
            # Cut highest SES:
            hises_ind = np.argmax(ses)%len(ses[0])
            hises_tdur_ind =  np.argmax(ses)//len(ses[0])

            hises_t0 = masktime[hises_ind]
            hises_tdur = self.tdurs[hises_tdur_ind]
            
            hises_mask = make_transit_mask(time=masktime, P=-1, t0=hises_t0,
                                           dur=hises_tdur)
            

            if np.max(ses[hises_tdur_ind][hises_mask])<0.5*np.max(ses):
                print('\nSingle Transit Candidate at t0={:.2f}\n'.format(hises_t0))
                ses_mask &= hises_mask

                single_TCE = pd.DataFrame.from_dict({'star_id':[self.lc.ID],'period':[-1],'mes':[np.max(ses)],'t0':[hises_t0], 'tdur':[hises_tdur]})


        
        for i_dur in range(len(self.tdurs)):


            dur = self.tdurs[i_dur]
            
            print('Star {1}: Searching tdur = {0:.3f}'.format(dur,self.lc.ID) )

            ses_i = self.ses[i_dur].copy()[ses_mask]
            num_i = self.num[i_dur].copy()[ses_mask]
            den_i = self.den[i_dur].copy()[ses_mask]

            if P_range is None:
                
                P_min = max( (dur * 24./1.426)**3. * (self.star_density) * dur_range[0]**3.,  dur*5.)
                P_max = min( (dur * 24./1.426)**3. * (self.star_density) * dur_range[1]**3.,  max(self.search_periods) )
            else:
                P_min, P_max = P_range

            if super_sample is None:
                super_sample=1
                dur=self.lc.exptime

                
            mid_time, num_hist, den_hist = FFA_Search_Duration_Downsample(masktime[ses_mask], num_i, den_i, dur=dur, exptime=self.lc.exptime, super_sample=super_sample)
            

            TCEs = FFA_TCE_Search(time=mid_time, num=num_hist, den=den_hist,
                                  cadence=dur/super_sample, kicid=self.lc.ID,
                                  P0_range=(P_min,P_max), dur=dur,
                                  **tce_search_kw)

            #else:
            #    mid_time, num_hist, den_hist = FFA_Search_Duration_Downsample(masktime[ses_mask], num_i, den_i, dur=dur, exptime=self.lc.exptime, super_sample=super_sample)
            
            #    TCEs = FFA_TCE_Search(time=mid_time, num=num_hist, den=den_hist,
            #                      cadence=dur/super_sample, kicid=self.lc.ID,
            #                      P0_range=(P_min,P_max), dur=dur,
            #                      **tce_search_kw)


            if len(TCEs.dropna())>0:
                tces_noharm = remove_TCE_harmonics(TCEs.to_numpy(), known_TCEs=None,
                                                      tolerance=1e9)

                
                for tce in tces_noharm:
            
                    #tce_per, tce_mes, tce_t0, tce_width = find_best_params_for_TCE(time=masktime, num=num, den=den,t0=tce[3], tdurs=self.tdurs, P=tce[1], texp=self.lc.exptime,)

                    TCE_append = [tce[0], tce[1], tce[2], tce[3], tce[4]]
                    all_tces = pd.concat([all_tces, pd.DataFrame([TCE_append],columns=['star_id','period','mes','t0','tdur'])])
            

        if len(all_tces.dropna())==0:
            return all_tces

        #tces_noharm = remove_TCE_harmonics(all_tces.to_numpy(), 
        #                                   tolerance=0.01)
        TCEs_checked = mask_highest_TCE_and_check_others(TCEs=all_tces.to_numpy(),
                            time=masktime, num=num, den=den, tdurs=self.tdurs)


        if not(single_TCE is None):
            if np.isnan(TCEs_checked).any():
                TCEs_checked = single_TCE.to_numpy()
            else:
                TCEs_checked = np.append(single_TCE.to_numpy(), TCEs_checked).reshape(-1,5)

                

        return pd.DataFrame(TCEs_checked, columns=['star_id','period','mes','t0','tdur'] )



    def Iterative_FFA_Search(self, niter=3, super_sample=5, dur_range=[0.25,1.5], tce_search_kw=None):

        print('Searching over Grid of {} transit durations:'.format(len(self.tdurs)) +
              '\n  dur range:    {:.3f}-{:.3f}'.format(min(self.tdurs), max(self.tdurs)) +
              '\n  period range: {:.3f}-{:.3f}\n'.format(min(self.search_periods),max(self.search_periods)) )

        all_tces = np.array([])
        i=0
        self.tce_mask = self.lc.mask.copy()
        while i<niter:
            
            i+=1

            ffa_tces = self.Search_TCEs_FFA(super_sample,dur_range,use_tce_mask=True, **tce_search_kw)

            if len(ffa_tces)==0:
                i=niter
            else:
                all_tces = np.append([all_tces], ffa_tces).reshape(-1, 5)

                print('\nTCEs after Iteration {}/{}'.format(i, niter))
                print(pd.DataFrame(all_tces, columns=['star_id','period','mes','t0','tdur']), '\n' )

                for tce in all_tces:
                    new_mask = make_transit_mask(time=self.lc.time, P=tce[1],
                                                  t0=tce[3], dur=tce[4]*1.5)
                    self.tce_mask &= new_mask


        
        if len(all_tces)==0:
            all_tces=None
        else:
            all_tces = remove_tces_with_bad_transit_coverage(all_tces, self.lc.time[self.lc.mask], cadence=self.lc.exptime,
                                                         min_transits=self.min_transits, ntdur=1.5, min_frac=0.75)

        self.TCEs = all_tces
            
        return pd.DataFrame(all_tces, columns=['star_id','period','mes','t0','tdur'])

    


    def Iterative_TCE_Search(self, niter=3, ses_cut=999999., threshold=7.0, check_vetoes=False, remove_hises=False, mad_window=7., fill_mode='reflect',**tce_search_kw):

        n=0
        ntces = 1
        candidates = np.array([])

        self.tce_mask = self.lc.mask.copy()

        print('Searching over Grid of {} periods and {} transit durations:'.format(len(self.search_periods), len(self.tdurs)) +
              '\n  dur range:    {:.3f}-{:.3f}'.format(min(self.tdurs), max(self.tdurs)) +
              '\n  period range: {:.3f}-{:.3f}'.format(min(self.search_periods),max(self.search_periods)) )

        while n<niter and ntces>0:
            n+=1
            print('\nStarting TCE Search Number {}'.format(n))

            if n>1:
                remove_hises=False
                check_vetoes=check_vetoes

                TCEs = self.Search_for_TCEs(calc_ses=True, threshold=threshold, ses_cut=ses_cut,remove_hises=remove_hises, return_df=False, use_tce_mask=True, check_vetoes=check_vetoes, fill_mode=fill_mode, **tce_search_kw )

            else:
                remove_hises=remove_hises
                check_vetoes=False
                
                TCEs = self.Search_for_TCEs(calc_ses=True, threshold=threshold, ses_cut=ses_cut,remove_hises=remove_hises, return_df=False, use_tce_mask=False, check_vetoes=check_vetoes, **tce_search_kw )


            if np.sum(np.isnan(TCEs[:,1]))>0:
                ntces=0
            else:
                if n>1:
                    TCEs = remove_TCE_harmonics(TCEs, candidates.reshape(-1,5))
                ntces = len(TCEs)

                for tce in TCEs:
                    self.tce_mask &= make_transit_mask(self.lc.time,P=tce[1],
                                                       t0=tce[3],dur=tce[4] )
                   
                candidates = np.append(candidates,TCEs)
            print('\n{0} TCEs added on Iteration {1}/{2}'.format(ntces,n,niter))
            print('Current Candidates:')
            print(pd.DataFrame(candidates.reshape(-1,5),columns=['star_id','period','mes','t0','tdur'] ) )

            
            if len(candidates)>10*5:
                ntces=0

            #if ntces > 0 and n<niter:   
            #    ses, num, den = self.Calculate_SES_by_Segment(mask = self.tce_mask.copy(),  window_width=mad_window )

        self.TCEs = candidates.reshape(-1,5)
        
        return pd.DataFrame(candidates.reshape(-1,5), columns=['star_id','period','mes','t0','tdur'])


    def get_max_mes(self, P, calc_ses=False):

        if calc_ses:
            ses, num, den = self.Calculate_SES()
        else:
            ses, num, den = self.ses.copy(), self.num.copy(), self.den.copy()

        fold_time = (self.lc.time[self.lc.mask]-self.lc.time[0])%P
        maxmes = 0.
        maxt0 = 0.
        for i in range(len(self.tdurs)):
            t_mes,mes = calc_mes(fold_time, num=num[i], den=den[i], P=P,
                           texp=self.lc.exptime, )

            if np.max(mes)>maxmes:
                maxmes=np.max(mes)
                maxt0 = t_mes[np.argmax(mes)]
                        

        return maxt0+self.lc.time[0], maxmes

    def get_second_max_mes(self, P, t0, calc_ses=False):

        if calc_ses:
            ses, num, den = self.Calculate_SES()
        else:
            ses, num, den = self.ses.copy(), self.num.copy(), self.den.copy()

        fold_time = (self.lc.time[self.lc.mask]-self.lc.time[0])%P
        maxmes = 0.
        maxt0 = 0.
        for i in range(len(self.tdurs)):
            t_mes,mes = calc_mes(fold_time, num=num[i], den=den[i], P=P,
                           texp=self.lc.exptime, )
            foldmask = np.abs(t_mes-t0)>1.


            if np.max(mes[foldmask])>maxmes:
                maxmes=np.max(mes)
                maxt0 = t_mes[np.argmax(mes[foldmask])]
                        

        return maxmes

    def calc_mes(self, P , ntran=2, calc_ses=False):

        if calc_ses:
            ses, num, den = self.Calculate_SES()
        else:
            ses, num, den = self.ses.copy(), self.num.copy(), self.den.copy()

        fold_time = (self.lc.time[self.lc.mask]-self.lc.time[0])%P
        foldtime_bins = np.digitize(fold_time,bins=np.arange(0,P,self.lc.exptime) )
        meslist = calc_mes_loop(foldtime_bins, num=num, den=den )
        
        return np.array(meslist)

        
        


    def plot_mes(self, t0, P, tdur, norm=False, ses_calc='mad', zoom=True, calc_ses=False, tce_mask=False, use_mask=None, plot_binflux=True, plot_all_dur=False):

        mask = self.lc.mask.copy()

        if P>max(self.lc.time)-min(self.lc.time):
            P = max(self.lc.time)-min(self.lc.time)

        if tce_mask:
            mask &= self.tce_mask.copy()

        if not(use_mask is None):
            mask &= use_mask

        if calc_ses:
            ses, num, den = self.Calculate_SES_by_Segment(mask=mask, var_calc=ses_calc)
        else:
            ses, num, den = self.ses.copy(), self.num.copy(), self.den.copy()


        if plot_all_dur:
            tdurs = self.tdurs.copy()
        else:
            tdur_sort = np.argsort(np.abs(self.tdurs.copy()-tdur))
            
            num = num[[tdur_sort[0],0,-1]]
            den = den[[tdur_sort[0],0,-1]]
            tdurs = self.tdurs.copy()[[tdur_sort[0],0,-1]]


        f, axes = plt.subplots(len(tdurs)+1, 1, sharex=True, figsize=(8, len(tdurs)*1.5), )

        fold_time = ((self.lc.time.copy()[mask] - t0 + P/4.) % P )

        mes_ylim = 8.
        mes_ymin = -3.

        max_mes = np.array([])
        
        for i,dur in enumerate(tdurs):

            t_mes,mes = calc_mes(fold_time, num=num[i], den=den[i], P=P, n_trans=1,
                                 texp=self.lc.exptime, norm=norm)

            max_mes = np.append(max_mes, np.max(mes) )
            
            axes[i+1].plot(t_mes-P/4., mes, lw=0.5, label='dur={:.3f}'.format(dur) )

            if max(mes-np.nanmedian(mes)) > mes_ylim:
                mes_ylim = max(mes)*1.1
                
            if min(mes-np.nanmedian(mes)) < mes_ymin:
                mes_ymin = min(mes)*1.1

            print('{0:.2f} duration max MES:{1:.2f}'.format(dur, np.max( mes )) )


        axes[0].plot(fold_time-P/4., self.lc.flux[mask]-1., '.', color='0.75', markersize=1,
                    )
        axes[0].set_ylabel('$\mathregular{\delta F/F}$')

        for i in range(1,len(axes)):
            axes[i].set_ylim(mes_ymin, mes_ylim)
            axes[i].axhline(7.1, color='k', ls='--')
            axes[i].axhline(0., color='k', ls='-')
            axes[i].set_ylabel('MES')
            axes[i].legend(loc='upper right', framealpha=.5)

        if plot_binflux:
            t_bin, f_bin = make_binned_flux(t=fold_time, f=self.lc.flux[mask]-1.,
                                            texp=tdur/3.)

            axes[0].plot(t_bin-P/4.,f_bin, 'o', markersize=2,)
            

        if P>4 and zoom:

            t_range = self.tdurs[np.argmax(max_mes)]
            axes[0].set_xlim(P/4.-3*t_range, P/4.+3*t_range)

        axes[-1].set_xlabel('Phase [days]')

        return axes




    

def make_transit_mask(time, P, t0, dur):

    if P<0:
        fold_time = time-t0
    else:
        fold_time = (time - t0 + P/2.)%P - P/2.
    
    mask = np.abs(fold_time) > 1.*dur
    
    return mask


def get_limb_darkening_grid(mission, grid_file='data/meta/limb_coeffs.h5'):

    ld_file = TRASH_DATA_DIR + grid_file
    limbs = pd.read_hdf(ld_file, mission)
    
    return limbs


KEPLER_LIMB_DARKENING_GRID = get_limb_darkening_grid('KEPLER')
TESS_LIMB_DARKENING_GRID = get_limb_darkening_grid('TESS')






def get_optimal_period_sampling(time, A = 5.217e-5, OS=3., rho=1., ntransits=3.):

    '''
    Calculates the optimal period spacing to search for transit signals (from Ofir+2014)
    time: time of each observation
    A: Constant related to M, R of star
    OS: Oversampling factor; recommended to be 2-5
    '''

    A = A * rho**(-1./3.) / OS
    
    f_max = 2*np.pi*1.661  #Set by Roche lobe of Sun-like star, can change this later
    f_min = ntransits*2.*np.pi/(max(time) - min(time)) # Minimum frequency calculated from observing time
    N_samp = np.ceil( ( f_max**(1./3.) - f_min**(1./3.) + A/3.) * (3./A) ) 
        
    freq_sampling = (A*np.arange(1., N_samp, 1)/3. + f_min**(1./3.)  - A/3.)**3.

    # convert frequency to period
    period_sampling = 2.*np.pi/freq_sampling[::-1]
    
    return period_sampling



def kep_period_sampling(time, P_min=None, P_max=None, n_tr=3., rho_star=1., d_leastsq=0.075):

    if P_min is None:
        P_min = 0.25 / np.sqrt(rho_star)
    if P_max is None:
        P_max = np.inf
    
    Pt = P_min
    periods = [P_min]

    baseline = max(time) - min(time)
    if n_tr>1:
        n_tr-=1
    while Pt < min(baseline/n_tr, P_max):
        
        d_dur = d_leastsq * (13./24.) * (Pt/365.)**(1./3.) * rho_star**(-1./3.)
        Pt += 4. * d_dur * (Pt/baseline)
        periods.append(Pt)
    
    return np.array(periods)





'''
def Detrend_and_Calculate_SES_to_df(time, flux, limb_dark, calc_depth=True, 
                                    tdurs=np.array([1.,1.5,2,2.5,3,3.5,4.5,5,6,7,8,9,10,11,12,13.,14.,15.,16.5,18.,19.5,21,23.,25.])/24.,  method='biweight', w=5.,var_calc='gap_mean', var_calc_window=10., detrend_model=False, texp=0.0204):

    
    ses_models = []
    num_models = []
    den_models = []

    detrend_masks = []
    detrend_flux = []

    ses_keys = ['ses_{0:.2f}d'.format(dur) for dur in tdurs]
    num_keys = ['num_{0:.2f}d'.format(dur) for dur in tdurs]
    den_keys = ['den_{0:.2f}d'.format(dur) for dur in tdurs]
    
    detrend_mask_keys = ['detrend_mask_{0:.2f}d'.format(dur) for dur in tdurs]
    detrend_flux_keys = ['detrend_flux_{0:.2f}d'.format(dur) for dur in tdurs]
    
    tdur_sigma = []


    for dur in tdurs:

        window_length = max(w*dur, 0.75)

        flux_detrended, flux_trend = flatten(time, flux, window_length=window_length, method=method, return_trend=True, edge_cutoff=window_length/2., break_tolerance=0.5*window_length)

        _, _, sig_clip = sigma_clip(time, flux_detrended, use_mad=False)

        flux_detrended[~sig_clip]=np.nan
        
        detrend_flux.append(flux_detrended )
        
        mask =  ~np.isnan(flux_detrended)

        detrend_masks.append(mask)
        pad_time, pad_flux, pad_truth = pad_time_series(x=time[mask], y=flux_detrended[mask],pad_end=True )

        
        flux_transform = ocwt(pad_flux-1.)

        if calc_depth:
            depth = mad(pad_flux)*1e6
        else:
            depth = calc_depth

        if detrend_model:
            mod_f = get_transit_signal(dur, depth, limb_dark=limb_dark, )
            mod_t = np.arange(0,len(mod_f))*texp
            model = flatten(mod_t, mod_f, window_length=w*dur, method=method)-1.
        else:
            model =  get_transit_signal(dur, depth, limb_dark=limb_dark, ) - 1.
            
        model_transform = ocwt( model )
                
        ses,num,den = calculate_SES(flux_transform, model_transform, window_size=var_calc_window*dur, var_calc=var_calc)
        
        ses = ses[pad_truth]
        num = num[pad_truth]
        den = den[pad_truth]

        tdur_sigma.append(np.median(den) )

        ses_zero_padded = np.zeros(len(mask), dtype=np.float32 )
        ses_zero_padded[mask] = ses

        num_zero_padded = np.zeros(len(mask), dtype=np.float32 )
        num_zero_padded[mask] = num

        den_zero_padded = np.zeros(len(mask), dtype=np.float32 )
        den_zero_padded[mask] = den
        
        ses_models.append( ses_zero_padded )
        num_models.append( num_zero_padded )
        den_models.append( den_zero_padded )

    keys = np.concatenate([detrend_flux_keys,detrend_mask_keys,ses_keys,num_keys,den_keys])
    data = np.concatenate([detrend_flux,detrend_masks,ses_models,num_models,den_models])

    keys = np.append(['time', 'pdcsap'], keys )
    data = np.concatenate([[time, flux], data ])
                     

    #print(np.shape(data), len(data) )
    
    df = pd.DataFrame( data=data.T, columns=keys,  )

    #lc.tdur_sigma = tdur_sigma
    
    return df
'''


def Detrend_and_Calculate_SES_for_all_Tdur_models(lc, tdurs=np.array([1.,1.5,2,2.5,3,3.5,4.5,5,6,7,8,9,10,11,12,13.,14.,15.,16.5,18.,19.5,21,23.,25.])/24., method='biweight', w=5.,var_calc='slat_mean',  var_calc_window=10.):
    
    lc.tdurs = tdurs
    
    ses_models = []
    num_models = []
    den_models = []

    detrend_masks = []
    detrend_flux = []

    ses_keys = ['ses_{0:.2f}d'.format(dur) for dur in tdurs]
    num_keys = ['num_{0:.2f}d'.format(dur) for dur in tdurs]
    den_keys = ['den_{0:.2f}d'.format(dur) for dur in tdurs]
    
    detrend_mask_keys = ['detrend_mask_{0:.2f}d'.format(dur) for dur in tdurs]
    detrend_flux_keys = ['detrend_flux_{0:.2f}d'.format(dur) for dur in tdurs]
    tdur_sigma = []


    limb_dark = np.concatenate([lc.stlr['limb1'],lc.stlr['limb2'],lc.stlr['limb3'],lc.stlr['limb4']])
    
    for dur in lc.tdurs:

        window_length = max(w*dur, 0.75)

        flux_detrended, flux_detrend = flatten(lc.time, lc.pdcsap, window_length=window_length, method=method, return_trend=True, edge_cutoff=window_length/2., break_tolerance=0.5*window_length)
        
        _, _, sig_clip = sigma_clip(lc.time, flux_detrended, use_mad=False)
        flux_detrended[~sig_clip]=np.nan

        detrend_flux.append(flux_detrended )
        
        mask = ~np.isnan(flux_detrended)
        detrend_masks.append(mask)
        
        pad_time, pad_flux, pad_truth = pad_time_series(x=lc.time[mask], y=flux_detrended[mask])
        
        flux_transform = ocwt(pad_flux-1.)
        depth = 250.
        model_transform = ocwt( get_transit_signal(dur, depth, limb_dark=limb_dark )-1. )
                
        ses,num,den = calculate_SES(flux_transform, model_transform, window_size=var_calc_window*dur, var_calc=var_calc)
        
        ses = ses[pad_truth]
        num = num[pad_truth]
        den = den[pad_truth]

        tdur_sigma.append (np.median(den) )

        ses_zero_padded = np.zeros(len(mask), dtype=np.float32 )
        ses_zero_padded[mask] = ses

        num_zero_padded = np.zeros(len(mask), dtype=np.float32 )
        num_zero_padded[mask] = num

        den_zero_padded = np.zeros(len(mask), dtype=np.float32 )
        den_zero_padded[mask] = den
        
        ses_models.append( ses_zero_padded )
        num_models.append( num_zero_padded )
        den_models.append( den_zero_padded )

    lc.ses = ses_models
    lc.num = num_models
    lc.den = den_models
    
    lc.detrend_masks = detrend_masks
    lc.detrend_flux = detrend_flux

    lc.ses_keys = ses_keys
    lc.num_keys = num_keys
    lc.den_keys = den_keys

    lc.detrend_mask_keys = detrend_mask_keys
    lc.detrend_flux_keys = detrend_flux_keys
    lc.tdur_sigma = tdur_sigma
    
    return lc




def FFA_Search_Duration_Downsample(time, num, den, dur, exptime, super_sample=5.):    

    dt =  dur/super_sample
    time_bins = np.arange(min(time), max(time)+dur/super_sample, dt)
    
    mid_time = (time_bins[:-1] + time_bins[1:])/2.
    norm_factor = dur/(super_sample * exptime)


    padded_time, padded_num, padded_truth = pad_time_series(time, num, in_mode='reflect', cadence=exptime)
    _, padded_den, _ = pad_time_series(time, den, in_mode='reflect',cadence=exptime)

    if dt<exptime:
        resample_factor = int(np.ceil(exptime/dt))
        norm_factor *= resample_factor
        padded_num_new, padded_time_new = resample(padded_num, len(padded_num)*resample_factor, t=padded_time)
        padded_den_new = resample(padded_den, len(padded_den)*resample_factor)

        padded_truth_new = np.interp(padded_time_new, padded_time, padded_truth)
        padded_truth_new[padded_truth_new<1]=0.
            
    
        num_hist,_ = np.histogram(padded_time_new, bins=time_bins, weights=padded_num_new*padded_truth_new/norm_factor)
        den_hist,_ = np.histogram(padded_time_new, bins=time_bins, weights=padded_den_new*padded_truth_new/norm_factor)

    else:
        num_hist,_ = np.histogram(padded_time, bins=time_bins, weights=padded_num/norm_factor)
        den_hist,_ = np.histogram(padded_time, bins=time_bins, weights=padded_den/norm_factor)
        
    

    return mid_time, num_hist, den_hist





def FFA_TCE_Search(time, num, den, cadence, dur, P0_range, kicid=99, progbar=True,
                   fill_gaps=False, threshold=7., check_vetoes=False, single_frac=0.7,
                   print_updates=True,return_df=True):

    
    time_cad = (time / cadence).astype(int)
    t0 = time[0]

    if fill_gaps:

        _,num_padded,_ = pad_time_series(x=time_cad, y=num, min_dif=1.01, cadence=1,
                                           in_mode='constant', 
                                            pad_end=False, fill_gaps=True, constant_val=0.)
    
        _,den_padded,_ = pad_time_series(x=time_cad, y=den, min_dif=1.01, cadence=1,
                                           in_mode='constant', 
                                            pad_end=False, fill_gaps=True, constant_val=0.)
    
    else:
        num_padded = num
        den_padded = den
    
    ses = num / np.sqrt(den)
    
    numlength = len(num_padded)
    max_periods = []
    max_mes = []

    get_Npad= lambda P, length: int( P * 2**np.ceil(np.log2(length/P) )  - length )

    Pcad0 =  np.arange(P0_range[0]//cadence, np.ceil(P0_range[1]/cadence), dtype=int)
    
    if progbar:
        period_iterable = tqdm(Pcad0)
    else:
        period_iterable = Pcad0
    
    
    detections = {'star_id':[], 'period':[], 'mes':[],'t0':[],'tdur':[]}

    
    
    for P0 in period_iterable:

        
        N_pad = get_Npad(P0,numlength)

        num_endpad = np.pad(num_padded, (0,N_pad), mode='constant').reshape((-1, P0) )
        den_endpad = np.pad(den_padded, (0,N_pad), mode='constant').reshape((-1, P0) )

        mes =  FFA(num_endpad) / np.sqrt( FFA( den_endpad) )

        #if P0*cadence>13.:
        #    print(P0*cadence, np.nanmax(mes), np.amax(mes))


        if np.nanmax(mes)>threshold:

            n_P, n_t0 = mes.shape
            
            dt0 = np.arange(0,n_t0+1) * cadence #+ cadence
            Ps = np.arange(0,n_P+1) #* cadence
            
            #split_indices = np.arange(0,numlength+N_pad,P0)
            #Ps = 0

            Pt = P0**2. / (P0 - (Ps / (n_P-1) ) ) * cadence
            
            #Pt = (P0 + (split_indices/P0) / (len(split_indices)  ) ) * cadence

            mes[np.isnan(mes)] = 0.
            
            max_args = np.unravel_index(np.argmax(mes,) , shape=mes.shape)
            max_mes = mes[max_args[0],max_args[1]]
                        
            best_period = Pt[max_args[0]]
            best_dt0 = dt0[max_args[1]]
            detected_mes = max_mes #mes[max_args[0],:]

            
            # Check TCE Vetoes
            if check_vetoes:
                
                one_tce_kwargs = {'time_bins':detected_time_bins, 'mes':detected_mes,
                                  'time_bin_t0':best_dt0,
                                  'dur':detected_duration, 'n_dur':5.}
                single_event_kwargs = {'fold_time':time%best_period, 'max_mes':max_mes,
                                       'ses':ses, 'peak_loc':best_dt0+time[0],
                                       'dur':dur, 'P':best_period,'frac':single_frac}
                gap_edge_kwargs = {'time':time, 't0':detected_peak_loc+t0,
                                   'P':P, 't_dur':dur}
                not_vetoed = ts.check_tce_vetoes(one_tce_kwargs,single_event_kwargs,
                                              gap_edge_kwargs)
                
            else:

                single_event_kwargs = {'fold_time':time%best_period, 'max_mes':max_mes,
                                       'ses':ses, 'peak_loc':best_dt0+time[0],
                                       'dur':dur, 'P':best_period, 'frac':single_frac}
                
                not_vetoed = not(check_tce_caused_by_single_event(**single_event_kwargs))
            
            
            if not_vetoed:

                if print_updates:
                    print('\rSTAR {4:09d}: TCE at P={0:.7f} days,    MES={1:.2f},    Tdur={2:.3f} days,    t0={3:.2f}'.format(best_period,max_mes,dur,best_dt0+t0, kicid))

                    #plot_detection(mes, dt0, Pt)
                    #plt.show()
                    
                detections['star_id'].append(kicid)
                detections['period'].append(best_period)
                detections['mes'].append(max_mes)
                detections['t0'].append(best_dt0+t0)
                detections['tdur'].append(dur)
            
                           
    df = pd.DataFrame( detections )
    # save final output
    if len(df)>0:
        
        cleaned_df = clean_tces_of_harmonics(df)

        if print_updates:
            print(cleaned_df)

        if return_df:
            return cleaned_df
        return cleaned_df.to_numpy()
    else:
        #print('\nNo Reliable TCEs found in STAR {0}'.format(kicid))
        if return_df:
            return df
        return np.array([[kicid, np.nan, np.nan, np.nan, np.nan]])





'''
def Search_for_TCEs_from_h5_file(lc_file_object, tcefile='results/tce_detection_test.h5', print_updates=True, dur_range=(.5, 2.)):

    data, metadata = lc_file_object
    
    time = data['time'].values.astype(np.float32)

    data_cols = data.columns

    num_keys = sorted([i for i in data_cols if i.startswith('num')])
    den_keys = sorted([i for i in data_cols if i.startswith('den')])
    mask_keys = sorted([i for i in data_cols if i.startswith('detrend_mask')])

    all_masks = data[mask_keys].values.T.astype('bool')
    all_num = data[num_keys].values.T.astype(np.float32)
    all_den = data[den_keys].values.T.astype(np.float32)

    rho_star = float(metadata['stlr']['b18_density'][0] )
    
    psamp = get_optimal_period_sampling(time, ntransits=3., OS=1., rho=rho_star)

    TCEs = Search_for_TCEs_in_all_Tdur_models(time=time, num=all_num, den=all_den, period_sampling=psamp, kicid=metadata['kic'], t_durs=metadata['tdurs'], ses_masks=all_masks, rho_star=rho_star, dur_range=dur_range, outputfile=tcefile, print_updates=print_updates)
    
    return TCEs
'''






def Search_for_TCEs_in_all_Tdur_models_2(time,num,den,ses,period_sampling,kicid,t_durs,rho_star=1.,dur_range=(.25, 1.5), outputfile='tce_detection_test.h5', print_updates=True, texp=0.0204, return_df=False, progbar=True, num_transits=2, check_vetoes=False, single_frac=0.9, norm_mes=False, threshold=7.1):


    '''
    Time: Time series data
    ses: the single event statistic from the previous step
    period_sampling: given from above
    transit duration: the width of the transit model used
    kicid: the KIC of the star used to identify it in files
    '''

    t0 = np.min(time)
    t0_time = time-t0
    
    detections = {'star_id':[], 'period':[], 'mes':[],'t0':[],'tdur':[]}

    buffer_count=0
    
    min_dur_buff = (13./24.) * (1./365.)**(1./3.) * rho_star**(-1./3.) * dur_range[0]
    max_dur_buff = (13./24.) * (1./365.)**(1./3.) * rho_star**(-1./3.) * dur_range[1]

    
    #calculate the MES for each period in period sampling
    if progbar:
        iterable =  tqdm(period_sampling[::-1])
    else:
        iterable = period_sampling[::-1]
        
    for P in iterable:
        
        mes_time = np.arange(0, P+texp, texp)
        fold_time = np.mod(t0_time, P)
        
        foldtime_bins = np.digitize(fold_time, bins=mes_time )
        number_transits = np.bincount(foldtime_bins)
        
        
        min_dur = P**(1./3.) * min_dur_buff
        max_dur = P**(1./3.) * max_dur_buff


        tdur_mask = t_durs <= max_dur
        tdur_mask &= t_durs >= min_dur
        tdur_mask &= t_durs < P/7.

        #print(tdur_mask)

        num_dur = num[tdur_mask]
        den_dur = den[tdur_mask]
        
        mes = calc_mes_loop(foldtime_bins, num_dur, den_dur)

        #print(np.argmax(  ) ) 

        if np.nanmax(mes) > threshold:

            all_maxmes = np.nanmax(mes, axis=1)
            dur_i = np.nanargmax(all_maxmes)
            mes_i = np.nanargmax(mes[dur_i])

            mes[np.isnan(mes)]  = 0.
            max_mes = all_maxmes[dur_i]

            detected_duration = t_durs[dur_i]
            detected_mes = mes[dur_i]
            detected_time_bins = mes_time
            detected_peak_loc = mes_time[mes_i]
            detected_ses = ses[tdur_mask][dur_i]
        
        
        
        #for i,dur in enumerate(t_durs):
                 
       #     if dur<=max_dur and dur>=min_dur and dur<P/10.:
                
                #bin_edge, mes = calc_mes(fold_time=fold_time,num=num[i], den=den[i],
                #                         norm=norm_mes, texp=texp,P=P, n_trans=num_transits)

                #if P<1000.:
                #    mes-=np.nanmedian(mes)
                #if norm_mes:
                #    mes /= mad(mes)

                #if np.max(mes) > max_mes:
                    
                #    max_mes=np.max(mes)

                #    detected_duration = dur
                #    detected_mes = mes
                #    detected_time_bins = bin_edge
                #    detected_peak_loc = bin_edge[np.argmax(mes)]
                #    detected_ses = ses[i]

    
        #if max_mes>7.1:

            # Check TCE Vetoes
            if check_vetoes:
                one_tce_kwargs = {'time_bins':detected_time_bins, 'mes':detected_mes,
                                  'time_bin_t0':detected_peak_loc,
                                  'dur':detected_duration, 'n_dur':5.}
                single_event_kwargs = {'fold_time':fold_time, 'max_mes':max_mes,
                                       'ses':detected_ses, 'peak_loc':detected_peak_loc,
                                       'dur':detected_duration, 'P':P,'frac':0.8}
                gap_edge_kwargs = {'time':time, 't0':detected_peak_loc+t0,
                                   'P':P, 't_dur':detected_duration}
                not_vetoed = check_tce_vetoes(one_tce_kwargs,single_event_kwargs,
                                              gap_edge_kwargs)
                #mad_mes_ratio_test = check_mad_mes_ratio(detected_mes, P)
                
            else:
                single_event_kwargs = {'fold_time':fold_time, 'max_mes':max_mes,
                                       'ses':detected_ses, 'peak_loc':detected_peak_loc,
                                       'dur':detected_duration, 'P':P, 'frac':single_frac}
                
                not_vetoed = not(check_tce_caused_by_single_event(**single_event_kwargs))
            
            if not_vetoed:

                if print_updates:
                    print('\rKIC{4:09d}: TCE at P={0:.7f} days,    MES={1:.2f},    Tdur={2:.3f} days,    t0={3:.2f}'.format(P,max_mes,detected_duration,detected_peak_loc+t0, kicid))

                # write detections to buffer
                buffer_count +=1
                detections['star_id'].append(kicid)
                detections['period'].append(P)
                detections['mes'].append(max_mes)
                detections['t0'].append(detected_peak_loc+t0)
                detections['tdur'].append(detected_duration)
                
    
    df = pd.DataFrame( detections )
    # save final output
    if buffer_count>0:
        
        cleaned_df = clean_tces_of_harmonics(df)

        if print_updates:
            print(cleaned_df)

        if return_df:
            return cleaned_df
        return cleaned_df.to_numpy()
    else:
        return np.array([[kicid, np.nan, np.nan, np.nan, np.nan]])

    return TCEs





def Search_for_TCEs_in_all_Tdur_models(time,num,den,ses,period_sampling,kicid,t_durs,rho_star=1.,dur_range=(.25, 1.5), outputfile='tce_detection_test.h5', print_updates=True, texp=0.0204, return_df=False, progbar=True, num_transits=2, check_vetoes=False, single_frac=0.9, norm_mes=False,threshold=7.):
    
    '''
    Time: Time series data
    ses: the single event statistic from the previous step
    period_sampling: given from above
    transit duration: the width of the transit model used
    kicid: the KIC of the star used to identify it in files
    '''

    t0 = np.min(time)
    t0_time = time-t0
    
    detections = {'star_id':[], 'period':[], 'mes':[],'t0':[],'tdur':[]}

    buffer_count=0
    
    min_dur_buff = (13./24.) * (1./365.)**(1./3.) * rho_star**(-1./3.) * dur_range[0]
    max_dur_buff = (13./24.) * (1./365.)**(1./3.) * rho_star**(-1./3.) * dur_range[1]


    #calculate the MES for each period in period sampling
    if progbar:
        iterable =  tqdm(period_sampling[::-1])
    else:
        iterable = period_sampling[::-1]
    for P in iterable:
        
        max_mes=threshold
        
        fold_time = np.mod(t0_time, P)
        min_dur = P**(1./3.) * min_dur_buff
        max_dur = P**(1./3.) * max_dur_buff
        
        for i,dur in enumerate(t_durs):
                 
            if dur<=max_dur and dur>=min_dur and dur<P/3.:
                
                bin_edge, mes = calc_mes(fold_time=fold_time,num=num[i], den=den[i],
                                         norm=norm_mes, texp=texp,P=P, n_trans=num_transits)

                #if P<1000.:
                #    mes-=np.nanmedian(mes)
                #if norm_mes:
                #    mes /= mad(mes)

                if np.max(mes) > max_mes:
                    
                    max_mes=np.max(mes)

                    detected_duration = dur
                    detected_mes = mes
                    detected_time_bins = bin_edge
                    detected_peak_loc = bin_edge[np.argmax(mes)]
                    detected_ses = ses[i]

        
        if max_mes>threshold:

            # Check TCE Vetoes
            if check_vetoes:
                one_tce_kwargs = {'time_bins':detected_time_bins, 'mes':detected_mes,
                                  'time_bin_t0':detected_peak_loc,
                                  'dur':detected_duration, 'n_dur':5.}
                single_event_kwargs = {'fold_time':fold_time, 'max_mes':max_mes,
                                       'ses':detected_ses, 'peak_loc':detected_peak_loc,
                                       'dur':detected_duration, 'P':P,'frac':0.8}
                gap_edge_kwargs = {'time':time, 't0':detected_peak_loc+t0,
                                   'P':P, 't_dur':detected_duration}
                not_vetoed = check_tce_vetoes(one_tce_kwargs,single_event_kwargs,
                                              gap_edge_kwargs)
                #mad_mes_ratio_test = check_mad_mes_ratio(detected_mes, P)
                
            else:
                single_event_kwargs = {'fold_time':fold_time, 'max_mes':max_mes,
                                       'ses':detected_ses, 'peak_loc':detected_peak_loc,
                                       'dur':detected_duration, 'P':P, 'frac':single_frac}
                
                not_vetoed = not(check_tce_caused_by_single_event(**single_event_kwargs))
            
            if not_vetoed:

                if print_updates:
                    print('\rKIC{4:09d}: TCE at P={0:.7f} days,    MES={1:.2f},    Tdur={2:.3f} days,    t0={3:.2f}'.format(P,max_mes,detected_duration,detected_peak_loc+t0, kicid))

                # write detections to buffer
                buffer_count +=1
                detections['star_id'].append(kicid)
                detections['period'].append(P)
                detections['mes'].append(max_mes)
                detections['t0'].append(detected_peak_loc+t0)
                detections['tdur'].append(detected_duration)
                
    
    df = pd.DataFrame( detections )
    # save final output
    if buffer_count>0:

        cleaned_df = clean_tces_of_harmonics(df)

        if print_updates:
            print(cleaned_df)

        if return_df:
            return cleaned_df
        return cleaned_df.to_numpy()
    else:
        return np.array([[kicid, np.nan, np.nan, np.nan, np.nan]])
        



    
def remove_TCE_harmonics(test_TCEs, known_TCEs=None, tolerance=0.0001):

    
    sorted_mes_index = np.argsort(-test_TCEs[:,2])

    if len(test_TCEs)==1:
        return test_TCEs
    
    if known_TCEs is None:
        test_TCEs_sorted = test_TCEs[sorted_mes_index[1:]]
        is_harmonic = np.zeros(len(test_TCEs)-1, dtype=bool)
        known_periods =  np.array([test_TCEs[sorted_mes_index[0],1]])
        known_t0s =  np.array([test_TCEs[sorted_mes_index[0],3]])
        add=True

    else:
        test_TCEs_sorted = test_TCEs[sorted_mes_index]
        is_harmonic = np.zeros(len(test_TCEs),dtype=bool)
        known_periods = known_TCEs[:,1]
        known_t0s = known_TCEs[:,3]
        add=False

    
    tce_periods = test_TCEs_sorted[:,1]
    tce_t0s = test_TCEs_sorted[:,3]
    tce_durs = test_TCEs_sorted[:,4]
    
    for i,p in enumerate( tce_periods ):


        harm_test = np.concatenate([(known_periods/p)%1. , (p/known_periods)%1.])


        if any(harm_test<tolerance) or any((1.-harm_test) < tolerance):
            if any( np.abs(known_t0s - tce_t0s[i]) < 2.*tce_durs[i]):
                is_harmonic[i] = True
            
        else:
            known_periods = np.append(known_periods, p)

    if add:
        test_TCEs_sorted = test_TCEs[(-test_TCEs[:,2]).argsort()]
        is_harmonic = np.append(False, is_harmonic)

        
    test_TCEs_noharm = test_TCEs_sorted[~is_harmonic]
    same_t0 = np.zeros(len(test_TCEs_noharm), dtype=bool)

    
    for i, tce in enumerate(test_TCEs_noharm[1:]):
        
        if any(np.abs(test_TCEs_noharm[:i+1,3]-tce[3])<2*tce[4]):
            
            same_t0[i+1] = True
    
    
    return  test_TCEs_noharm[~same_t0]




    
def clean_tces_of_harmonics(tces,tolerance=0.01):
                
    periods = tces['period'].values
    t0s =  tces['t0'].values
    good_mes = []
    good_mes2 = []

    for i,p in enumerate(periods[::-1]):
        highest_mes = 0.
        same_t0s = np.abs( tces['t0'].values - t0s[i] ) < 0.03
        same_periods =  np.abs( (periods - p )/p) < 0.02
                
        highest_mes = np.max( tces['mes'].values[ same_periods ])
        good_mes.append(highest_mes )

    
    cut  =  np.isin(tces['mes'].values, np.unique( good_mes ) )
    
    cleaned_tces ={'star_id':tces['star_id'].values[cut],
                   'period':tces['period'].values[cut], 
                   'mes':tces['mes'].values[cut],
                   't0':tces['t0'].values[cut],
                   'tdur':tces['tdur'].values[cut],
}
    
    tce_count = 0
    
    super_clean_period = [ ]
        
    for i,p in enumerate(cleaned_tces['period']):
            
        add=True
            
        for n in [1.,3./2., 4./3.,5./3., 7./3., 2., 5./2., 7/2., 6./5.,17./3., 3., 4., 5., 6., 7., 8., 9., 10.,]:
                
            condition1 = np.abs( (cleaned_tces['period']-n*p)/cleaned_tces['period'] )
            condition2 = np.abs( (cleaned_tces['period']-(p/n))/cleaned_tces['period'] )

            is_harm1 = np.logical_and(condition1<tolerance, condition1>0.) 
            is_harm2 = np.logical_and(condition2<tolerance, condition2>0.)
            
            condition = np.logical_or(is_harm1, is_harm2)
            
            if condition.any() and cleaned_tces['mes'][i] < np.max(cleaned_tces['mes'][condition]):
                                
                add=False
            
        if add:
            super_clean_period.append( cleaned_tces['period'][i] )
                

    cut2 = np.isin( cleaned_tces['period'], np.unique( super_clean_period ) )
    
    final_tces = pd.DataFrame({'star_id':cleaned_tces['star_id'][cut2],
                               'period':cleaned_tces['period'][cut2],
                               'mes':cleaned_tces['mes'][cut2],
                               't0':cleaned_tces['t0'][cut2],
                               'tdur':cleaned_tces['tdur'][cut2],
    } )
            
    noharm_tces = final_tces.drop_duplicates(keep='last')
    
    return noharm_tces






'''
def mask_TCEs_and_run_transit_search_again(TCEs, lc_file_object):

    
    data, meta = lc_file_object

    time = data['time'].values.astype(np.float32)
    flux = data['pdcsap'].values.astype(np.float32)
    
    tr_mask = np.ones_like(time, dtype=bool)

    for i,row in TCEs.iterrows():

        P, t0, tdur = row['period'], row['t0_bkjd'], row['tdur']*1.5
        tr_mask &= ~transit_mask(time, period=P, duration=tdur, T0=t0, )
        

    limbs = np.concatenate([meta['stlr']['limb1'],meta['stlr']['limb2'],meta['stlr']['limb3'],meta['stlr']['limb4']])


    print('Detrending Light Curves ... ')

    pad_time, pad_flux, pad_mask = pad_time_series(time[tr_mask], flux[tr_mask], )
    
    
    lc_df = Detrend_and_Calculate_SES_to_df(pad_time, pad_flux, limb_dark=limbs, calc_depth=250., tdurs=meta['tdurs'])
    

    data_cols = data.columns

    num_keys = sorted([i for i in data_cols if i.startswith('num')])
    den_keys = sorted([i for i in data_cols if i.startswith('den')])
    mask_keys = sorted([i for i in data_cols if i.startswith('detrend_mask')])

    
    all_masks = lc_df[mask_keys].values.T.astype('bool')
    
    all_num = lc_df[num_keys].values.T.astype(np.float32)
    all_den = lc_df[den_keys].values.T.astype(np.float32)

    rho_star = float(meta['stlr']['b18_density'][0] )
    
    psamp = get_optimal_period_sampling(time, ntransits=3., OS=1., rho=rho_star)

    print('Searching for new TCEs ... ')
    
    new_TCEs = Search_for_TCEs_in_all_Tdur_models(time=time[tr_mask], num=all_num, den=all_den, period_sampling=psamp, kicid=meta['kic'], t_durs=meta['tdurs'], ses_masks=all_masks, rho_star=rho_star, )

    return new_TCEs
'''

    


def check_tce_happens_once(time_bins, mes, time_bin_t0, dur, n_dur=4.5, P_lim=10. ):


    if np.max(time_bins)<P_lim:
        width_tolerance = np.max(time_bins)/4.
    else:
        width_tolerance = min(dur*n_dur, np.max(time_bins)/10.)

    out_of_transit_mask = np.abs(time_bins - time_bin_t0) > width_tolerance

    if dur > np.max(time_bins)/3.:
        return False
    
    if np.max(mes[out_of_transit_mask]) > 7.1:
        return False
    else:
        return True



def choose_highest_mes_not_caused_by_one_event(fold_time, time_bins, mes, ses, dur, P, texp=0.0204, P_lim=50., frac=0.9):

    dominated_by_one_event=True
    max_mes = 10

    while max_mes >= 7.1 and dominated_by_one_event:

        max_mes = np.max(mes)
        peak_loc = time_bins[np.argmax(mes)]

        dominated_by_one_event =  check_tce_caused_by_single_event(fold_time, max_mes, ses, peak_loc, dur, P, texp=texp, P_lim=P_lim, frac=frac)

        if dominated_by_one_event:
            near_event = np.abs(time_bins - peak_loc) < dur
            mes[near_event] = -99.


    return peak_loc, max_mes





def check_tce_caused_by_single_event(fold_time, max_mes, ses, peak_loc, dur, P, texp=0.0204, P_lim=50., frac=0.7):

    fold_time_shift = (fold_time - peak_loc + P/2)%P - P/2.
    
    in_transit = np.abs(fold_time_shift) < dur
    ses_in_transit = ses[in_transit]
    
    if P<P_lim:
        if np.max(ses_in_transit) > max_mes:
            return True
        return False

    if np.max(ses_in_transit)/max_mes>frac:
        return True
    else:
        return False



def check_if_TCE_at_gap_edge(time, t0, P, t_dur, texp=0.0204):

    dt = time[1:]-time[:-1]
    gaps =  dt>0.5*t_dur
    gaps_bool = np.logical_or( np.concatenate([gaps, [True]]), np.concatenate([[True], gaps]))

    in_transit = make_transit_mask(time, P=P, dur=t_dur, t0=t0 )

    transit_points_near_gaps = np.logical_and(gaps_bool, in_transit)

    N_transits_in_gaps = np.sum( transit_points_near_gaps)
    N_transits = np.sum( in_transit )

    fraction_near_gaps =  N_transits_in_gaps/N_transits
    
    if fraction_near_gaps>0.33:
        return True
    else:
        return False


def check_tce_vetoes(one_tce_kwargs, single_event_kwargs, gap_edge_kwargs):

    if check_tce_happens_once(**one_tce_kwargs):
        if not(check_tce_caused_by_single_event(**single_event_kwargs) ):
               if not( check_if_TCE_at_gap_edge(**gap_edge_kwargs) ):
                   return True

    return False



def check_mad_mes_ratio(mes, P, thresh=7.1):

    if P>10.:
        return np.max(mes)/mad(mes[np.abs(mes)>1e-9]) > thresh
    else:
        return True
    


def plot_mes(lc_file_object, period, t0, mes_ls='-', texp=0.0204, mes_norm=False ):
    
    P=period

    data, metadata = lc_file_object

    time = data['time'].values.astype(float)

    tdur = metadata['tdurs']
    rho_star = metadata['stlr']['b18_density'][0]

    data_cols = data.columns.values.astype(str)
    ses_cols = sorted([i for i in data_cols if i.startswith('ses')])
    num_cols = sorted([i for i in data_cols if i.startswith('num')])
    den_cols = sorted([i for i in data_cols if i.startswith('den')])

    mask_cols = sorted([i for i in data_cols if i.startswith('detrend_mask')])
    flux_cols = sorted([i for i in data_cols if i.startswith('detrend_flux')])
    
    ses = data[ses_cols].values.T.astype(float)
    num = data[num_cols].values.T.astype(float)
    den = data[den_cols].values.T.astype(float)

    mask = data[mask_cols].values.T.astype(bool)
        
    f, axes = plt.subplots(len(tdur), 1, sharex=True, figsize=(10, len(tdur)*1.3), )
    
    fold_time = (time - t0 + P/2.)%P
    max_mes = 7.09
    min_mes = 0.

    i_max = 0
    
    for i in range(1,len(axes)):
        
        detrend_mask = mask[i]
        
        ses_masked = ses[i][detrend_mask]
        time_masked = fold_time[detrend_mask]
        
        num_masked = num[i][detrend_mask]
        den_masked = den[i][detrend_mask]

        
        bin_edge, mes = calc_mes_from_dn(fold_time=time_masked,
                                    num=num_masked, den=den_masked,
                                         P=period, norm=mes_norm, texp=texp)

        if period<10.:

            if period<3.:
                mes-=np.nanpercentile(mes, [25])[0]
            else:
                mes-=np.nanmedian(mes)

        if np.min(mes)<min_mes:
            min_mes = np.min(mes)

        if np.max(mes)>max_mes:
            max_mes = np.max(mes)
            i_max = i

        
        # find peaks in the MES distribution
        detect_width = np.array([.5, 2.])*tdur[i]/texp
        peak_ind, peak_props = find_peaks(mes, width=detect_width, height=5, prominence=5)

        #print(peak_props['widths'])

        # return number and location of peaks
        num_peaks = len(peak_ind)
        peak_loc = bin_edge[np.argmax(mes)]

        max_dur = (13./24.) * (period/365.)**(1./3.) * rho_star**(-1./3.) * 2.0
        min_dur = (13./24.) * (period/365.)**(1./3.) * rho_star**(-1./3.) * 0.5

        if tdur[i] < max_dur and tdur[i] > min_dur:
            axes[i].plot(bin_edge-P/2., mes, ls=mes_ls, label='Tdur={0:.2f} days'.format(tdur[i]), color='dodgerblue',  )

        else:
            axes[i].plot(bin_edge-P/2., mes,  ls=mes_ls, label='Tdur={0:.2f} days'.format(tdur[i]), color='lightslategrey' )
        
        if num_peaks>0:
            axes[i].plot(bin_edge[peak_ind]-P/2., mes[peak_ind], 'ro')
        
        axes[i].set_ylabel('MES[$\sigma_{mad}$]', fontsize=12)
        axes[i].axhline(7.1, ls='--', color='k')    
        axes[i].axhline(0., ls='-', color='k')    
        axes[i].legend(framealpha=0.)


    flux_to_use=i_max
    flux = data[flux_cols].values.T.astype(float)[flux_to_use]
    flux_mask = mask[flux_to_use]
    
    mean_flux = histogram1d(fold_time[flux_mask], weights=flux[flux_mask], bins=750, range=(0,period))/histogram1d(fold_time[flux_mask], bins=750, range=(0,period))
    flux_bins = np.linspace( 0, period, 750  )

        
    axes[0].plot(fold_time-P/2., 1e3*(flux-1.), '.', markersize=0.3, color='dodgerblue')
    axes[0].plot(flux_bins-P/2., 1e3*(mean_flux-1.), '-', color='salmon', label='P = {0:.6f} days'.format(period) )

    axes[0].set_ylim(np.nanmin( 1e3*(flux-1.)), np.nanmax( 1e3*(flux-1.)) )
    axes[0].legend(framealpha=0.)

    axes[0].set_ylabel('Flux [ppt]', fontsize=12)
    axes[-1].set_xlabel('Phase [Days]')
    
    for ax in axes:
        ax.set_xlim(-0.03*P-P/2.,P/2.+0.03*P)
        
    for ax in axes[1:]:
        ax.set_ylim(min_mes, max_mes + 0.1*max_mes)
        
    plt.tight_layout()
    f.subplots_adjust(hspace=0)

    print(max_mes)
            





def estimate_transit_depth(fold_time, flux, t0, width,):

    in_transit = np.abs(fold_time-t0) <= 0.25*width
    best_depth = (1.-np.median(flux[in_transit]))*1.0e6
    
    return best_depth





def find_best_params_for_TCE(time, num, den, tdurs, P, texp, t0, harmonics = [1., 1.5, 2., 3.], ntransits=3):
    
    best_period = P
    best_tdur = 0.
    max_mes = 0.
    best_t0 = 0.

    min_time = min(time)
    
    t0_time = time-min_time
    
    for n in harmonics:
        
        mes_time = np.arange(0, n*P+texp, texp)
        fold_time = np.mod(t0_time, n*P)
        
        foldtime_bins = np.digitize(fold_time, bins=mes_time )
        number_transits = np.bincount(foldtime_bins)
        mes = calc_mes_loop(foldtime_bins, num, den)

        mes[:,number_transits<ntransits]=0.
        
        if np.nanmax(mes)>max_mes:
            
            all_maxmes = np.nanmax(mes, axis=1)
            dur_i = np.nanargmax(all_maxmes)
            mes_i = np.nanargmax(mes[dur_i])
            
            best_tdur = tdurs[dur_i]
            max_mes = all_maxmes[dur_i]

            #mes_dur = mes[dur_i]
            best_t0 = mes_time[mes_i] + texp/2. + min_time
            
            best_period = n*P

    if np.abs(best_t0-t0)<1.:
        return best_period, max_mes, best_t0, best_tdur
    else:
        return  best_period, 0., best_t0, best_tdur




def mask_highest_TCE_and_check_others(TCEs, time, num, den, tdurs, threshold=7.,mes_norm=False,mes_frac=0.85,):

    texp = np.nanmin(np.abs(time[1:]-time[:-1]) )
    
    tces_sorted = TCEs[TCEs[:,2].argsort()][::-1]
    top_tce = tces_sorted[0]
    
    # Assume highest S/N TCE is real, and mask it
    tr_mask = make_transit_mask(time, P=top_tce[1],t0=top_tce[3],dur=top_tce[4] )
    true_TCE_list = np.zeros(len(tces_sorted), dtype=bool)
    true_TCE_list[0]=True
    
    # Check if TCEs remain after masking the higher S/N TCEs. 
    for i in range(1, len(tces_sorted)):

        _, period, mes_orig, t0, tdur = tces_sorted[i]

        if num.ndim==2:
            tdur_index = np.argmin(np.abs(tdur - tdurs) )
            num_masked = num[tdur_index][tr_mask]
            den_masked = den[tdur_index][tr_mask]
        else:
            num_masked = num[tr_mask]
            den_masked = den[tr_mask]

        fold_time = ((time[tr_mask] - t0 + period/2.) % period )
        mes_time, mes = calc_mes(fold_time,num_masked,den_masked,period,n_trans=1,texp=texp)

        
        max_mes = np.nanmax(mes)
        num_points, _ = np.histogram(fold_time, bins=np.arange(0,period+texp,texp))
        ntransits = num_points[np.argmax(mes)]
        new_t0 = mes_time[np.nanargmax(mes)]

        t0_diff = np.abs(period/2.-new_t0)

        if (max_mes>threshold and max_mes>mes_frac*mes_orig and ntransits>1 and t0_diff<tdur/2.):
            tr_mask &= make_transit_mask(time=time, P=period, dur=tdur, t0=t0)
            print('tce at {:.5f}: Original MES = {:.2f} Modified MES = {:.2f}, t0 diff={:.2f}, MAYBE REAL?'.format(period, mes_orig, max_mes, t0_diff))
            true_TCE_list[i]=True

            #plt.plot(mes_time, mes)
            #plt.show()
            
        else:
            #print('tce at {:.5f}: Modified MES = {:.2f}, t0 diff={:.2f}, FAKE!'.format(period, max_mes, t0_diff))
            true_TCE_list[i]=False
            
    return tces_sorted[true_TCE_list]




def check_for_minimum_transit_coverage(time, P, t0, tdur, cadence, min_transits=2, ntdur=1.5, min_frac=0.8,):

    tn=t0
    n_good_transits=0

    if P<0:
        return False

    while tn<max(time):

        #print(str(tn), end='\r')
        t_min, t_max = tn-ntdur * tdur, tn + ntdur*tdur
        event_cut = np.logical_and(time>t_min, time<t_max)

        tn+=P
        
        if sum(event_cut)<min_frac * (2*ntdur * tdur/cadence):
            n_good_transits+=1
        if n_good_transits>min_transits:
            return True
            

    return n_good_transits>=min_transits
        

        
def remove_tces_with_bad_transit_coverage(tces, time, cadence=None, min_transits=2, ntdur=2, min_frac=0.75):

    if cadence is None:
        cadence = np.nanmin(time[1:]-time[:-1])

    good_tces = np.ones(len(tces), dtype=bool)
    
    for i,tce in enumerate(tces):
        star, period, mes, t0, tdur = tce

        good_tces[i] = check_for_minimum_transit_coverage(time, period, t0, tdur, cadence, min_transits, ntdur, min_frac)

    return tces[good_tces]
        
        
    
        


