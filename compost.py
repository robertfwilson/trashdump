
'''
COMbination of _ recyclebin Objctsn idk comopst egtfbfwugfbwieurl

or

Bin Collector
'''

from .utils import *
from .dump import *
from .lightcurves import * 
import trashdump.recyclebin as rb


class Compost(object):

    def __init__(self, time, flux, flux_err, ID, TCEs, det_kw_list, flags=None, mask=None, trend=None, mission='KEPLER', sector_dates=None, qflags=None):
        
        #number of detrtending methods that will be used
        num_methods = len(det_kw_list)

        #initializing
        self.bin_col = []
        #lcs = []

        #iterating through the number of methods 
        for i in range(num_methods):
            #making the lightcurve object
            keplc = LightCurve(time=time, flux=flux, flux_err=flux_err, flags=flags, qflags=qflags,mission=mission, ID=ID, sector_dates=sector_dates)
            
            #flatten lc depending on the det_kw_list
            keplc.flatten(det_kw_list[i])

            #masking unwanted things
            keplc.mask_badguys(sig=5.)
            keplc.mask_bad_gap_edges(sig=4.)


            #making the dump object
            kepler_dump = Dump(keplc, min_transits=3, gap_size=2.)

            #making the recyclebin object
            rbin = rb.RecycleBin(Dump=kepler_dump, TCEs = TCEs.to_numpy())

            self.bin_col.append(rbin)

    def get_all_vetting_metrics(self,tce_num):
        result_list = [r.get_all_vetting_metrics(tce_num) for r in self.bin_col]

        results = {f'{k}_{j}': v for j,d in enumerate(result_list) for k, v in d.items()}

        return results







