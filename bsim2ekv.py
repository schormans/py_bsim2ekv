"""
Main file for py_bsim2ekv. Performs BSIM to EKV conversion, by implementing
the method described by Kayal and Stefanovic.
"""

import importlib
import mpmath as mp
import numpy as np
import ekv
from datetime import datetime
from datetime import timedelta
import extrafuncs as ef
from scipy.optimize import least_squares
import copy
import matplotlib.pyplot as plt

importlib.reload(ef)

#GLOBAL GMIN DEF:
global_gmin = 1e-12

#GLOBAL LMIN (MINIMUM TRANSISTOR LENGTH) DEF:
global_lmin = 180e-9

#TARGET DATA LENGTH AFTER DECIMATION
global_dec_targ = 50

#ENABLE DECIMATION
global_dec_flag = True

#GLOBAL LENGTH CUTOFF
#EKV doesn't handle some short channel effects well
#define a length here, and the following resid funcs
#will check if L is lower.
# find_lambda_ucrit, find_ucrit, find_lambda
#If so, give a zero residual.
#Default to zero, mostly only useful in edge cases

global_len_cutoff = 0


#LIST OF STAGES TO NEVER DECIMATE
#(some datasets need precision)
global_dec_blacklist = [
    'vto_phi_gamma',
    'lambda_ucrit',
    'kp_e0',
    'lk_q02'
]

#DEFINE DEFAULTS FOR DATA INGEST
#If no params are passed to data_ingest, the default params get passed instead

default_params = {

    "wlmax" : mp.mpf('10e-6'),
    "wslengths" : [1.8e-07, 2.2e-07, 2.5e-07, 2.9e-07, 3.25e-07, 3.6e-07, 4.3e-07, 5.0e-07, 5.75e-07, 6.5e-07, 7.2e-07],
    "nlwidths" : [0.22e-6, 0.44e-6],
    "temps" : [-40.0, -20.0, 27, 80, 120],
    "vbsweep" : np.linspace(-0.9,0,10),
    # all voltages below are nominal bulk ref'd as per EKV convention!
    # if bulk voltage is changed, this is accounted for by including -vb term where necessary
    "vdlinearnom" : 50e-3, #nominal linear region drain voltage
    "vdsatnom" : 1.5, #nominal saturation region drain voltage
    "vgsatnom" : 1.1, #nominal saturation region gate voltage
    "vdd" : 1.8,
    "vss" : 0,

}

#empty filenamedict below: demonstrates required structure, but is not used:

default_filenamedict = {
    "wl_is_temp_data" : "",
    "wl_vp_vg_temp_data" : "",
    "wl_id_vg_vb_linear_data" : "",
    "wl_id_vg_linear_temp_data" : "",
    "wl_ib_vg_sat_data" : "",
    "wl_noise_data" : "",
    "ws_is_data" : "",
    "ws_vp_vg_data" : "",
    "ws_id_vg_linear_data" : "",
    "ws_id_vg_sat_temp_data" : "",
    "ws_id_vd_sat_data" : "",
    "nl_is_data" : "",
    "nl_vp_vg_data" : "",
    "sat_region_len_isdata" : ""
}


class DataStore:
    def __init__(self,filenamedict,**kwargs):
        #initialize the store as a copy of the filenamedict
        #data_ingest overwrites the attr value with the np.ndarray
        #default_params sets the default parameters, but these
        #can be overwritten by passing a dict of kwargs
        for k,v in filenamedict.items():
            setattr(self, k, v)
        for k,v in default_params.items():
            setattr(self, k, v)
        for k,v in kwargs.items():
            setattr(self, k, v)


def data_ingest(filenamedict, #dict with k:v as {"dataset_name" : "file path"}
                **params #e.g. wslengths=[0.18e-6,0.22e-6,0.5e-6.....]
                ):
    """
    Potential params:
    wlmax
    wslengths
    nlwidths
    temps
    vbsweep
    If supplied, these will be used, otherwise defaults are used.
    """
    #create DataStore with filenamedict
    #this initializes datalump with the attributes set to filepath strings
    #other params will be set to defaults
    datalump = DataStore(filenamedict)

    #if fresh params were passed in **params, set them.
    for k,v in params.items():
        setattr(datalump,k,v)


    ingest_func_selector = {
        "wl_is_temp_data" : ["temp", "is/2"], #0 params [temp,is/2]
        "wl_vp_vg_temp_data" : ["VG","VP","temps"], #1 param [VG,VP,temps]
        "wl_id_vg_vb_linear_data" : ["VG","ID","vbsweep"], #1 param [VG,ID,vbsweep]
        "wl_id_vg_linear_temp_data" : ["VG","ID","temps"], #1 param [VG,ID,temps]
        "wl_ib_vg_sat_data" : ["VG","IB"], #0 params [VG,IB]
        "wl_noise_data" : ["freq","PSD"], #0 params [freq,psd]
        "ws_is_data" : ["len","is/2"], #0 params [len,is/2]
        "ws_vp_vg_data" : ["VG","VP","wslengths"], #1 param [vg,vp,wslengths]
        "ws_id_vg_linear_data" : ["VG","ID","wslengths"], #1 param [vg,id,wslengths]
        "ws_id_vg_sat_temp_data" : ["VG","ID","wslengths","temps"], #2 params [vg,id,wslengths,temps]
        "ws_id_vd_sat_data" : ["VD","ID","wslengths"], #1 param [vd,id,wslengths]
        "nl_is_data" : ["wid", "is/2"], #0 params [wid,is/2]
        "nl_vp_vg_data" : ["VG","VP","nlwidths"], #1 param [vg,vp,nlwidths]
        "sat_region_len_isdata" : ["LEN","ID"] #0 params
    }

    def ingest_func(dataset,x,y,*params) -> np.ndarray:
        #construct paramlist from *params
        paramlol = [] #param list of lists
        namelist = [] #eventual namelist
        for k in params:
            origklist = getattr(datalump,k)
            #now convert to list of names
            nlist = []
            for elem in origklist:
                #nlist.append(str(str(k)+'='+str(elem)).replace(".","_").replace("-","neg"))
                nlist.append(str(str(k)+'='+str(elem)))
            paramlol.append(nlist)

        fullparamlist = ef.recursing_combinations(ef.paramsumstr,paramlol)
        for elem in fullparamlist:
            for a in x,y:
                namelist.append(a+elem)

        #now we should have a complete namelist
        #print(namelist)
        return np.genfromtxt(
            filenamedict[dataset],
            delimiter=",",
            skip_header=6,
            names=namelist
        )
        
    for k,v in ingest_func_selector.items():
        setattr(datalump,k,ingest_func(k,*v))
        
    return datalump


def full_run(datalump: DataStore, mos: ekv.MOS, proclist=[]) -> None:
    #full list of function handles
    #run each function on datalump and mos in sequence
    #at the end of the run, mos should have a fitted set of attributes
    #if no proclist is provided, use this default
    if not(proclist):
        proclist = [
            'vto_phi_gamma',
            'leta',
            'weta',
            'lk_q0',
            'kp_e0',
            'dl',
            'lambda_ucrit',
            'iba_ibb_ibn', #ib params twice (often won't converge after 1x)
            'iba_ibb_ibn',
            'tcv',
            'bex',
            'ucex',
            'kf_af'
        ]
    grandstart = datetime.now()
    for dex in range(len(proclist)):
        print(f'Running stage {dex+1} of {len(proclist)}: {proclist[dex]}')
        secstart = datetime.now()
        stage = proclist[dex]
        # if stage == 'lk_q0':
        #     lkdef = mos.LK #store default LK value
        #     print(f'Stored default mos LK value of {lkdef}')
        #     result = lsq_wrapper(stage,mos,datalump)
        #     mos.LK = lkdef
        #     print(f'Restored default LK value, lsq always settles poorly for LK')
        # else:    
        #     result = lsq_wrapper(stage,mos,datalump)
        result = lsq_wrapper(stage,mos,datalump)
        print(f'Stage complete!')
        print(f'Result vector for this stage: {result.x}')
        print(f'Time taken for this stage: {datetime.now() - secstart}')
    print('All stages complete!')
    print(f'Total time taken: {datetime.now() - grandstart}')
    print('Check MOS W,L, and T of MOS object before running more calculations.')
    return None
        

def VTH(mos,vs,vd,vg):
    terms = [
        mos.VTOt(),
        mos.deltaV_RSCE(),
        mos.GAMMAEFF(vs,vd,vg)*mp.sqrt(mos.V_S_OR_D_EFF(vs)),
        -mos.GAMMA*mp.sqrt(mos.PHI)
    ]
    return sum(terms)

def VTH_ALT(mos: ekv.MOS, vgl: list ,vs,vd,idsl=[]):
    # alternate method for calculating VTH, 2nd deriv max
    # find VG @ max(d2ID/dVG2)
    # idsl is an optional param, if provided, we're working
    # with an empirical dataset, otherwise we're calculating
    # from the mos object passed through
    if not(any(idsl)):
        idsl = [mos.I_DS(vs,vd,vg) for vg in vgl]
    didl = ef.simplediff(vgl, idsl)
    #d2idl = ef.simplediff(vgl, didl)
    pointdex = list([abs(i) for i in didl]).index(max([abs(i) for i in didl]))
    # y = mx + c
    x = vgl[pointdex]
    m = max([abs(i) for i in didl])
    y = abs(idsl[pointdex])
    c = y - m*x
    return -c/m


def resid_wrapper(x: np.ndarray, stage: str, mos: ekv.MOS, datalump: DataStore) -> None:
    resid_func = resid_func_lookup[stage][0]
    dat = getattr(datalump,resid_func_lookup[stage][1])
    if global_dec_flag and stage not in global_dec_blacklist:
        # decimator is enabled by default, otherwise just proceed
        dat = ef.simple_dec(dat,global_dec_targ)
    namelist = dat.dtype.names
    residuallist = []
    for a in np.arange(len(namelist)):
        if not(a%2): #dataseries is in pairs, so jump two at a time
            residuallist = np.append(residuallist,
                                     resid_func(
                                         datalump,
                                         mos,
                                         x,
                                         dat,
                                         namelist,
                                         a
                                     ))
    return residuallist

#All 'resid_func' possibilities defined below
    
def find_vto_phi_gamma(datalump: DataStore,
                       mos: ekv.MOS,
                       x: np.ndarray, #VTO, PHI, GAMMA
                       dat: np.ndarray,
                       namelist: np.ndarray,
                       a: int
                       ):
    #first set width and length to max setting in datastore
    mos.W = datalump.wlmax
    mos.L = datalump.wlmax
    #special case func. we are working with vp = f(vg) across temp,
    #but only want the temp=27°C case for this test. if temps[a/2]=/=27, disregard
    if not(datalump.temps.index(27)):
        print('NO NOMINAL TEMP DATA FOUND IN wl_vp_vg_temp_data!')
        pass
    elif datalump.temps[int(a/2)] == 27:
        #run least squares and return residuals
        mos.VTO = x[0]
        mos.PHI = x[1]
        mos.GAMMA = x[2]
        vgl = dat[namelist[a]] #gate voltage list
        vpl = dat[namelist[a+1]] #pinchoff voltage list
        return [np.float64(mos.VP(vp,vg,vg) - vp) for vg,vp in zip(vgl,vpl)]
    else:
        return 0


def find_leta(datalump: DataStore,
              mos: ekv.MOS,
              x: np.ndarray, #LETA
              dat: np.ndarray,
              namelist: np.ndarray,
              a: int
              ):
    mos.W = datalump.wlmax
    mos.L = datalump.wslengths[int(a/2)]
    mos.LETA = x[0]
    vgl = dat[namelist[a]] #gate voltage list
    vpl = dat[namelist[a+1]] #pinchoff voltage list
    return [np.float64(mos.VP(vp,vg,vg) - vp) for vg,vp in zip(vgl,vpl)]

def find_weta(datalump: DataStore,
              mos: ekv.MOS,
              x: np.ndarray, #WETA
              dat: np.ndarray,
              namelist: np.ndarray,
              a: int
              ):
    mos.L = datalump.wlmax
    mos.W = datalump.nlwidths[int(a/2)]
    mos.WETA = x[0]
    vgl = dat[namelist[a]] #gate voltage list
    vpl = dat[namelist[a+1]] #pinchoff voltage list
    return [np.float64(mos.VP(vp,vg,vg) - vp) for vg,vp in zip(vgl,vpl)]

def vth_resid(datalump: DataStore,
              mos: ekv.MOS,
              x: np.ndarray, #LK, Q0
              dat: np.ndarray,
              namelist: np.ndarray,
              a: int
              ):
    #LK and Q0 should be converted to float first in this instance
    x = [float(a) for a in x]
    #changed from original spec; find LK later in DL section
    mos.LK = x[0]
    mos.Q0 = x[1]
    mos.W = datalump.wlmax
    mos.L = datalump.wslengths[int(a/2)]
    vgl = dat[namelist[a]] #gate voltage list
    vpl = dat[namelist[a+1]] #pinchoff voltage list

    vs = 0

    # find vp_intercept for this run
    vth_from_vp_intercept = ef.interp_find_x_intercept(vgl,vpl)
    # print(vp_intercept)
    
    # vp_vth = mp.findroot(lambda vg: mos.VP(vs,vg,vg),
    #                 [-datalump.vdd,datalump.vdd],
    #                 solver='bisect')

    #vthavg = (vth_from_vp_intercept + vp_vth)/2
    
    calc_vth = VTH(mos,vs,vth_from_vp_intercept,vth_from_vp_intercept)
    return np.float64(calc_vth - vth_from_vp_intercept)
    

def vth_resid2(datalump: DataStore,
              mos: ekv.MOS,
              x: np.ndarray, #LK, Q0
              dat: np.ndarray,
              namelist: np.ndarray,
              a: int
              ):
    #LK and Q0 should be converted to float first in this instance
    #x = [float(a) for a in x]
    #changed from original spec; find LK later in DL section
    mos.LK = x[0]
    mos.Q0 = x[1]
    mos.W = datalump.wlmax
    mos.L = datalump.wslengths[int(a/2)]
    vgl = dat[namelist[a]] #gate voltage list
    idsl = dat[namelist[a+1]] #drain current list
    vs = 0
    # find vth_intercept for this run
    vth_bsim = VTH_ALT(mos,vgl,vs,datalump.vdlinearnom,idsl=idsl)
    # here we use alternate vth calc, using ws_id_vg_linear_data
    ndat = datalump.ws_id_vg_linear_data
    nl = ndat.dtype.names
    nvgl = ndat[nl[a]]
    nidl = ndat[nl[a+1]]   
    calc_vth = VTH_ALT(mos,vgl,vs,datalump.vdlinearnom)
    return np.float64(calc_vth - vth_bsim)

def find_kp_e0(datalump: DataStore,
               mos: ekv.MOS,
               x: np.ndarray, #KP, E0
               dat: np.ndarray,
               namelist: np.ndarray,
               a: int
               ):
    mos.W = datalump.wlmax
    mos.L = datalump.wlmax
    vb = datalump.vbsweep[int(a/2)] # current vb value
    vgl = dat[namelist[a]] # gate voltage list
    idsl = dat[namelist[a+1]] # drain current list
    mos.KP = x[0]
    mos.E0 = x[1]
    vs = 0
    vd = datalump.vdlinearnom
    return [
        np.float64(mos.I_DS(vs-vb,vd-vb,vg-vb)*1e6 - ids*1e6)
        for vg,ids in zip(vgl,idsl)
    ]

def find_kp(datalump: DataStore,
               mos: ekv.MOS,
               x: np.ndarray, #KP
               dat: np.ndarray,
               namelist: np.ndarray,
               a: int
               ):
    mos.W = datalump.wlmax
    mos.L = datalump.wlmax
    vb = datalump.vbsweep[int(a/2)] # current vb value
    vgl = dat[namelist[a]] # gate voltage list
    idsl = dat[namelist[a+1]] # drain current list
    mos.KP = x[0]
    vs = 0
    vd = datalump.vdlinearnom
    return [
        np.float64(mos.I_DS(vs-vb,vd-vb,vg-vb)*1e6 - ids*1e6)
        for vg,ids in zip(vgl,idsl)
    ]


def find_e0(datalump: DataStore,
               mos: ekv.MOS,
               x: np.ndarray, #E0
               dat: np.ndarray,
               namelist: np.ndarray,
               a: int
               ):
    mos.W = datalump.wlmax
    mos.L = datalump.wlmax
    vb = datalump.vbsweep[int(a/2)] # current vb value
    vgl = dat[namelist[a]] # gate voltage list
    idsl = dat[namelist[a+1]] # drain current list
    mos.E0 = x[0]
    vs = 0
    vd = datalump.vdlinearnom
    return [
        np.float64(mos.I_DS(vs-vb,vd-vb,vg-vb)*1e6 - ids*1e6)
        for vg,ids in zip(vgl,idsl)
    ]

def find_kp_e0_lk(datalump: DataStore,
               mos: ekv.MOS,
                  x: np.ndarray, #KP, E0, LK
               dat: np.ndarray,
               namelist: np.ndarray,
               a: int
               ):
    mos.W = datalump.wlmax
    mos.L = datalump.wlmax
    vb = datalump.vbsweep[int(a/2)] # current vb value
    vgl = dat[namelist[a]] # gate voltage list
    idsl = dat[namelist[a+1]] # drain current list
    mos.KP = x[0]
    mos.E0 = x[1]
    mos.LK = x[2]
    vs = 0
    vd = datalump.vdlinearnom
    return [
        np.float64(mos.I_DS(vs-vb,vd-vb,vg-vb)*1e6 - ids*1e6)
        for vg,ids in zip(vgl,idsl)
    ]
    

def find_dl(datalump: DataStore,
              mos: ekv.MOS,
              x: np.ndarray, #DL
              dat: np.ndarray,
              namelist: np.ndarray,
              a: int
              ):
    mos.DL = x[0]
    vs = 0
    vd = datalump.vdlinearnom
    mos.W = datalump.wlmax
    mos.L = datalump.wslengths[int(a/2)]
    vgl = dat[namelist[a]]
    idsl = dat[namelist[a+1]]
    return [np.float64(mos.I_DS(vs,vd,vg)*1e6 - ids*1e6) for vg, ids in zip(vgl,idsl)]

def find_dl_lk(datalump: DataStore,
              mos: ekv.MOS,
              x: np.ndarray, #DL
              dat: np.ndarray,
              namelist: np.ndarray,
              a: int
              ):
    mos.DL = x[0]
    mos.LK = x[1]
    vs = 0
    vd = datalump.vdlinearnom
    mos.W = datalump.wlmax
    mos.L = datalump.wslengths[int(a/2)]
    vgl = dat[namelist[a]]
    idsl = dat[namelist[a+1]]
    return [np.float64(mos.I_DS(vs,vd,vg)*1e6 - ids*1e6) for vg, ids in zip(vgl,idsl)]

def find_lambda_ucrit(datalump: DataStore,
              mos: ekv.MOS,
              x: np.ndarray, #LAMBDA, UCRIT
              dat: np.ndarray,
              namelist: np.ndarray,
              a: int
              ):
    weight = weightfunc(datalump.wslengths[int(a/2)])
    mos.LAMBDA = x[0]
    mos.UCRIT = x[1]
    vdl = dat[namelist[a]]
    idsl = dat[namelist[a+1]]
    vs = 0
    vg = datalump.vgsatnom
    mos.W = datalump.wlmax
    mos.L = datalump.wslengths[int(a/2)]
    if mos.L < global_len_cutoff:
        return 0
    return [
        abs(np.float64(mos.I_DS(vs,vd,vg) -ids)*1e6*weight)
        for vd,ids in zip(vdl,idsl)
    ]

def find_lambda_ucrit_kp(datalump: DataStore,
              mos: ekv.MOS,
              x: np.ndarray, #LAMBDA, UCRIT
              dat: np.ndarray,
              namelist: np.ndarray,
              a: int
              ):
    weight = weightfunc(datalump.wslengths[int(a/2)])
    mos.LAMBDA = x[0]
    mos.UCRIT = x[1]
    mos.KP = x[2]
    vdl = dat[namelist[a]]
    idsl = dat[namelist[a+1]]
    vs = 0
    vg = datalump.vgsatnom
    mos.W = datalump.wlmax
    mos.L = datalump.wslengths[int(a/2)]
    if mos.L < global_len_cutoff:
        return 0
    return [
        abs(np.float64(mos.I_DS(vs,vd,vg) -ids)*1e6*weight)
        for vd,ids in zip(vdl,idsl)
    ]

def find_lambda_ucrit_deriv(datalump: DataStore,
              mos: ekv.MOS,
              x: np.ndarray, #LAMBDA, UCRIT
              dat: np.ndarray,
              namelist: np.ndarray,
              a: int
              ):
    # find based on dId/dVd
    weight = weightfunc(datalump.wslengths[int(a/2)])
    mos.LAMBDA = x[0]
    mos.UCRIT = x[1]
    vdl = dat[namelist[a]]
    idsl = ef.simplediff(vdl,dat[namelist[a+1]])
    vs = 0
    vg = datalump.vgsatnom
    mos.W = datalump.wlmax
    mos.L = datalump.wslengths[int(a/2)]
    if mos.L < global_len_cutoff:
        return 0
    return [
        abs(np.float64(mos.num_diff_dirty(lambda x: mos.I_DS(vs,x,vg), vd, mos.nudge) -ids)*1e6*weight)
        for vd,ids in zip(vdl,idsl)
    ]
    

def find_lambda_ucrit_resid_deriv(datalump: DataStore,
              mos: ekv.MOS,
              x: np.ndarray, #LAMBDA, UCRIT
              dat: np.ndarray,
              namelist: np.ndarray,
              a: int
              ):
    # find based on d(resid)/dVd
    weight = weightfunc(datalump.wslengths[int(a/2)])
    mos.LAMBDA = x[0]
    mos.UCRIT = x[1]
    vdl = dat[namelist[a]]
    #idsl = ef.simplediff(vdl,dat[namelist[a+1]])
    idsl = dat[namelist[a+1]]
    vs = 0
    vg = datalump.vgsatnom
    mos.W = datalump.wlmax
    mos.L = datalump.wslengths[int(a/2)]
    if mos.L < global_len_cutoff:
        return 0
    return ef.simplediff(vdl,[
        abs(np.float64( (mos.I_DS(vs,vd,vg) -ids))*1e6*weight)
        for vd,ids in zip(vdl,idsl)
    ])

def find_lambda_ucrit_kp_resid_deriv(datalump: DataStore,
              mos: ekv.MOS,
              x: np.ndarray, #LAMBDA, UCRIT
              dat: np.ndarray,
              namelist: np.ndarray,
              a: int
              ):
    # find based on d(resid)/dVd
    weight = weightfunc(datalump.wslengths[int(a/2)])
    mos.LAMBDA = x[0]
    mos.UCRIT = x[1]
    mos.KP = x[2]
    vdl = dat[namelist[a]]
    #idsl = ef.simplediff(vdl,dat[namelist[a+1]])
    idsl = dat[namelist[a+1]]
    vs = 0
    vg = datalump.vgsatnom
    mos.W = datalump.wlmax
    mos.L = datalump.wslengths[int(a/2)]
    if mos.L < global_len_cutoff:
        return 0
    return ef.simplediff(vdl,[
        abs(np.float64( (mos.I_DS(vs,vd,vg) -ids))*1e6*weight)
        for vd,ids in zip(vdl,idsl)
    ])


def find_ucrit(datalump: DataStore,
              mos: ekv.MOS,
              x: np.ndarray, #UCRIT
              dat: np.ndarray,
              namelist: np.ndarray,
              a: int
              ):
    if not(datalump.temps.index(27)):
        print('NO NOMINAL TEMP DATA FOUND IN wl_vp_vg_temp_data!')
        return None
    elif datalump.temps[int((a%(len(datalump.wslengths)-1))/2)] == 27:
        #run least squares and return residuals
        weight = weightfunc(datalump.wslengths[int(a*(len(datalump.wslengths)/len(namelist)))])
        mos.W = datalump.wlmax
        mos.L = datalump.wslengths[int(a*(len(datalump.wslengths)/len(namelist)))]
        mos.UCRIT = x[0]
        vgl = dat[namelist[a]]
        idsl = dat[namelist[a+1]]
        vs = 0
        vd = datalump.vdsatnom
        if mos.L < global_len_cutoff:
            return 0
        return [
            abs(np.float64(mos.I_DS(vs,vd,vg) - ids)*1e6*weight)
            for vg,ids in zip(vgl,idsl)
        ]
    else:
        return 0

def find_lambda(datalump: DataStore,
              mos: ekv.MOS,
              x: np.ndarray, #LAMBDA
              dat: np.ndarray,
              namelist: np.ndarray,
              a: int
              ):
    # weighting: fit of minimum length transistor is 10x more important
    # than longest ws device.
    weight = weightfunc(datalump.wslengths[int(a/2)])
    mos.W = datalump.wlmax
    mos.L = datalump.wslengths[int(a/2)]
    mos.LAMBDA = x[0]
    vdl = dat[namelist[a]]
    idsl = dat[namelist[a+1]]
    vg = datalump.vgsatnom
    vs = 0
    if mos.L < global_len_cutoff:
        return 0
    return [
        abs(np.float64(mos.I_DS(vs,vd,vg)*1e6 - ids*1e6)*weight) for vd,ids in zip(vdl,idsl)
    ]
    

def find_ib_params(
        datalump: DataStore,
        mos: ekv.MOS,
        x: np.ndarray, #IBA, IBB, IBN
        dat: np.ndarray,
        namelist: np.ndarray,
        a: int
):
    mos.W = datalump.wlmax
    mos.L = datalump.wlmax
    mos.IBA = mp.mpf(x[0])
    mos.IBB = mp.mpf(x[1])
    mos.IBN = mp.mpf(x[2])
    vs = 0
    vd = datalump.vdsatnom
    vgl = dat[namelist[a]]
    idbl = dat[namelist[a+1]]
    outlist = [
        np.float64((mp.mpf(mos.I_DB(vs,vd,vg))*mp.mpf(1e12) - mp.mpf(datalump.vdsatnom*global_gmin)*mp.mpf(1e12) - mp.mpf(idb)*mp.mpf(1e12)))
        for vg,idb in zip(vgl,idbl)
    ]
    return outlist
    

def find_tcv(datalump: DataStore,
              mos: ekv.MOS,
              x: np.ndarray, #TCV
              dat: np.ndarray,
              namelist: np.ndarray,
              a: int
              ):
    mos.W = datalump.wlmax
    mos.L = datalump.wlmax
    mos.TCV = x[0]
    mos.T = mp.mpf(datalump.temps[int(a/2)]) + 273.15
    vgl = dat[namelist[a]]
    vpl = dat[namelist[a+1]]
    outval = [
        np.float64(mos.VP(vp,vg,vg) - vp) for vg,vp in zip(vgl,vpl)
    ]
    mos.T = mos.Tnom
    return outval

def find_bex(datalump: DataStore,
              mos: ekv.MOS,
              x: np.ndarray, #BEX
              dat: np.ndarray,
              namelist: np.ndarray,
              a: int
              ):
    mos.W = datalump.wlmax
    mos.L = datalump.wlmax
    mos.BEX = x[0]
    mos.T = datalump.temps[int(a/2)] + 273.15
    vgl = dat[namelist[a]]
    idsl = dat[namelist[a+1]]
    vs = 0
    vd = datalump.vdlinearnom
    outval = [
        np.float64(mos.I_DS(vs,vd,vg)*1e6 - ids*1e6) for vg,ids in zip(vgl,idsl)
    ]
    mos.T = mos.Tnom
    return outval

def find_ucex(datalump: DataStore,
              mos: ekv.MOS,
              x: np.ndarray, #UCEX
              dat: np.ndarray,
              namelist: np.ndarray,
              a: int
              ):
    mos.W = datalump.wlmax
    mos.T = mp.mpf(datalump.temps[int((a%len(datalump.temps))/2)]) + 273.15
    mos.L = mp.mpf(datalump.wslengths[int((len(datalump.wslengths)/len(namelist))*a)])
    mos.UCEX = x[0]
    vgl = dat[namelist[a]]
    idsl = dat[namelist[a+1]]
    vs = 0
    vd = datalump.vdsatnom
    outval = [
        np.float64(mos.I_DS(vs,vd,vg)*1e6 - ids*1e6) for vg,ids in zip(vgl,idsl)
    ]
    mos.T = mos.Tnom
    return outval

def find_kf_af(datalump: DataStore,
              mos: ekv.MOS,
               x: np.ndarray, #KF, AF, thermfudge
              dat: np.ndarray,
              namelist: np.ndarray,
              a: int
              ):
    mos.W = datalump.wlmax
    mos.L = datalump.wlmax
    mos.KF = mp.mpf(x[0])
    mos.AF = mp.mpf(x[1])
    mos.thermfudge = mp.mpf(x[2])
    freql = dat[namelist[a]] #list of freqs
    psdl = dat[namelist[a+1]] #list of PSDs
    vs = 0
    vd = datalump.vdsatnom
    vg = datalump.vgsatnom
    meanpsd = np.mean(psdl)
    return [
        np.float64(
            (mos.Sthermal(vs,vd,vg) + mos.Sflicker(vs,vd,vg,freq) - psd)/meanpsd
        ) for freq,psd in zip(freql,psdl)
    ]
    


# DEFINE WEIGHTING FUNCTION TO MAKE FITTING MIN
# LENGTH DEVICE MOST IMPORTANT
# CAN BE TWEAKED TO AFFECT ALL LAMBDA/UCRIT STAGES

def weightfunc(
        length
):
    return 2/((length**3)*1e18)

##########################################
##########################################
# DEFINE PLOTTING FUNCTIONS HERE
##########################################
##########################################

def plotwrapper(
        stage: str,
        datalump: DataStore,
        mos: ekv.MOS
) -> None:
    # check if plotfunc exists for this stage, if not, quit
    if not(resid_func_lookup[stage][4]):
        print('no plots for this stage')
        return
    # if func is in exceptions list, just run the func and quit
    if resid_func_lookup[stage][4][0] in plot_exceptions:
        resid_func_lookup[stage][4][0](mos,datalump)
        plt.xlabel(resid_func_lookup[stage][4][1])
        plt.ylabel(resid_func_lookup[stage][4][2])
        plt.title(f'Plot for stage: {stage}')
        return
    # else get on with it
    # first job: print experimental BSIM data
    # special case for vto_phi_gamma plot
    if stage=='vto_phi_gamma':
        print('all vg_vp curves will be plotted here, but red line should match with 27°C curve')
    data = getattr(datalump,resid_func_lookup[stage][1])
    plt.figure()
    nl = data.dtype.names
    pltf = plt.plot
    if stage=="kf_af":
        pltf = plt.loglog
    for a in range(len(nl)):
        if not(a%2):
            pltf(data[nl[a]],data[nl[a+1]], '--k')
            pltf(data[nl[a]],[resid_func_lookup[stage][4][0](mos,x,y,datalump,a)
                                  for x,y in zip(data[nl[a]],data[nl[a+1]])], '-r')
    plt.xlabel(resid_func_lookup[stage][4][1])
    plt.ylabel(resid_func_lookup[stage][4][2])
    plt.title(f'Plot for stage: {stage}')

def plot_vg_vp(
        mos: ekv.MOS,
        x,
        y,
        datalump,
        a
):
    #usually don't need y, but required for VP plots (circ dep...)
    return mos.VP(y,x,x)

def plot_vths(
        mos: ekv.MOS,
        datalump
):
    #this one is different to most plotting funcs
    #need to reconstruct stuff from scratch
    #generate data for vth plot
    intercepts = []
    vths = []
    dat = datalump.ws_vp_vg_data
    nl = dat.dtype.names
    for i in range(len(datalump.wslengths)):
        mos.W = datalump.wlmax
        mos.L = datalump.wslengths[i]
        intercepts.append(ef.interp_find_x_intercept(
            dat[nl[2*i]], dat[nl[2*i+1]]
        ))
        vths.append(VTH(mos,0,intercepts[i],intercepts[i]))
    plt.figure()
    plt.plot(datalump.wslengths,intercepts,'--k')
    plt.plot(datalump.wslengths,vths,'-r')


def plot_vths_alt(
        mos: ekv.MOS,
        datalump
):
    #this one is different to most plotting funcs
    #need to reconstruct stuff from scratch
    #generate data for vth plot
    vths_bsim = []
    vths = []
    dat = datalump.ws_id_vg_linear_data
    nl = dat.dtype.names
    vs = 0
    vd = datalump.vdlinearnom
    for i in range(len(datalump.wslengths)):
        mos.W = datalump.wlmax
        mos.L = datalump.wslengths[i]
        vgl = dat[nl[2*i]]
        idsl = dat[nl[2*i+1]]
        vths_bsim.append(VTH_ALT(mos,vgl,vs,vd,idsl=idsl))
        vths.append(VTH_ALT(mos,vgl,vs,vd))
    plt.figure()
    plt.plot(datalump.wslengths,vths_bsim,'--k')
    plt.plot(datalump.wslengths,vths,'-r')
    

def plot_wl_id_vg_vb_linear(
        mos: ekv.MOS,
        x,
        y,
        datalump,
        a
):
    vb = datalump.vbsweep[int(a/2)]
    vs = 0
    vd = datalump.vdlinearnom
    return mos.I_DS(vs-vb,vd-vb,x-vb)

def plot_ws_id_vg_linear_data(
        mos: ekv.MOS,
        x,
        y,
        datalump,
        a
):
    mos.W = datalump.wlmax
    mos.L = datalump.wslengths[int(a/2)]
    vs = 0
    vd = datalump.vdlinearnom
    return mos.I_DS(vs,vd,x)

def plot_ws_id_vd_sat_data(
        mos: ekv.MOS,
        x,
        y,
        datalump,
        a
):
    mos.W = datalump.wlmax
    mos.L = datalump.wslengths[int(a/2)]
    vs = 0
    vg = datalump.vgsatnom
    return mos.I_DS(vs,x,vg)

def plot_ws_id_vd_sat_deriv_data(
        mos: ekv.MOS,
        datalump,
):
    #since we need to modify bsim data before plotting
    #need a custom func, this is on the plot exception list
    plt.figure()
    mos.W = datalump.wlmax
    vs = 0
    vg = datalump.vgsatnom
    dat = datalump.ws_id_vd_sat_data
    nl = dat.dtype.names
    for a in range(len(datalump.wslengths)):
        mos.L = datalump.wslengths[int(a)]
        plt.plot(dat[nl[2*a]],ef.simplediff(dat[nl[2*a]],dat[nl[2*a+1]]),'--k')
        plt.plot(
            dat[nl[2*a]],
            [
                mos.num_diff_dirty(lambda x: mos.I_DS(vs,x,vg), vd, mos.nudge)
                for vd in dat[nl[2*a]]
            ],
            '-r'
        )


def plot_ws_id_vg_sat_notemp(
        mos: ekv.MOS,
        datalump
):
    dat = datalump.ws_id_vg_sat_temp_data
    nl = dat.dtype.names
    plt.figure()
    for a in range(len(nl)):
        if not(a%2):
            if datalump.temps[int((a%(len(datalump.wslengths)-1))/2)] == 27:
            #run least squares and return residuals
                mos.W = datalump.wlmax
                mos.L = datalump.wslengths[int(a*(len(datalump.wslengths)/len(nl)))]
                vs = 0
                vd = datalump.vdsatnom
                plt.plot(
                    dat[nl[a]],dat[nl[a+1]],'--k'
                )
                plt.plot(
                    dat[nl[a]], [
                        mos.I_DS(vs,vd,vg) for vg in dat[nl[a]]
                    ], '-r'
                )

def plot_sat_id_vs_len(
        mos: ekv.MOS,
        datalump
):
    print("BSIM data vs EKV calculated saturation current vs length:")
    # abs of currents here so the plot works in logscale for pmos
    dat = datalump.sat_region_len_isdata
    plt.figure()
    plt.loglog(dat["LEN"],[abs(d) for d in dat["ID"]],'--k')
    vs = 0
    mos.W = datalump.wlmax
    ekv_idl = []
    for l in dat["LEN"]:
        mos.L = l
        ekv_idl.append(abs(mos.I_DS(vs,datalump.vdsatnom,datalump.vgsatnom)))
    plt.loglog(dat["LEN"],ekv_idl,'-r')


def plot_ib_params(
        mos: ekv.MOS,
        x,
        y,
        datalump,
        a
):
    vs = 0
    vd = datalump.vdsatnom
    # include gmin correction here by default
    outd = {
        'n' : mos.I_DB(vs,vd,x) + global_gmin*vd,
        'p' : mos.I_DB(vs,vd,x) - global_gmin*vd
    }
    
    return outd[mos.polarity]

def plot_wl_id_vg_linear_temp(
        mos: ekv.MOS,
        x,
        y,
        datalump,
        a
):
    mos.W = datalump.wlmax
    mos.L = datalump.wlmax
    mos.T = mp.mpf(datalump.temps[int(a/2)]) + 273.15
    vs = 0
    vd = datalump.vdlinearnom
    outval = mos.I_DS(vs,vd,x)
    mos.T = mos.Tnom
    return outval

def plot_wl_noise(
        mos: ekv.MOS,
        x,
        y,
        datalump,
        a
):
    mos.W = datalump.wlmax
    mos.L = datalump.wlmax
    vs = 0
    vd = datalump.vdsatnom
    vg = datalump.vgsatnom
    return np.float64((mos.Sthermal(vs,vd,vg) + mos.Sflicker(vs,vd,vg,x)))

#plotting exceptions, for plotting funcs that don't follow normal route in plotwrapper
#funcs included in this list are expected to sort themselves outputs

plot_exceptions = [
    plot_vths,
    plot_vths_alt,
    plot_ws_id_vg_sat_notemp,
    plot_sat_id_vs_len,
    plot_ws_id_vd_sat_deriv_data
]
        
#dictionary lookup table, points lsq_wrapper and resid_wrapper in the right direction
# 'stage' : [<resid_func handle>, <dataset in datalump>, <initial x>, <mos attrs> , <lsqkwargs>]
# initial x: array of either numbers or strings, if string, will pull attribute from
# mos with that name. if conversion is bad, try tweaking these values
# <mos attrs> gives the names of the attributes being worked on, required for final set of values
# when lsq_wrapper resolves, could be used in resid funcs, but not required. Will be the same
# as <initial x> if initial values are pulled from mos defaults
# list needs at least 4 elements, doesn't always have lsqkwargs though.
resid_func_lookup = {
    'vto_phi_gamma' : [
        find_vto_phi_gamma, #function that returns residuals
        'wl_vp_vg_temp_data', #dataset name (attribute of datalump)
        [0.4, 0.2, 0.5], #initial conditions
        ['VTO', 'PHI', 'GAMMA'], #mos attribute names
        [
            plot_vg_vp,
            'Gate voltage (V)',
            'Pinchoff voltage (V)'
        ], #plotting function and axis labels
        {'method' : 'lm'} #additional params
    ],
    'leta' : [
        find_leta,
        'ws_vp_vg_data',
        ['LETA'],
        ['LETA'],
        [],
        {'method' : 'lm'}
    ],
    'weta' : [
        find_weta,
        'nl_vp_vg_data',
        ['WETA'],
        ['WETA'],
        [],
        {'method' : 'lm'}
    ],
    'lk_q0' : [
        vth_resid,
        'ws_vp_vg_data',
        ['LK', 'Q0'],
        ['LK', 'Q0'],
        [
            plot_vths,
            'Length (m)',
            'VTH (V)'
        ],
        {'method' : 'lm',
         'xtol' : 1e-12,
         'ftol' : 1e-12
         }
    ],
    'lk_q02' : [
        vth_resid2,
        'ws_id_vg_linear_data',
        ['LK', 'Q0'],
        ['LK', 'Q0'],
        [
            plot_vths_alt,
            'Length (m)',
            'VTH (V)'
        ],
        {'method' : 'lm',
         'xtol' : 1e-12,
         'ftol' : 1e-12
         }
    ],
    'kp_e0' : [
        find_kp_e0,
        'wl_id_vg_vb_linear_data',
        ['KP', 'E0'],
        ['KP', 'E0'],
        [
            plot_wl_id_vg_vb_linear,
            'Gate voltage (V)',
            'Drain current (A)'
        ],
        {
            'bounds' : (1e-22, np.inf)
            #'method' : 'lm'
        }
    ],
    'kp_e0_lk' : [
        find_kp_e0_lk,
        'wl_id_vg_vb_linear_data',
        ['KP', 'E0', 'LK'],
        ['KP', 'E0', 'LK'],
        [
            plot_wl_id_vg_vb_linear,
            'Gate voltage (V)',
            'Drain current (A)'
        ],
        {
            'bounds' : (1e-22, np.inf)
            #'method' : 'lm'
        }
    ],
    'kp_e02' : [
        find_kp_e0,
        'wl_id_vg_vb_linear_data',
        ['KP', 'E0'],
        ['KP', 'E0'],
        [
            plot_wl_id_vg_vb_linear,
            'Gate voltage (V)',
            'Drain current (A)'
        ],
        {
            #'bounds' : (1e-22, np.inf)
            'method' : 'lm'
        }
    ],
    'kp' : [
        find_kp,
        'wl_id_vg_vb_linear_data',
        ['KP'],
        ['KP'],
        [
            plot_wl_id_vg_vb_linear,
            'Gate voltage (V)',
            'Drain current (A)'
        ],
        {
            #'bounds' : (1e-22, np.inf)
            'method' : 'lm'
        }
    ],
    'e0' : [
        find_e0,
        'wl_id_vg_vb_linear_data',
        ['E0'],
        ['E0'],
        [
            plot_wl_id_vg_vb_linear,
            'Gate voltage (V)',
            'Drain current (A)'
        ],
        {
            #'bounds' : (1e-22, np.inf)
            'method' : 'lm'
        }
    ],
    'dl' : [
        find_dl,
        'ws_id_vg_linear_data',
        [0],
        ['DL'],
        [
            plot_ws_id_vg_linear_data,
            'Gate voltage (V)',
            'Drain current (A)'
        ],
        {
            'method' : 'trf',
            'bounds' : (-0.9*global_lmin, 0)
        }
    ],
    'dl_lk' : [
        find_dl_lk,
        'ws_id_vg_linear_data',
        [0, 'LK'],
        ['DL', 'LK'],
        [
            plot_ws_id_vg_linear_data,
            'Gate voltage (V)',
            'Drain current (A)'
        ],
        {
            'method' : 'trf',
            'bounds' : (
                (-0.9*global_lmin, 0),
                (0, np.inf)
            )
        }
    ],
    'lambda_ucrit' : [
        find_lambda_ucrit,
        'ws_id_vd_sat_data',
        ['LAMBDA', 'UCRIT'],
        ['LAMBDA', 'UCRIT'],
        [
            plot_ws_id_vd_sat_data,
            'Drain voltage (V)',
            'Drain current (A)'
        ],
        {
            'bounds' : ((0,0), (4,np.inf))
        }
    ],
    'lambda_ucrit_kp' : [
        find_lambda_ucrit_kp,
        'ws_id_vd_sat_data',
        ['LAMBDA', 'UCRIT', 'KP'],
        ['LAMBDA', 'UCRIT', 'KP'],
        [
            plot_ws_id_vd_sat_data,
            'Drain voltage (V)',
            'Drain current (A)'
        ],
        {
            'bounds' : ((0,0,0), (4,np.inf,1))
        }
    ],
    'lambda_ucrit_deriv' : [
        find_lambda_ucrit_deriv,
        'ws_id_vd_sat_data',
        ['LAMBDA', 'UCRIT'],
        ['LAMBDA', 'UCRIT'],
        [
            plot_ws_id_vd_sat_deriv_data,
            'Drain voltage (V)',
            'Deriv Drain current wrt Drain voltage (A/V)'
        ],
        {
            'bounds' : ((0,0), (4,np.inf))
        }
    ],
    'lambda_ucrit_resid_deriv' : [
        find_lambda_ucrit_resid_deriv,
        'ws_id_vd_sat_data',
        ['LAMBDA', 'UCRIT'],
        ['LAMBDA', 'UCRIT'],
        [
            plot_ws_id_vd_sat_data,
            'Drain voltage (V)',
            'Drain current (A)'
        ],
        {
            'bounds' : ((0,0), (4,np.inf))
        }
    ],
    'lambda_ucrit_kp_resid_deriv' : [
        find_lambda_ucrit_kp_resid_deriv,
        'ws_id_vd_sat_data',
        ['LAMBDA', 'UCRIT', 'KP'],
        ['LAMBDA', 'UCRIT', 'KP'],
        [
            plot_ws_id_vd_sat_data,
            'Drain voltage (V)',
            'Drain current (A)'
        ],
        {
            'bounds' : ((0,0,0), (4,np.inf,1))
        }
    ],
    'ucrit' : [ #by default, don't use this stage in runs(?)
        find_ucrit,
        'ws_id_vg_sat_temp_data', #special case needs to get nominal temp set
        ['UCRIT'],
        ['UCRIT'],
        [
            plot_ws_id_vg_sat_notemp,
            'Gate voltage (V)',
            'Drain current (A)'
        ],
        {
            'bounds' : (0,np.inf)
        }
    ],
    'lambda' : [
        find_lambda,
        'ws_id_vd_sat_data',
        ['LAMBDA'],
        ['LAMBDA'],
        [
            plot_sat_id_vs_len,
            'Length (m)',
            'Drain current (A)'
        ],
        {
            'bounds' : (0,4)
        }
    ],
    'iba_ibb_ibn' : [
        find_ib_params,
        'wl_ib_vg_sat_data',
        ['IBA', 'IBB', 'IBN'],
        ['IBA', 'IBB', 'IBN'],
        [
            plot_ib_params,
            'Gate voltage (V)',
            'Body current I_{DB} (A)'
        ],
        {
            'method' : 'lm'
        }
    ],
    'tcv' : [
        find_tcv,
        'wl_vp_vg_temp_data',
        ['TCV'],
        ['TCV'],
        [],
        {}
    ],
    'bex' : [
        find_bex,
        'wl_id_vg_linear_temp_data',
        ['BEX'],
        ['BEX'],
        [
            plot_wl_id_vg_linear_temp,
            'Gate voltage (V)',
            'Drain current (A)'
        ],
        {}
    ],
    'ucex' : [
        find_ucex,
        'ws_id_vg_sat_temp_data',
        ['UCEX'],
        ['UCEX'],
        [],
        {
            'bounds' : (0,np.inf)
        }
    ],
    'kf_af' : [
        find_kf_af, #also thermfudge...
        'wl_noise_data',
        ['KF','AF','thermfudge'],
        ['KF','AF','thermfudge'],
        [
            plot_wl_noise,
            'Frequency (Hz)',
            'PSD ($V/\\sqrt{Hz}$)'
        ],
        {
            'method' : 'lm'
        }
    ]
}

def lsq_wrapper(stage: str, mos: ekv.MOS, datalump: DataStore):
    """
    stage is a key of resid_func_lookup, provides the following:
    func to get residuals
    dataset to pull from datalump
    x0 values in np.array (pull from mos)
    any extra **kwargs to send to least_squares
    """

    # unpack x0
    x0 = copy.deepcopy(resid_func_lookup[stage][2])
    # process x0, if number, leave, if string, grab attribute from mos
    for a in range(len(x0)):
        if isinstance(x0[a],str):
            x0[a] = getattr(mos,x0[a])
        
    result = least_squares(
        resid_wrapper,
        x0,
        args=(
            stage,
            mos,
            datalump
        ),
        **resid_func_lookup[stage][5]
    )
    # final attribute value setting happens here
    for a in range(len(result.x)):
        setattr(mos,resid_func_lookup[stage][3][a],result.x[a])

    plotwrapper(stage, datalump, mos)
    
    # return the result for logging purposes, but the func above mutates mos
    return result
