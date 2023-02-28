import numpy as np
import xarray as xr
from intake_esm import DerivedVariableRegistry

# CAM to CMIP names
cam_dvr = DerivedVariableRegistry()


@cam_dvr.register(variable="cltisccp", query={'variable':['CLDTOT_ISCCP']})
def cam_to_cltisccp(ds):
    ds['cltisccp'] = ds['CLDTOT_ISCCP']
    # check on units
    return ds

@cam_dvr.register(variable="clhisccp", query={'variable':['FISCCP1_COSP']})
def cam_to_clhisccp(ds):
    p = ds['FISCCP1_COSP'].cosp_prs
    if np.max(p) > 2000:
        pslice = slice(44000,None)
    else:
        pslice = slice(440,None)
    ds['clhisccp'] = ds['FISCCP1_COSP'].sel(cosp_prs=pslice, cosp_tau=slice(0.3,None)).sum(dim=("cosp_prs","cosp_tau"))
    return ds

@cam_dvr.register(variable="clmisccp", query={'variable':['FISCCP1_COSP']})
def cam_to_clmisccp(ds):
    p = ds['FISCCP1_COSP'].cosp_prs
    if np.max(p) > 2000:
        pslice = slice(68000, 44000)
    else:
        pslice = slice(680, 440)
    ds['clmisccp'] = ds['FISCCP1_COSP'].sel(cosp_prs=pslice, cosp_tau=slice(0.3,None)).sum(dim=("cosp_prs","cosp_tau"))
    return ds

@cam_dvr.register(variable="cllisccp", query={'variable':['FISCCP1_COSP']})
def cam_to_cllisccp(ds):
    p = ds['FISCCP1_COSP'].cosp_prs
    if np.max(p) > 2000:
        pslice = slice(None, 68000)
    else:
        pslice = slice(None, 680)
    ds['cllisccp'] = ds['FISCCP1_COSP'].sel(cosp_prs=pslice, cosp_tau=slice(0.3,None)).sum(dim=("cosp_prs","cosp_tau"))
    return ds

@cam_dvr.register(variable="clisccp", query={'variable':['FISCCP1_COSP']})
def cam_to_clisccp(ds):
    ds['clisccp'] = ds['FISCCP1_COSP']
    return ds


@cam_dvr.register(variable="cltmisr", query={'variable':['CLD_MISR']})
def cam_to_cltmisr(ds):
    tmp = ds['CLD_MISR'].sel(cosp_tau = slice(0.3, None)).sum(dim=('cosp_tau', 'cosp_htmisr'))
    tmp.name = 'cltmisr'
    tmp.attrs['long_name'] = 'MISR Total Cloud Fraction (tau > 0.3)'
    ds['cltmisr'] = tmp
    # check on units
    return ds

@cam_dvr.register(variable="cltcalipso", query={'variable':['CLDTOT_CAL']})
def cam_to_cltcalipso(ds):
    ds['cltcalipso'] = ds['CLDTOT_CAL']
    return ds

@cam_dvr.register(variable="clhcalipso", query={'variable':['CLDHGH_CAL']})
def cam_to_cltcalipso(ds):
    ds['clhcalipso'] = ds['CLDHGH_CAL']
    return ds

@cam_dvr.register(variable="clmcalipso", query={'variable':['CLDMED_CAL']})
def cam_to_cltcalipso(ds):
    ds['clmcalipso'] = ds['CLDMED_CAL']
    return ds

@cam_dvr.register(variable="cllcalipso", query={'variable':['CLDLOW_CAL']})
def cam_to_cltcalipso(ds):
    ds['cllcalipso'] = ds['CLDLOW_CAL']
    return ds

@cam_dvr.register(variable="clcalipso", query={'variable':['CLD_CAL']})
def cam_to_clcalipso(ds):
    ds['clcalipso'] = ds['CLD_CAL']
    return ds

@cam_dvr.register(variable="cltmodis", query={'variable':['CLTMODIS']})
def cam_to_cltmodis(ds):
    ds['cltmodis'] = ds['CLTMODIS']
    # check on units
    return ds
