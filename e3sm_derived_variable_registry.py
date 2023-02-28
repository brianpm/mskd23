import numpy as np
import xarray as xr
from intake_esm import DerivedVariableRegistry

# 
# Derived variables 
# 

# E3SM to CAM names
e3sm_dvr = DerivedVariableRegistry()

@e3sm_dvr.register(variable='SWCF', query={'variable':['rsut','rsutcs']})
def e3sm_to_SWCF(ds):
    ds['SWCF'] = ds['rsutcs'] - ds['rsut']
    ds['SWCF'].attrs = {'units': 'W/m2', 'long_name': "shortwave cloud radiative effect",
    "derived_by": "intake-esm"}
    return ds

@e3sm_dvr.register(variable='LWCF', query={'variable':['rlut','rlutcs']})
def e3sm_to_LWCF(ds):
    ds['LWCF'] = ds['rlut'] - ds['rlutcs']
    ds['LWCF'].attrs = {'units': 'W/m2', 'long_name': "longwave cloud radiative effect",
    "derived_by": "intake-esm"}
    return ds

@e3sm_dvr.register(variable='CLDTOT_CAL', query={'variable':['cltcalipso']})
def e3sm_to_CLDTOT_CAL(ds):
    ds['CLDTOT_CAL'] = ds['cltcalipso']
    return ds

@e3sm_dvr.register(variable="clhisccp", query={'variable':['clisccp']})
def e3sm_to_clhisccp(ds):
    ds['clhisccp'] = ds['clisccp'].sel(plev=slice(44000,None), tau=slice(0.3,None)).sum(dim=("plev","tau"))
    return ds

@e3sm_dvr.register(variable="clmisccp", query={'variable':['clisccp']})
def e3sm_to_clmisccp(ds):
    ds['clmisccp'] = ds['clisccp'].sel(plev=slice(68000, 44000), tau=slice(0.3,None)).sum(dim=("plev","tau"))
    return ds

@e3sm_dvr.register(variable="cllisccp", query={'variable':['clisccp']})
def e3sm_to_cllisccp(ds):
    ds['cllisccp'] = ds['clisccp'].sel(plev=slice(None, 68000), tau=slice(0.3,None)).sum(dim=("plev","tau"))
    return ds

@e3sm_dvr.register(variable="OMEGA500", query={"variable":["wap"]})
def e3sm_to_omega500(ds):
    ds['OMEGA500'] = ds['wap'].sel(plev=50000)
    return ds