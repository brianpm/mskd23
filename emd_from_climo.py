import xarray as xr
import numpy as np
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO)
import computations as comp
from computations import Model, Satellite # DataClasses!

#
# Functions
#
def check_dim_increasing(dset, dimName, flip=False):
    """Check that dimension values are increasing. If not, [optionally] flip them."""
    vals = dset[dimName]
    d = np.diff(vals)
    if np.all(d > 0):
        is_increasing = True
    elif np.all(d < 0):
        is_increasing = False
    else:
        raise ValueError(f"The dimension {dimName} is not monotonic.")
    if flip:
        if not is_increasing:
            logging.warning(f"Reversing dimension: {dimName}")
            return dset.reindex({dimName:list(reversed(dset[dimName]))})
        else:
            return dset
    else:
        logging.info(f"[check_dim_increasing] result says that {is_increasing = }")

#
# The main function
#
def main(sat=None, model=None):

    if not isinstance(sat, Satellite):
        sat = Satellite(sat)

    cl_name = sat.data_var_name
    ds_sat = sat.load_climo()
    time_range_str = sat.time_range_str  # only after load_climo!
    sat_climo = ds_sat[cl_name].squeeze()

    if not isinstance(model, Model):
        model = Model(model, sat.name, time_range_str)

    ds_model = model.load_data()
    start_year = model.daterange.split('-')[0]
    end_year = model.daterange.split('-')[1]
    if start_year is not None:
        start_year = f"{start_year}-01-01"
    if end_year is not None:
        end_year = f"{end_year}-12-31"

    model_climo = ds_model[model.data_var_name].sel(time=slice(start_year, end_year)).sel({model.tau_name:slice(0.3,None)}).mean(dim='time')
    if hasattr(model_climo, 'compute'):
        print("Force `compute()`")
        model_climo = model_climo.compute()
    model_climo = check_dim_increasing(model_climo, model.pres_name, flip=True)

    sat_climo = check_dim_increasing(sat_climo, sat.pres_name, flip=True)

    # RENAME DIMS
    if sat.tau_name != "tau":
        sat_climo = sat_climo.rename({sat.tau_name:"tau"})
    if sat.pres_name != "ctp":
        sat_climo = sat_climo.rename({sat.pres_name:"ctp"})
    model_climo = model_climo.rename({model.tau_name:"tau", model.pres_name:"ctp"})

    logging.info(f"{sat_climo.shape = }, {model_climo.shape = }")

    emdist = comp.emd_loop(sat_climo, model_climo, 'tau', 'ctp')
    oloc = Path("/Users/brianpm/Desktop")
    outname = oloc / f"{model.name.lower()}_{sat.name.lower()}_{time_range_str}_emd_nanversion.nc"
    emdist.to_netcdf(outname)
    print(f"Finished and wrote: {outname}")

#
# as a command -- without arguments, so have to edit for now.
#
if __name__ == "__main__":

    satellite_name = "MISR"
    model_name = "CAM6"

    satellite = Satellite(satellite_name)
    # Time ranges:
    # CAM6 1979 - 2014
    # CAM5 2001 - 2010
    # CAM4 2001 - 2010
    # E3SM 1870 - 2014
    if model_name in ['CAM4', 'CAM5']:
        trange = "2001-2016"
    else:
        trange = "2000-2016"

    gcm = Model(model_name, satellite.name, trange)
    
    main(sat=satellite, model=gcm)
    print("*DONE*")