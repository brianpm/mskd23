#
# This module is where computation is done
#
from pathlib import Path
import xarray as xr
import numpy as np
from dataclasses import dataclass
import logging
logger = logging.getLogger(__name__)
# import esmlab  # DEPRECATED ... import error with python 3.10
logger.setLevel(logging.DEBUG)
#
# module data
#
days_per_month = xr.DataArray(np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]), dims=("month",), coords={"month":np.arange(1,13)})
assert np.sum(days_per_month) == 365  # we don't care about leap years

# general utilities

def get_coslat(data, latname='lat'):
    return np.cos(np.radians(data[latname]))


def lonFlip(data):
    # NOTE: this assumes global values
    tmplon = data['lon']
    tmpdata = data.roll(lon=len(tmplon) // 2, roll_coords=True)
    lonroll = tmpdata['lon'].values
    if tmplon.min() >= 0:
        # flip to -180:180
        tmpdata = tmpdata.assign_coords({'lon': np.where(lonroll >= 180, lonroll - 360, lonroll)})
    else:
        # flip from -180:180 to 0:360
        # tmpdata = tmpdata.assign_coords({'lon': ((lonroll + 360) % 360)})
        tmpdata = tmpdata.assign_coords({'lon': lonroll%360})
    return tmpdata


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


# averaging

def get_monthly_climo(data):
    return data.groupby('time.month').mean(dim='time')


def get_climo(data):
    """Return time-mean climatology.
    If monthly data: weight by month length directly and average.
    If `month` is a dimension (and `time` is not), weight by standard days-per-month and average months.

    If `time` is a dimension, an attribute called date_bounds will be included to report the time being averaged.
    -- 
    """
    if 'time' in data.dims:
        dpm = data.time.dt.daysinmonth
        tmp = data.weighted(dpm).mean(dim='time', keep_attrs=True)
        tmp.attrs['date_bounds'] = f"{data.time[0].dt.strftime('%Y%m').item()} - {data.time[-1].dt.strftime('%Y%m').item()}"
        return tmp
        # tmp = get_monthly_climo(data)
    elif 'month' in data.dims:
        # assume montly climo
        return data.weighted(days_per_month).mean(dim='month')
    else:
        raise IOError("Sorry, that data does not look like monthly climo or time series.")


def month_avg(da):
    """Average monthly data, weighted by days per month."""
    month_length = da.time.dt.days_in_month
    return da.weighted(month_length).mean(dim='time')


def global_average_np(fld, wgt, verbose=False):
    """
    A simple, pure numpy global average.
    fld: an input ndarray
    wgt: a 1-dimensional array of weights
    wgt should be same size as one dimension of fld
    """

    s = fld.shape
    for i in range(len(s)):
        if np.size(fld, i) == len(wgt):
            a = i
            break
    fld2 = np.ma.masked_invalid(fld)
    if verbose:
        print("(global_average)-- fraction input missing: {}".format(fraction_nan(fld)))
        print("(global_average)-- fraction of mask that is True: {}".format(np.count_nonzero(fld2.mask) / np.size(fld2)))
        print("(global_average)-- apply ma.average along axis = {} // validate: {}".format(a, fld2.shape))
    avg1, sofw = np.ma.average(fld2, axis=a, weights=wgt, returned=True) # sofw is sum of weights

    return np.ma.average(avg1)


def global_average(data, weights=None):
    if weights is None:
        weights = get_coslat(data)
    return data.weighted(weights).mean(keep_attrs=True)


def get_zonal_climo(data):
    """Zonal average of the climatology from monthly files _or_ monthly climo."""
    tmp = get_climo(data)
    return tmp.mean(dim='lon')

def rm_land(dset, lf):
    """Return dset with np.nan where lf <= 0. 
    If lat dimensions are same size, but not equal, will reassign
    lat from dset to lf and report largest difference.
    """
    assert hasattr(dset, 'lat'), f'No lat in data set, {dset.coords = }'
    assert len(dset['lat']) == len(lf['lat']), f"NO GOOD! data: {len(dset['lat'])}, land: {len(lf['lat'])}"
    if np.count_nonzero(dset['lat'] == lf['lat']) != len(lf['lat']):
        print(f"[rm_land] Latitudes mismatch. Largest discrepancy is {np.max(np.absolute(dset['lat'].values - lf['lat'].values))} degrees.")
        lf = lf.assign_coords({"lat":dset["lat"], "lon":dset["lon"]})
    return xr.where(lf <= 0, dset, np.nan)



def get_land_ocean_avg(data, landmask, weights=None):
    land_data = xr.where(landmask > 0, data, np.nan)
    ocean_data = xr.where(landmask <= 0, data, np.nan)
    land_average = global_average(land_data, weights=weights)
    ocean_average = global_average(ocean_data, weights=weights)
    return land_average, ocean_average


def get_overlapping_time(ds1, ds2, return_indices=None):
    # a function to get the data for overlapping months
    #
    # The calendar months where the data sets overlap
    #
    # indices in ds1 that correspond to months in ds2
    ds1_overlap_indices = np.nonzero([(i.item() in ds2.time.dt.strftime("%Y%m")) for i in ds1.time.dt.strftime("%Y%m")])[0]
    ds2_overlap_indices = np.nonzero([(i.item() in ds1.time.dt.strftime("%Y%m")) for i in ds2.time.dt.strftime("%Y%m")])[0]
    ds1ovrlap = ds1.isel(time=ds1_overlap_indices)
    ds2ovrlap = ds2.isel(time=ds2_overlap_indices)
    if return_indices:
        return ds1ovrlap, ds2ovrlap, ds1_overlap_indices, ds2_overlap_indices
    else:
        return ds1ovrlap, ds2ovrlap


def weighted_binned(x, y, bins):
    """Binned statistic weighted by cos(lat).
    Uses scipy binned_statistic, but correctly weights the data and accounts for
    missing values by setting the weight to zero where either x or y are missing.
    
    note: right now applies binned_statistic twice, but investigate whether it would be more efficient to use the bin assignment from the first call to directly do the averaging. 
    
    return
    weighted mean in bins, bin edges, bin assignment
    """
    import scipy.stats as ss
    wgt = get_coslat(x)
    wgt_array = wgt.broadcast_like(x)
    # wherever the data is invalid, weight should be set to zero:
    wgt_array = xr.where(np.logical_or(np.isnan(x), np.isnan(y)), 0, wgt_array)
    # first get the sum of weights in each bin
    wgt_bs, wgt_edg, wgt_num = ss.binned_statistic(x.values.flatten(), wgt_array.values.flatten(), statistic=np.nansum, bins=bins)
    # next get the sum of wgt_array * y
    wy_bs, wy_edg, wy_num = ss.binned_statistic(x.values.flatten(), (y*wgt_array).values.flatten(), statistic=np.nansum, bins=bins)
    # calculate the average by dividing by the sum of weights
    return wy_bs/wgt_bs, wy_edg, wy_num



def binned_mean(xvar, yvar, xbins):
    """
    a "fast" binned statistic that works in the vertical based on 2d independent variable
    The use case for this was to plot `clcalipso` in bins of Omega500. 
    
    xvar : the independent variable (2 dimensional)
    yvar : the dependent variable (can be 2 or 3 dimensional)
    xbins : the bin edges to use
    """
    # the weighted average in each bin is SUM[w(bin)*x(bin)] / SUM[w(bin)]
    # THIS PART ONLY NEEDS DOING ONCE:
    binnumber = np.digitize(xvar, xbins) ## says which bin each (time,lat,lon) points belongs in
    # Define area-weights
    # wgt = np.cos(np.radians(xvar.lat))
    wgt = get_coslat(xvar)
    warr = wgt.broadcast_like(xvar)  ## weights the same size as xvar

    nbins = len(xbins) # Need one more bin
    # print(f"{nbins = }")

    xrank = len(xvar.shape)
    yrank = len(yvar.shape)
    xvarnan = np.isnan(xvar) # compute this once, and use below.
    if xrank == yrank:
        # arrays need to be only valid data in both xvar and yvar
        dataValid = ~(xvarnan | np.isnan(yvar))  # the values that we keep
        weightedData = (warr*yvar).values[dataValid]
        binAssign = binnumber[dataValid]   ## only keep the ones where xvar&yvar are valid
        # flatcount = np.bincount(binAssign.ravel())  ## counts the values in each bin (reduces to size of bins)
        wgtValid = warr.values[dataValid]
        sumWeightsPerBin = np.bincount(binAssign.ravel(), weights=wgtValid.ravel(), minlength=nbins+1)
        sumWeightedDataPerBin = np.bincount(binAssign.ravel(), weights=(weightedData).ravel(), minlength=nbins+1)
        # print(f"{sumWeightsPerBin = }, {sumWeightedDataPerBin = }")
        weightedAveragePerBin = sumWeightedDataPerBin / sumWeightsPerBin
    elif xrank == (yrank-1):
        # only allow one dimension extra in y, identify it by dim name:
        loopDimName = set(yvar.dims).difference(xvar.dims).pop()
        # <alt version>
        # yDimsInX = [d in xvar.dims for d in yvar.dims]
        # assert np.sum(yDimsInX) == 1, "Dimension names do not match"
        # loopDimName = yvar.dims[yDimsInX.index(False)]
        # </alt version>
        # initialize the result array
        weightedAveragePerBin = np.zeros((nbins+1, len(yvar[loopDimName])))
        # print(f"{weightedAveragePerBin.shape = }")
        for k, lev in enumerate(yvar[loopDimName]):
            ylev = yvar.isel({loopDimName:k}) # this should be same shape as xvar
            dataValid = ~(xvarnan | np.isnan(ylev))  # the values that we keep
            weightedData = (warr*ylev).values[dataValid]
            binAssign = binnumber[dataValid]
            # print(f"{k = }, {binAssign.shape = }, uniq: {np.unique(binAssign)}")
            wgtValid = warr.values[dataValid]
            sumWeightsPerBin = np.bincount(binAssign.ravel(), weights=wgtValid.ravel(), minlength=nbins)
            sumWeightedDataPerBin = np.bincount(binAssign.ravel(), weights=(weightedData).ravel(), minlength=nbins)
            # print(f"{sumWeightsPerBin = }, {sumWeightedDataPerBin = }")
            weightedAveragePerBin[:,k] = sumWeightedDataPerBin / sumWeightsPerBin
    else:
        raise NotImplementedError(f"Sorry, dimensions are too different: {xvar.shape = }, {yvar.shape = }")

    return weightedAveragePerBin

#
# ERROR STATISTICS 
# 

def bias(measured, observed):
    """The bias is the global mean of the difference."""
    assert len(measured.shape) == 2, "Need 2-d input"
    assert len(observed.shape) == 2, "Need 2-d input"
    if measured.shape != observed.shape:
        print(f"Regrid required. Will interpolate observed {observed.shape} to measured grid {measured.shape}.")
        observed = observed.interp_like(measured)
    difference = measured - observed
    return global_average(difference).item()


def wgt_rmse(fld1, fld2, wgt=None):
    """Calculated the area-weighted RMSE.

    Inputs are 2-d spatial fields, fld1 and fld2 with the same shape.
    They can be xarray DataArray or numpy arrays.

    Input wgt is the weight vector, expected to be 1-d, matching length of one dimension of the data.

    Returns a single float value.
    """
    assert len(fld1.shape) == 2,     "Input fields must have exactly two dimensions."
    if fld1.shape != fld2.shape:
        print(f"Regrid required. Will interpolate fld2 {fld2.shape} to fld1 grid {fld1.shape}.")
        fld2 = fld2.interp_like(fld1)
    if wgt is None:
        wgt = get_coslat(fld1)
    if isinstance(fld1, xr.DataArray) and isinstance(fld2, xr.DataArray):
        return (np.sqrt(((fld1 - fld2)**2).weighted(wgt).mean())).values.item()
    else:
        check = [len(wgt) == s for s in fld1.shape]
        if ~np.any(check):
            raise IOError(f"Sorry, weight array has shape {wgt.shape} which is not compatible with data of shape {fld1.shape}")
        check = [len(wgt) != s for s in fld1.shape]
        dimsize = fld1.shape[np.argwhere(check).item()]  # want to get the dimension length for the dim that does not match the size of wgt
        warray = np.tile(wgt, (dimsize, 1)).transpose()   # May need more logic to ensure shape is correct.
        warray = warray / np.sum(warray) # normalize
        wmse = np.sum(warray * (fld1 - fld2)**2)
        return np.sqrt( wmse ).item()
 

def _weighted_corr(x,y, w, dim=None):
    xw = x.weighted(w)
    yw = y.weighted(w)
    # weighted covariance:
    devx = x - xw.mean(dim=dim)
    devy = y - yw.mean(dim=dim)
    devxy = devx * devy
    covxy = devxy.weighted(w).mean(dim=dim)
    denom = np.sqrt(xw.var()*yw.var())
    return covxy / denom


# NMSE is composed of the 
# * unconditional bias: U = ( (mean(x_m) - mean(x_o)) / σ_o)**2
# * conditional bias: C = (r - σ_m/σ_o)**2
# * phasing error: P = (1 - r**2)
def nmse_components(measured, observed, weights=None, is_averaged=None):
    """Inputs all need to be Xarray DataArrays."""
    if weights is None:
        weights = get_coslat(measured)
    if (is_averaged is not None):
        if is_averaged:
            do_average = False
            time_mean_m = measured
            time_mean_o = observed
        else:
            do_average = True
    else:
        do_average = True
    if do_average:
        time_mean_m = measured.mean(dim='time')
        time_mean_o = observed.mean(dim='time')
    wgt,_ = xr.broadcast(weights, time_mean_m)
    mean_m = time_mean_m.weighted(weights).mean(dim=["lat","lon"]).compute()  # esmlab.weighted_mean(time_mean_m, dim=["lat","lon"], weights=wgt)
    mean_o = time_mean_o.weighted(weights).mean(dim=["lat","lon"]).compute()  # esmlab.weighted_mean(time_mean_o, dim=["lat","lon"], weights=wgt)
    sigma_m = time_mean_m.weighted(weights).std(dim=["lat","lon"]).compute()  # esmlab.weighted_std(time_mean_m, dim=["lat","lon"], weights=wgt)
    sigma_o = time_mean_o.weighted(weights).std(dim=["lat","lon"]).compute()  # esmlab.weighted_std(time_mean_o, dim=["lat","lon"], weights=wgt)
    corr = _weighted_corr(time_mean_m, time_mean_o, weights, dim=["lat","lon"])  # esmlab.weighted_corr(time_mean_m, time_mean_o, dim=["lat","lon"], weights=wgt, return_p=False)
    U = (((mean_m - mean_o)/sigma_o)**2).compute()
    C = ((corr - (sigma_m/sigma_o))**2).compute()
    P = (1 - corr**2).compute()
    svr = (sigma_m / sigma_o)**2 * (U+C+P)  # scaled variance ratio:
                                            # indicates whether conditional bias arises from too much 
                                            # (SVR > NMSE) or too little (SVR < NMSE) spatial variance
    if hasattr(U, "item"):
        U = U.item()
    if hasattr(C, "item"):
        C = C.item()
    if hasattr(P, "item"):
        P = P.item()
    if hasattr(svr, "item"):
        svr = svr.item()
    return U, C, P, svr

# binned statistics (dynamical regimes)


#
# Earth Mover's Distance (alternative to relative entropy)
#
import ot  # PythonOT = optimal transport
def emd_calc(a, b, M):
    """ This is the exact solver for EMD. 
        Calculates G0 the transport matrix,
        but returns SUM(G0*M) which is the distance.

        Using ot.emd rather than ot.emd2 because I found
        emd2 did not return the same result, and could 
        not diagnose the issue. Compared this result to
        `pyemd` package and it is the same. 
    """
    # print(f"{a.shape = }, {b.shape = }, {M.shape = }")
    G0 = ot.emd(a, b, M) # This is the optimal transport matrix
    return (G0*M).sum()  # This is the Wasserstein distance (divide by G0.sum() if it is not 1)


def emd_loop(arr1, arr2, tau_dim_name, z_dim_name):
    """This provides the interface to the EMD calculation.

    arr1 : xarray DataArray
    arr2 : xarray DataArray
    tau_dim_name : str, name of first dimension to use
    z_dim_name : str, name of second dimension to use

    Assumes that we have two-dimensions that need to be used
    to calculate the cost matrix (distance), and then it flattens
    them into a single dimension.
    Other dimensions are flattened into a single (second) dimension
    in order to loop over only one dimension.

    To try to avoid errors/warnings, tries to only do calculation
    for the entries that are valid.
    The EMD result is then put into an array and "unstacked" to produce
    an xarray DataArray with dimensions like arr1/arr2 (with tau and z dims gone).

    note: `pot` (python optimal transport) is used for the EMD calculation.
          `pot` used here for the distance matrix (uses Euclidean distance; assumes simple integer distances of the 2d grid)
    note: `joblib` is used to parallelize the loop.

    """
    from joblib import Parallel, delayed

    tau = arr1[tau_dim_name]
    ctp = arr1[z_dim_name]
    stau = np.arange(0, len(tau))
    sctp = np.arange(0, len(ctp))
    mtau, mctp = np.meshgrid(stau, sctp)
    nodes = np.column_stack([mtau.ravel(), mctp.ravel()])
    M = ot.dist(nodes, metric='euclidean')  # This is same as dmatrix, ot.dist(nodes,nodes,metric='euclidean')
    # M = distance_matrix(nodes, nodes)  # scipy's function, default is euclidean distance
    
    # normalize so sum of histograms is 1.0
    na = arr1 / arr1.sum(dim=(tau_dim_name, z_dim_name))
    nb = arr2 / arr2.sum(dim=(tau_dim_name, z_dim_name))

    # we carefully rearrange the 2d histogram to be one-dimensional and consistent with the distance matrix M
    avals = na.stack(z=(tau_dim_name, z_dim_name))
    bvals = nb.stack(z=(tau_dim_name, z_dim_name))

    all_dims = avals.dims
    other_dims = list(all_dims)
    other_dims.pop(other_dims.index('z'))
    avalsstack = avals.stack(d=other_dims)
    bvalsstack = bvals.stack(d=other_dims)

    emd = np.empty_like(avalsstack.isel(z=0))
    emd[:] = np.nan
    valid_data = np.isclose(avalsstack.sum(dim='z'), bvalsstack.sum(dim='z'))
    print(f"{valid_data.shape = }, {avalsstack.shape = }")
    valid_data_inds = np.argwhere(valid_data)
    print(valid_data_inds)

    results = Parallel(n_jobs=8)(delayed(emd_calc)(avalsstack.isel(d=i).squeeze().values, bvalsstack.isel(d=i).squeeze().values, M) for i in valid_data_inds)

    # results = []
    # for i in valid_data_inds:
    #     input1 = avalsstack.isel(d=i).squeeze().values
    #     input2 = bvalsstack.isel(d=i).squeeze().values
    #     # print(f"{input1.shape = }, {input2.shape = }")
    #     results.append(emd_calc(input1, input2, M))

    for j, i in enumerate(valid_data_inds):
        emd[i] = results[j] 

    print(f"{emd.shape = }")
    emdxr = (xr.DataArray(emd, dims="d", coords={"d":avalsstack["d"]}, name='EMD')).unstack()  # should be ([time], lat, lon)
    return emdxr


#
# Specialty functions
#
def cosp_isccp_to_thml(clisccp, p_name="cosp_prs", tau_name="cosp_tau"):
    """Derive total, high, mid, and low cloud fraction from ISCCP histogram."""
    # in order to keep missing data as missing rather than zero, 
    # do one dim at a time with a min_count kwarg. (at least for actual obs)
    #
    # Note: COSP grid is different from ISCCP grid, so need to span a couple levels
    # To use expressions, use `query`

    assert p_name in clisccp.dims, f"p_name given as {p_name} is not in dimensions: {clisccp.dims}"
    assert tau_name in clisccp.dims, f"tau_name given as {tau_name} is not in dimensions: {clisccp.dims}"
    
    # sometimes (CAM6) the cosp_prs is in Pa instead of hPa:
    if clisccp[p_name].max() > 2000.:
        print("INFO: going to convert pressure from apparent Pa to hPa.")
        pres_hPa = clisccp[p_name] * 0.01
        clisccp = clisccp.assign_coords({p_name:pres_hPa})
    
    tot_p_expr = f"{p_name} >= 0"
    tot_t_expr = f"{tau_name} >= 0.3"
    cltisccp = clisccp.query({p_name: tot_p_expr, tau_name: tot_t_expr}).squeeze().sum(dim=p_name, min_count=1).sum(dim=tau_name, min_count=1).compute()
    cltisccp = clisccp.sum(dim=tau_name, min_count=1).sum(dim=p_name, min_count=1).compute()
    cltisccp.name = 'cltisccp'
    cltisccp.attrs['long_name'] = "Total cloud fraction"

    low_p_expr = f"{p_name} >= 680.0"
    low_t_expr = f"{tau_name} >= 0.3"
    cllisccp = clisccp.query({p_name: low_p_expr, tau_name: low_t_expr}).squeeze().sum(dim=p_name, min_count=1).sum(dim=tau_name, min_count=1).compute()
    cllisccp.name = 'cllisccp'
    cllisccp.attrs['long_name'] = "Low-topped cloud"
    
    mid_p_expr = f"({p_name} <= 680.) & ({p_name} >= 440.)"
    mid_t_expr = f"{tau_name} >= 0.3"
    clmisccp = clisccp.query({p_name: mid_p_expr, tau_name: mid_t_expr}).squeeze().sum(dim=p_name, min_count=1).sum(dim=tau_name, min_count=1).compute()
    clmisccp.name = 'clmisccp'
    clmisccp.attrs['long_name'] = "Mid-topped cloud"
    
    hgh_p_expr = f"{p_name} <= 440."
    hgh_t_expr = f"{tau_name} >= 0.3"
    clhisccp = clisccp.query({p_name:hgh_p_expr, tau_name:hgh_t_expr}).squeeze().sum(dim=p_name, min_count=1).sum(dim=tau_name, min_count=1).compute()
    clhisccp.name = 'clhisccp'
    clhisccp.attrs['long_name'] = "High-topped cloud"
    
    return cltisccp, clhisccp, clmisccp, cllisccp



#
# I/O
#
def adjust_monthly_time(ds):
    assert 'time_bnds' in ds  # require time_bnds to derive average time
    bnd_dims = ds['time_bnds'].dims
    time_correct = ds['time_bnds'].mean(dim=bnd_dims[1])
    time_correct.attrs = ds['time'].attrs
    ds = ds.assign_coords({"time":time_correct})
    ds = xr.decode_cf(ds)
    return ds


def open_cesm_dataset(fils):
    if isinstance(fils,list) and (len(fils)==1):
        ds = xr.open_dataset(fils[0], decode_times=False)
    elif isinstance(fils,list):
        ds = xr.open_mfdataset(fils, decode_times=False, combine='by_coords')
    else:
        ds = xr.open_dataset(fils, decode_times=False)
    assert 'time_bnds' in ds  # require time_bnds to derive average time
    bnd_dims = ds['time_bnds'].dims
    # print(f"The time bounds dimension is {bnd_dims[1]}")
    time_correct = ds['time_bnds'].mean(dim=bnd_dims[1])
    time_correct.attrs = ds['time'].attrs
    ds = ds.assign_coords({"time":time_correct})
    ds = xr.decode_cf(ds)
    return ds


#
# Define Model and Satellite dataclasses
#
@dataclass
class Model:
    name: str
    satellite: str
    daterange: str
    def __post_init__(self):
        if self.satellite == "ISCCP":
            if self.name == "E3SM":
                self.data_var_name = 'clisccp'
                self.tau_name = 'tau'
                self.pres_name = 'plev'
            else:
                self.data_var_name = 'FISCCP1_COSP'
                self.tau_name = 'cosp_tau'
                self.pres_name = 'cosp_prs'
        elif self.satellite == "MODIS":
            if "CAM" in self.name:
                self.data_var_name = 'CLMODIS'
                self.tau_name = 'cosp_tau_modis'
                self.pres_name = 'cosp_prs'
            else:
                self.data_var_name = None
        elif self.satellite == "MISR":
            if "CAM" in self.name:
                self.data_var_name = 'CLD_MISR'
                self.tau_name = 'cosp_tau'
                self.pres_name = 'cosp_htmisr'
            else:
                self.data_var_name = None            

    def load_data(self):
        """A method to automatically load the COSP data."""
        if self.name == "CAM6":
            data_root = Path("/Volumes/Drudonna/f.e21.FHIST_BGC.f09_f09_mg17.CMIP6-AMIP.001_cosp1/atm/proc/tseries/month_1")
            data_files = sorted(data_root.glob(f"f.e21.FHIST_BGC.f09_f09_mg17.CMIP6-AMIP.001_cosp1.cam.h0.{self.data_var_name}.*.nc"))            
        elif self.name == "CAM5":
            data_root = Path("/Volumes/Drudonna/cam5_1deg_release_amip/atm/proc/tseries/month_1")
            data_files = sorted(data_root.glob(f"cam5_1deg_release_amip.cam.h0.{self.data_var_name}.*.nc"))
        elif self.name == "CAM4":
            data_root = Path("/Volumes/Drudonna/cam4_1deg_release_amip/atm/proc/tseries/month_1")
            data_files = sorted(data_root.glob(f"cam4_1deg_release_amip.cam.h0.{self.data_var_name}.*.nc"))
        elif self.name == "E3SM":
            data_root = Path("/Volumes/Drudonna/E3SM-1-0_amip/remapped")
            data_files = sorted(data_root.glob(f"{self.data_var_name}_CFmon_E3SM-1-0_amip_r1i1p1f1_gr_*.nc"))
            logger.debug(f"{len(data_files) = }")
        assert len(data_files) >= 1, f"Sorry, did not find data files for {self.name}, {self.data_var_name}"
        if "CAM" in self.name:
            if len(data_files) > 1:
                ds = xr.open_dataset(data_files[0], decode_times=False)
            else:
                ds = xr.open_mfdataset(data_files, decode_times=False)
            timevals = ds['time_bnds'].mean(dim='nbnd')
            timevals.attrs = ds['time'].attrs
            ds = ds.assign_coords({"time":timevals})
            ds = xr.decode_cf(ds)
        else:
            if len(data_files) > 1:
                ds = xr.open_mfdataset(data_files)
            else:
                # just one file, take it out of list
                ds = xr.open_dataset(data_files[0])
        return ds


@dataclass
class Satellite:
    name: str
    def __post_init__(self):
        if self.name == "ISCCP":
            self.data_var_name = "clisccp"
            self.tau_name = 'levtau'
            self.pres_name = 'levpc'
        elif self.name == "MODIS":
            self.tau_name = "tau"
            self.pres_name = "pres"
            self.data_var_name = "clmodis"
        elif self.name == "MISR":
            self.tau_name = "tau"
            self.pres_name = "cth"
            self.data_var_name = 'clmisr'
    
    def load_climo(self):
        """Method to load the climo file.
        """
        if self.name == "ISCCP":
            self.time_range_str = "2000-2016"
            return xr.open_dataset("/Volumes/Drudonna/ISCCPH/processed/fv09/ISCCP-Basic.HGG.GLOBAL.10KM.nan_climo.2000-2016.nc")
        elif self.name == "MODIS":
            self.time_range_str = "2002-2022"
            ds = xr.open_dataset("/Volumes/Drudonna/MODIS/MCD06COSP_M3_MODIS.062.nan_climo.2002-2022.nc")
            ds = ds.sel({self.tau_name: slice(0.3,None)})
            return ds
        elif self.name == "MISR":
            self.time_range_str = "2001-2020"
            ds = xr.open_dataset(f"/Volumes/Drudonna/MISR/misr_L3_V7/remapped/clmisr_nan_climo_obs4MIPs_MISR_V7_{self.time_range_str}.nc")
            ds = ds.sel({self.tau_name: slice(0.3,None)})
            return ds
        else:
            raise NotImplementedError(f"Sorry, no support for satellite called {self.name}")

    def load_monthly(self):
        """Method to load the monthly (remapped) satellite data.
        """
        if self.name == "ISCCP":
            self.data_var_name = 'n_pctaudist'
            ds = xr.open_mfdataset("/Volumes/Drudonna/ISCCPH/remapped/ISCCP-Basic.HGG.GLOBAL.10KM.*.nc")
            self.time_range_str = f"{ds.time.dt.year.min().item()}-{ds.time.dt.year.max().item()}"
            return ds
        elif self.name == "MODIS":
            self.data_var_name = 'CLMODIS'
            ds = xr.open_mfdataset("/Volumes/Drudonna/MODIS/processed/remapped/fv09/MCD06COSP_M3_MODIS.*.nc")
            self.time_range_str = f"{ds.time.dt.year.min().item()}-{ds.time.dt.year.max().item()}"
            ds = ds.sel({self.tau_name: slice(0.3,None)})
            return ds
        elif self.name == "MISR":
            self.data_var_name = 'clMISR'
            ds = xr.open_mfdataset("/Volumes/Drudonna/MISR/misr_L3_V7/remapped/clMISR_obs4MIPs_MISR_V7_*.nc")
            self.time_range_str = f"{ds.time.dt.year.min().item()}-{ds.time.dt.year.max().item()}"
            ds = ds.sel({self.tau_name: slice(0.3,None)})
            return ds
        else:
            raise NotImplementedError(f"Sorry, no support for satellite called {self.name}")