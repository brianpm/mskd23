from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import xarray as xr


@dataclass
class Regions:
    name: str
    latitudes: slice
    label: str

    def region_definition(self):
        """Make a string that specified the latitude bounds."""
        val1 = self.latitudes.start
        if val1 > 0:
            hem1 = "N"
        elif val1 < 0:
            hem1 = "S"
        else:
            hem1 = ""
        val2 = self.latitudes.stop
        if val2 > 0:
            hem2 = "N"
        elif val2 < 0:
            hem2 = "S"
        else:
            hem2 = ""
        if np.absolute(val1) > np.absolute(val2):
            return f"{np.absolute(val2)}{hem2}-{np.absolute(val1)}{hem1}"
        else:
            return f"{np.absolute(val1)}{hem1}-{np.absolute(val2)}{hem2}"


def fig_output(figobj, filnam, clobber_fig=False):
    """Check whether to overwrite figure."""
    if not isinstance(filnam, Path):
        assert isinstance(filnam, str)
        filnam = Path(filnam)
    if filnam.is_file():
        print(f"File already exists ... specify whether to overwrite: {clobber_fig}.")
        if clobber_fig:
            figobj.savefig(filnam, dpi=172, bbox_inches="tight")
        else:
            print("Did not save output file.")
    else:
        figobj.savefig(filnam, dpi=172, bbox_inches="tight")


def plot_lines_by_omega(
    binned,
    bincenter,
    colorkey,
    regionspec,
    fig=None,
    ax=None,
    xvarname=None,
    yvarname=None,
    **kwargs
):
    """
    binned -> dict of regions, which are dict of x(bins)
           -> entries need to be labeled by regions in regionspec,
              -> and those are then dict of models/obs
    nregions = len(regions) = len(binned) ?
    bctr : bin centers
    colorkey : dict of colors based on label
    regionspec : dict of instances of Region
    """
    nregions = len(regionspec)
    if (fig is None) and (ax is None):
        fig, ax = plt.subplots(
            figsize=(6.25, 5), nrows=nregions, sharex=True, constrained_layout=True
        )
        a = ax.ravel()

    for i, r in enumerate(regionspec):
        for j, b in enumerate(binned[r]):
            if isinstance(binned[r][b], xr.Dataset):
                if yvarname in binned[r][b]:
                    bdata = binned[r][b][yvarname]
                else:
                    continue
            else:
                # assume array is correct
                bdata = binned[r][b]
            a[i].plot(
                bincenter, bdata, label=b, color=colorkey[b], linewidth=3, **kwargs
            )
            # a[i].fill_between(bctr, binned[r][b]-binned_std[r][b], binned[r][b]+binned_std[r][b], alpha=0.5, color=colorkey[b], clip_on=False)
        if r == "socean":
            a[i].set_title(
                f"{regionspec[r].label} ({regionspec[r].region_definition()})", loc="left"
            )
        else:
            a[i].set_title(regionspec[r].label, loc="left")

    if xvarname is not None:
        ax[-1].set_xlabel(xvarname, fontsize=14)
    else:
        ax[-1].set_xlabel("$\omega_{500}$ [hPa d$^{-1}$]", fontsize=14)
        [a.set_xlim([-100, 100]) for a in ax]
        [a.set_xticks([-90, -60, -30, 0, 30, 60, 90]) for a in ax]
        [a.spines.bottom.set_bounds((-90, 90)) for a in ax]
    if yvarname is not None:
        if yvarname == "relative_entropy":
            [a.set_ylabel("$\mathcal{D}_{\mathrm{KL}}$", fontsize=14) for a in ax]
        elif yvarname == "EMD":
            [a.set_ylabel("EMD", fontsize=14) for a in ax]
            [a.set_ylim([0, 1.5]) for a in ax]
            [a.set_yticks(np.arange(0, 1.5, 0.5)) for a in ax]
            [a.spines.left.set_bounds((0, 1.5)) for a in ax]
        else:
            [a.set_ylabel(yvarname, fontsize=14) for a in ax]
    ax[0].legend(ncol=2)
    [a.spines["top"].set_visible(False) for a in ax]
    [a.spines["right"].set_visible(False) for a in ax]
    return fig, ax
