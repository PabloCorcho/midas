from abc import ABC, abstractmethod
from midas.utils import gaussian1d_conv
import numpy as np

class EmissionGridModel(ABC):
    """TODO..."""

    @abstractmethod
    def get_spectra(self):
        raise NotImplementedError("Method not implemented on child class")
    
    @abstractmethod
    def load_grid(self):
        raise NotImplementedError("Method not implemented on child class")
    
    def cut_wavelength(self, wl_min, wl_max):
        mask = (self.wavelength >= wl_min) & (self.wavelength <= wl_max)
        if not mask.any():
            raise ValueError("Empty wavelength range")
        else:
            self.wavelength = self.wavelength[mask]
            self.sed_grid = self.sed_grid[mask]


class BidimensionalGridMixin:

    def interpolate_sed(self, new_wl_edges):
        """Flux-conserving interpolation.

        params
        -----
        - new_wl_edges: bin edges of the new interpolated points.
        """
        new_wl = (new_wl_edges[1:] + new_wl_edges[:-1]) / 2
        dwl = np.diff(new_wl_edges)
        ori_dwl = np.hstack(
            (np.diff(self.wavelength),
             self.wavelength[-1] - self.wavelength[-2]))
        print(' Interpolating SED Grid')
        new_sed_grid = np.empty(
            shape=(new_wl.size, *self.sed_grid.shape[1:]),
            dtype=float)
        for i in range(self.sed_grid.shape[1]):
            for j in range(self.sed_grid.shape[2]):
                f = np.interp(new_wl_edges, self.wavelength,
                              np.cumsum(self.sed_grid[:, i, j] * ori_dwl))
                new_flux = np.diff(f) / dwl
                new_sed_grid[:, i, j] = new_flux
        self.sed_grid = new_sed_grid
        self.wavelength = new_wl

    def convolve_sed(self, profile=gaussian1d_conv,
                     **profile_params):
        """Convolve the SSP spectra with a given LSF."""
        print(' [SSP] Convolving SSP SEDs')
        for i in range(self.sed_grid.shape[1]):
            for j in range(self.sed_grid.shape[2]):
                self.sed_grid[:, i, j] = profile(self.sed_grid[:, i, j],
                                              **profile_params)

    def get_mass_lum_ratio(self, wl_range):
        """Compute the mass-to-light ratio within a giveng wavelength range."""
        pts = np.where((self.wavelength >= wl_range[0]) &
                       (self.wavelength <= wl_range[1]))[0]
        self.mass_to_lum = np.empty((self.metallicities.size,
                                     self.ages.size)
                                    )
        for i in range(self.metallicities.size):
            for j in range(self.ages.size):
                self.mass_to_lum[i, j] = 1/np.mean(
                    self.sed_grid[pts, i, j])
