#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 09:06:17 2022

@author: pablo
"""
import numpy as np
from extinction import ccm89
from .cosmology import cosmo

# Constants
H_mass = 1.6735575e-27  # Hydrogen mass in kg
M_sun = 1.989e30  # Sun mass in kg
H_to_Sun = M_sun / H_mass
kpc_to_cm = 3.086e21


class RTModel(object):
    """
    Pseudo-radiative transfer model.

    Refs used in the model:
        - Calzetti et al 1994
        https://ui.adsabs.harvard.edu/abs/1994ApJ...429..582C/abstract
        - Cardelli et al 1989
        - Nelson et al 2018
        https://arxiv.org/pdf/1707.03395.pdf
    """

    def __init__(self, wave, redshift, galaxy, grid_resolution_kpc=1):
        print('> Initialising pseudo RT model')
        print(' [RT] -- Params:\n    · grid resolution: {} kpc'
              .format(grid_resolution_kpc))
        self.grid_resolution_kpc = grid_resolution_kpc
        self.wavelength = np.array(wave, dtype=np.float64)
        self.redshift = redshift
        self.extinction_curve = ccm89(self.wavelength, a_v=1.0, r_v=3.1)
        # Exponent for dust-to-gas metallicity relation
        self.gamma = np.zeros_like(self.wavelength)
        self.gamma[:] = 1.6
        self.gamma[self.wavelength < 2000] = 1.35

        self.N_hydrogen_0 = 2.1e21  # cm^-2
        self.Z_sun = 0.0127

        self.compute_phase_function()
        # Gas particles properties
        if len(galaxy.gas) > 0:
            self.gas_cells_volume = (
                galaxy.gas['Masses'] / galaxy.gas['Density']
                ) / (cosmo.H0.value/100)**3  # cKpc^3
            self.gas_met = galaxy.gas['GFM_Metallicity']
            self.gas_pos = galaxy.gas['ProjCoordinates']
            # -- Expressed in 1e10 Msun
            self.gas_mass_hydrogen = (galaxy.gas['GFM_Metals'][:, 0]
                                      * galaxy.gas['NeutralHydrogenAbundance']
                                      * galaxy.gas['Masses'])
        else:
            print(
                ' [RT] · No gas particles to compute extinction were found\n')
            self.gas_cells_volume = np.array([0])
            self.gas_met = np.array([0])
            self.gas_pos = np.array([[0], [0], [0]])
            self.gas_mass_hydrogen = np.array([0])
        # Stellar particles and grid
        self.stars_pos = galaxy.stars['ProjCoordinates']

        self.min_x, self.max_x = (np.min(self.stars_pos[0, :]),
                                  np.max(self.stars_pos[0, :]))
        self.min_y, self.max_y = (np.min(self.stars_pos[1, :]),
                                  np.max(self.stars_pos[1, :]))
        self.x_bin_edges = np.arange(self.min_x - grid_resolution_kpc/2,
                                     self.max_x + grid_resolution_kpc/2,
                                     grid_resolution_kpc)
        self.y_bin_edges = np.arange(self.min_y - grid_resolution_kpc/2,
                                     self.max_y + grid_resolution_kpc/2,
                                     grid_resolution_kpc)
        self.create_grid()

    def compute_phase_function(self):
        """..."""
        self.albedo = np.zeros_like(self.wavelength)
        y = np.log10(self.wavelength)
        # UV
        regime = np.where((self.wavelength > 1000) & (self.wavelength < 3460)
                          )[0]
        self.albedo[regime] = 0.43 + 0.366 * (
            1 - np.exp(-(y[regime] - 3)**2/0.2))
        # Optical
        regime = np.where((self.wavelength > 3460) & (self.wavelength < 8000)
                          )[0]
        self.albedo[regime] = - 0.48 * y[regime] + 2.41

        self.anisotropy_scatter = 1 - 0.561 * np.exp(
            - np.abs(y - 3.3112)**2.2 / 0.17)
        self.phase_function = (
            self.anisotropy_scatter * (1 - self.albedo)**0.5
            + (1 - self.anisotropy_scatter) * (1 - self.albedo))

    def compute_tau_abs(self, Z_gas, N_hydrogen):
        """..."""
        tau_abs = (self.extinction_curve * (1 + self.redshift)**-0.5
                   * (Z_gas/self.Z_sun)**self.gamma
                   * N_hydrogen/self.N_hydrogen_0)
        return tau_abs

    def compute_tau(self, **tau_abs_args):
        """..."""
        tau_tot = self.compute_tau_abs(**tau_abs_args) * self.phase_function
        return tau_tot

    def compute_extinction(self, **tau_abs_args):
        """..."""
        tau = self.compute_tau(**tau_abs_args)
        if tau.max() == 0:
            return np.ones_like(self.wavelength), np.zeros_like(self.wavelength)
        else:
            extinction = 1/tau * (1 - np.exp(-tau))
            return extinction, tau

    def create_grid(self):
        """..."""
        print(' [RT] --> Creating grid with ({},{}) elements'.format(
            self.x_bin_edges.size, self.y_bin_edges.size))
        self.x_gas_binnumb = np.digitize(self.gas_pos[0, :], self.x_bin_edges,
                                         right=False)
        self.y_gas_binnumb = np.digitize(self.gas_pos[1, :], self.y_bin_edges,
                                         right=False)

    def compute_nh_column_density(self, star_x, star_y, star_z):
        """..."""
        x_bin = np.digitize(star_x, self.x_bin_edges, right=False)
        y_bin = np.digitize(star_y, self.y_bin_edges, right=False)

        los_gas_idx = np.where((self.x_gas_binnumb == x_bin)
                               & (self.y_gas_binnumb == y_bin))[0]

        los_gas_z_pos = self.gas_pos[2, los_gas_idx]
        gas_argsort_idx = np.argsort(los_gas_z_pos)
        star_z_pos_idx = np.searchsorted(los_gas_z_pos[gas_argsort_idx],
                                         star_z)
        star_z_pos_idx = np.clip(star_z_pos_idx, a_min=0,
                                 a_max=los_gas_z_pos.size)
        # Hydrogen mass
        H_masses = self.gas_mass_hydrogen[los_gas_idx][gas_argsort_idx
                                                       ][:star_z_pos_idx]
        H_total_mass = np.sum(H_masses)
        if H_total_mass <= 0:
            return 0, 0
        # Mass-weighted metallicity
        cell_met = self.gas_met[los_gas_idx][gas_argsort_idx][:star_z_pos_idx]
        mass_weighted_met = np.sum(cell_met * H_masses) / H_total_mass

        N_hydrogen = H_total_mass * 1e10 * H_to_Sun / (
            self.grid_resolution_kpc * kpc_to_cm)**2
        return N_hydrogen, mass_weighted_met


if __name__ == '__main__':
    from generate_grid import Galaxy
    from matplotlib import pyplot as plt
    import h5py
    from astropy import units as u
    f = h5py.File('tng100_subhalo_175251.hdf5', 'r')
    stars = f['PartType4']
    gas = f['PartType0']
    galaxy = Galaxy(stars=stars,
                    gas=gas,
                    gal_spin=np.array([38.9171, 70.069, -140.988]),
                    gal_vel=np.array([457.701, -275.459, -364.912]),
                    gal_pos=np.array([29338., 1754.63, 73012.9]))
    galaxy.set_to_galaxy_rest_frame()
    galaxy.proyect_galaxy(galaxy.spin)
    # projection_vetor = np.array([-1/galaxy.spin[0], 1/galaxy.spin[1], 0])
    # galaxy.proyect_galaxy(projection_vetor)
    f.close()

    wave = np.linspace(1000, 8000, 1000) * u.angstrom
    model = RTModel(wave=wave, redshift=0.01, galaxy=galaxy)
    # model.compute_extinction(Z_gas=0.02, N_hydrogen=2e22)
    NH, Z = model.compute_nh_column_density(galaxy.stars['ProjCoordinates'][0, 15000],
                                    galaxy.stars['ProjCoordinates'][1, 15000],
                                    galaxy.stars['ProjCoordinates'][2, 15000])

    plt.figure()
    plt.plot(model.compute_extinction(Z_gas=Z, N_hydrogen=NH))