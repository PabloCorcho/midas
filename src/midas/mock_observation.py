#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 13:20:54 2021

@author: pablo
"""
from astropy import units as u
from .smoothing_kernel import CubicSplineKernel, GaussianKernel
from .pseudoRT_model import RTModel
from ppxf.ppxf_util import gaussian_filter1d
import numpy as np
from . import cosmology


class Observation(object):
    """todo."""

    def __init__(self, SSP, Instrument, Galaxy, kernel_args=None):
        print('-' * 50 + '\n [OBSERVATION]  Initialising observation\n'
              + '-' * 50)
        # Instrument that will observe the galaxy
        self.instrument = Instrument
        # Simple Stellar Populations Model for generating synthetic SEDs
        self.SSP = SSP
        self.prepare_SSP()
        # Galaxy to be observed
        self.galaxy = Galaxy
        # Synthetic cube
        self.cube = np.zeros((self.instrument.wavelength.size,
                              self.instrument.det_x_n_bins,
                              self.instrument.det_y_n_bins
                              )
                             ) * (self.SSP.L_lambda[0, 0].unit * u.Msun)
        self.cube_extinction = np.zeros_like(self.cube)
        # Physical data where all the maps (kinematics, SFH) will be stored
        self.phys_data = {}

    def prepare_SSP(self):
        """Interpolate SSP models to Instrumental resolution and apply LSF convolution."""
        # This method is extremelly usefull in order to speed up the computation time.
        print(' [Observation]  Interpolating SSP models to instrumental resolution')
        self.SSP.cut_models(self.instrument.wavelength_edges.min() * .9, self.instrument.wavelength_edges.max() * 1.1)
        lsf_sigma = np.interp(self.SSP.wavelength.value, self.instrument.wl_lsf, self.instrument.lsf)
        self.SSP.convolve_sed(profile=gaussian_filter1d,
                              **dict(sig=lsf_sigma / np.median(np.diff(self.SSP.wavelength.value))))
        self.SSP.interpolate_sed(self.instrument.wavelength_edges)

    def compute_los_emission(self, stellar=True, nebular=False,
                             dust_extinction=None, kinematics=False):
        """Compute the emission along the LOS."""
        if stellar:
            ssp_metallicities = self.SSP.metallicities
            ssp_ages = self.SSP.ages
            ssp_wave = self.SSP.wavelength
            n_stellar_part = self.galaxy.stars['Masses'].size
            # Store the stellar masses to provide the SFH
            self.phys_data['sfh'] = np.zeros((self.cube.shape[1],
                                              self.cube.shape[2],
                                              ssp_ages.size)) * u.Msun
            self.phys_data['tot_stellar_mass'] = np.zeros((self.cube.shape[1],
                                                           self.cube.shape[2]))
            # Store stellar kinematics
            self.phys_data['stellar_kin_mass_weight'] = np.zeros(
                (self.cube.shape[1],
                 self.cube.shape[2])
                )
            # Initialise pseudo radiative transfer element
            self.rtmodel = RTModel(wave=ssp_wave,
                                   redshift=self.instrument.redshift,
                                   galaxy=self.galaxy)

            print(' [Observation] Computing stellar spectra for'
                  + ' {} star particles'.format(n_stellar_part))
            part_i = 0
            while part_i < n_stellar_part:
                # if part_i > 50:
                #    break
                print("\r Particle --> {}, Completion: {:2.2f} %".format(
                    part_i, part_i/n_stellar_part * 100), end='', flush=True)
                # Particle data
                mass, age, metals = (
                    self.galaxy.stars['GFM_InitialMass'][part_i] * 1e10 * u.Msun,
                    self.galaxy.stars['ages'][part_i] * u.Gyr,
                    self.galaxy.stars['GFM_Metallicity'][part_i]
                    )
                x_pos, y_pos, z_pos = (
                    self.galaxy.stars['ProjCoordinates'][:, part_i])
                vel_z = self.galaxy.stars['ProjVelocities'][2, part_i]
                # SSP SED
                sed = self.SSP.compute_burstSED(age, metals)
                sed *= mass
                # Dust extinction
                # NH, Z = self.rtmodel.compute_nh_column_density(
                #     x_pos, y_pos, z_pos)
                # ext = self.rtmodel.compute_extinction(
                #     Z_gas=Z, N_hyldrogen=NH)
                # sed_extinction = sed * ext * u.angstrom / u.angstrom
                # Kernel Smoothing matrix
                r = np.sqrt((self.instrument.X_kpc - x_pos)**2
                             + (self.instrument.Y_kpc - y_pos)**2)
                ker = self.galaxy.kernel.kernel(r).T
                # (blue/red)shifting spectra
                redshift = vel_z / 3e5
                wave = ssp_wave * (1 + redshift)
                # Store physical properties
                self.phys_data['stellar_kin_mass_weight'] += (
                    vel_z * mass.value * ker)
                self.phys_data['tot_stellar_mass'] += mass.value * ker
                # Interpolation to instrumental resolution
                cumulative = np.cumsum(sed * np.diff(wave)[0])
                sed = np.interp(self.instrument.wavelength_edges, wave,
                                cumulative)
                sed = np.diff(sed) / (self.instrument.delta_wave * u.angstrom)

                # cumulative_extinction = cumsum(sed_extinction
                #                                * diff(wave)[0])
                # sed_extinction = interp(self.instrument.wavelength_edges, wave,
                #                         cumulative_extinction)
                # sed_extinction = diff(sed_extinction) / (
                #     self.instrument.delta_wave * u.angstrom)
                # Final kernel-weighted contribution to the cube
                self.cube += (sed[:, np.newaxis, np.newaxis].value
                              * ker[np.newaxis, :, :]) * sed.unit
                # self.cube_extinction += (
                #     sed_extinction[:, newaxis, newaxis].value
                #     * ker[newaxis, :, :]) * sed.unit

                part_i += 1
                # if part_i > 100:
                #     break
        self.phys_data['stellar_kin_mass_weight'] /= (self.phys_data['tot_stellar_mass'])
        print('\n [Observation] Stellar spectral computed successfully')
        self.cube = self.cube.to(u.erg/u.s/u.angstrom)
        self.cube_extinction = self.cube_extinction.to(u.erg/u.s/u.angstrom)
        # Converting luminosity to observed fluxes
        self.luminosity_to_flux()

    def luminosity_to_flux(self):
        """todo."""
        print(' [Observation] Converting luminosities to observed fluxes for'
              + ' a galaxy located at z={:.4f}'.format(
                  self.instrument.redshift))
        L_dist = cosmology.cosmo.luminosity_distance(
            self.instrument.redshift).to(u.cm).value
        self.cube = self.cube / (4 * np.pi * L_dist**2)
        self.cube_extinction = self.cube_extinction / (4 * np.pi * L_dist**2)

    def add_noise(self, Noise):
        """todo."""
        pass

# Mr Krtxo
