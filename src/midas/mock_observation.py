#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 13:20:54 2021

@author: pablo
"""
from astropy import units as u
from smoothing_kernel import CubicSplineKernel, GaussianKernel
from pseudoRT_model import RTModel
from ppxf.ppxf_util import gaussian_filter1d
import numpy as np
from numpy import zeros, zeros_like, interp, newaxis, sqrt, cumsum, diff, meshgrid, clip, pi, random
import cosmology

class ObserveGalaxy(object):
    """todo."""

    def __init__(self, SSP, Instrument, Galaxy, kernel_args=None):
        print('-' * 50 + '\n [OBSERVATION]  Initialising observation\n'
              + '-' * 50)
        # Simple Stellar Populations Model for generating synthetic SEDs
        self.SSP = SSP
        # Instrument that will observe the galaxy
        self.instrument = Instrument

        # Interpolate SSP models to Instrumental resolution
        print(' [Observation]  Interpolating SSP models to instrumental resolution')
        self.SSP.cut_models(self.instrument.wavelength_edges.min() * .9, self.instrument.wavelength_edges.max() * 1.1)
        self.lsf_sigma = np.interp(self.SSP.wavelength.value, self.instrument.wl_lsf, self.instrument.lsf)
        self.SSP.convolve_sed(profile=gaussian_filter1d,
                              **dict(sig=self.lsf_sigma / np.median(np.diff(self.SSP.wavelength.value))))
        self.SSP.interpolate_sed(self.instrument.wavelength_edges)

        # Galaxy to be observed
        self.galaxy = Galaxy
        # Synthetic cube
        self.cube = zeros((self.instrument.wavelength.size,
                           self.instrument.det_x_n_bins,
                           self.instrument.det_y_n_bins
                           )
                          ) * (self.SSP.L_lambda[0, 0].unit * u.Msun)
        self.cube_extinction = zeros_like(self.cube)
        # Smoothing kernel
        self.X_kpc, self.Y_kpc = meshgrid(self.instrument.det_x_bins_kpc,
                                          self.instrument.det_y_bins_kpc)
        # self.kernel = CubicSplineKernel(dim=2, h=.5)
        self.kernel = GaussianKernel(mean=0, sigma=.3)

    def compute_los_emission(self, stellar=True, nebular=False,
                             dust_extinction=None, kinematics=False):
        """Compute the emission along the LOS."""
        if stellar:
            ssp_metallicities = self.SSP.metallicities
            ssp_ages = self.SSP.ages
            ssp_wave = self.SSP.wavelength
            n_stellar_part = self.galaxy.stars['Masses'].size
            # Store the stellar masses to provide the SFH
            self.star_formation_history = zeros((self.cube.shape[1],
                                                 self.cube.shape[2],
                                                 ssp_ages.size)) * u.Msun
            # Store stellar kinematics
            self.velocity_field = zeros((self.cube.shape[1],
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
                print("\r Particle --> {}, Completion: {:2.2f} %".format(
                    part_i, part_i/n_stellar_part * 100), end='', flush=True)
                mass, age, metals = (
                    self.galaxy.stars['GFM_InitialMass'][part_i] * 1e10 * u.Msun,
                    self.galaxy.stars['ages'][part_i] * u.Gyr,
                    self.galaxy.stars['GFM_Metallicity'][part_i]
                    )
                x_pos, y_pos, z_pos = (
                    self.galaxy.stars['ProjCoordinates'][:, part_i])
                # SSP SED
                sed = self.SSP.compute_burstSED(age, metals)
                sed *= mass
                # Dust extinction
                # NH, Z = self.rtmodel.compute_nh_column_density(
                #     x_pos, y_pos, z_pos)
                # ext = self.rtmodel.compute_extinction(
                #     Z_gas=Z, N_hyldrogen=NH)
                # sed_extinction = sed * ext * u.angstrom / u.angstrom
                # Kernel Smoothing
                r = sqrt((self.X_kpc - x_pos)**2
                         + (self.Y_kpc - y_pos)**2)
                ker = self.kernel.kernel(r).T
                # (blue/red)shifting spectra
                vel_z = self.galaxy.stars['ProjVelocities'][2, part_i]
                redshift = vel_z / 3e5
                wave = ssp_wave * (1 + redshift)
                # Interpolation to instrumental resolution
                cumulative = cumsum(sed * diff(wave)[0])
                sed = interp(self.instrument.wavelength_edges, wave,
                             cumulative)
                sed = diff(sed) / (self.instrument.delta_wave * u.angstrom)

                # cumulative_extinction = cumsum(sed_extinction
                #                                * diff(wave)[0])
                # sed_extinction = interp(self.instrument.wavelength_edges, wave,
                #                         cumulative_extinction)
                # sed_extinction = diff(sed_extinction) / (
                #     self.instrument.delta_wave * u.angstrom)
                # Final kernel-weighted contribution to the cube
                self.cube += (sed[:, newaxis, newaxis].value
                              * ker[newaxis, :, :]) * sed.unit
                # self.cube_extinction += (
                #     sed_extinction[:, newaxis, newaxis].value
                #     * ker[newaxis, :, :]) * sed.unit
                # Star formation History and Stellar Kinematics map
                #self.star_formation_history[:, :, age_idx] += mass * ker
                self.velocity_field += vel_z * mass.value * ker

                part_i += 1
                # if part_i > 100:
                #     break
        print(' [Observation] Stellar spectral computed successfully')
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
        self.cube = self.cube / (4 * pi * L_dist**2)
        self.cube_extinction = self.cube_extinction / (4 * pi * L_dist**2)

    def add_noise(self):
        """todo."""
        self.noise = (2*random.rand(self.cube.size)-1).reshape(
            self.cube.shape
            ) * 1e-17 * (u.erg/u.s/u.angstrom)
        self.cube += self.noise

# Mr Krtxo
