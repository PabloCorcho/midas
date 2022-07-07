#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 13:20:54 2021

@author: pablo
"""
from astropy import units as u
import numpy as np
from scipy.ndimage import gaussian_filter

# from .smoothing_kernel import CubicSplineKernel, GaussianKernel
from .pseudoRT_model import RTModel
from ppxf.ppxf_util import gaussian_filter1d
from . import cosmology
from .utils import fast_interpolation
from .cosmology import cosmo


class Observation(object):
    """todo."""

    def __init__(self, SSP, Instrument, Galaxy):
        self.rtmodel = None
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
                             )
        # Physical data where all the maps (kinematics, SFH) will be stored

    def prepare_SSP(self):
        """Interpolate SSP models to Instrumental resolution and apply LSF convolution."""
        # This method is extremelly usefull in order to speed up the computation time.
        print(' [Observation]  Interpolating SSP models to instrumental resolution')
        self.SSP.cut_models(self.instrument.wavelength_edges.min().value * .9,
                            self.instrument.wavelength_edges.max().value * 1.1)
        lsf_sigma = np.interp(self.SSP.wavelength,
                              self.instrument.wl_lsf,
                              self.instrument.lsf)
        self.SSP.convolve_sed(profile=gaussian_filter1d,
                              **dict(sig=lsf_sigma /
                                     np.median(np.diff(self.SSP.wavelength))))
        self.SSP.interpolate_sed(self.instrument.wavelength_edges.value)

    def compute_los_emission(self, stellar=True, nebular=False,
                             dust_extinction=True, kinematics=True,
                             mass_to_light_wave=[5450, 5550]):
        """Compute the emission along the LOS."""
        if stellar:
            ssp_wave = self.SSP.wavelength
            mlr_mask = (ssp_wave > mass_to_light_wave[0]
                        ) & (ssp_wave > mass_to_light_wave[1])
            n_stellar_part = self.galaxy.stars['Masses'].size

            print(' [Observation] Computing stellar spectra for'
                  + ' {} star particles'.format(n_stellar_part))

            # Store new properties computed on-the-fly
            self.galaxy.stars['xbin'] = - np.ones(n_stellar_part, dtype=int)
            self.galaxy.stars['ybin'] = - np.ones(n_stellar_part, dtype=int)
            self.galaxy.stars['MLR'] = np.full(n_stellar_part,
                                               fill_value=np.nan)
            if dust_extinction:
                lambda_tau = 5500
                self.galaxy.stars['tau_5500'] = np.full(n_stellar_part,
                                                        fill_value=np.nan)
                # Initialise pseudo-radiative transfer module
                self.rtmodel = RTModel(wave=ssp_wave,
                                       redshift=self.instrument.redshift,
                                       galaxy=self.galaxy,
                                       grid_resolution_kpc=0.5)

            for part_i in range(n_stellar_part):
                print("\r Particle --> {}, Completion: {:2.2f} %".format(
                    part_i, part_i/n_stellar_part * 100), end='', flush=True)
                # Particle data
                mass, age, metals = (
                    self.galaxy.stars['GFM_InitialMass'][part_i].copy()
                    * 1e10/cosmo.h,
                    self.galaxy.stars['ages'][part_i].copy() * 1e9,
                    self.galaxy.stars['GFM_Metallicity'][part_i].copy()
                    )
                mass, age, metals, wind = (
                    self.galaxy.stars['GFM_InitialMass'][part_i]
                    * 1e10/cosmo.h,
                    self.galaxy.stars['ages'][part_i] * 1e9,
                    self.galaxy.stars['GFM_Metallicity'][part_i],
                    self.galaxy.stars['wind'][part_i])
                if wind:
                    continue
                x_pos, y_pos, z_pos = (
                    self.galaxy.stars['ProjCoordinates'][:, part_i].copy() / cosmo.h)
                vel_z = self.galaxy.stars['ProjVelocities'][2, part_i].copy()

                xbin = np.digitize(x_pos,
                                   self.instrument.det_x_bin_edges_kpc) - 1
                ybin = np.digitize(y_pos,
                                   self.instrument.det_y_bin_edges_kpc) - 1
                if ((xbin <= 0) | (ybin <= 0) |
                    (xbin >= self.instrument.det_x_n_bins) |
                        (ybin >= self.instrument.det_y_n_bins)):
                    # particle out of FoV
                    continue

                self.galaxy.stars['xbin'][part_i] = xbin
                self.galaxy.stars['ybin'][part_i] = ybin

                particle_pos = xbin, ybin
                # Stellar population synthesis --------------------------------
                sed = self.SSP.compute_burstSED(age, metals)
                self.galaxy.stars['MLR'][part_i] = np.mean(sed[mlr_mask])
                sed *= mass
                # Dust extinction ---------------------------------------------
                if dust_extinction:
                    NH, Z = self.rtmodel.compute_nh_column_density(
                        x_pos, y_pos, z_pos)
                    ext, tau = self.rtmodel.compute_extinction(Z_gas=Z,
                                                               N_hydrogen=NH)
                    tau_5500 = np.interp(lambda_tau, ssp_wave, tau)
                    self.galaxy.stars['tau_5500'][part_i] = tau_5500
                    sed = sed * ext
                # Kinematics --------------------------------------------------
                if kinematics:
                    # (blue/red)shifting spectra
                    redshift = vel_z / 3e5
                    wave = ssp_wave * (1 + redshift)
                # Instrumental resolution -------------------------------------
                sed = fast_interpolation(
                    sed, np.diff(wave)[0], wave,
                    self.instrument.wavelength_edges.value,
                    self.instrument.delta_wave)
                # Cube storage ------------------------------------------------
                self.cube[:, particle_pos[0], particle_pos[1]] += sed
                # if part_i > 100:
                #     break
        print('\n [Observation] LOS emission computed successfully')
        # Converting luminosity to observed fluxes
        self.luminosity_to_flux()

    def luminosity_to_flux(self):
        """todo."""
        print(' [Observation] Converting luminosities to observed fluxes for'
              + ' a galaxy located at z={:.4f}'.format(
                  self.instrument.redshift))
        l_dist = cosmology.cosmo.luminosity_distance(
            self.instrument.redshift).to(u.cm).value
        self.cube = self.cube * u.Lsun.to('erg/s') / (4 * np.pi * l_dist**2)

    def add_noise(self, Noise):
        """todo."""
        pass

    @staticmethod
    def gaussian_smooth(ndimage, sigma):
        """Apply multidimensional gaussian smoothing."""
        smooth = gaussian_filter(input=ndimage, sigma=sigma)
        return smooth
# Mr Krtxo \(ﾟ▽ﾟ)/
