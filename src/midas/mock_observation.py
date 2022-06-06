#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 13:20:54 2021

@author: pablo
"""

from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy.io import fits
from scipy.stats import binned_statistic_2d
from smoothing_kernel import CubicSplineKernel, GaussianKernel
from pseudoRT_model import RTModel
from ppxf.ppxf_util import log_rebin, gaussian_filter1d
import os
import numpy as np
from numpy import (
    array, logspace, linspace, arange, where, zeros, zeros_like, float32, interp,
    newaxis, sqrt, sum, cumsum, diff, mean, median, cross, dot, sin, cos,
    arccos, meshgrid, clip, linalg, pi, random, savetxt)
import yaml


# Cosmological model used for computing angular diameter distance...
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# Mapping between scale factor H(t)/H0 to look-back times (stellar ages)
scale_f = array(list(map(cosmo.scale_factor, logspace(-5, 2, 100))))
age_f = array(
    [cosmo.lookback_time(z_i).value for z_i in logspace(-5, 2, 100)])


class Instrument(object):
    pass


class WEAVE_Instrument(Instrument):
    """
    This class represents the WEAVE LIFU instrument.
    """
    configfile = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'cube_templates', 'WEAVE', 'weave_instrument.yml')
    wl_lsf, lsf = np.loadtxt(os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'cube_templates', 'WEAVE', 'weave_lsf'))

    def __init__(self, **kwargs):
        print('-' * 50
              + '\n [INSTRUMENT] Initialising instrument: WEAVE LIFU\n'
              + '-' * 50)
        print('\n [INSTRUMENT] Loading default parameters for WEAVE LIFU\n -- configfile: '
              + self.configfile)
        with open(self.configfile, 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
            self.field_of_view = tuple(config['fov']) * u.arcsec  # arcsec
            self.delta_wave = config['delta_wave']
            self.wave_init = config['lambda_init']
            self.wave_end = config['lambda_end']
            self.blue_arm_npix = config['blue_npix']
            self.red_arm_npix = config['red_npix']
            self.pixel_size = config['pixel_size'] * u.arcsec

        # Build the wavelenth array
        self.wavelength = arange(self.wave_init, self.wave_end,
                                 self.delta_wave) * u.angstrom
        # Wavelenght array for blue and red arms (used for saving each cube separately)
        self.wavelength_blue = arange(
            self.wave_init,
            self.wave_init + self.delta_wave * self.blue_arm_npix,
            self.delta_wave) * u.angstrom
        self.wave_blue_pos = where(
            self.wavelength <= self.wavelength_blue[-1])[0]
        self.wavelength_red = arange(
            self.wave_end - self.delta_wave * self.red_arm_npix, self.wave_end,
            self.delta_wave) * u.angstrom
        self.wave_red_pos = where(self.wavelength >= self.wavelength_red[0])[0]
        self.wavelength_edges = arange(self.wave_init - self.delta_wave / 2,
                                       self.wave_end + self.delta_wave / 2,
                                       self.delta_wave) * u.angstrom
        self.wavelength_arms = {'BLUE': self.wavelength_blue,
                                'RED': self.wavelength_red}

        print('\n  · FoV: ({:.1f}, {:.1f})'.format(self.field_of_view[0],
                                                   self.field_of_view[1])
              + '\n  · Pixel size: {:.2f}'.format(self.pixel_size)
              + '\n  · Delta(Lambda) (angstrom): {:.1f}'.format(self.delta_wave)
              + '\n  · Wavelength range (angstrom): {:.2f} - {:.2f}'.format(
                  self.wave_init, self.wave_end)
              + '\n    -> Blue arm: {:.2f} - {:.2f} ({} pixels)'.format(
            self.wavelength_blue[0], self.wavelength_blue[-1],
            self.wavelength_blue.size)
              + '\n    -> Red arm: {:.2f} - {:.2f} ({} pixels)'.format(
            self.wavelength_red[0], self.wavelength_red[-1],
            self.wavelength_red.size))

        self.redshift = kwargs.get('z', 0.05)
        self.get_pixel_physical_size()
        self.detector_bins()
        print('\n [INSTRUMENT] specific parameters for observation'
              + '\n  · Source redshift: {:.4f}'.format(self.redshift)
              + '\n  · Pixel physical size: {:.4f}'.format(self.pixel_size_kpc)
              )

    def detector_bins(self):
        """todo."""
        self.det_x_n_bins = int(self.field_of_view[0].value
                                / self.pixel_size.value)
        self.det_y_n_bins = int(self.field_of_view[1].value
                                / self.pixel_size.value)
        self.x_fov_kpc = self.det_x_n_bins * self.pixel_size_kpc
        self.y_fov_kpc = self.det_y_n_bins * self.pixel_size_kpc
        self.det_x_bin_edges_kpc = arange(-self.x_fov_kpc.value/2,
                                          self.x_fov_kpc.value/2
                                          + self.pixel_size_kpc.value/2,
                                          self.pixel_size_kpc.value)
        self.det_x_bins_kpc = (self.det_x_bin_edges_kpc[:-1]
                               + self.det_x_bin_edges_kpc[1:]) / 2
        self.det_y_bin_edges_kpc = arange(-self.y_fov_kpc.value/2,
                                          self.y_fov_kpc.value/2
                                          + self.pixel_size_kpc.value/2,
                                          self.pixel_size_kpc.value)
        self.det_y_bins_kpc = (self.det_y_bin_edges_kpc[:-1]
                               + self.det_y_bin_edges_kpc[1:]) / 2

    def get_pixel_physical_size(self):
        """todo."""
        ang_dist = cosmo.angular_diameter_distance(self.redshift
                                                   ).to('kpc')
        self.pixel_size_kpc = (ang_dist.value
                               * self.pixel_size.to('radian').value) * u.kpc

    def bin_particles(self):
        """todo."""
        self.stat, _, _, self.binnumb = binned_statistic_2d(
            x=self.X, y=self.Y, values=None,
            statistic='count',
            bins=[self.det_x_bin_edges_kpc,
                  self.det_y_bin_edges_kpc],
            expand_binnumbers=True)
        x_out = where(
            (self.binnumb[0, :] == self.det_x_n_bins + 1))[0]
        y_out = where(
            (self.binnumb[1, :] == self.det_y_n_bins + 1))[0]
        self.binnumb[0, x_out] = 0
        self.binnumb[1, y_out] = 0


class Galaxy(object):
    """
    Def.

    This class represents a galaxy from IllustrisTNG simulations and contains
    different elements such as stellar and gas particles.
    """

    def __init__(self, **kwargs):
        self.name = kwargs.get('name', "")
        self.stars_params = kwargs.get('stars', None)
        self.build_stars()
        self.gas_params = kwargs.get('gas', None)
        self.build_gas()
        self.spin = kwargs.get('gal_spin', zeros(3))  # kpc/(km/s)
        self.velocity = kwargs.get('gal_vel', zeros(3))  # kpc/(km/s)
        self.position = kwargs.get('gal_pos', zeros(3))  # kpc/(km/s)

    def build_stars(self):
        """todo."""
        self.stars = {}
        if self.stars_params is not None:
            for key in list(self.stars_params.keys()):
                self.stars[key] = self.stars_params[key][()]
        if 'ages' not in self.stars.keys():
            self.stars['ages'] = interp(
                self.stars['GFM_StellarFormationTime'],
                scale_f[::-1], age_f[::-1])

    def build_gas(self):
        """todo."""
        self.gas = {}
        if self.gas_params is not None:
            for key in list(self.gas_params.keys()):
                self.gas[key] = self.gas_params[key][()]

    def set_to_galaxy_rest_frame(self):
        """todo."""
        if self.stars_params is not None:
            self.stars['Velocities'] += -self.velocity[newaxis, :]
            self.stars['Coordinates'] += -self.position[newaxis, :]
        if self.gas_params is not None:
            self.gas['Velocities'] += -self.velocity[newaxis, :]
            self.gas['Coordinates'] += -self.position[newaxis, :]
        self.velocity = zeros(3)
        self.position = zeros(3)

    def proyect_galaxy(self, orthogonal_vector):
        """todo."""
        norm = orthogonal_vector / sqrt(sum(orthogonal_vector**2))
        self.proyection_vector = norm
        b = cross(array([0, 0, 1]), norm)
        b /= sqrt(sum(b**2))
        theta = arccos(dot(array([0, 0, 1]), norm))
        q0, q1, q2, q3 = (cos(theta/2), sin(theta/2)*b[0],
                          sin(theta/2)*b[1], sin(theta/2)*b[2])
        # Quartenion matrix
        Q = zeros((3, 3))
        Q[0, 0] = q0**2 + q1**2 - q2**2 - q3**2
        Q[1, 1] = q0**2 - q1**2 + q2**2 - q3**2
        Q[2, 2] = q0**2 - q1**2 - q2**2 + q3**2
        Q[0, 1] = 2*(q1*q2 - q0*q3)
        Q[0, 2] = 2*(q1*q3 + q0*q2)
        Q[1, 0] = 2*(q1*q2 + q0*q3)
        Q[1, 2] = 2*(q2*q3 - q0*q1)
        Q[2, 0] = 2*(q1*q3 - q0*q2)
        Q[2, 1] = 2*(q3*q2 + q0*q1)
        # New basis
        u, v, w = (dot(Q, array([1, 0, 0])),
                   dot(Q, array([0, 1, 0])),
                   dot(Q, array([0, 0, 1])))
        # Change of basis matrix
        W = array([u, v, w])
        inv_W = linalg.inv(W)
        if self.stars_params is not None:
            self.stars['ProjCoordinates'] = array([
                sum(self.stars['Coordinates'] * u[newaxis, :], axis=1),
                sum(self.stars['Coordinates'] * v[newaxis, :], axis=1),
                sum(self.stars['Coordinates'] * w[newaxis, :], axis=1)
                ])
            self.stars['ProjVelocities'] = array([
                sum(self.stars['Velocities'] * inv_W[0, :][newaxis, :],
                    axis=1),
                sum(self.stars['Velocities'] * inv_W[1, :][newaxis, :],
                    axis=1),
                sum(self.stars['Velocities'] * inv_W[2, :][newaxis, :],
                    axis=1)
                ])
        if self.gas_params is not None:
            self.gas['ProjCoordinates'] = array([
                sum(self.gas['Coordinates'] * u[newaxis, :], axis=1),
                sum(self.gas['Coordinates'] * v[newaxis, :], axis=1),
                sum(self.gas['Coordinates'] * w[newaxis, :], axis=1)
                ])
            self.gas['ProjVelocities'] = array([
                sum(self.gas['Velocities'] * inv_W[0, :][newaxis, :],
                    axis=1),
                sum(self.gas['Velocities'] * inv_W[1, :][newaxis, :],
                    axis=1),
                sum(self.gas['Velocities'] * inv_W[2, :][newaxis, :],
                    axis=1)
                ])


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
        self.SSP.interpolate_sed(self.instrument.wavelength_edges)
        self.lsf_sigma = np.interp(self.SSP.wavelength.value, self.instrument.wl_lsf, self.instrument.lsf)
        self.SSP.convolve_sed(profile=gaussian_filter1d,
                              **dict(sig=self.lsf_sigma / np.median(np.diff(self.SSP.wavelength.value))))
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
                z_idx = ssp_metallicities.searchsorted(metals)
                z_idx = clip(z_idx, 0, ssp_metallicities.size - 1)
                age_idx = ssp_ages.searchsorted(age.to('yr'))
                age_idx = clip(age_idx, 0, ssp_ages.size - 1)
                sed = self.SSP.L_lambda[z_idx, age_idx].flux.copy()
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
                self.star_formation_history[:, :, age_idx] += mass * ker
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
        L_dist = cosmo.luminosity_distance(
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
