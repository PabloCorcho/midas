from astropy import units as u
from scipy.stats import binned_statistic_2d
import os
import numpy as np
import yaml
from . import cosmology

class Instrument(object):
    pass

class IFU(Instrument):
    spectral_axis = True
    spatial_axis = True


class WEAVE_Instrument(IFU):
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
        self.wavelength = np.arange(self.wave_init, self.wave_end,
                                 self.delta_wave) * u.angstrom
        # Wavelenght array for blue and red arms (used for saving each cube separately)
        self.wavelength_blue = np.arange(
            self.wave_init,
            self.wave_init + self.delta_wave * self.blue_arm_npix,
            self.delta_wave) * u.angstrom
        self.wave_blue_pos = np.where(
            self.wavelength <= self.wavelength_blue[-1])[0]
        self.wavelength_red = np.arange(
            self.wave_end - self.delta_wave * self.red_arm_npix, self.wave_end,
            self.delta_wave) * u.angstrom
        self.wave_red_pos = np.where(self.wavelength >= self.wavelength_red[0])[0]
        self.wavelength_edges = np.arange(self.wave_init - self.delta_wave / 2,
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
        self.det_x_bin_edges_kpc = np.arange(-self.x_fov_kpc.value/2,
                                             self.x_fov_kpc.value/2
                                             + self.pixel_size_kpc.value/2,
                                             self.pixel_size_kpc.value)
        self.det_x_bins_kpc = (self.det_x_bin_edges_kpc[:-1]
                               + self.det_x_bin_edges_kpc[1:]) / 2
        self.det_y_bin_edges_kpc = np.arange(-self.y_fov_kpc.value/2,
                                             self.y_fov_kpc.value/2
                                             + self.pixel_size_kpc.value/2,
                                             self.pixel_size_kpc.value)
        self.det_y_bins_kpc = (self.det_y_bin_edges_kpc[:-1]
                               + self.det_y_bin_edges_kpc[1:]) / 2
        # Create a grid of positions
        self.X_kpc, self.Y_kpc = np.meshgrid(self.det_x_bins_kpc, self.det_y_bins_kpc)

    def get_pixel_physical_size(self):
        """todo."""
        ang_dist = cosmology.cosmo.angular_diameter_distance(self.redshift
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
        x_out = np.where(
            (self.binnumb[0, :] == self.det_x_n_bins + 1))[0]
        y_out = np.where(
            (self.binnumb[1, :] == self.det_y_n_bins + 1))[0]
        self.binnumb[0, x_out] = 0
        self.binnumb[1, y_out] = 0

# Mr Krtxo \(ﾟ▽ﾟ)/