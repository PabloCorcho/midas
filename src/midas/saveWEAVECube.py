#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 08:05:09 2022

@author: pablo
"""
import yaml
from astropy.io import fits
from datetime import date
from numpy import array, float32, ones_like

"""
Example of WEAVE Apertiff collaboration simulated cubes content
Filename: cubes/20170930/stackcube_1004122.fit
No.    Name      Ver    Type      Cards   Dimensions   Format
  0  PRIMARY       1 PrimaryHDU     470   ()      
  1  BLUE_DATA     1 ImageHDU        72   (176, 153, 4825)   float32   
  2  BLUE_IVAR     1 ImageHDU        72   (176, 153, 4825)   float32   
  3  BLUE_DATA_NOSS    1 ImageHDU        72   (176, 153, 4825)   float32   
  4  BLUE_IVAR_NOSS    1 ImageHDU        72   (176, 153, 4825)   float32   
  5  BLUE_SENSFUNC    1 ImageHDU        14   (4825,)   float32   
  6  BLUE_DATA_COLLAPSE3    1 ImageHDU        72   (176, 153)   float32   
  7  BLUE_IVAR_COLLAPSE3    1 ImageHDU        72   (176, 153)   float32   
  
"""
class SaveWEAVECube(object):
    def __init__(self, observation, filename):
        self.basic = None
        self.observation = observation
        self.filename = filename
        self.get_observation_info()

        self.primary_hdr = None
        self.data_hdr = None
        print('[SAVING] -- *** SAVING OBSERVATION ***')

        self.build_primary()
        self.build_data()
    def get_observation_info(self):
        """
        blah...
        :return:
        """
        # Basic stuff common to both arms
        description = {
            'AUTHOR': ('Corcho-Caballero P.', ''),
            'INTRUME': ('Mock-WEAVE', 'Instrument use for observations'),
            'BUNIT': ('10^-16 * erg * s^-1 cm^-2 Angstrom^-1', 'flux unit'),
            'OBJECT': (self.observation.galaxy.name, 'Target object'),
            'FILENAME': (self.filename, 'File name'),
            'MED_VEL': (self.observation.instrument.redshift * 3e5,
                        'systemic velocity km/s'),
            'RMS_VEL': (0.0, 'rms of systemic velocity (km/s)')}

        wcs = {
            'SIMPLE': ('T', 'file conforms to FITS standard'),
            'BITPIX': (32, 'number of bits per data pixel'),
            'NAXIS': (len(self.observation.cube.shape),
                      'number of data axes'),
            'NAXIS1': (self.observation.cube.shape[1],
                       'length of data axis 1'),
            'NAXIS2': (self.observation.cube.shape[2],
                       'length of data axis 1'),
            'NAXIS3': (self.observation.instrument.wavelength_blue.size,
                       'length of data axis 1'),
            'EXTEND': ('T', 'FITS dataset may contain extensions'),
            'ORIG_RA': (0.0, ''),
            'ORIG_DEC': (0.0, ''),
            'CRVAL1': (0.0, 'RA at CRPIX1 in deg'),
            'CRPIX1': (self.observation.cube.shape[1] // 2,
                       'Ref pixel for WCS'),
            'CRVAL2': (0.0, 'DEC at CRPIX2 in deg'),
            'CRPIX2': (self.observation.cube.shape[2] // 2, 'Ref pixel for WCS'),
            # CRVAL3 must be overwritten by each arm configuration
            'CRVAL3': (self.observation.instrument.wave_init,
                       'DEC at CRPIX2 in deg'),
            'CDELT3': (self.observation.instrument.delta_wave, ''),
            'CRPIX3': (1.0, 'Ref pixel for WCS'),
            # 'WCSNAME': ('TELESCOPE', ''), Not necessary when there is only one WCS
            'CTYPE1': ('RA---TAN', 'Variable measured by the WCS'),
            'CUNIT1': ('deg', 'Units'),
            'CTYPE2': ('DEC--TAN', 'Variable measured by the WCS'),
            'CUNIT2': ('deg', 'Units'),
            'CTYPE3': ('AWAV', 'Variable measured by the WCS'),
            'CUNIT3': ('Angstrom', 'Units'),
            'CD1_1': (-self.observation.instrument.pixel_size.to('deg').value,
                      'Pixels in degress for X-axis'),
            'CD1_2': (0.0, ''),
            'CD1_3': (0.0, ''),
            'CD2_1': (0.0, ''),
            'CD2_2': (self.observation.instrument.pixel_size.to('deg').value,
                      'Pixels in degress for Y-axis'),
            'CD2_3': (0.0, ''),
            'CD3_1': (0.0, ''),
            'CD3_2': (0.0, ''),
            'CD3_3': (self.observation.instrument.delta_wave,
                      'Linear dispersion (Angstrom/pixel)'),
        }
        self.basic = {'blue': {**wcs.copy(), **description.copy()},
                      'red': {**wcs.copy(), **description.copy()}}
        # Fill specific field for each arm
        self.basic['blue']['NAXIS3'] = self.observation.instrument.wavelength_blue.size
        self.basic['blue']['CRVAL3'] = self.observation.instrument.wavelength_blue[0].value
        self.basic['red']['NAXIS3'] = self.observation.instrument.wavelength_red.size
        self.basic['red']['CRVAL3'] = self.observation.instrument.wavelength_red[0].value

    def create_arms(self, cube):
        """
        Provide a cube and returns the corresponding arms
        :return: red and blue arms
        """
        blue_cube = cube[self.observation.instrument.wave_blue_pos, :, :]
        red_cube = cube[self.observation.instrument.wave_red_pos, :, :]
        return blue_cube, red_cube

    def build_primary(self):
        with open('/home/pablo/WEAVE-Apertiff/input/weave_specifics/'
                  + 'cube_header/PRIMARY_keys.pkl', 'rb') as f:
            data_primary = pickle.load(f)

        basic = {'SIMU': ('IllustrisTNG',
                          'Simulation suite from which data was taken'),
                 'ID': (self.observation.galaxy.name,
                        'Object ID referring to the simulation'),
                 'AUTHOR1': ('Corcho-Caballero P.', ''),
                 'AUTHOR2': ('pablo.corcho@uam.es', 'contact'),
                 'DATE': (str(date.today()))
                 }
        kernel_args = {
                  'KERNEL': (self.observation.kernel.name,
                             'Kernel used for distributing particles light')}
        for key in self.observation.kernel.kernel_params.keys():
            kernel_args['KER_'+key] = (
                self.observation.kernel.kernel_params[key],
                'Kernel parameter')

        primary_keys = {**basic, **kernel_args, **data_primary}

        print(' [SAVING] --> Creating HEADER for PRIMARY extension')
        self.primary_hdr = fits.Header()
        for key in primary_keys.keys():
            try:
                self.primary_hdr[key] = primary_keys[key]
            except:
                self.primary_hdr[key] = ''
        print(self.primary_hdr)
        print(' [SAVING] --> Creating PRIMARY HDU extension')
        self.primary_hdu = fits.PrimaryHDU(header=self.primary_hdr)

    def build_data(self):
        print(' [SAVING] --> Creating DATA extensions')
        # Fill the headers with the standard information
        with open('/home/pablo/WEAVE-Apertiff/input/weave_specifics/'
                  + 'cube_header/BLUE_DATA_keys.pkl', 'rb') as f:
            blue_data_keys = pickle.load(f)
        with open('/home/pablo/WEAVE-Apertiff/input/weave_specifics/'
                  + 'cube_header/RED_DATA_keys.pkl', 'rb') as f:
            red_data_keys = pickle.load(f)
        # Include the observation information
        blue_data_keys = {**blue_data_keys, **self.basic['blue']}
        red_data_keys = {**red_data_keys, **self.basic['red']}
        # Create fits header and fill with values
        blue_header, red_header = fits.Header(), fits.Header()
        for blue_key in blue_data_keys:
            try:
                blue_header[blue_key] = blue_data_keys[blue_key]
            except:
                print(blue_key, ' key failed')
        for red_key in red_data_keys:
            try:
                red_header[red_key] = red_data_keys[red_key]
            except Exception:
                print(red_key, ' key failed')
        blue_ivar_header = blue_header.copy()
        blue_ivar_header['BUNIT'] = ('1/(10^-32 * erg * s^-1 cm^-2 Angstrom^-1)', 'ivar unit')
        red_ivar_header = red_header.copy()
        red_ivar_header['BUNIT'] = ('1/(10^-32 * erg * s^-1 cm^-2 Angstrom^-1)', 'ivar unit')
        # Get the data for red and blue arms separately
        blue_cube_data, red_cube_data = self.create_arms(
            self.observation.cube)
        blue_cube_variance, red_cube_variance = self.create_arms(
            self.observation.cube_variance)
        # Create HDU images
        blue_data = fits.ImageHDU(array(blue_cube_data / 1e-16,
                                        dtype=float32), name='BLUE_DATA',
                                  header=blue_header)
        blue_ivar = fits.ImageHDU(array(1e-32 / blue_cube_variance,
                                        dtype=float32),
                                  name='BLUE_IVAR', header=blue_header)

        red_data = fits.ImageHDU(array(red_cube_data / 1e-16, dtype=float32),
                                 name='RED_DATA', header=red_header)
        red_ivar = fits.ImageHDU(array(1e-32 / red_cube_variance, dtype=float32),
                                 name='RED_IVAR',
                                 header=red_ivar_header)

        # Create collapsed images and remove unnecessary entries -------------------------------------------------------
        blue_collapsed_hdr = blue_header.copy()
        red_collapsed_hdr = red_header.copy()
        blue_collapsed_ivar_hdr = blue_ivar_header.copy()
        red_collapsed_ivar_hdr = red_ivar_header.copy()
        # Anything related to the spectral axis is removed
        del blue_collapsed_hdr['*3']
        del red_collapsed_hdr['*3']
        del blue_collapsed_ivar_hdr['*3']
        del red_collapsed_ivar_hdr['*3']
        # Collapsed HDU images
        blue_data_collapsed = fits.ImageHDU(array(blue_cube_data.sum(axis=0) / 1e-16, dtype=float32),
                                            name='BLUE_DATA_COLLAPSE3', header=blue_collapsed_hdr)
        blue_ivar_collapsed = fits.ImageHDU(array(1e32 / blue_cube_variance.sum(axis=0), dtype=float32),
                                            name='BLUE_IVAR_COLLAPSE3', header=blue_collapsed_ivar_hdr)
        red_data_collapsed = fits.ImageHDU(array(red_cube_data.sum(axis=0) / 1e-16, dtype=float32),
                                           name='RED_DATA_COLLAPSE3', header=red_collapsed_hdr)
        red_ivar_collapsed = fits.ImageHDU(array(1e32 / red_cube_variance.sum(axis=0), dtype=float32),
                                           name='RED_IVAR_COLLAPSE3', header=red_collapsed_ivar_hdr)

        # Response function for converting ADU's into physical units ---------------------------------------------------
        with open('/home/pablo/WEAVE-Apertiff/input/weave_specifics/'
                  + 'cube_header/BLUE_SENSFUNC_keys.pkl', 'rb') as f:
            blue_sensfunc_keys = pickle.load(f)
        with open('/home/pablo/WEAVE-Apertiff/input/weave_specifics/'
                  + 'cube_header/RED_SENSFUNC_keys.pkl', 'rb') as f:
            red_sensfunc_keys = pickle.load(f)
        blue_sensfunc_hdr, red_sensfunc_hdr = fits.Header(), fits.Header()
        for key in blue_sensfunc_keys:
            blue_sensfunc_hdr[key] = blue_sensfunc_keys[key]
        for key in red_sensfunc_keys:
            red_sensfunc_hdr[key] = red_sensfunc_keys[key]
        blue_sensfunc_data = ones_like(self.observation.instrument.wavelength_blue)
        red_sensfunc_data = ones_like(self.observation.instrument.wavelength_red)

        blue_sensfunc = fits.ImageHDU(array(blue_sensfunc_data, dtype=float32), name='BLUE_SENSFUNC',
                                      header=blue_sensfunc_hdr)
        red_sensfunc = fits.ImageHDU(array(red_sensfunc_data, dtype=float32), name='RED_SENSFUNC',
                                     header=blue_sensfunc_hdr)
        # Saving cubes
        self.save_cube(blue_data, blue_ivar, blue_sensfunc, blue_data_collapsed, blue_ivar_collapsed,
                       self.filename + '_blue')
        self.save_cube(red_data, red_ivar, red_sensfunc, red_data_collapsed, red_ivar_collapsed,
                       self.filename + '_red')

    def save_cube(self, data, ivar, sensfunc, collapsed, ivar_collapsed, filename):
        hdul = fits.HDUList([self.primary_hdu, data, ivar, sensfunc, collapsed])
        print('[SAVING] --> Saving cube file at {}'.format(filename + '.fits.gz'))
        hdul.writeto(filename + '.fits.gz', overwrite=True)
        hdul.close()

# Mr Krtxo.
