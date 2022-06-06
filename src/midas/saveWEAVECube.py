#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import yaml
import os
from astropy.io import fits
from datetime import date
from numpy import array, float32, ones_like, ones

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
    # Path to cube template files
    header_templates_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'cube_templates', 'WEAVE', 'headers')
    # Number of extension of L1 WEAVE data
    n_extensions = 8
    extension_names = ['PRIMARY', '<arm>_DATA', '<arm>_IVAR', '<arm>_DATA_NOSS',
                       '<arm>_IVAR_NOSS', '<arm>_SENSFUNC', '<arm>_DATA_COLLAPSE3',
                       '<arm>_IVAR_COLLAPSE3']
    # Normalization value for flux
    flux_norm = 1e-18
    def __init__(self, observation, filename):
        print('[SAVING] -- *** SAVING OBSERVATION AS WEAVE DATA CUBE ***')
        self.filename = filename
        self.observation = observation
        # First it loads the templates used for building the headers of each extensino
        self.load_header_templates()
        # Modify some values according to the simulated data
        self.modify_keyword_values()
        # Handling data format and arm splitting
        self.build_extensions()

    def modify_keyword_values(self):

        try:
            self.headers
        except Exception:
            raise NameError(' [SAVING] ERROR: HEADER TEMPLATES NOT PROPERLY LOADED')
        print(' [SAVING] Replacing HEADER values according to simulated data')
        for arm in self.headers.keys():
            for extension in self.headers[arm].keys():
                hdr = self.headers[arm][extension]
                if 'PROV0000' in hdr.keys():
                    hdr['PROV0000'] = self.filename

                # keywords corresponding to the simulation
                if extension == 'PRIMARY':
                    hdr['S_DATA'] = ('IllustrisTNG100-1', 'Simulation suite used')
                    hdr['S_MED_VEL'] = (self.observation.instrument.redshift * 3e5, 'systemic velocity km/s')
                    hdr['SIM_OBJID'] = self.observation.galaxy.name
                    hdr['SIM_KERNEL'] = (self.observation.kernel.name, 'Kernel used for particle smoothin')
                    for key in self.observation.kernel.kernel_params.keys():
                        hdr['SI_KER_PAR_' + key] = (
                            self.observation.kernel.kernel_params[key],
                            'Kernel parameter')
                    hdr['AUTHOR1'] = ('Pablo Corcho-Caballero (UAM/MQ)', '')
                    hdr['AUTHOR2'] =  ('pablo.corcho@uam.es', 'contact')
                    hdr['DATE'] = (str(date.today()))
                # Set WCS parameters
                if 'CRVAL1' in hdr.keys():
                    # spatial axes
                    hdr['NAXIS1'] = (self.observation.cube.shape[2], 'length of data axis 1')
                    hdr['CRVAL1'] = (30.0, 'RA at CRPIX1 in deg')
                    hdr['CRPIX1'] = (self.observation.cube.shape[2] // 2, 'Ref pixel for WCS')
                    hdr['CD1_1'] = (-self.observation.instrument.pixel_size.to('deg').value,
                                    'Pixels in degress for X-axis')
                    
                    if 'CRVAL2' in hdr.keys():
                        hdr['NAXIS2'] = (self.observation.cube.shape[1], 'length of data axis 2')
                        hdr['CRVAL2'] = (30.0, 'DEC at CRPIX2 in deg')
                        hdr['CRPIX2'] = (self.observation.cube.shape[1] // 2, 'Ref pixel for WCS')
                        hdr['CD2_2'] = (self.observation.instrument.pixel_size.to('deg').value,
                                        'Pixels in degress for Y-axis')
                    # spectral axis
                    if 'CRVAL3' in hdr.keys():
                        hdr['NAXIS3'] = (self.observation.instrument.wavelength_arms[arm].size, 'length of data axis 3')
                        hdr['CRVAL3'] = (self.observation.instrument.wavelength_arms[arm][0].value, 'DEC at CRPIX2 in deg')
                        hdr['CRPIX3'] = (1.0, 'Ref pixel for WCS')
                        hdr['CD3_3'] = (self.observation.instrument.delta_wave,
                                        'Linear dispersion (Angstrom/pixel)')

                self.headers[arm][extension] = hdr

    def create_arms(self, cube):
        """
        Provide a cube and returns the corresponding arms
        :return: red and blue arms
        """
        blue_cube = cube[self.observation.instrument.wave_blue_pos, :, :]
        red_cube = cube[self.observation.instrument.wave_red_pos, :, :]
        return {'BLUE':blue_cube, 'RED':red_cube}

    def load_header_templates(self):
        print(' [SAVING] Loading header templates from\n  --> ', self.header_templates_path)
        self.headers = {}
        # Iterate over both arms
        for arm in ['blue', 'red']:
            print(' --> ARM: ', arm.upper())
            # Create each extension header
            self.headers[arm.upper()] = {}
            for i in range(self.n_extensions):
                ext_name = self.extension_names[i].replace('<arm>',
                                                           arm.upper())
                print('       · EXTENSION: ' + ext_name)
                hdr_i = fits.Header()
                file_path = os.path.join(self.header_templates_path,
                                         arm + '_header_{}.yml'.format(i))
                with open(file_path) as file:
                    hdr_template = yaml.safe_load(file)
                hdr_i = self.map_dict_to_hdr(hdr_template, hdr_i)
                self.headers[arm.upper()][ext_name] = hdr_i

    def map_dict_to_hdr(self, dict_, hdr_):
        for key in dict_:
            if len(dict_[key]) == 1:
                hdr_[key] = dict_[key][0]
            elif len(dict_[key]) == 2:
                # Assumes that the second element corresponds to a comment
                hdr_[key] = tuple(dict_[key])
            else:
                print(key + 'keyword values has length > 2 --> IGNORING')
        return hdr_

    def build_extensions(self):
        print(' [SAVING] --> Creating HDU data extensions')
        # Fill the headers with the standard information
        try:
            self.headers
        except Exception:
            raise NameError(' [SAVING] ERROR: HEADER TEMPLATES NOT PROPERLY LOADED')
        print(' [SAVING] Replacing HEADER values according to simulated data')
        for arm in self.headers.keys():
            hdu_image_list = []
            for extension in self.headers[arm].keys():
                hdr = self.headers[arm][extension]
                if extension == 'PRIMARY':
                    hdu_image_list.append(fits.PrimaryHDU(header=hdr));
                elif extension == arm + '_DATA':
                    arm_data = self.create_arms(self.observation.cube)[arm]
                    arm_data = array(arm_data / self.flux_norm, dtype=float32)
                    hdu = fits.ImageHDU(data=arm_data, name=extension, header=hdr)
                    hdu_image_list.append(hdu)
                elif extension == arm + '_IVAR':
                    arm_data = self.create_arms(self.observation.cube_variance)[arm]
                    arm_data = array(self.flux_norm**2 / arm_data, dtype=float32)
                    hdu = fits.ImageHDU(data=arm_data, name=extension, header=hdr)
                    hdu_image_list.append(hdu)
                elif extension == arm + '_DATA_NOSS':
                    try:
                        self.observation.sky
                        arm_data = self.create_arms(self.observation.cube + self.observation.sky)[arm]
                        arm_data = array(arm_data / self.flux_norm, dtype=float32)
                        hdu = fits.ImageHDU(data=arm_data, name=extension, header=hdr)
                    except Exception:
                        print(' [SAVING] WARNING: NO SKY MODEL\n   --> ' + extension
                              + ' extension data will be empty')
                        hdu = fits.ImageHDU(name=extension)
                    hdu_image_list.append(hdu)
                elif extension == arm + '_IVAR_NOSS':
                    if self.observation.sky is not None:
                        arm_data = self.create_arms(self.observation.cube_variance
                                                    + self.observation.sky**2)[arm] / self.flux_norm**2
                        arm_data = array(1/arm_data, dtype=float32)
                        hdu = fits.ImageHDU(data=arm_data, name=extension, header=hdr)
                    else:
                        print(' [SAVING] WARNING: NO SKY MODEL\n   --> ' + extension
                              + ' extension data will be empty')
                        hdu = fits.ImageHDU(name=extension)
                    hdu_image_list.append(hdu)
                elif extension == arm + '_DATA_COLLAPSE3':
                    arm_data = self.create_arms(self.observation.cube)[arm]
                    arm_data = array(arm_data / self.flux_norm, dtype=float32)
                    arm_data = arm_data.sum(axis=0)
                    hdu = fits.ImageHDU(data=arm_data, name=extension, header=hdr)
                    hdu_image_list.append(hdu)
                elif extension == arm + '_IVAR_COLLAPSE3':
                    arm_data = self.create_arms(self.observation.cube_variance)[arm]
                    arm_data = array(self.flux_norm ** 2 / arm_data, dtype=float32)
                    arm_data = arm_data.sum(axis=0)
                    hdu = fits.ImageHDU(data=arm_data, name=extension, header=hdr)
                    hdu_image_list.append(hdu)
                elif extension == arm + '_SENSFUNC':
                    arm_data = ones(self.observation.instrument.wavelength_arms[arm].size, dtype=float32)
                    hdu = fits.ImageHDU(data=arm_data, name=extension, header=hdr)
                    hdu_image_list.append(hdu)

            hdul = fits.HDUList(hdu_image_list)
            filename = self.filename + "_" + arm + ".fits.gz"
            print("[SAVING] --> Saving cube file at {}".format(filename))
            hdul.writeto(filename, overwrite=True)
            hdul.close()
            del hdul

# Mr Krtxo.
