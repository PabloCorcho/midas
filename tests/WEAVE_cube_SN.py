#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 16:10:05 2022

@author: pablo
"""

from astropy.io import fits
import numpy as np
from matplotlib import pyplot as plt
from FAENA.read_cube import CALIFACube
from scipy.ndimage import gaussian_filter

logflux_edges = np.linspace(-20, -15, 51)
logflux_bins = (logflux_edges[:-1] + logflux_edges[1:]) / 2

logsnr_edges = np.linspace(-1, 3, 51)
logsnr_bins = (logsnr_edges[:-1] + logsnr_edges[1:]) / 2

hdul = fits.open(
    '/home/pablo/WEAVE-Apertiff/mock_cubes/227583/galaxy_ID_227583_xsl_BLUE.fits.gz')

flux = hdul[1].data * 1e-18
sigma = 1e-18/np.sqrt(hdul[2].data)
hdul.close()

g_flux = gaussian_filter(flux, sigma=[0, 1, 1])

snr = flux / sigma

mask = (flux > 0) & (snr > 0.1) & (snr < 1e3)

logflux = np.log10(flux[mask].flatten())
logsnr = np.log10(snr[mask].flatten())

H, _, _ = np.histogram2d(logflux, logsnr, bins=[logflux_edges, logsnr_edges])

p = np.polyfit(logflux, logsnr, 1, rcond=1e-20)
p_3 = np.polyfit(logflux, logsnr, 3, rcond=1e-20)

lin_fit = np.poly1d(p)
cubic_fit = np.poly1d(p_3)

plt.figure()
plt.contourf(logflux_bins, logsnr_bins, np.log10(H.T))
plt.plot(logflux_bins, lin_fit(logflux_bins))
plt.plot(logflux_bins, cubic_fit(logflux_bins))
plt.colorbar()
plt.ylim(logsnr_edges.min(), logsnr_edges.max())
plt.xlim(logflux_edges.min(), logflux_edges.max())
plt.show()