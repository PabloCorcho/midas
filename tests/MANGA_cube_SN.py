#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 15:56:15 2022

@author: pablo
"""

from astropy.io import fits
import numpy as np
from matplotlib import pyplot as plt
from FAENA.read_cube import MANGACube
from glob import glob

logflux_edges = np.linspace(-20, -15, 51)
logflux_bins = (logflux_edges[:-1] + logflux_edges[1:]) / 2

logsnr_edges = np.linspace(-1, 3, 51)
logsnr_bins = (logsnr_edges[:-1] + logsnr_edges[1:]) / 2

cube_paths = glob('/home/pablo/obs_data/MANGA/cubes/*')

all_H = np.zeros((logflux_bins.size, logsnr_bins.size))
all_p_fit = []
all_p_cubic_fit = []
for i, cube_path in enumerate(cube_paths):
    cube = MANGACube(path=cube_path, abs_path=True)
    cube.get_flux()
    cube.get_wavelength(to_rest_frame=True)

    allsnr = cube.flux / cube.flux_error

    mask = (cube.flux > 1e-20) & (allsnr > 0.1) & (allsnr < 1e3)

    logflux = np.log10(cube.flux[mask].flatten())
    logsnr = np.log10(allsnr[mask].flatten())

    p = np.polyfit(logflux, logsnr, 1, rcond=1e-20)

    p_3 = np.polyfit(logflux, logsnr, 3, rcond=1e-20)

    lin_fit = np.poly1d(p)
    cubic_fit = np.poly1d(p_3)
    all_p_fit.append(p)
    all_p_cubic_fit.append(p_3)
    H, _, _ = np.histogram2d(logflux, logsnr, bins=[logflux_edges, logsnr_edges])
    all_H += H

    plt.figure()
    plt.contourf(logflux_bins, logsnr_bins, np.log10(H.T))
    plt.plot(logflux_bins, lin_fit(logflux_bins))
    plt.plot(logflux_bins, cubic_fit(logflux_bins))
    plt.colorbar()
    plt.ylim(logsnr_edges.min(), logsnr_edges.max())
    plt.xlim(logflux_edges.min(), logflux_edges.max())
    plt.close()

    cube.close_cube()

# %%
mean_lin_fit = np.poly1d(np.mean(all_p_fit, axis=0))
mean_cubic_fit = np.poly1d(np.mean(all_p_cubic_fit, axis=0))

all_H_cond = all_H / np.sum(all_H * np.diff(logsnr_edges), axis=1)[:, np.newaxis]
all_H_cum = np.cumsum(all_H_cond * np.diff(logsnr_edges), axis=1)

plt.figure(figsize=(8, 8))
plt.contourf(logflux_bins, logsnr_bins, np.log10(all_H.T),
             vmax=np.log10(all_H.max()), vmin=np.log10(all_H.max()*1e-4))
plt.colorbar(label=r'$\log_{10}(n_{pix})$')
plt.contour(logflux_bins, logsnr_bins, all_H_cum.T, levels=[0.16, 0.5, 0.84],
            linewidths=[1, 2, 1], colors='fuchsia')
plt.plot(logflux_bins, mean_lin_fit(logflux_bins), ls='--', c='k')
plt.plot(logflux_bins, mean_cubic_fit(logflux_bins), c='k')

plt.grid()
plt.ylim(logsnr_edges.min(), logsnr_edges.max())
plt.xlim(logflux_edges.min(), logflux_edges.max())
plt.xlabel(r'$\log_{10}(F)$')
plt.ylabel(r'$\log_{10}(SNR)$')
# %%
plt.figure()
plt.plot(logflux_bins,
         10**(mean_lin_fit(logflux_bins) - mean_cubic_fit(logflux_bins)))
plt.xlabel(r'$\log_{10}(F)$')
plt.ylabel(r'$\frac{SNR_{lin}}{SNR_{cubic}}$', fontsize=14)