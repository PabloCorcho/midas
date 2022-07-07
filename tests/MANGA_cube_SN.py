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
from scipy.optimize import curve_fit

logflux_edges = np.linspace(-20, -15, 51)
logflux_bins = (logflux_edges[:-1] + logflux_edges[1:]) / 2

logsnr_edges = np.linspace(-1, 3, 51)
logsnr_bins = (logsnr_edges[:-1] + logsnr_edges[1:]) / 2

cube_paths = glob('/home/pablo/obs_data/MANGA/cubes/*')

mag_lim = 24  # mag / arcsec**2
g_band_wl = 4770
# band_wl = 6500
band_width = 50
c_ligth_aa = 3e18
f_lim = (10**(-0.4 * (mag_lim + 48.60)) * c_ligth_aa / g_band_wl**2
         )
bands = np.array([4000, 4500, 5000, 5500, 6000, 6500, 7000])
bands = np.arange(3800, 7400, 100)


def snr_model(logflux, logflux_0, logsigma_0):
    """..."""
    logsnr = (logflux - logsigma_0
              - 0.5 * np.log10(1 + 10**logflux / 10**logflux_0))
    return logsnr


all_H = np.zeros((len(bands), logflux_bins.size, logsnr_bins.size))

for i, cube_path in enumerate(cube_paths):
    cube = MANGACube(path=cube_path, abs_path=True)
    cube.get_flux()
    cube.get_wavelength(to_rest_frame=True)

    allsnr = cube.flux / cube.flux_error / cube.pixel_size
    for j, band in enumerate(bands):
        wl_mask = (cube.wl < band + band_width
                   ) & (cube.wl > band - band_width)

        mask = ((cube.flux > 1e-20) & (allsnr > 0.1) & (allsnr < 1e4)
                & wl_mask[:, np.newaxis, np.newaxis])

        logflux = np.log10(cube.flux[mask].flatten() / 0.5**2)
        logsnr = np.log10(allsnr[mask].flatten())

        H, _, _ = np.histogram2d(logflux, logsnr,
                                 bins=[logflux_edges, logsnr_edges])
        H_cond = H / np.sum(H * np.diff(logsnr_edges), axis=1)[:, np.newaxis]
        H_cum = np.cumsum(H_cond * np.diff(logsnr_edges), axis=1)
        all_H[j] += H

    cube.close_cube()
    if i > 20:
        break

# %%
all_H_cond = all_H / np.sum(all_H * np.diff(logsnr_edges)[np.newaxis, :],
                            axis=2)[:, :, np.newaxis]
all_H_cum = np.cumsum(all_H_cond * np.diff(logsnr_edges)[np.newaxis, :],
                      axis=2)

logsnr_p50 = logsnr_bins[np.argmin(np.abs(all_H_cum - 0.5), axis=2)]
n_per_flux = all_H.sum(axis=2)

sigma_p50 = 1 / (np.log(n_per_flux + 1.1))
# sigma_p50 /= np.min(sigma_p50)

# p = np.polyfit(logflux_bins, logsnr_p50, 1, w=1/sigma_p50, rcond=1e-20)
# lin_fit = np.poly1d(p)
# %%
fits = []

fig, axs = plt.subplots(ncols=6, nrows=6, sharex=True,
                        sharey=True, figsize=(17, 17))
axs = axs.flatten()
for i in range(len(bands)):
    popt, pcov = curve_fit(snr_model,
                           xdata=logflux_bins, ydata=logsnr_p50[i],
                           sigma=sigma_p50[i],
                           bounds=([-21, -21], [-10, -10])
                           )
    fits.append(popt)
    ax = axs[i]
    mappable = ax.pcolormesh(logflux_bins, logsnr_bins, np.log10(all_H[i].T),
                             vmax=np.log10(all_H.max()),
                             vmin=np.log10(all_H.max()*1e-4),
                             cmap='cividis')
    # plt.colorbar(mappable, ax=ax, label=r'$\log_{10}(n_{pix})$',
    #              orientation='horizontal')

    ax.plot(logflux_bins, logsnr_p50[i], c='fuchsia', lw=2,
            label='Running median')
    ax.fill_between(logflux_bins,
                    logsnr_p50[i] - sigma_p50[i],
                    logsnr_p50[i] + sigma_p50[i],
                    color='fuchsia', alpha=0.2)
    ax.plot(logflux_bins, snr_model(logflux_bins, *popt), c='k')
    ax.axvline(np.log10(f_lim), c='k', ls='--', lw=3, alpha=0.5)
    ax.minorticks_on()
    # ax.grid(visible=True, which='major')
    # ax.grid(visible=True, which='minor', alpha=0.3)
    ax.set_xlabel(r'$\log_{10}(F_{%s})$' % bands[i])


ax = fig.add_axes((0.92, 0.1, 0.01, 0.8))
plt.colorbar(mappable, ax, label=r'$\log_{10}(n_{pix})$')

ax = axs[0]
ax.set_ylim(logsnr_edges.min(), logsnr_edges.max())
ax.set_xlim(logflux_edges.min(), logflux_edges.max())
ax.set_ylabel(r'$\log_{10}(SNR~{\rm per}~\AA)$')

fig.subplots_adjust(wspace=0.2, hspace=0.2)
fig.savefig('manga_logflux_snr.png', bbox_inches='tight')

fits = np.array(fits)
# %%
fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(len(bands)):
    ax.plot(logflux_bins, logsnr_p50[i], lw=1,
            label='{}'.format(bands[i]))
    ax.fill_between(logflux_bins,
                    logsnr_p50[i] - sigma_p50[i],
                    logsnr_p50[i] + sigma_p50[i],
                    alpha=0.3)
    ax.set_ylim(logsnr_edges.min(), logsnr_edges.max())
    ax.set_xlim(logflux_edges.min(), logflux_edges.max())
    ax.set_xlabel(r'$\log_{10}(F_{%s})$' % bands[i])
    ax.set_ylabel(r'$\log_{10}(SNR)$')

ax.axvline(np.log10(f_lim), c='k', ls='--', lw=2, alpha=0.5)
ax.minorticks_on()
ax.grid(visible=True, which='major')
ax.grid(visible=True, which='minor', alpha=0.3)
# ax.legend(ncol=3)

# %%


def sigma0_model(wl, a, b, c):
    """..."""
    logsigma = a + (wl/b)**c
    return logsigma


def f0_model(wl, A, wl_0, a, b):
    """..."""
    WL = wl/wl_0
    f0 = A * WL**a / (1 + WL**(a + b))
    return f0

# def f0_model(wl, A, wl_0, a):
#     """..."""
#     WL = wl/wl_0
#     f0 = A * np.exp(a * np.abs((wl - wl_0) / wl_0))
#     return f0


sigma0_popt, sigma_pcov = curve_fit(sigma0_model,
                                    xdata=bands,
                                    ydata=fits[:, 1],
                                    maxfev=1000000
                                    # bounds=([, -5], [5000, 5])
                                    )

f0_popt, f0_pcov = curve_fit(f0_model,
                             xdata=bands,
                             ydata=10**fits[:, 0],
                             maxfev=1000000,
                             p0=[3e-17, 5900, 3, 5]
                             )

# %%
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.plot(bands, 10**fits[:, 0], ls='-', marker='o', c='k')
plt.plot(bands, f0_model(bands, *f0_popt), c='r')
# plt.plot(bands, f0_model(bands, 3e-17, 5900, 3, 5))
plt.xlabel(r'$\lambda~(\AA)$')
plt.ylabel(r'$F_0$')

eq = (r'${:.2}\times'.format(f0_popt[0]) + r'\frac{(\lambda /'
      + r'{:.0f}'.format(f0_popt[1]) + ')^{'
      + '{:.1f}'.format(f0_popt[2]) + r'}}'
      + r'{ 1 + (\lambda /' + r'{:.0f}'.format(f0_popt[1])
      + ')^{' + '{:.1f}'.format(f0_popt[3]) + '}'
      + '}$'
      )

plt.annotate(eq, xy=(.5, .05), xycoords='axes fraction', ha='center')

plt.subplot(122)
plt.plot(bands, fits[:, 1], ls='-', marker='o',  c='k')
plt.plot(bands, sigma0_model(bands, *sigma0_popt), c='r')
plt.ylim(fits[:, 1].min(), fits[:, 1].max())

eq = (r'$\log_{10}(\sigma_0) = ' + '{:2.1f} +'.format(sigma0_popt[0])
      + r'(\lambda /'
      + r'{:.0f}'.format(sigma0_popt[1]) + ')^{'
      + '{:.1f}'.format(sigma0_popt[2]) + r'}$'
      )

plt.annotate(eq, xy=(.5, .95), xycoords='axes fraction',
             ha='center', va='top')

plt.xlabel(r'$\lambda~(\AA)$')
plt.ylabel(r'$\log_{10}(\sigma_0)$')
plt.subplots_adjust(wspace=0.3)
plt.savefig('manga_model_fits.png', bbox_inches='tight')
# plt.annotate('${}'.format(sigma_popt[0])
#              +'+(\frac{\lambda}' +'{%s}' % sigma_popt[1]
#              + ')^{%s}$' % sigma_popt[2],
#              xy=(.9, .9),
#              xycoords='axes fraction')
# %%
logsnr_g_band = snr_model(np.log10(f_lim),
                          np.log10(f0_model(g_band_wl, *f0_popt)),
                          sigma0_model(g_band_wl, *sigma0_popt))
