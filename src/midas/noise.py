#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 15:56:15 2022

@author: pablo
"""
import numpy as np


# mag_lim = 24 # mag / arcsec**2
# band_wl = 4686
# sn_lim = 40
# c_ligth_aa = 3e18
# f_lim = (10**(-0.4 * (mag_lim + 48.60)) * c_ligth_aa / band_wl**2
#           /weave_instrument.pixel_size.to('arcsec').value) # erg/s/AA/cm2

def snr_model(logflux, logflux_0, logsigma_0):
    """..."""
    logsnr = (logflux - logsigma_0
              - 0.5 * np.log10(1 + 10**logflux / 10**logflux_0))
    return logsnr


def logsigma0_model(wl, a=-18.4, b=3464, c=-5.9):
    """..."""
    logsigma = a + (wl/b)**c
    return logsigma


def logf0_model(wl, A=2.17e-15, wl_0=6052., a=3.9, b=3.2):
    """..."""
    WL = wl/wl_0
    f0 = A * WL**a / (1 + WL**(a + b))
    return f0


def noise_model(wave, flux):
    """Return the expected SNR corresponding to the input flux values."""
    logflux_0 = logf0_model(wave)
    logsigma_0 = logsigma0_model(wave)
    snr = snr_model(np.log10(flux), logflux_0, logsigma_0)
    return snr

# Mr Krtxo \(ﾟ▽ﾟ)/
