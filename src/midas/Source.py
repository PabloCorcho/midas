#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 12:19:36 2022

@author: pablo
"""


class Source(object):
    """
    This class represents an astronomical object that will be observed by some
    instrument.
    """
    ra = None
    dec = None
    redshift = None
    
    def compute_sed(self, elements):
        pass

# Mr Krtxo
