# -*- coding: utf-8 -*-
"""Uirapuru Module.

This module provides a predefined Instrument. Uirapuru is a transit telescope.
"""
from .instrument import Instrument
from astropy import units as u
from pytz import timezone
# --------------------
# Equipment Definition: UIRAPUDU.
# --------------------
lat= -7.211637 * u.deg;
lon= -35.908138 * u.deg;
elev= 553 * u.m
Alt= 84
Az=0
fwhm = 15
timezone = timezone("America/Recife")
#backend = CallistoSpectrometer()
Uirapuru = Instrument(name='Uirapuru', lon=lon, lat=lat, elev=elev, timezone=timezone, verbose=True, Alt=Alt, Az=Az, fwhm = fwhm)
# --------------------
