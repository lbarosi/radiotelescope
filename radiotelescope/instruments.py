# -*- coding: utf-8 -*-
"""
Este módulo provê instrumentos e backends já inicializados com  informações padrão.

Instrumento: minihorn.
Controller: linuxbox.
Backend: RTLSDR.
"""
from datetime import datetime
import os
import sys
import time
from astropy import units as u
import pytz
import radiotelescope
import radiotelescope.GNURadio.GNUController as GNURADIO

# Definindo Instrumento
lat = -7.211637 * u.deg;
lon = -35.908138 * u.deg;
elev = 553 * u.m
Alt = 84
Az = 0
fwhm = 15
timezone = pytz.timezone("America/Recife")
minihorn = radiotelescope.Instrument(name='Cornetinha', lon=lon, lat=lat, elev=elev, timezone=timezone, verbose=True, Alt=Alt, Az=Az, fwhm=fwhm)

linuxbox = radiotelescope.LinuxBox(name="linuxbox", interface="WAN", user="bingo", remote_port=22, remote_IP="192.168.15.81", local_folder="../data/raw/GNURADIO/", remote_folder = "~/SDR/")

# Definindo Modos
modes = {"COLD":"01", "WARM":"02", "HOT":"03", "SKY":"59"}
RTLSDR = radiotelescope.GNURadioBackend(controller=linuxbox, instrument=minihorn, modes=modes, name="SDR_01", DEVICE="RTL2838")
