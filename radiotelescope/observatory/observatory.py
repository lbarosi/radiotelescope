# -*- coding: utf-8 -*-
"""
This module provides the class `Observations`.

Holds informations necessary to identify the radiotelescope in use.
The class has properties like a dictionary and very few methods. An `Instrument` is something in a place, with a poiting, connected to a backend.
"""
#------------------
# Imports
#------------------
import os
import sys
from astropy import units as u
import astropy.coordinates as coord
from astropy.coordinates import EarthLocation
from astropy.time import Time, TimeDelta
import numpy as np
import pandas as pd
from pytz import timezone
from skyfield  import api
from skyfield.api import load
from skyfield.api import Loader

#------------------

class Observations:
    """This is a stub."""

    def __init__(self, ephemeris="de440s.bsp", instrument=None, t_start=None, duration=None):
        """Instantiate and go."""
        self._ephemeris = ephemeris
        self._load = Loader('../data/auxiliary/')
        self._eph = self._load(self.ephemeris)
        self._earth = self._eph['earth']
        self._instrument = instrument
        self._t_start = t_start
        self._duration = duration
        self.timevector = None
        self.pointings = None
        self._ts = api.load.timescale()
        return

    @property
    def ephemeris(self):
        return self._ephemeris

    @ephemeris.setter
    def ephemeris(self, ephemeris):
        self._ephemeris = ephemeris

    @property
    def t_start(self):
        return self._t_start

    @t_start.setter
    def t_start(self, t_start):
        self._t_start = t_start

    @property
    def duration(self):
        return self._duration

    @duration.setter
    def duration(self, duration):
        self._duration = duration

    @property
    def instrument(self):
        """Retorna o instrument."""
        return self._instrument

    @instrument.setter
    def instrument(self, instrument):
        """Define o instrumento."""
        self._instrument = instrument

    def make_timevector(self, t_start=None, duration=None, delta=1*u.h):
        """Faz vetor de tempo.

        Vetor com timestamps a partir de t_start, com intervalo delta e duração duration, no formato de tempo do skyfield.
        """
        if not t_start:
            t_start = self.t_start
        if not duration:
            duration = self.duration
        delta = delta.to(u.s).value
        steps = duration.to(u.s).value / delta
        vec = np.arange(steps)
        # Astropy timevector is fast to create.
        timelist = Time(t_start, scale='utc') + np.arange(steps)*TimeDelta(delta, format='sec', scale='tai')
        # Convert to skyfield timescale for later use.
        self.timevector = self._ts.from_astropy(timelist)
        return self


    def make_pointings(self, timevector=None, observer=None, Alt=None, Az=None):
        """Faz apontamentos para observador.

        Args:
            timevector (type): vetor de tempos para determinação das posições. Defaults to None.
            observer (type): objeto `observer` contém informações de latitude, longitude e altitude para observador. Defaults to None.
            Alt (type): altitude `Alt`, coordenada horizontal para apontamento. Defaults to None.
            Az (type): azimute `Az`, coordenada horizontal para apontamento. Defaults to None.

        Returns:
            DataFrame: colunas TIME, RA e DEC.

        """
        if not timevector:
            timevector = self.timevector
        if not observer:
            observer = self.instrument.observatory
        if not Alt:
            Alt = self.instrument.Alt
        if not Az:
            Az = self.instrument.Az
        # skyfield para obter coordenadas rapidamente.
        ra, dec, _ = observer.at(timevector).from_altaz(alt_degrees=Alt, az_degrees=Az).radec(self._ts.J2000)
        # astropy para manipular e desempactor vetor de coordenadas.
        pointings_sky = coord.SkyCoord(ra.hours, dec.degrees, unit=(u.hourangle, u.deg), frame='icrs', equinox='J2000')
        RA = pointings_sky.ra.radian
        DEC = pointings_sky.dec.radian
        TIME = timevector.tai
        self.pointings = pd.DataFrame(zip(TIME, RA, DEC), columns=["TIME", "RA", "DEC"])
        return self



def main():
    """Run the main dummy function."""
    message = ''.join([
    '\n\n This is instrument module of uirapuru package.',
    '\n Check Documentation\n'
    ])
    print(message)
    return None

if __name__ == "__main__":
    main()
