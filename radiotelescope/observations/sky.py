# -*- coding: utf-8 -*-
"""
Este módulo contém a classe SKY

PACKAGE: Radiotelecope
AUTHOR: Luciano Barosi
DATE: 07.05.2022
"""
# General Imports
from adjustText import adjust_text
import sys
import numpy as np
import pandas as pd
import itertools
# Plotting
from matplotlib.colorbar import Colorbar
import matplotlib.cm as cm
import matplotlib.dates as mdates
import matplotlib.colors as colors
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator,
    FormatStrFormatter, AutoMinorLocator)
# Astropy
import astropy.coordinates as coord
import astropy.units as u
from astropy.time import Time, TimeDelta
from astropy.visualization import time_support
from astropy.wcs import WCS
from astropy.table import QTable
# Catalogs
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
# Date and Time
from datetime import datetime
from pytz import timezone
# Special packages
from skyfield  import api
from skyfield.api import load
from skyfield.api import Loader

# --------------------
# Local objects to keep track during observations.
# --------------------
LOCAL_OBJECTS = ['sun', 'moon', 'mercury barycenter', 'venus barycenter',
                 'mars barycenter', 'jupiter barycenter', 'saturn barycenter' ]
# --------------------
# Very strong radisources
# --------------------
RADIOSOURCES = ['Cassiopeia A', 'Centaurus A', 'Cygnus A', 'Fornax A',
                'Hercules A', 'Hydra A', 'Pictor A', 'Puppis A', 'Sagittarius A*', 'Taurus A', 'Virgo A']
# --------------------
# GNSS satellite constelation TLE data: orbits
# Constelações principais: GPS, GALILEO, GLONASS, BEIDOU, QZS
# Constelação IRIDIUM opera fora da banda
# --------------------
TLE_urls = ['https://celestrak.com/NORAD/elements/gps-ops.txt', 'https://celestrak.com/NORAD/elements/glo-ops.txt', 'https://celestrak.com/NORAD/elements/galileo.txt', 'https://celestrak.com/NORAD/elements/beidou.txt']
# --------------------
# plate_carre WCS
# --------------------
WCS_PLATE = WCS(naxis=2)
WCS_PLATE.wcs.crpix = [0., 0.]
WCS_PLATE.wcs.cdelt = [1, 1]
WCS_PLATE.wcs.crval = [180, 0]
WCS_PLATE.wcs.ctype = ["RA---CAR", "DEC--CAR"]
# --------------------
# Mollweide WCS
# --------------------
WCS_MOL = WCS(naxis=2)
WCS_MOL.wcs.crpix = [0., 0.]
WCS_MOL.wcs.cdelt = [1, 1]
WCS_MOL.wcs.crval = [180, 0]
WCS_MOL.wcs.ctype = ["RA---MOL", "DEC--MOL"]

class Sky:

    def __init__(self, ephemeris="de440s.bsp", t_start=None, duration=None,
                 FWHM=None, instrument=None, local_object=LOCAL_OBJECTS):
        """Instantiate and go."""
        self._ephemeris = ephemeris
        self._load = Loader('../data/auxiliary/')
        self._eph = self._load(self.ephemeris)
        self._earth = self._eph['earth']
        self._instrument = instrument
        self._t_start = t_start
        self._duration = duration
        self._FWHM = FWHM
        self.timevector = None
        self.pointings = None
        self.local_objects = local_objects
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
    def local_objects(self):
        return self._local_objects

    @local_objects .setter
    def local_objects (self, local_objects ):
        self._local_objects  = local_objects


    def _get_altaz_from_radec(self, observer, df):
        """Get AltAz to use with pandas apply."""
        try:
            name = df["NAME"]
        except KeyError as e:
            name = None
        try:
            obj_type = df["TYPE"]
        except KeyError as e:
            obj_type = None
        ra = (df["RA"]*u.deg).to(u.hourangle).value
        dec = df["DEC"]
        timestamp = self._ts.tai_jd(df["TIME"])
        celestial = api.Star(ra_hours=ra, dec_degrees=dec)
        pos = observer.at(timestamp).observe(celestial)
        alt, az, _ = pos.apparent().altaz()
        return obj_type, name, timestamp.tai, alt.degrees, az.degrees


    def get_altaz_from_radec(self, observer=None, objects = None):
        df = objects.apply(lambda row: self._get_altaz_from_radec(observer, row), axis = 1, result_type ='expand')
        if df.shape[1] == 3:
            df.columns = ["TIME", "ALT", "AZ"]
        if df.shape[1] == 4:
            df.columns = ["NAME", "TIME", "ALT", "AZ"]
        if df.shape[1] == 5:
            df.columns = ["TYPE", "NAME", "TIME", "ALT", "AZ"]
        return df


    def _get_radec_target(self, observer=None, objects=None, timevector=None):
        """Get coordinates of objects and returns astropy Skycoord."""
        _ra, _dec, _ = observer.at(timevector).observe(objects).radec(self._ts.J2000)
        coords = coord.SkyCoord(ra=_ra.hours, dec=_dec.degrees,
                                unit=(u.hourangle, u.deg), frame='icrs',
                                equinox='J2000')
        return coords


    def _pos(obj, observer, timevector):
        pos = (obj - observer).at(timevector)
        return pos


    def _get_satellites(obj, observer, timevector, Alt, Az):
        pos = _pos(obj, observer, timevector)
        ra, dec, dist = pos.radec()
        cone = observer.at(timevector).from_altaz(alt_degrees=Alt, az_degrees=Az).separation_from(pos)
        RA = coord.Longitude(ra._degrees, u.deg, wrap_angle=180*u.deg).to(u.rad).value
        sats = da.from_array([timevector.tai, RA, dec.radians, cone.radians, [obj.name] * len(timevector)])
        return sats


    def _get_satellites_df(obj, observer, timevector, Alt, Az):
        pos = _pos(obj, observer, timevector)
        ra, dec, dist = pos.radec()
        cone = observer.at(timevector).from_altaz(alt_degrees=Alt, az_degrees=Az).separation_from(pos)
        RA = coord.Longitude(ra._degrees, u.deg, wrap_angle=180*u.deg).to(u.rad).value
        sats = pd.DataFrame({"TIME": timevector.tai,
                             "RA": RA,
                             "DEC": dec.radians,
                             "ANGLE": cone.radians,
                             "NAME": [obj.name] * len(timevector)
                             })
        return sats


    def get_satellites(timevector=None, observer=None, Alt=None, Az=None,      reload=False):

        if not timevector:
            timevector = self.timevector
        if not observer:
            observer = self.instrument.observatory
        if not Alt:
            Alt = self.instrument.Alt
        if not Az:
            Az = self.instrument.Az
        # --------------------
        # Loading satellite data.
        # --------------------
        TLE = TLE_urls
        satellites = []
        for url in TLE:
            satellites.append(sky_loader.tle_file(url, reload=reload))
        gnss_all = list(itertools.chain(*satellites))
        # --------------------
        # Generate positions in dataframe.
        # --------------------
        # TLE are geocentric, observer should also be geocentric.
        objects = []
        for ii, satellite in enumerate(gnss_all):
            sat_obj = _get_satellites_df(satellite, observer, timevector, Alt, Az)
            objects.append(sat_obj)

        df = pd.concat(objects)
        df['TIME'] = df['TIME'].astype(float)
        df['RA'] = df['RA'].astype(float)
        df['DEC'] = df['DEC'].astype(float)
        df['ANGLE'] = df['ANGLE'].astype(float)
        df['NAME'] = df['NAME'].astype(str)

        return df



# -----------------
# Verificar ainda
# -----------------
    def beam_on_sky(self):
        """Collect all celestials in one dataframe."""
        pulsares_csv = obs.load_pulsares()
        radiosources_csv = obs.load_radiosources()
        nvss_csv = obs.load_nvss_catalog()
        df_celestials = pd.concat(
            [pulsares_csv.query("DEC >-20 & DEC < 10 & S1400>10")[["PSRJ", "RA", "DEC"]].rename(columns = {"PSRJ": "NAME"}),
             nvss_csv.query("DEC >-20 & DEC < 10 & S1400>10000")[["NVSS", "RA", "DEC"]].rename(columns = {"NVSS": "NAME"}),
             radiosources_csv[["SOURCE", "RA", "DEC"]].rename(columns = {"SOURCE": "NAME"})]
        )
        df = self.get_star_cone(objects = df_celestials)
        df_local_objects = self.get_local_objects_cone()
        FWHM = self.instrument.fwhm
        df_gnss_satellites = self.get_satellites().query("ANGLE < @FWHM")
        df = pd.concat([df, df_local_objects, df_gnss_satellites])
        df["TIME"] = pd.to_datetime(Time(df.TIME.values, format='jd', scale = "tai").to_datetime())
        df.set_index('TIME', inplace=True)
        df = df.sort_index()
        df.reset_index(inplace = True)
        df_obs = df
        return df


# -----------------
# Verificar ainda
# -----------------
    def get_all_beam(self, query_string_nvss = "S1400>10000", query_string_psr = "S1400>10"):
        """Collect information from all celestials of interest in a single dataframe."""
        df_01 = self.get_local_objects()
        df_02 = self.get_satellites()
        df_03 = self.get_star_cone(load_nvss_catalog().query(query_string_nvss))
        df_04 = self.get_star_cone(load_pulsares().query(query_string_psr))
        df_05 = self.get_star_cone(load_radiosources())
        df_list = [df_01, df_02, df_03, df_04, df_05]
        list_objects = ["LOCAL", "GNSS", "NVSS", "PSR", "RADIOA"]
        dfcat = []
        for ii, item in enumerate(df_list):
            if not item.empty:
                item["TYPE"] = list_objects[ii]
                dfcat.append(item)
        df = pd.concat(dfcat)
        return df



# -----------------
# Verificar ainda
# -----------------
    def get_local_objects(self, objects = None, CONE = True):
        """Get positions of local objects during observation."""
        # --------------------
        # Generate positions in dataframe.
        # --------------------
        if self.instrument is not None:
            timevector = self.timevector
            fwhm = self.instrument.fwhm
            observer = self._earth + self.instrument.observatory
            if objects is None:
                objects = self.local_objects
            object_list = []
            for sky_object in objects:
                pos = (observer - self._eph[sky_object]).at(timevector)
                ra, dec, dist = pos.radec()
                cone = observer.at(timevector).from_altaz(alt_degrees=self.instrument.Alt, az_degrees=self.instrument.Az).separation_from(pos)
                df = pd.DataFrame(zip(timevector.tai, ra._degrees, dec.degrees, cone.degrees, dist.km),
                                  columns=['TIME', 'RA', 'DEC', 'ANGLE', 'DISTANCE'])
                df['NAME'] = [sky_object.split(" ")[0]] * len(timevector)
                object_list.append(df)
            objects_df = pd.concat(object_list)
            if CONE:
                df = objects_df[objects_df.ANGLE < fwhm/2]
        else:
            print("Instrument not set")
            df = None
        return df



# -----------------
# Verificar ainda
# -----------------
    def get_star_cone(self, objects = None, CONE = True):
        """Populate dataframe with distant objects in beam."""
        # --------------------
        # Generate positions in dataframe.
        # --------------------
        instrument_ok = self.instrument is not None
        df_ok = not objects.empty
        if instrument_ok and df_ok:
            timevector = self.timevector
            fwhm = self.instrument.fwhm
            observer = self._earth + self.instrument.observatory
            object_list = []
            for index, star in objects.iterrows():
                ra = (star.RA*u.deg).to(u.hourangle).value
                dec = star.DEC
                celestial = api.Star(ra_hours=ra, dec_degrees=dec)
                pos = observer.at(timevector).observe(celestial)
                ra, dec, dist = pos.radec()
                cone = observer.at(timevector).from_altaz(alt_degrees=self.instrument.Alt, az_degrees=self.instrument.Az).separation_from(pos)
                df = pd.DataFrame(zip(timevector.tai, ra._degrees, dec.degrees, cone.degrees, dist.km),
                                  columns=['TIME', 'RA', 'DEC', 'ANGLE', 'DISTANCE'])
                df['NAME'] = [star.NAME] * len(timevector)
                object_list.append(df)
            objects_df = pd.concat(object_list)
            if CONE:
                df = objects_df[objects_df.ANGLE < fwhm/2]
        else:
            df = pd.DataFrame()
        return df


# -------------------------------------
# Funções do Módulo - Já Verificadas
# -------------------------------------
def fetch_nvss_catalogs(filename = "../data/auxiliary/nvss_radiosources.csv", DEC_FILTER = "<10 && >-30", S1400_mjy = ">100", query = "DEC >-20 & DEC < 10 & S1400>100"):
    """Fetch astroquery vizier nvss catalog.

    Args:
        DEC_FILTER (type): Filter for Vizier `DEC_FILTER`. Defaults to "<10 && >-30".
        S1400_mjy (type): Filter for Vizier `S1400_mjy`. Defaults to ">10".

    Returns:
        DataFrame

    """
    nvss = "VIII/65/nvss"
    catalog = Vizier(catalog=nvss, columns=['*', '_RAJ2000', '_DEJ2000', 'S1400'], column_filters={"_DEJ2000":"<10 && >-30", "S1.4":S1400_mjy}, row_limit=-1).query_constraints()[nvss]
    df = catalog.to_pandas()[['_RAJ2000','_DEJ2000','NVSS', 'S1.4']]
    df.columns = ['RA','DEC','NAME', 'S1400']
    df = df[["NAME", "RA", "DEC", "S1400"]]
    df = df.query(query)
    try:
        df.to_csv(filename, encoding="utf-8", index=False)
        print("arquivo salvo em disco: {}".format(filename))
    except IOError as err:
        print(err + "\n arquivo não foi salvo em disco")
    return df

def fetch_pulsar_catalogs(filename = "../data/auxiliary/pulsares.csv", query = "DEC >-20 & DEC < 10 & S1400>10"):
    """Baixa catálogo de pulsares B/psr/psr.

    Baixa catálogo e salva em formato csv no computador local.
    """
    pulsar_table = "B/psr/psr"
    catalog = Vizier(catalog=pulsar_table, columns=['*', 'RA2000', 'DE2000', 'S1400'], row_limit=-1)
    pulsares = catalog.query_constraints()[pulsar_table]
    df = pulsares.to_pandas()[['PSRJ', 'RA2000', 'DE2000', 'Dist', 'P0', 'DM', 'S1400']]
    df.dropna(subset=['Dist', 'S1400'], inplace=True)
    # Filtrando fluxos maiores do que 1mJy
    df.columns = ['NAME', 'RA', 'DEC', 'DIST', 'P0', 'DM', 'S1400']
    df = df.query(query)
    try:
        df.to_csv(filename, encoding="utf-8", index=False)
        print("arquivo salvo em disco: {}".format(filename))
    except IOError as err:
        print(err + "\n arquivo não foi salvo em disco")
    return df

def fetch_radiosources(filename = "../data/auxiliary/radiosources.csv", radiosources=RADIOSOURCES):
    """Baixa radiofontes selecionadas.

    Args:
        radiosources (type): Description of parameter `radiosources`. Defaults to RADIOSOURCES.

    Returns:
        type: Description of returned object.

    """
    s = Simbad()
    radio = s.query_objects(radiosources)
    df = radio.to_pandas()[["MAIN_ID", "RA", "DEC"]]
    df["NAME"] = radiosources
    df["RA"] = coord.Angle(df.RA, unit="hour").degree
    df["DEC"] = coord.Angle(df.DEC, unit="deg").degree
    try:
        df.to_csv(filename, encoding="utf-8", index=False)
        print("arquivo salvo em disco: {}".format(filename))
    except IOError as err:
        print(err + "\n arquivo não foi salvo em disco")
    return df

def load_nvss_catalog(filename = "../data/auxiliary/nvss_radiosources.csv", **kwargs):
    """Load previously saved data from NVSS catalog. If file does not exist, fetch data from vizier with kwargs.

    Args:
        filename (type): Defaults to "../data/auxiliary/nvss_radiosources.csv".
        **kwargs (type): Description of parameter `**kwargs`.

    Returns:
        type: Description of returned object.

    """
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError as err:
        print("sources not found on local disk, trying Vizier...")
        df = fetch_nvss_catalogs(**kwargs)
    return df

def load_pulsares(filename = "../data/auxiliary/pulsares.csv", **kwargs):
    """Load previously saved data from pulsar catalog. If file does not exist, fetch data from vizier with kwargs.

    Args:
        filename (type): Defaults to "../data/auxiliary/pulsares.csv".
        **kwargs (type): Description of parameter `**kwargs`.

    Returns:
        type: Description of returned object.

    """
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError as err:
        print("sources not found on local disk, trying Vizier...")
        df = fetch_pulsar_catalogs(**kwargs)
    return df

def load_radiosources(filename = "../data/auxiliary/radiosources.csv"):
    """Load selected sources. Fetch if file not found.

    Args:
        filename (type): Description of parameter `filename`. Defaults to "../data/auxiliary/radiosources.csv".

    Returns:
        type: Description of returned object.

    """
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError as err:
        print("sources not found on local disk, trying SIMBAD...")
        df = fetch_radiosources(radiosources)
    return df

def get_galactic_equator(size = 720):
    """Return data to plot galactic plane.

    Args:
        size (type): Number of points to plot `size`. Defaults to 720.

    Returns:
        type: Skycoord object.

    """
    l = np.linspace(0,360,size)
    b = np.zeros(size)
    gal_plane = coord.SkyCoord(l,b, unit=u.deg, frame="galactic")
    return gal_plane
