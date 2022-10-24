# -*- coding: utf-8 -*-
"""
Este módulo contém a classe OBSERVATIONS

PACKAGE: Radiotelecope
AUTHOR: Luciano Barosi
DATE: 20.10.2022
"""
# General Imports
from datetime import datetime
import itertools
import sys
# Third Party Imports
import astropy.coordinates as coord
import astropy.units as u
from astropy.time import Time, TimeDelta
from matplotlib.colorbar import Colorbar
import matplotlib.cm as cm
import matplotlib.dates as mdates
import matplotlib.colors as colors
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from pytz import timezone
# Local imports
import radiotelescope.backend.backend as backend
from radiotelescope.backend.backend import Backend as Backend
# --------------------
# main class definition.
# --------------------
class Observations:
    """Short summary.
    """
    def __init__(self, t_start=None, duration=None, backend=None, path=None):
        """Instantiate and go."""
        self._t_start = t_start
        self._duration = duration
        self.t_end = None
        self._backend = backend
        self._path = path
        self.data = None
        self.data_cal = None
        self.timezone = None
        return


    @property
    def t_start(self):
        """Return the duration of observation."""
        return self._t_start


    @t_start.setter
    def t_start(self, t_start):
        """Set the initial time of observations."""
        self._t_start = t_start
        return


    @property
    def duration(self):
        """Return the duration of observation."""
        return self._duration

    @duration.setter
    def duration(self, duration):
        """Set the initial time of observations."""
        self._duration = duration
        return


    @property
    def backend(self):
        """Return the object set as backend."""
        return self._backend


    @backend.setter
    def backend(self, backend):
        """Set the backend object from backend Class."""
        self._backend = backend
        return


    @property
    def path(self):
        """Return the duration of observation."""
        return self._path


    @path.setter
    def path(self, path):
        """Set the initial time of observations."""
        self._path = path
        return

    def initialize(self):

        if self.backend is None:
            self.timezone = "UTC"
        else:
            self.timezone = self.backend.instrument.timezone
        if self.t_start is None:
            self.t_start = pd.Timestamp.today().tz_localize("UTC").floor("D")
        if self.duration is None:
            self.duration = pd.Timedelta(12, unit="h")
        self.t_end = self.t_start + self._duration
        return self


    def filter_data(self, df, begin = None, freqs = None, duration = None, sampling = None):
        """Filter data from bachend in frequency, duration or sampling."""
        if begin is None:
            begin = df.index[0]
        else:
            begin = begin
        end = df.index[-1]
        if freqs is not None:
            freqmin = freqs[0]
            freqmax = freqs[1]
        else:
            freqmin = df.columns.min()
            freqmax = df.columns.max()
        if duration is not None:
            end = begin + pd.Timedelta(seconds = duration.to(u.s).value)
            #check if it fits inside original interval
            if end > df.index[-1]:
                end = df.index[-1]
        mask = df.columns.where((df.columns >= freqmin) & (df.columns < freqmax)).dropna()
        df = df.loc[begin:end][mask]
        if sampling is not None:
            df = df.resample(sampling).mean()
        return df


    def load_observation(self, mode="59", extension="fit"):  # SKY mode default.
        """Check existent datafiles and retrieve information for observation as given parameters."""
        # Read filenames and parse timestamps
        filenames = self.backend._get_filenames(extension=extension, modes=mode).filenames
        filenames = filenames.loc[self.t_start:self.t_end]
        try:
            df = self.backend.load_measurement(filenames=filenames, mode=mode, extension=extension)
            self.data = df
        except ValueError as err:
            print("No data found.")
            return None
        return self


    def calibrate(self, duration):
        pass


    def plot_waterfall(self, df = None, freqs = None, duration = None, sampling = None, colorbar = True):
        freqs = freqs
        duration = duration
        sampling = sampling
        if (freqs is not None) or (duration is not None) or (sampling is not None):
            # You may filter data inplace.
            df = self.filter_data(df, freqs = freqs, sampling = sampling, duration = duration)
        # limits
        freqs = df.columns
        begin = df.index[0]
        end = df.index[-1]
        mt = mdates.date2num((begin, end))
        hfmt = mdates.DateFormatter('%Y-%m-%d\n%H:%M:%S')
        fmt_major = mdates.MinuteLocator(interval = 30)
        fmt_minor = mdates.MinuteLocator(interval = 15)
        #----------------
        # average spectrum.
        SN = df.mean(axis=0)/df.std(axis=0)
        flux = df.median(axis = 1)
        #----------------
        # create grid format.
        fig = plt.figure(figsize=(16, 10))
        grid = plt.GridSpec(10, 9, hspace=0.0, wspace=0.1)
        spectrum_ax = fig.add_subplot(grid[2:-2, :-1])
        hor_fig = fig.add_subplot(grid[0:2, :-1], xticklabels=[], sharex=spectrum_ax)
        ver_fig = fig.add_subplot(grid[2:-2, -1], sharey=spectrum_ax)
        #--------------------------
        main = spectrum_ax.imshow(df.T, aspect='auto', extent = [ mt[0], mt[-1], freqs[-1], freqs[0]], cmap = cm.inferno)
        spectrum_ax.set_ylabel("Frequencies (MHz)")
        spectrum_ax.set_xlabel("Time")
        spectrum_ax.xaxis.set_major_formatter(hfmt)
        if colorbar:
            cbax = fig.add_subplot(grid[9,:-1])
            cb = Colorbar(ax = cbax, mappable = main, orientation = "horizontal", ticklocation = "bottom")
        #--------------------------
        # plot averaged spectrum in the vertical.
        # plot averaged spectrum in the vertical.
        ver_fig.plot(SN, freqs, c = 'red')
        ver_fig.grid()
        ver_fig.yaxis.tick_right()
        ver_fig.yaxis.set_label_position('right')
        ver_fig.set_xlabel("S/N")
        # plot averaged spectrum in the vertical.
        hor_fig.plot(df.index, flux, c = 'red')
        hor_fig.set_ylabel("dB")
        hor_fig.xaxis.tick_top()
        hor_fig.xaxis.set_minor_locator(fmt_minor)
        hor_fig.xaxis.set_major_formatter(hfmt)
        hor_fig.grid()
        return fig

    def plot_ts(self, df = None, begin = None, duration = None, freqs = None):
        if begin is None:
            begin = self.t_start
        if df is None:
            df = self.df_data
        df_filter = self.filter_data(df.loc[begin:], duration = duration, freqs = freqs)
        fig = self.plot_timeseries(df_filter, self.df_sky)
        return fig

#--------------------------------------
# Utilidades para gráficos
#--------------------------------------
def _plot_waterfall(data = None, ax = None):
    freqs = data.columns
    begin = data.index[0]
    end = data.index[-1]
    mt = mdates.date2num((begin, end))
    hfmt = mdates.DateFormatter('%Y-%m-%d\n%H:%M:%S')
    fmt_major = mdates.MinuteLocator(interval = 30)
    fmt_minor = mdates.MinuteLocator(interval = 15)
    #--------------------------
    if ax is None:
        fig, ax = plt.subplots(figsize = (18,6))
    main = ax.imshow(data.T, aspect='auto', extent = [ mt[0], mt[-1], freqs[-1], freqs[0]], cmap = cm.inferno)
    ax.set_ylabel("Frequencies (MHz)")
    ax.set_xlabel("Time")
    ax.xaxis.set_major_formatter(hfmt)
    return ax


def plot_mosaic(data = None, ax = None):
    axes = [];
    num = len(data)
    if (num == 3) or (num > 4):
            cols = 3
    else:
        cols = 2
    rows = int(np.ceil(num / cols))
    dfs = np.empty([rows, cols], dtype = "object")
    ii = 0
    if rows > 1:
        for row in np.arange(rows):
            for col in np.arange(cols):
                dfs[row, col] = data[ii]
                ii += 1
    else:
        for col in np.arange(cols):
            dfs[0, col] = data[ii]
            ii += 1
    if ax is None:
        fig, ax = plt.subplots(nrows = rows, ncols = cols, figsize=(18,6))
    if rows > 1:
        for row in np.arange(rows):
            for col in np.arange(cols):
                axes.append(_plot_waterfall(data = dfs[row, col], ax = ax[row, col]));
    else:
        for col in np.arange(cols):
            axes.append(_plot_waterfall(data = dfs[0,col], ax = ax[col]));
    return axes

#--------------------------------------
