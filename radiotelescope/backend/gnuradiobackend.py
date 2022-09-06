# -*- coding: utf-8 -*-
"""
Este módulo contém a classe GNURadioBackend e seus métodos, representando dispositivo controlador de radiotelescópio, conectado a um objeto da classe Controller, algum tipo de computador. Este objeto específico tem a captura de dados ocorrendo por meio de um script GNURadio rodando em background.

PACKAGE: Radiotelecope
AUTHOR: Luciano Barosi
DATE: 26.04.2022
"""
from datetime import datetime
import os
from glob import glob
import logging
import pathlib
import psutil
import time
import sys

from astropy import units as u
from astropy.io import fits
from astropy.table import Table
import numpy as np
import pandas as pd
from paramiko.auth_handler import SSHException
from scipy.signal import savgol_filter as savgol_filter
# ------------------
# local imports
# ------------------
# import radiotelescope.backend.controller as Controller
# import radiotelescope.backend.instrument as Instrument
import radiotelescope.misc.multiprocess as multiprocess
import radiotelescope.misc.utils as utils
import radiotelescope.backend.backend as backend
from radiotelescope.backend.backend import Backend as Backend
import radiotelescope.misc.utils as utils
# Preparando log ----------------------------------------------------
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
# -------------------------------------------------------------------

# ------------------
# Implementação de backends: RTLSDRpower
# ------------------
class GNURadioBackend(Backend):
    """Backend GNURadio conectado remota ou localmente e seus métodos de controle."""

    def __init__(self, bandwidth=None, controller=None, instrument=None,
                 gain=None, integration_time=None, modes=None, name=None,
                 nominal_slope=None, observing_time=None, temperature=None,
                 **kwargs):
        self.slope = None
        self.NF = None
        self.freqs = None
        self.filenames = None
        self.GNUScript = None
        self.RTLSDR = None
        super().__init__(bandwidth=bandwidth, controller=controller,
                         instrument=instrument, gain=gain, integration_time=integration_time, modes=modes,
                         name=name, nominal_slope=nominal_slope,
                         observing_time=observing_time, temperature=temperature,
                         **kwargs)


    def load_measurement(self, filenames=None, mode=None, extension="fit"):
        """Implementa método com nome padrão para o carregamento de arquivos.

        Cada implementação de backend pode ter arquivos diferentes.
        Todos tem o mesmo nome como wrapper para método específico do backend.
        """
        if not filenames:
            filenames = self.filenames
        if mode:
            filenames = filenames[filenames.mode == mode]
        if extension == "fit":
            files = [
                file for file in filenames if file.split(".")[-1] == extension
                ]
            result = self.fits2df(files)
        elif extension == "prn":
            # To be implemented
            pass
        else:
            print("Method for extension {} not implemented.".format(extension))
            result = None
        return result

    def save_measurement(self):
        """Não implementado para esta classe ainda."""
        pass

    def scan_command(self, rtlsdr=None, GNUScript=None, folder=None, mode="SKY", channels=4096, bandwidth=None, gain=50, freq=1240, n_integration=100, n_samples=1000, duration=300, csv=False, fit=True):
        name = str(folder) + self.name
        mode = self.modes[mode]
        duration = utils.parse_time(duration)
        vec_length = channels
        samp_rate = bandwidth
        freq = freq * 1e6
        csvflag = " --csv " if csv else ""
        fitflag = " --fit " if csv else ""
        command = "python " + (str(GNUScript) + " --rtlsdr " + rtlsdr \
                  + " --name " + str(name) + " --mode " + str(mode) \
                  + " --vec_length " + str(vec_length) + " --samp_rate " \
                  + str(int(samp_rate)) + " --gain " + str(gain) + " --freq "\
                  + str(int(freq)) + " --n_integration " + str(n_integration)\
                  + " --n_samples " + str(n_samples) + " --duration " \
                  + str(duration) + csvflag + " " + fitflag)
        return command

    def start_tcp(self):
        try:
            is_connected = self.is_connected
        except AttributeError:
            is_connected = False
            pass
        if is_connected:
            command = "rtl_tcp -a " + self.controller.remote_IP + "&"
            self.controller.run_remote(command = command)
        else:
            logger.error("Backend not connected to any remote")
        return self

    def stop_tcp(self, **kwargs):
        timeout=kwargs.pop("timeout", 1500)
        PID=kwargs.pop("PID", 0)
        try:
            is_connected = self.is_connected
        except AttributeError:
            is_connected = False
            pass
        if is_connected:
            time_start = time.perf_counter()
            while psutil.pid_exists(PID):
                result = True
                if time.perf_counter() - time_start > timeout:
                    logger.error("Taking too long ot die.")
                    break
            command = "pkill rtl_tcp"
            self.controller.run_remote(command=command)
        else:
            logger.error("Backend not connected to any remote")
        return self

    def observe(self, **kwargs):
        try:
            is_connected = self.is_connected
        except AttributeError:
            is_connected = False
            pass
        rtlsdr = str(kwargs.pop("rtlsdr", self.RTLSDR))
        GNUScript = str(kwargs.pop("GNUScript", self.GNUScript))
        tcp = True if "tcp" in rtlsdr else False

        #RTLGLOBAL = "RTL_GLOBAL=" + "\"" + rtlstring + "\""
        #rtlfile = pathlib.Path(pathlib.Path(os.path.abspath(GNUScript)).parent, "rtlstring.py")
        #with open(rtlfile, 'w') as file:
        #    file.write(RTLGLOBAL)
        command = self.scan_command(rtlsdr=rtlsdr, GNUScript=GNUScript, **kwargs)
        if tcp:
            self.start_tcp()
            # Try to kill and wait for killing to finish before proceed.
            job1 = multiprocess.run_daemon(thread=self.controller.run, kwargs = {"command":command})
            job2 =  multiprocess.run_daemon(thread=self.stop_tcp, kwargs = {"PID":job1.pid})
        else:
            if is_connected:
                job1 = multiprocess.run_daemon(thread=self.controller.remote.run, kwargs = {"command":command})
            else:
                job1 = multiprocess.run_daemon(thread=self.controller.remote.run, kwargs = {"command":command})
        return


    def calibrate(self, data=None, dcold=None):
        """Não Implementado ainda."""
        pass
