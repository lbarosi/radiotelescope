# -*- coding: utf-8 -*-
"""
Este módulo contém a Classe Backend e seus métodos, representando dispositivo controlador de radiotelescópio, conectado a um objeto da classe Controller, algum tipo de computador.

As classes Backend são RTLSDRpower, CallistoSpectrometer e GNUradio.

PACKAGE: Radiotelecope
AUTHOR: Luciano Barosi
DATE: 09.04.2022
"""
from abc import ABC, abstractmethod
from datetime import datetime
import os
from glob import glob
import logging
import pathlib
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
import callisto
import radiotelescope.misc.multiprocess as multiprocess
import radiotelescope.misc.utils as utils
from radiotelescope.backend.backend import Backend as Backend
import radiotelescope.misc.utils as utils
# Preparando log ----------------------------------------------------
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# ------------------
# Implementação de backends: RTLSDRpower
# ------------------
class RTLSDRpowerBackend(Backend):
    """Backend RTLSDRpower conectado remota ou localmente e seus métodos de controle."""

    def __init__(self, bandwidth=None, controller=None, instrument=None,
                 gain=None, integration_time=None, modes=None, name=None,
                 nominal_slope=None, observing_time=None, temperature=None,
                 **kwargs):
        super().__init__(bandwidth=bandwidth, controller=controller,
                         instrument=instrument, gain=gain, integration_time=integration_time, modes=modes,
                         name=name, nominal_slope=nominal_slope,
                         observing_time=observing_time, temperature=temperature,
                         **kwargs)


    @staticmethod
    def _csv2df(filename=None):
        """Lê arquivo CSV gerador por RTLPOWER e converte em dataframe."""
        if filename:
            # Dados de rtl_power são csv que podem ter números de colunas distintos.
            # Começa lendo tamanho das linhas
            with open(filename, 'r') as temp_f:
                col_count = [ len(ll.split(",")) for ll in temp_f.readlines() ]
            column_names = np.arange(max(col_count))
            df = pd.read_csv(filename, header=None, delimiter=",", names=column_names)
            # Descarta valores de colunas que são maiores do que as demais
            # Número de medidas é total subtraído das colunas de informação
            # Date Time F_start F_end BW N_s
            n_meas = min(col_count) - 6
            # Corrige o caso de n = 1 que aparece com 2 colunas em rtl_power
            if n_meas ==2:
                n_meas = 1
            # Frequencias F_Start + N * bandwidth
            freq_table = ( df[2].to_numpy().reshape(-1,1) +
                           ( df[4].to_numpy().reshape(-1,1) *
                             np.arange(n_meas+1).reshape(1,-1)
                            )
                         ) / 1e6
            # Leituras em dB
            db_table = df.loc[:,6:].to_numpy()
            db_size = db_table.shape[1]
            # Datas e Horários das leituras concatenados
            times_table = df.loc[:,:1].sum(axis = 1).to_numpy().reshape(-1,1)
            del(df)
            # Monta dataframes com conjuntos de medidas e concatena
            df_list = [ np.hstack( [times_table,
                                    freq_table[:,ii].reshape(-1,1),
                                    db_table[:,ii].reshape(-1,1)
                                    ]
                                    )
                                    for ii in np.arange(db_size)
                                    ]
            df_data = np.vstack(df_list)
            df_data = pd.DataFrame(df_data)
            # Massageia dados
            df_data.columns = ["Date", "Freq", "Flux_dB"]
            df_data["Date"] = pd.to_datetime(df_data["Date"])
            result = df_data.pivot_table(values='Flux_dB',
                                         index='Date',
                                         columns='Freq').rename_axis(index = None, columns = None
                                         )
        return result

    @staticmethod
    def csvs2df(filenames=None):
        if filenames:
            data = []
            for file in filenames.files:
                data.append(RTLSDRpowerBackend._csv2df(filename = file))
            result = pd.concat(data)
        else:
            print("Função cv2df chamada sem argumentos.")
            result = None
        return result

    def csv2fit(self, path=None, filename = None, overwrite = False, **kwargs):
        """Lê arquivo CSV no formato de RTLPOWER e converte em dataframe concatenado."""
        if not path:
            path = os.path.abspath(self.controller.local_folder)
        filename = os.path.abspath(filename)
        df = RTLSDRpowerBackend._csv2df(filename)
        filename_fits = os.path.join(os.getcwd(), path, filename.split(".")[-2] + ".fit")
        MODE = filename.split(".")[-1].split("_")[-1]
        DATE = datetime.today().strftime("%Y%m%d")
        DATE_START = df.index[0].strftime("%Y%m%d")
        DATE_END = df.index[-1].strftime("%Y%m%d")
        TIME_START =   df.index[0].strftime("%H%M%S")
        TIME_END =   df.index[0].strftime("%H%M%S")
        time_array = (df.index - df.index[0]).total_seconds()
        freq_array = df.columns.to_numpy()
        time_size = time_array.size
        freq_size = freq_array.size
        data = df.T.to_numpy()
        #--------------------
        # header
        #--------------------
        header = fits.Header()
        header["SIMPLE"] = "T"
        header["BITPIX"] = "16"
        header["NAXIS"] = 2
        header["NAXIS1"] = time_size
        header["NAXIS2"] = freq_size
        header["EXTEND"] = "T"
        header["COMMENT"] = "FITS (Flexible Image Transport System) format defined in Astronomy and"
        header["COMMENT"] = "Astrophysics Supplement Series v44#p363, v44#p371, v73#p359, v73#p365."
        header["COMMENT"] = "Contact the NASA Science Office of Standards and Technology for the   "
        header["COMMENT"] = "FITS Definition document #100 and other FITS information.             "
        header["DATE"] = DATE
        header["CONTENT"] = "Radio flux density - " + self.name + "@" + self.instrument.name
        header["ORIGIN"] = "PB"
        header["TELESCOP"] = self.instrument.name
        header["INSTRUME"] = self.name
        header["OBJECT"] = MODE  # object
        header["DATE-OBS"] = DATE_START
        header["TIME-OBS"] = TIME_START
        header["DATE-END"] = DATE_END
        header["TIME-END"] = TIME_END
        header["BZERO"] = 0.
        header["BSCALE"] = 1.
        header["BUNIT"] = 'digits'
        header["CTYPE1"] = 'Time [UT]'
        header["CTYPE2"] = 'Frequency [MHz]'
        header["OBS_LAT"] = self.instrument.lat.value
        header["OBS_LAC"] = 'S'
        header["OBS_LON"] = self.instrument.lon.value
        header["OBS_LOC"] = 'W'
        header["OBS_ALT"] = self.instrument.elev.value
        #--------------------
        # header from kwargs
        for key, value in kwargs:
            header[key] = value
        #--------------------
        primary_HDU = fits.PrimaryHDU(header = header, data = data )
        table_hdu = fits.table_to_hdu(Table([[time_array], [freq_array]], names = ("TIME", "FREQUENCY")))
        hdul = fits.HDUList([primary_HDU, table_hdu])
        pathlib.Path(filename_fits).parents[0].mkdir(parents=True, exist_ok=True)
        hdul.writeto(filename_fits, overwrite = overwrite)
        return

    def load_measurement(self, filenames=None, mode=None, extension = "csv"):
        """Carrega dataframe com dados contidos na lista de arquivos indicada no argumento, selecionando os modos e permitindo definir a extensão dos arquivos."""
        if not filenames:
            filenames = self.filenames
        if mode:
            filenames = filenames[filenames["mode"] == mode]
        if extension == "fit":
            files = [ file for file in filenames.files if file.split(".")[-1]==extension ]
            result = self.fits2df(filenames = files)
        elif extension == "csv":
            files = [ file for file in filenames.files if file.split(".")[-1]==extension ]
            result = RTLSDRpowerBackend.csvs2df(filenames = files)
        else:
            print("Method for extension {} not implemented.".format(extension))
            result = None
        return result

    def save_measurement(self):
        """Salva medidas localmente se o backend for remoto."""
        try:
            is_connected = self.is_connected
        except AttributeError:
            is_connected = False
            pass
        if is_connected:
            try:
                result = self.controller.connect().sync_files()
            except SSHException as err:
                print("Deu pobrema no SSH!")
        else:
            logger.error("Method implemented only for remote backends.")
        return

    def scan(self, duration="15m", band=None, bandwidth="1", gain=50, integration="1s", filename=None, **kwargs):
        """Executa scan específico do backend."""
        scan = ("rtl_power -f " + str(band[0]) + ":" + str(band[1]) + ":" + str(bandwidth) + " -g" + str(gain) + " -i " + integration + " -e " + duration + " " + filename)
        print(scan)
        try:
            is_connected = self.is_connected
        except AttributeError:
            is_connected = False
            pass
        if is_connected:
            result = self.controller.run_remote(scan)
        else:
            result = self.controller.run(scan)
        return None

    def scan_mp(self, duration="15m", band=None, bandwidth="1", gain=50, integration="1s", filename=None, monitor=False, interval=30, **kwargs):
        """Executa scan em modo detached, liberando a execução do script enquando a medição é realizada."""
        scan = ("rtl_power -f " + str(band[0]) + ":" + str(band[1]) + ":" + str(bandwidth) + " -g" + str(gain) + " -i " + integration + " -e " + duration + " " + filename)
        print(scan)
        try:
            is_connected = self.is_connected
        except AttributeError:
            is_connected = False
            pass
        if is_connected:
            # run_detach quebra via ssh. Opção run_remote sem chamada de stdout tmb não é bloqueante.
            self.controller.run_remote(command=scan, stdout=False)
        else:
            result = multiprocess.run_detached(target=self.controller.run, command=scan, monitor=monitor, interval=interval)
        return

    def log_record(self, **kwargs):
        """Registra parâmetros da medição em arquivo de LOG."""
        try:
            is_connected = self.is_connected
        except AttributeError:
            is_connected = False
            pass
        cols = ["origin", "duration", "band", "bandwidth", "gain", "integration", "filename", "date", "backend",  "instrument", "controller"]
        path = os.path.join(self.controller.local_folder, "log/",self.name + "_" + self.instrument.name + "_" + "file_log.csv")
        pathlib.Path(path).parents[0].mkdir(parents=True, exist_ok=True)
        origin = "remote" if is_connected else "local"
        info = {"origin":origin, "backend": self.name, "instrument":self.instrument.name, "controller":self.controller.name}
        info = {**info, **kwargs}
        record = pd.DataFrame([info])[cols]
        record.to_csv(path, mode="a", header=(not os.path.exists(path)), index=False)
        return


    def observe_mp(self, duration="15m", band=None, bandwidth="1", gain=50, integration="1s", mode="01", interval=30):
        """Realiza observação, determinando nome único e parâmetros de medida, chamando as funções específicas no backend em modo detached."""
        try:
            is_connected = self.is_connected
        except AttributeError:
            is_connected = False
            pass
        NOW = datetime.now().strftime("%Y%m%d_%H%M%S")
        if is_connected:
            filename = self.controller.remote_folder + self.name + "_" + NOW + "_" + mode + ".csv"
        else:
            filename = self.controller.local_folder + self.name + "_" + NOW + "_" + mode + ".csv"
        scan_kwargs = {"duration":duration, "band":band, "bandwidth":bandwidth, "gain":gain, "integration":integration, "filename":filename, "name":self.name, "interval":interval}
        result = self.scan_mp(**scan_kwargs)
        self.log_record(date=NOW, **scan_kwargs)
        return

    def observe(self, duration="15m", band=None, bandwidth="1", gain=50, integration="1s", mode="01"):
        """Realiza observação, determinando nome único e parâmetros de medida, chamando as funções específicas no backend local ou remoto conforme o caso."""
        NOW = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            is_connected = self.is_connected
        except AttributeError:
            is_connected = False
            pass
        if is_connected:
            filename = self.controller.remote_folder + self.name + "_" + NOW + "_" + mode + ".csv"
        else:
            filename = self.controller.local_folder + self.name + "_" + NOW + "_" + mode + ".csv"
        scan_kwargs = {"duration":duration, "band":band, "bandwidth":bandwidth, "gain":gain, "integration":integration, "filename":filename}
        result = self.scan(**scan_kwargs)
        self.log_record(date=NOW, **scan_kwargs)
        return

    def calibrate(self):
        observe_dict = {"band":["50M", "1700M"], "bandwidth":"1M",
                        "gain":"50", "integration":"15m", "mode":"01"}
        self.observe_mp(**observe_dict)
        return
