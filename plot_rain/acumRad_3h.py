#!/usr/bin/env python
# -*- coding: utf-8 -*-

# FUENTE

import matplotlib 
import matplotlib.font_manager as fm
import matplotlib.dates as mdates
import matplotlib.font_manager as font_manager

font_dirs = ['/media/nicolas/Home/Jupyter/Sebastian/AvenirLTStd-Book']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
font_list = font_manager.createFontList(font_files)
font_manager.fontManager.ttflist.extend(font_list)
# 
matplotlib.rcParams['font.family'] = 'Avenir LT Std'
matplotlib.rcParams['font.size'] = 18

#PAQUETES

import pandas as pd
import numpy as np 
import glob 
import pylab as pl
import pylab as plt
import datetime as dt
import datetime
import os

from cprv1 import cprv1
from wmf import wmf

import multiprocessing
from multiprocessing import Pool
import time
# from mpl_toolkits.basemap import Basemap
# import netCDF4

import funciones_sora as fs

def logger(orig_func):
    '''logging decorator, alters function passed as argument and creates
    log file. (contains function time execution)
    Parameters
    ----------
    orig_func : function to pass into decorator
    Returns
    -------
    log file
    '''
    import logging
    from functools import wraps
    import time
    logging.basicConfig(filename = 'acumRad_3h.log',level=logging.INFO)
    @wraps(orig_func)
    def wrapper(*args,**kwargs):
        start = time.time()
        f = orig_func(*args,**kwargs)
        date = dt.datetime.now().strftime('%Y-%m-%d %H:%M')
        took = time.time()-start
        log = '%s:%s:%.1f sec'%(date,orig_func.__name__,took)
        print log
        logging.info(log)
        return f
    return wrapper

@logger
def plot_acum_radar():
    
    #DEFINICION DE COSAS
    rutafig= '/media/nicolas/Home/Jupyter/Soraya/Op_Alarmas/Result_to_web/operacional/acum_radar/'
    selfN = cprv1.Nivel(codigo=260,user='sample_user',passwd='s@mple_p@ss',SimuBasin=True)
    codigos= selfN.infost.index[:]
    #setting the est indexes and order.
    pos_1st= [51,36,37,40,54,64]
    codigos = np.delete(codigos,pos_1st)
    codigos = np.insert(codigos,0,260)
    #time windows
    windows_t= ['30m_ahead','10m','5m','3h']#,'6h','24h','3d']

    #PLOTS
    for window_t in windows_t[:2]:
        print window_t
        if window_t == '30m_ahead':
            start = pd.to_datetime('2019-05-09 22:00')
#             start= fs.round_time(dt.datetime.now())
            end = start + pd.Timedelta('30m')
        else:
            end = pd.to_datetime('2019-05-09 22:00')
#             end = fs.round_time(dt.datetime.now())
            start= end -  pd.Timedelta(window_t)
    
        path_basins= '/media/nicolas/maso/Mario/basins/'
        
        #acumulado para la cuenca mas grande
        path_radtif = '/media/nicolas/maso/Soraya/op_files/radar/tifs/260-'+window_t+'.tif'
        dfAll,rvec = fs.get_radar_rain(start,end,300.,path_basins+'%s.nc'%(260),codigos,accum=True,
                            path_tif = path_radtif,#+start.strftime('%Y%m%d%H%M')+'_'+start.strftime('%Y%m%d%H%M')+'.tif',
                            meanrain_ALL=True)
        dfAll.to_csv(rutafig+window_t+'/dfAcum'+window_t+'.csv')
        
        #plots
        for codigo in codigos[0:]:
            if '%s.nc'%(codigo) in os.listdir(path_basins):
                cu = wmf.SimuBasin(rute=path_basins+'%s.nc'%(codigo))
                a,b = wmf.read_map_raster(path_radtif)
                vec_rain = cu.Transform_Map2Basin(a,b)
                #Plot
                fig = pl.figure(figsize=(10,12))
                ax = fig.add_subplot(111)
                if window_t == '30m_ahead':
                    fs.plot_basin_rain(cu,vec_rain,codigo,window_t='30m',ax=ax,cbar=True)
                else:
                    fs.plot_basin_rain(cu,vec_rain,codigo,window_t='30m',ax=ax,cbar=True)
                ax.set_title('Est. ' +str(codigo) +' | '+ selfN.infost.loc[codigo].nombre +'\n'+pd.to_datetime(start).strftime('%Y%m%d%H%M')+'-'+pd.to_datetime(end).strftime('%Y%m%d%H%M'))
                pl.savefig(rutafig+window_t+'/'+selfN.infost.loc[codigo].slug+'.png',bbox_inches='tight')
    
#     return dfAll

@logger
def processs_multiple_plots():
    from multiprocessing import Pool
    if __name__ == '__main__':
        p = Pool(10)
        p.map(plot_acum_radar)
        p.close()
        p.join()

if __name__ == '__main__':
    p = multiprocessing.Process(target=plot_acum_radar, name="")
    p.start()
    time.sleep(60*10) # in seg (seg*min)
    p.terminate()
    p.join()
    print 'plot_acum_radar executed'
    
#EJECUCION
# print dt.datetime.now()
# plot_acum_radar()
# print dt.datetime.now()

#COPIA A SAL
# res = os.system('scp '+rutafig++' socastillogi@192.168.1.74:/var/www/kml/01_Redes')