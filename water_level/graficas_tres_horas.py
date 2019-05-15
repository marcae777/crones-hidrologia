#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  Copyright 2018 MCANO <mario.cano@siata.gov.co>
import cprv1.cprv1 as cpr
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import os
import numpy as np
import multiprocessing
import time
import wmf.wmf as wmf
from multiprocessing import Pool

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
    logging.basicConfig(filename = 'reporte_nivel.log',level=logging.INFO)
    @wraps(orig_func)
    def wrapper(*args,**kwargs):
        start = time.time()
        f = orig_func(*args,**kwargs)
        date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        took = time.time()-start
        log = '%s:%s:%.1f sec'%(date,orig_func.__name__,took)
        print log
        logging.info(log)
        return f
    return wrapper

@logger
def data_base_query():
    return self.level_all(calidad=True)

def convert_to_risk(df):
    df = self.risk_df(df)
    return df[df.columns.dropna()]

def risk_report(df):
    return self.make_risk_report_current(df)

def plot_level(codigo):
    try:
        resolution='3h' # will be argument later on
        path = '/media/nicolas/Home/Jupyter/MarioLoco'
        if resolution == '3h':
            folder = 'tres_horas'
        obj = cpr.Nivel(codigo = codigo,user='sample_user',passwd='s@mple_p@ss',SimuBasin=False)
        levantamiento = pd.read_csv('%s/ultimos_levantamientos/%s.csv'%(path,codigo),index_col=0)
        filepath = '%s/%s.png'%(path+'/real_time/'+folder,obj.info.slug)
        obj.plot_operacional(df[codigo]/100.0,levantamiento,resolution,filepath=filepath)
        r = os.system('scp %s mcano@siata.gov.co:/var/www/mario/realTime/%s'%(filepath,folder))
    except:
        print 'error in plot %s'%codigo

        
def plot_level(codigo):
    resolution='3h' # will be argument later on
    path = '/media/nicolas/Home/Jupyter/MarioLoco'
    if resolution == '3h':
        folder = 'tres_horas'
    obj = cpr.Nivel(codigo = codigo,user='sample_user',passwd='s@mple_p@ss',SimuBasin=False)
    levantamiento = pd.read_csv('%s/ultimos_levantamientos/%s.csv'%(path,codigo),index_col=0)
    filepath = '%s/%s.png'%(path+'/real_time/'+folder,obj.info.slug)
    obj.plot_operacional(df[codigo]/100.0,levantamiento,resolution,filepath=filepath)
    print(codigo)

@logger
def processs_multiple_plots():
    p = Pool(10)
    p.map(plot_level, list(df.columns))
    p.close()
    p.join()

@logger
def process_multiple_plots_looping():
    self = cpr.Nivel(codigo = 99,user='sample_user',passwd='s@mple_p@ss',SimuBasin=False)
    for codigo in df.columns:
        plot_level(codigo)   
    os.system('scp /media/nicolas/Home/Jupyter/MarioLoco/real_time/tres_horas/* mcano@siata.gov.co:/var/www/mario/realTime/tres_horas/')
    os.system('scp /media/nicolas/Home/Jupyter/MarioLoco/real_time/tres_horas/* mcano@siata.gov.co:/var/www/mario/realTime/tres_horas/')
    os.system('scp /media/nicolas/Home/Jupyter/MarioLoco/real_time/tres_horas/* mcano@siata.gov.co:/var/www/mario/realTime/tres_horas/')
    
@logger        
def convert_series_to_risk(self,level):
    '''level: pandas Series, index = codigos de estaciones'''
    risk = level.copy()
    colors = ['green','gold','orange','red','red','black']
    for codigo in level.index:
        try:
            risks = cpr.Nivel(codigo = codigo,user='sample_user',passwd='s@mple_p@ss').risk_levels
            risk[codigo] = colors[int(self.convert_level_to_risk(level[codigo],risks))]
        except:
            risk[codigo] = 'black'
    return risk

self = cpr.Nivel(codigo = 99,user='sample_user',passwd='s@mple_p@ss',SimuBasin=False)
df = data_base_query() #dataframe level

if __name__ == '__main__':
    p = multiprocessing.Process(target=process_multiple_plots_looping, name="")
    p.start()
    time.sleep(290) # wait near 5 minutes to kill process
    p.terminate()
    p.join()