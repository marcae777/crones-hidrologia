#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  Copyright 2018 MCANO <mario.cano@siata.gov.co>
import cprv1.cprv1 as cpr
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import os
import numpy as np
os.system('scp /home/nicolas/reportes_nivel.log mcano@siata.gov.co:/var/www/mario/nivel_tres_horas.log')
os.system('scp /home/nicolas/self_code/Crones/niveles_riesgo_error.log mcano@siata.gov.co:/var/www/mario/nivel_tres_horas_cron.log')

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
    logging.basicConfig(filename = 'reportes_nivel.log',level=logging.INFO)
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
    return self.level_all()

def convert_to_risk(df):
    df = self.risk_df(df)
    return df[df.columns.dropna()]

@logger
def risk_report(df):
    return self.make_risk_report_current(df)


@logger
def processs_multiple_plots():
    from multiprocessing import Pool
    if __name__ == '__main__':
        p = Pool(10)
        p.map(plot_level, list(df.columns))
        p.close()
        p.join()

@logger
def process_multiple_plots_looping():
    for codigo in df.columns:
        plot_level(codigo)   
        
def level_local(start,end):
    start = pd.to_datetime(start).strftime('%Y-%m-%d %H:%M:00')
    end = pd.to_datetime(end).strftime('%Y-%m-%d %H:%M:00')
    return self.read_sql("SELECT fecha,nivel from hydro where codigo = '%s' and fecha between '%s' and '%s'"%(self.codigo,start,end)).set_index('fecha')['nivel']

def level_local_all(start,end):
    start = pd.to_datetime(start).strftime('%Y-%m-%d %H:%M:00')
    end = pd.to_datetime(end).strftime('%Y-%m-%d %H:%M:00')
    return self.read_sql("SELECT fecha,nivel,codigo from hydro where fecha between '%s' and '%s'"%(start,end)).set_index(['codigo','fecha']).unstack(0)['nivel']

def plot_level(serie,codigo,resolution,folder,path = '/media/nicolas/Home/Jupyter/MarioLoco'):
    #try: # will be argument later on
    obj = cpr.Nivel(codigo = codigo,user='sample_user',passwd='s@mple_p@ss',SimuBasin=False)
    levantamiento = pd.read_csv('%s/ultimos_levantamientos/%s.csv'%(path,codigo),index_col=0)
    filepath = '%s/%s.png'%(path+'/real_time/'+folder,obj.info.slug)
    obj.plot_operacional(serie,levantamiento,resolution,filepath=filepath)
    r = os.system('scp %s mcano@siata.gov.co:/var/www/mario/realTime/%s'%(filepath,folder))
    #except:
    #    print 'error in plot %s'%codigo
#PROCESS
self = cpr.Nivel(codigo = 99,user='sample_user',passwd='s@mple_p@ss',SimuBasin=False)
end = self.round_time()
start = end - datetime.timedelta(days=30)
df = self.level_all(start,end,calidad=True)
resoluciones = ['24h','72h','30d']
folders = ['diario','tres_dias','treinta_dias']
def td(days):
    return datetime.timedelta(days=days)
timedeltas = [td(days=1),td(days=3),td(days=30)]
codigos = self.infost.index
@logger
def reporte():
    for resolution,folder,timedelta in zip(resoluciones,folders,timedeltas):
        for codigo in codigos:
            try:
                data = df.loc[end-timedelta:]
                plot_level(data[codigo]/100.0,codigo,resolution,folder)
            except:
                pass
        print resolution,folder,end-timedelta
reporte()
