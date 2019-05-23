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

self = cpr.Nivel(codigo=260,user='sample_user',passwd = 's@mple_p@ss',SimuBasin=True)
self.nc_path = self.info.nc_path
end = datetime.datetime.now()
start = self.round_time(end - datetime.timedelta(minutes=30))
posterior = end + datetime.timedelta(minutes=30)
rain = self.radar_rain(start,posterior)
rain_vect = self.radar_rain_vect(start,posterior)
codigos = self.infost.index
df = pd.DataFrame(index = rain_vect.index,columns=codigos)

for codigo in df.columns:
    mask_path = '/media/nicolas/maso/Mario/mask/mask_%s.tif'%(codigo)
    try:
        mask_map = wmf.read_map_raster(mask_path)
        mask_vect = self.Transform_Map2Basin(mask_map[0],mask_map[1])
    except AttributeError:
        print 'mask:%s-name:%s'%(codigo,self.infost.loc[codigo,'nombre'])
        mask_vect = None
        df = df.drop(codigo,axis=1)
        
    if mask_vect is not None:
        mean = []
        for date in rain_vect.index:
            try:
                mean.append(np.sum(mask_vect*rain_vect.loc[date])/np.sum(mask_vect))
            except:
                print 'mean:%s'%codigo
        if len(mean)>0:
            df[codigo] = mean
            
self = cpr.Nivel(codigo=260,user='sample_user',passwd = 's@mple_p@ss',SimuBasin=True)
sql = cpr.SqlDb('hydrology','sample_user','localhost','s@mple_p@ss',3306,'data_data')

def get_id(codigo):
    sql = cpr.SqlDb('hydrology','sample_user','localhost','s@mple_p@ss',3306,'data_data')
    id = sql.read_sql("select id from meta_basin where codigo='%s'"%codigo).iloc[0].id
    return id

field = 'radar_rain'

def insert_data_databasin(field,df):
    df = df.copy()
    for date,series in df.iterrows():
        #print(date)
        codigo = series
        rounded = np.round(series.values,5)
        field_values = str(tuple(np.array(rounded,str)))
        values = str(tuple(zip([date.strftime('%Y-%m-%d %H:%M')]*series.index.size,np.array(list(map(lambda x:get_id(x),series.index)),str),np.array(rounded,str))))
        field_values = str(tuple(np.array(rounded,str)))
        on_dup = ' ON DUPLICATE KEY UPDATE data_databasin.%s = VALUES(data_databasin.%s)'%(field,field)
        statement = ("INSERT INTO data_databasin (data_databasin.date,data_databasin.fk_id,data_databasin.%s) VALUES "%field+values[1:-1])+on_dup
        sql.execute_sql(statement)
        
insert_data_databasin(field,df)
def select(pos):
    try:
        pos = int(pos)
        colors = ['green','#FAF16A','orange','red','red']
        return colors[pos]
    except:
        return '#D8E0E8'
    
def convert_to_risk(df):
    df = self.risk_df(df)
    return df[df.columns.dropna()]

df = self.level_all(start,end)
risk_df = convert_to_risk(df.copy())
df_colors = risk_df.T.applymap(lambda x:select(x))
field = 'water_level'

def insert_data_databasin(field,df):
    df = df.copy()
    for date,series in df.iterrows():
        #print(date)
        codigo = series
        try:
            rounded = np.round(series.values,5)
        except:
            rounded = series.values
        field_values = str(tuple(np.array(rounded,str)))
        values = str(tuple(zip([date.strftime('%Y-%m-%d %H:%M')]*series.index.size,np.array(list(map(lambda x:get_id(x),series.index)),str),np.array(rounded,str))))
        on_dup = ' ON DUPLICATE KEY UPDATE data_databasin.%s = VALUES(data_databasin.%s)'%(field,field)
        statement = ("INSERT INTO data_databasin (data_databasin.date,data_databasin.fk_id,data_databasin.%s) VALUES "%field+values[1:-1])+on_dup
        sql.execute_sql(statement)
        
insert_data_databasin(field,df)
field = 'water_level_color'
insert_data_databasin(field,df_colors)
#vsup
remote = cpr.SqlDb(**cpr.info.REMOTE)
vsup = remote.read_sql('SELECT * FROM estaciones where sv=1')

dfvsup = pd.DataFrame(index=df.index,columns=np.array(vsup.Codigo.values,int))
def read_surface_water_level(start,end,codigo):
    start = pd.to_datetime(start).strftime('%Y-%m-%d %H:%M') 
    end =  pd.to_datetime(end).strftime('%Y-%m-%d %H:%M')
    return "select fecha_hora,speedavg from velocidad_superficial_rio where fecha_hora between '%s' and '%s' and cliente = %s"%(start,end,codigo)

for codigo in dfvsup.columns:
    data= remote.read_sql(read_surface_water_level(start,end,codigo))
    data[data==-999] = np.NaN
    data = data.set_index('fecha_hora')['speedavg']
    dfvsup[codigo] = data.resample('5min').max().reindex(dfvsup.index)
dfvsup[dfvsup==0.0] = np.NaN
insert_data_databasin('water_surface_velocity',dfvsup)
