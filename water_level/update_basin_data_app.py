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

sql = cpr.SqlDb('hydrology','sample_user','localhost','s@mple_p@ss',3306,'data_data')


def read(start,end):
    start = pd.to_datetime(start).strftime('%Y-%m-%d %H:%M') 
    end =  pd.to_datetime(end).strftime('%Y-%m-%d %H:%M')
    fields = 'fk_id, date,water_level, radar_rain,water_surface_velocity, water_level_color'
    data = sql.read_sql("SELECT %s from data_databasin where date between '%s' and '%s'"%(fields,start,end))
    data[['fk_id','water_level','radar_rain','water_surface_velocity','water_level_color']]
    return data

end = datetime.datetime.now()
start = end - datetime.timedelta(minutes=155)
df = read(start,end)

if df.index.size == 2232:
    pass
elif df.index.size>2232:
    df = read(start,end-datetime.timedelta(minutes=5))
else:
    df = read(start,end+datetime.timedelta(minutes=5))
columns = ['color','hour','pk','water_level_history_path','radar_rain_history_path','statistical_model_path','picture_path','camera_path','path','name','location','longitude','latitude','water_level','radar_rain','water_surface_velocity']
def color_value(color):
    try:
        colors = {'#D8E0E8':0,'green':1,'#FAF16A':2,'orange':3,'red':4}
        return colors[color]
    except:
        return 0
df['color_value'] = list(map(lambda x:color_value(x),df.water_level_color))
def get_id(codigo):
    sql = cpr.SqlDb('hydrology','sample_user','localhost','s@mple_p@ss',3306,'data_data')
    id = sql.read_sql("select id from meta_basin where codigo='%s'"%codigo).iloc[0].id
    return id

order = df.groupby('fk_id')['color_value'].sum().sort_values()
df = df.set_index('fk_id').sort_values('date').loc[order.index].drop('color_value',axis=1).reset_index()
df.columns = ['id']+list(df.columns[1:])
meta = sql.read_sql('select id,nombre,three_hours_image_path,slug from meta_basin')
df[df==0.0] = np.NaN
data = pd.merge(df, meta, on='id')
data = data.drop('id',axis=1)
data['hour'] = map(lambda x:pd.to_datetime(x).strftime('%H:%M'),list(data.date))
data['date'] = map(lambda x:pd.to_datetime(x).strftime('%Y-%m-%d %H:%M:00'),list(data.date))
data = data.set_index('date')
if data.index.size == 2232:
    data.to_csv('/home/nicolas/Dev/fullapp/src/staticfiles/data.csv')
else:
    print('something went wrong')
print('done')
