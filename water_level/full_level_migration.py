#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  Copyright 2018 MCANO <mario.cano@siata.gov.co>
import cprv1.cprv1 as cpr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os

def duplicate_existing_table(self,table_name):
    '''
    inserts data into SQL table from list of fields and values
    Parameters
    ----------
    table_name   = SQL db table name
    Returns
    -------
    Sql sentence,str
    '''
    df = self.read_sql('describe %s'%table_name)
    df['Null'][df['Null']=='NO'] = 'NOT NULL'
    df['Null'][df['Null']=='YES'] = 'NULL'
    sentence = 'CREATE TABLE %s ('%table_name
    if df[df['Extra']=='auto_increment'].empty:
        pk = None
    else:
        pk = df[df['Extra']=='auto_increment']['Field'].values[0]
    for id,serie in df.iterrows():
        if (serie.Default=='0') or (serie.Default is None):
            row = '%s %s %s'%(serie.Field,serie.Type,serie.Null)
        else:
            if (serie.Default == 'CURRENT_TIMESTAMP'):
                serie.Default = "DEFAULT %s"%serie.Default
            elif serie.Default == '0000-00-00':
                serie.Default = "DEFAULT '1000-01-01 00:00:00'"
            else:
                serie.Default = "DEFAULT '%s'"%serie.Default
            row = '%s %s %s %s'%(serie.Field,serie.Type,serie.Null,serie.Default)
        if serie.Extra:
            row += ' %s,'%serie.Extra
        else:
            row += ',' 
        sentence+=row
    if pk:
        sentence +='PRIMARY KEY (%s)'%pk
    else:
        sentence = sentence[:-1]
    sentence +=');'
    return sentence


def default_values(self,table_name):
    describe_table = self.read_sql('describe %s'%table_name)
    not_null = describe_table['Default'].notnull()
    default_values = describe_table[['Field','Default','Type']][not_null].set_index('Field')[['Default','Type']]
    return default_values



def siata_remote_data_to_transfer(start,end,*args,**kwargs):
    remote = cpr.Nivel(**cpr.info.REMOTE)
    codigos_str = '('+str(list(local.infost.index)).strip('[]')+')'
    parameters = tuple([codigos_str,local.fecha_hora_query(start,end)])
    df = remote.read_sql('SELECT * FROM datos WHERE cliente in %s and %s'%parameters)
    return df

def filter_data_to_transfer(start,end,local_path=None,remote_path=None,**kwargs):
    transfer = siata_remote_data_to_transfer(start,end,**kwargs)
    def convert(x):
        try:
            value = pd.to_datetime(x).strftime('%Y-%m-%d')
        except:
            value = np.NaN
        return value
    transfer['fecha'] = transfer['fecha'].apply(lambda x:convert(x))
    transfer = transfer.loc[transfer['fecha'].dropna().index]
    if local_path:
        transfer.to_csv(local_path)
        if remote_path:
            os.system('scp %s %s'%(local_path,remote_path))
    return transfer

def export_tables(self,tablas):
    for tabla in tablas:
        initia = datetime.datetime.now()
        filename = '%s.csv'%tabla
        local_path = '/media/nicolas/maso/Mario/data_migration/%s'%filename
        remote_path = "mcano@siata.gov.co:data_migration/%s"%filename
        self.read_sql('select * from %s'%tabla).to_csv(local_path)
        print(os.system('scp %s %s'%(local_path,remote_path)))
        print('filename:%s,took:%s'%(filename,datetime.datetime.now()-inicia))
        
        

remote = cpr.Nivel(**cpr.info.REMOTE)
local = cpr.Nivel(codigo=99,user='sample_user',passwd='s@mple_p@ss')

initial_run = datetime.datetime.now()
df = remote.read_sql("select codigo,FechaInstalacion from estaciones where red='Nivel'")
s = pd.to_datetime(df['FechaInstalacion'])
s[s<'2000-01-01'] = np.NaN
initial = pd.to_datetime("2017-11-26")
daily = pd.date_range(initial,datetime.datetime.now())
sundays = daily[daily.strftime('%a') =='dom']
# arguments
for count in np.arange(0,len(sundays)-1):
    inicia = datetime.datetime.now()
    start = sundays[count]
    end   = sundays[count+1]
    date_format = '%Y-%m-%d'
    filename = 'weekly_level_%s_%s.csv'%(start.strftime(date_format),end.strftime(date_format))
    local_path = '/media/nicolas/maso/Mario/weekly_data/%s'%filename
    remote_path = "mcano@siata.gov.co:weekly_data/%s"%filename
    filter_data_to_transfer(start,end,local_path,remote_path)
    print('%d out of %s:filename:%s,took:%s'%(count,len(sundays),filename,datetime.datetime.now()-inicia))
print('TOOK %s'%(datetime.datetime.now()-initial_run))
