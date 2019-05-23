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
import multiprocessing

def cron():
    def convert_to_risk(df):
        df = self.risk_df(df)
        return df[df.columns.dropna()]

    #PROCESS
    self = cpr.Nivel(codigo = 99,user='sample_user',passwd='s@mple_p@ss',SimuBasin=False)
    end = datetime.datetime.now()
    start = end - datetime.timedelta(minutes=150)
    df = self.level_all(start,end,calidad=True)
    risk_df = convert_to_risk(df.copy())

    def select(pos):
        try:
            colors = ['green','#F7DD1E','#F9A43E','#FC3A3A','#FC3A3A']
            return colors[pos]
        except IndexError:
            return '#D8E0E8'

    filepath = "http://siata.gov.co/mario/realTime/tres_horas/"
    data_list = []
    for codigo in risk_df.index[::-1]:
        positions = np.array(risk_df.loc[codigo].values,int)
        color = map(lambda x:select(x),positions)
        data = pd.DataFrame(color,index=risk_df.loc[codigo].index)
        data['hour'] = risk_df.loc[codigo].index.strftime('%H:%M')
        data['nombre'] = "%s | "%codigo+str(self.infost.loc[codigo,'nombre'])
        data['path']= filepath + self.infost.loc[codigo,'slug']+'.png'
        data['location'] = self.infost.loc[codigo,'municipio']
        data['telefono'] = 'No'
        data['celular'] = 'No'
        data['waterlevel'] = df.T.loc[codigo]
        data.columns = ['color','hour','name','path','location','phone','mobile','waterlevel']
        data_list.append(data)

    data_list = pd.concat(data_list)
    data_list.name = data_list.name.str.replace(' - Nivel','')
    filename = 'heatmap_data.csv'
    data_list.to_csv(filename)
    statement = 'scp %s mcano@siata.gov.co:/var/www/mario/realTime/risk_levels_chart/%s'%(filename,filename)
    print(os.system(statement))
    print(os.system(statement))
    print(os.system(statement))

    
if __name__ == '__main__':
    p = multiprocessing.Process(target=cron, name="")
    p.start()
    time.sleep(290) # wait near 5 minutes to kill process
    p.terminate()
    p.join()