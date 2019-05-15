#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  CRP.py
#
#  Copyright 2018 MCANO <mario.cano@siata.gov.co>
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  CRP.py
#
#  Copyright 2018 MCANO <mario.cano@siata.gov.co>

import cprv1.cprv1 as cpr
import datetime
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates

self = cpr.Nivel(codigo=99,user='sample_user',passwd='s@mple_p@ss',SimuBasin=True)
date = datetime.datetime.now()

fechas = []
#def reporte_diario(self,date):
end = pd.to_datetime(pd.to_datetime(date).strftime('%Y-%m-%d')+' 23:50') - datetime.timedelta(days=1)
start = (end-datetime.timedelta(days=6)).strftime('%Y-%m-%d 00:00')
folder_path = '/media/nicolas/Home/Jupyter/MarioLoco/reporte_diario/%s'%end.strftime('%Y%m%d')
folder_name = end.strftime('%Y%m%d')
os.system('mkdir %s'%folder_path)
df = self.level_all(start,end,calidad=True)

from matplotlib.patches import Rectangle
try:
    df = df.T.drop([1013,1014,195,196]).T
except:
    pass
daily = df.resample('D').max()

rdf = self.risk_df(daily)
# niveles de riesgo en el último día
last_day_risk = rdf[rdf.columns[-1]].copy()
last_day_risk = last_day_risk[last_day_risk>0.0].sort_values(ascending=False).index
rdf = rdf.loc[rdf.max(axis=1).sort_values(ascending=False).index]
rdf = rdf[rdf.max(axis=1)>0.0]
rdf = rdf.fillna(0)
labels = []
for codigo,nombre in zip(self.infost.loc[rdf.index].index,self.infost.loc[rdf.index,'nombre'].values):
    labels.append('%s | %s'%(codigo,nombre))
rdf.index = labels
def to_col_format(date):
    return (['L','M','MI','J','V','S','D'][int(date.strftime('%u'))-1]+date.strftime('%d'))
rdf.columns = map(lambda x:to_col_format(x),rdf.columns)
import sys
# sys.setdefaultencoding() does not exist, here!
reload(sys)  # Reload does the trick!
sys.setdefaultencoding('UTF8')
self.plot_risk_daily(rdf,figsize=(14,20))
plt.savefig(folder_path+'/reporte_nivel.png',bbox_inches='tight')
remote_path = 'mcano@siata.gov.co:/var/www/mario/reporte_diario/'
query = "rsync -r %s %s/"%(folder_path+'/reporte_nivel.png',remote_path+end.strftime('%Y%m%d'))
os.system(query)
#Graficas
fontsize = 25
font = {'size'   :fontsize}
plt.rc('font', **font)
filepath = None
try:
    obj = cpr.Nivel(codigo=260,user='sample_user',passwd='s@mple_p@ss',SimuBasin=True)
    end = pd.to_datetime(pd.to_datetime(date).strftime('%Y-%m-%d')+' 23:55') - datetime.timedelta(days=1)
    start = (end-datetime.timedelta(days=6)).strftime('%Y-%m-%d 00:00')
    print('end=%s'%end)
    radar_rain = obj.radar_rain_vect(start,end)
    diario = radar_rain.resample('D').sum()
    rain = obj.radar_rain(start,end)
    fig = plt.figure(figsize=(20,20))
    for pos,dia in enumerate(diario.index):
        ax = fig.add_subplot(3,3,pos+1)
        obj.rain_area_metropol(diario.loc[dia].values/1000.0,ax)
        ax.set
        plt.gca().set_title(rdf.columns[pos])
    plt.savefig(folder_path+'/lluvia_diaria.png',bbox_inches='tight')
    remote_path = 'mcano@siata.gov.co:/var/www/mario/reporte_diario/'
    query = "rsync -r %s %s/"%(folder_path+'/lluvia_diaria.png',remote_path+folder_name)
    os.system(query)
    took = (datetime.datetime.now()-date)    
    print(took)
except:
    print('ERROR %s'%codigo)
    os.system('rm -r /media/nicolas/maso/Mario/user_output/radar')
    os.system('mkdir /media/nicolas/maso/Mario/user_output/radar')
    obj = cpr.Nivel(codigo=260,user='sample_user',passwd='s@mple_p@ss',SimuBasin=True)
    radar_rain = obj.radar_rain_vect(start,end)
    diario = radar_rain.resample('D').sum()
    rain = obj.radar_rain(start,end)
    fig = plt.figure(figsize=(20,20))
    for pos,dia in enumerate(diario.index):
        ax = fig.add_subplot(3,3,pos+1)
        obj.rain_area_metropol(diario.loc[dia].values/1000.0,ax)
        ax.set
        plt.gca().set_title(rdf.columns[pos])
    plt.savefig(folder_path+'/lluvia_diaria.png',bbox_inches='tight')
    remote_path = 'mcano@siata.gov.co:/var/www/mario/reporte_diario/'
    query = "rsync -r %s %s/"%(folder_path+'/lluvia_diaria.png',remote_path+folder_name)
    os.system(query)
    took = (datetime.datetime.now()-date)    
    print(took)

for num,codigo in enumerate(np.array(last_day_risk,int)):
    try:
        obj = cpr.Nivel(codigo=codigo,user='sample_user',passwd='s@mple_p@ss',SimuBasin=False)
        series = df.loc[daily.index[-1]-datetime.timedelta(days=1):][codigo]
        plt.figure()
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(25,4))
        obj.plot_level(series/100.0,
                        lamina='max',
                        risk_levels=np.array(obj.risk_levels)/100.0,
                        legend=False,
                        resolution='m',
                        ax=ax1,
                        scatter_size=40)
        alpha=0.2
        bat = pd.read_csv('/media/nicolas/Home/Jupyter/MarioLoco/ultimos_levantamientos/%s.csv'%codigo,index_col=0)
        ymax = max([bat['y'].max(),(obj.risk_levels[-1])/100.0])
        lamina = 'max'
            # plot section
        if series.dropna().index.size == 0:
            lamina = 0.0
        else:
            if lamina == 'current':
                x,lamina = (series.dropna().index[-1],series.dropna().iloc[-1])
            else:
                x,lamina = (series.argmax(),series.max())

        sections =obj.plot_section(bat,
                                ax = ax2,
                                level=lamina/100.0,
                                riskLevels=np.array(obj.risk_levels)/100.0,
                                xSensor=obj.info.x_sensor,
                                scatterSize=50)
        major_locator        = mdates.DayLocator(interval=5)
        formater = '%H:%M'
        ax1.xaxis.set_major_formatter(mdates.DateFormatter(formater))
        ax1.set_xlabel(u'Fecha')
        ax2.spines['top'].set_color('w')
        ax2.spines['right'].set_color('w')
        ax2.spines['right'].set_color('w')
        ax2.set_ylim(bat['y'].min(),ymax)
        ax1.set_ylim(bat['y'].min(),ymax)
        ax1.set_title(u'código: %s'%codigo)
        ax2.set_title('Profundidad en el canal')
        ax2.set_ylabel('Profundidad [m]')
        #ax1.set_xlabel('03 Mayo - 04 Mayo')
        ax1.annotate(u'máximo', (mdates.date2num(series.argmax()), series.max()/100.0), xytext=(10, 10),textcoords='offset points',fontsize=fontsize)
        #file = 'section_%s.png'%(num+1)
        ax2.set_title(obj.info.nombre)
        #filepath = 'reportes_amva/%s.png'%codigo
        for tick in ax1.get_xticklabels():
            tick.set_rotation(45)
        filepath = folder_path+'/'+obj.info.slug+'.png'
        plt.savefig(filepath,bbox_inches='tight')
        sending = 'rsync %s %s'%(filepath,remote_path+folder_name+'/')
        print(sending)
        os.system(sending)
        os.system('rsync %s %s'%(filepath,remote_path+folder_name+'/'))
        os.system('rsync %s %s'%(filepath,remote_path+folder_name+'/'))
        try:
            print('getting radar')
            obj = cpr.Nivel(codigo=codigo,user='sample_user',passwd='s@mple_p@ss',SimuBasin=True)
            print('done')
            obj = cpr.Nivel(codigo=codigo,user='sample_user',passwd='s@mple_p@ss',SimuBasin=True)
            start = series.argmax()-datetime.timedelta(hours=4)
            end = series.argmax()+datetime.timedelta(hours=1)
            obj.gif(start,end)
            filepath = '/media/nicolas/maso/Mario/user_output/gifs/%s/*.gif'%obj.file_format(start,end)
            scp = 'scp %s %s'%(filepath,remote_path+folder_name+'/')
            print(os.system(scp))
            print(os.system(scp))
            print(os.system(scp))
        except:
            print('no gif for : %s'%codigo)
    except:
        print('ERROR: %s didnt work'%codigo) 


print(last_day_risk)
