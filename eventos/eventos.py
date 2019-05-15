#!/usr/bin/env python
# -*- coding: utf-8 -*-

#  Copyright 2018 scastil <socastillogi@unal.edu.co>

import numpy as np 
import pandas as pd 
import pylab as pl 
import datetime as dt 
import os 
from wmf import wmf 
import json
from cprv1 import cprv1
import alarmas as al
import pickle

import matplotlib.font_manager as fm
import glob

import netCDF4

#funciones eventos

def MeanHietogramRad_basins(start,end,rutaNC,Dt,cuenca,codigos):
    '''
    Lee .nc's en 101Radar_Class en el periodo y frecuencia indicados y saca el hietograma promedio por cuencas
    siempre que hayan mascaras de estas. Divide por 1000 el campo de radar por dentro.
    
    '''
    #hora UTC
    startUTC,endUTC = start + pd.Timedelta('5 hours'), end + pd.Timedelta('5 hours')
    fechaI,fechaF,hora_1,hora_2 = startUTC.strftime('%Y-%m-%d'), endUTC.strftime('%Y-%m-%d'),startUTC.strftime('%H:%M'),endUTC.strftime('%H:%M')
    #Obtiene las fechas por dias
    datesDias = pd.date_range(fechaI, fechaF,freq='D')

    a = pd.Series(np.zeros(len(datesDias)),index=datesDias)
    a = a.resample('A').sum()
    Anos = [i.strftime('%Y') for i in a.index.to_pydatetime()]

    datesDias = [d.strftime('%Y%m%d') for d in datesDias.to_pydatetime()]

    ListDays = []
    ListRutas = []
    for d in datesDias:
        try:
            L = glob.glob(rutaNC + d + '*.nc')
            ListRutas.extend(L)
            for i in L:
                if i[-11:].endswith('extrapol.nc'):
                    ListDays.append(i[-32:-20])
                else:
                    ListDays.append(i[-23:-11])
        except:
            print 'mierda'
    #Organiza las listas de dias y de rutas
    ListDays.sort()
    ListRutas.sort()
    datesDias = [dt.datetime.strptime(d[:12],'%Y%m%d%H%M') for d in ListDays]
    datesDias = pd.to_datetime(datesDias)
    #Obtiene las fechas por Dt
    textdt = '%d' % Dt
    #Agrega hora a la fecha inicial
    if hora_1 <> None:
            inicio = fechaI+' '+hora_1
    else:
            inicio = fechaI
    #agrega hora a la fecha final
    if hora_2 <> None:
            final = fechaF+' '+hora_2
    else:
            final = fechaF
    datesDt = pd.date_range(inicio,final,freq = textdt+'s')

    #Obtiene las posiciones de acuerdo al dt para cada fecha
    PosDates = []
    pos1 = [0]
    for d1,d2 in zip(datesDt[:-1],datesDt[1:]):
            pos2 = np.where((datesDias<d2) & (datesDias>=d1))[0].tolist()
            if len(pos2) == 0:
                    pos2 = pos1
            else:
                    pos1 = pos2
            PosDates.append(pos2)

    # acumular dentro de la cuenca.
    cu = wmf.SimuBasin(rute= cuenca)
    # hora local
    datesDt = datesDt - dt.timedelta(hours=5)
    datesDt = datesDt.to_pydatetime()
    #Index de salida en hora local
    rng= pd.date_range(start,end, freq=  textdt+'s')
    df = pd.DataFrame(index = rng,columns=codigos)
    
#     meanM=[]

    for dates,pos in zip(datesDt[1:],PosDates):
            rvec = np.zeros(cu.ncells)
            try:
                    for c,p in enumerate(pos):
                            #Lee la imagen de radar para esa fecha
                            g = netCDF4.Dataset(ListRutas[p])
    #                         print ListRutas[p]
                            RadProp = [g.ncols, g.nrows, g.xll, g.yll, g.dx, g.dx]
                            #Agrega la lluvia en el intervalo 
                            rvec += cu.Transform_Map2Basin(g.variables['Rain'][:].T/ (12*1000.0),RadProp)
                            #Cierra el netCDF
                            g.close()
            except Exception, e:
                    print 'error - zero field '
                    rvec = np.zeros(cu.ncells)
            mean = []
            
#             meanM.append(rvec.mean())
            
            for codigo in codigos:
                if 'mask_%s.tif'%(codigo) in os.listdir('/media/nicolas/maso/Mario/mask/'):
                    mask_path = '/media/nicolas/maso/Mario/mask/mask_%s.tif'%(codigo)
                    mask_map = wmf.read_map_raster(mask_path)
                    mask_vect = cu.Transform_Map2Basin(mask_map[0],mask_map[1])
                else:
                    mask_vect = None
                if mask_vect is not None:
                #for date in rain_vect.index:
                    try:
                        mean.append(np.sum(mask_vect*rvec)/np.sum(mask_vect))
                    except:
                        mean.append(np.nan)
                # se actualiza la media de todas las mascaras en el df.
            df.loc[dates.strftime('%Y-%m-%d %H:%M:%S')]=mean  
    
    return df

#plotN_history

def plotN_history(est,level_ob,selfN,path_fuentes,path_evs,path_bandas,timedeltaEv,rng1,set_timing=False,n_pronos=None,rutafig=None):
    # fig properties
    fonttype = fm.FontProperties(fname=path_fuentes)
    pl.rc('axes',labelcolor='#4f4f4f')
    pl.rc('axes',linewidth=1.25)
    pl.rc('axes',edgecolor='#4f4f4f')
    pl.rc('text',color= '#4f4f4f')
    pl.rc('text',color= '#4f4f4f')
    pl.rc('xtick',color='#4f4f4f')
    pl.rc('ytick',color='#4f4f4f')
    fonttype.set_size(12)
    legendfont=fonttype.copy()
    legendfont.set_size(16.5)
    colors=['#C7D15D','#3CB371', '#22467F']

    if set_timing == True:
        Dt = int(level_ob.index[level_ob.size-1] - level_ob.argmax())

        # set timing whti the shapes.
        if n_pronos is not None:    
            texty = 0.42
            if n_pronos.loc[est][2] > level_ob[level_ob.argmax()]:
                serieNob=level_ob.shift(-6)#la mitad de pasos de 5  min en una hora, talque quede en el centro de la fig.
                Nob_steps = -12*3
                serieNob2 = pd.Series(serieNob[Nob_steps:].values,index=np.arange(0,serieNob[Nob_steps:].size))
                scatter_posx= serieNob2.size
            else:
                Nob_steps = (-12*3 )- Dt 
                serieNob2 = pd.Series(level_ob[Nob_steps:].values,index=np.arange(0,level_ob[Nob_steps:].size))
                scatter_posx= serieNob2.size + 6 # 30 min ahead.

        else:
            texty = 0.34
            Nob_steps = (-12*3 )- Dt 
            serieNob2 = pd.Series(level_ob[Nob_steps:].values,index=np.arange(0,level_ob[Nob_steps:].size))
            scatter_posx= serieNob2.size + 6 # 30 min ahead.
    else:
        serieNob2 = level_ob
        texty = 0.34

    #bandas y ob, ev max.
    dfEv=pd.read_csv(path_evs+'dfNevs5m_'+str(est)+'.csv',index_col=0)
    Nbandas=pd.read_csv(path_bandas+'bandas5m_'+str(est)+'.csv',index_col=0)
    fig=pl.figure(figsize=(12,6),dpi=110)
    ax=fig.add_subplot(111)
    ax.fill_between(np.arange(Nbandas['0.1'].size),Nbandas['0.25'],Nbandas['0.75'], color = 'gray', alpha = 0.3,label='$P_{25-75}$')
    ax.fill_between(np.arange(Nbandas['0.1'].size),Nbandas['0.1'],Nbandas['0.9'], color = 'c', alpha = 0.3,label='$P_{10-90}$')
    ax.plot(Nbandas['0.5'],color='darkcyan',label='Mediana',lw=4)
    evmax=dfEv.max().argmax()
    serie_evmax = dfEv[evmax]
    ax.plot(serie_evmax,c='darkcyan',lw=2,label=u'$Ev_{máx}$ '+str(evmax[:-3]),ls='--',alpha=0.6)
    ax.plot(serieNob2, c='k', lw=3.5)
    if n_pronos is not None:
        ax.scatter(scatter_posx,n_pronos.loc[est][2],c='crimson',marker='v',s=110,label=u'Nmax30')
        # ax.errorbar(scatter_posx,n_pronos.loc[est][2], ecolor='crimson', yerr= [10,20])

    # properties
    ax.set_title('Est. '+str(est)+' | ' + str(selfN.infost[selfN.infost.keys()[1]].sort_index().loc[est]),fontsize=18.5,fontproperties=fonttype)
    xticks=np.arange(0,Nbandas.shape[0],60/timedeltaEv)
    ax.set_xticks(xticks)
    ax.set_xticklabels(rng1)
    pl.yticks(fontproperties=fonttype)
    pl.xticks(fontproperties=fonttype)
    ax.tick_params(labelsize=14)
    ax.set_ylabel('Nivel $[cm]$',fontproperties=fonttype,fontsize=18.5)
    ax.set_xlabel(u'Tiempo respecto al $N_{máx}$',fontproperties=fonttype,fontsize=18.5)

    # nrisks
    riskcolor=['k','green','yellow','orange','r']    
    Qrisks=np.insert(selfN.infost[['n1','n2','n3','n4']].loc[est].values,0,0)
    if ax.get_yticks()[-1] > Qrisks[-1]:
        pass
    else:
        ax.set_ylim(0,Qrisks[-1]+10)
    for index,qrisk in enumerate(Qrisks[:-1]):    
        ax.axvspan(73,75,Qrisks[index]/ax.get_ylim()[1],Qrisks[index+1]/ax.get_ylim()[1],color=riskcolor[index+1])
    ax.set_xlim(-1,76)

    # text
    ytext = ax.get_ylim()[0] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * texty
    xtext = ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.565
    text = ax.text(xtext,ytext,str(dfEv.shape[1])+' eventos entre '+ selfN.infost['fecha_instalacion'].sort_index().loc[est].strftime('%Y-%m')+' y 2018-10',
            fontsize=16.25,
            bbox=dict(edgecolor='#c1c1c1',facecolor='w',boxstyle='round,pad=0.27'),
            fontproperties=fonttype)
    #legend
    legend = ax.legend(fontsize=15,bbox_to_anchor =(1,-0.150),ncol=4,prop=legendfont)
    if rutafig is not None:
        pl.savefig(rutafig+'Nshape_'+str(est)+'.png',bbox_inches='tight',bbox_extra_artists=[legend,text])

def plotPrad_history(est,dfrad,endR,selfN,pathevR,pathbandasR,path_fuentes,timedeltaEv,rng1,posteriorR=None,rutafig=None):
    # fig properties
    fonttype = fm.FontProperties(fname=path_fuentes)
    pl.rc('axes',labelcolor='#4f4f4f')
    pl.rc('axes',linewidth=1.25)
    pl.rc('axes',edgecolor='#4f4f4f')
    pl.rc('text',color= '#4f4f4f')
    pl.rc('text',color= '#4f4f4f')
    pl.rc('xtick',color='#4f4f4f')
    pl.rc('ytick',color='#4f4f4f')
    fonttype.set_size(12)
    legendfont=fonttype.copy()
    legendfont.set_size(16.5)
    colors=['#C7D15D','#3CB371', '#22467F']
    
    dfEv=pd.read_csv(pathevR+'dfPmeanEv5m_'+str(est)+'.csv',index_col=0)
    Nbandas=pd.read_csv(pathbandasR+'bandasP5m_'+str(est)+'.csv',index_col=0)
    fig=pl.figure(figsize=(12,6),dpi=110)
    ax=fig.add_subplot(111)

    ax.fill_between(np.arange(Nbandas['0.1'].size),Nbandas['0.25'],Nbandas['0.75'], color = 'gray', alpha = 0.3,label='$P_{25-75}$')
    ax.fill_between(np.arange(Nbandas['0.1'].size),Nbandas['0.1'],Nbandas['0.9'], color = 'c', alpha = 0.3,label='$P_{10-90}$')
    ax.plot(Nbandas['0.5'],color='darkcyan',label='Mediana',lw=4)
    evmax=dfEv.cumsum().max().argmax()
    serie_obs = dfEv[evmax].cumsum()
    ax.plot(serie_obs,c='darkcyan',lw=2,label=u'$Ev_{máx}$ '+str(evmax[:-3]),ls='--',alpha=0.6)
    ax.set_title('Est. '+str(est)+' | ' + str(selfN.infost[selfN.infost.keys()[1]].sort_index().loc[est]),fontsize=18.5,fontproperties=fonttype)
    ax.plot(dfrad[est][:endR].cumsum().values[1:],color='k',lw=3, label='$P_{obs.}$')
    if posteriorR is not None:
        ax.plot(dfrad[est][:posteriorR].cumsum().values[1:],ls='--',c='k',lw=3, label='$P_{pron.}$')

    legend = ax.legend(fontsize=15,bbox_to_anchor =(1,-0.150),ncol=3,prop=legendfont)
    xticks=np.arange(0,Nbandas.shape[0],60/timedeltaEv)
    ax.set_xticks(xticks)
    ax.set_xticklabels(rng1)
    pl.yticks(fontproperties=fonttype)
    pl.xticks(fontproperties=fonttype)
    ax.tick_params(labelsize=14)
    ax.set_ylabel('$P_{Acum.}$ promedio en la cuenca $[mm]$',fontproperties=fonttype,fontsize=18.5)
    ax.set_xlabel(u'Tiempo respecto al $N_{máx}$',fontproperties=fonttype,fontsize=18.5)

    ytext = ax.get_ylim()[0] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.42
    xtext = ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.575
    text = ax.text(xtext,ytext,str(dfEv.shape[1])+' eventos entre '+ selfN.infost['fecha_instalacion'].sort_index().loc[est].strftime('%Y-%m')+' y 2018-10',
            fontsize=16.25,
            bbox=dict(edgecolor='#c1c1c1',facecolor='w',boxstyle='round,pad=0.27'),
            fontproperties=fonttype)
    ax.set_xlim(-1,73)
    if rutafig is not None:
        pl.savefig(rutafig+'Pshape_'+str(est)+'.png',bbox_inches='tight',bbox_extra_artists=[legend,text])
        