import numpy as np 
import pandas as pd 
import datetime as dt 
import os 
from wmf import wmf 
import json
from cprv1 import cprv1
import alarmas as al
import pickle
import eventos as ev

import multiprocessing
from multiprocessing import Pool

import warnings
warnings.filterwarnings('ignore')

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
    logging.basicConfig(filename = 'plotHistory2.log',level=logging.INFO)
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
def asses_plotHistory(start,end,posteriorR,N_in2,n_pronos,dfrad,timedeltaEv,rng1,path_fuentes,path_evs,path_bandas,pathevR,pathbandasR,rutafig):
    for est in  N_in2:
        # consulta nivel
        selfn = cprv1.Nivel(codigo=est, user='soraya', passwd='12345') 
        level=selfn.level(start,end)    
        # hay dato en los ultimos 5 min ?
        if np.isnan(level[-5:].mean()) or np.isnan(level[level.size-1]) == True:
            print '- No data in last 5 min. en Est.'+str(est)
            pass
        else:
            #hay datos
            level=level.resample('5T').mean()
            level_ob = pd.Series(level.values,index=np.arange(0,level.size))
            # tiene pron. est. ?
            if est in n_pronos.index:
                # es igual a cero ?
                if (n_pronos.loc[est][1:4]).all() == 0:
                    print '- PE in Est.'+str(est)+'= 0'
                    # si ha llovido en las ultimas 3 h.
                    if dfrad[est][start:end].sum() > 0: 
                        print 'However, Graph level3h for Est.'+str(est)
                        ev.plotN_history(est,level_ob,selfn,path_fuentes,path_evs,path_bandas,timedeltaEv,rng1,set_timing=True,rutafig=rutafig+'/Nivel/')
                else:
                    print 'Graph level3h + pronos for Est.'+str(est)
                    ev.plotN_history(est,level_ob,selfn,path_fuentes,path_evs,path_bandas,timedeltaEv,rng1,set_timing=True,n_pronos=n_pronos,rutafig=rutafig+'/Nivel/')
            # si ha llovido en las ultimas 3 h.
            elif dfrad[est][start:end].sum() > 0: 
                print 'Graph level3h for Est.'+str(est)
                ev.plotN_history(est,level_ob,selfn,path_fuentes,path_evs,path_bandas,timedeltaEv,rng1,set_timing=True,rutafig=rutafig+'/Nivel/')
            else:
                print '- No rain in last 3h for levelgraph in Est.'+str(est)
    
    #plots de lluvia
        # si ha llovido en las ultimas 3 h.
        if dfrad[est][start:end].sum() > 0 or dfrad[est][start:posteriorR].sum() > 0: 
            print 'Graph Pmean3h for Est.'+str(est)
            ev.plotPrad_history(est,dfrad,end,selfn,pathevR,pathbandasR,path_fuentes,timedeltaEv,rng1,posteriorR=posteriorR,rutafig=rutafig+'/Rad/')
        else:
            print '- No rain in last 3h for Pacumgraph Est.'+str(est)
        
# rutas necesarias
path_fuentes='/media/nicolas/Home/Jupyter/Sebastian/AvenirLTStd-Book/AvenirLTStd-Book.otf'
path_evs = '/media/nicolas/maso/Soraya/data/historicosN/eventos/'
path_bandas = '/media/nicolas/maso/Soraya/data/historicosN/bandas/'
pathevR= '/media/nicolas/maso/Soraya/data/historicosRadar/eventos/'
pathbandasR= '/media/nicolas/maso/Soraya/data/historicosRadar/bandas/'
rutafig= '/media/nicolas/Home/Jupyter/Soraya/Op_Alarmas/Result_to_web/plotHistoricos/'


# lee pronostico
ruta_config= '/media/nicolas/Home/Jupyter/Soraya/git/Alarmas/04_web_hidrologia/hidrologia/configfile_web.md'
listconfig = al.get_rutesList(ruta_config)
f=open(al.get_ruta(listconfig,'ruta_estadistico'))
n_pronos1=pickle.load(f)
f.close()
n_pronos = pd.DataFrame(n_pronos1,index= map(int,np.array(n_pronos1).T[0]))

# fechas radar
endR =  pd.to_datetime(dt.datetime.now().strftime('%Y-%m-%d %H:%M'))
startR = endR -  pd.Timedelta('3 hours')
startR, endR = pd.to_datetime(startR) , pd.to_datetime(endR)
posteriorR = endR + pd.Timedelta('30 min')
Dt=300
rutaNC= '/media/nicolas/Home/nicolas/101_RadarClass/'
cuenca= '/media/nicolas/maso/Mario/basins/260.nc'
#cuencas que me interesan.
N_in2 = np.array([ 91,  92,  93,  94,  96,  98,  99, 101, 106, 108, 109, 115, 116,
       124, 140, 143, 166, 179, 182, 183, 186, 187, 236, 238, 240, 245,
       251, 259, 260, 268])

#radar df
dfrad = ev.MeanHietogramRad_basins(startR,endR,rutaNC,Dt,cuenca,N_in2)

# fechas nivel
end= endR 
start= startR - pd.Timedelta('3 hours')

#Tick labels for evolution figures
timedeltaEv=5 #min,
# label time
hours=np.arange(-3,4)
rng1=[]
for i in range(hours.size):
    if hours[i]<0:
        rng1.append('-0'+str(np.abs(hours[i]))+':00')
    else:
        rng1.append('0'+str(np.abs(hours[i]))+':00')
rng1=np.array(rng1)

asses_plotHistory(start,end,posteriorR,N_in2,n_pronos,dfrad,timedeltaEv,rng1,path_fuentes,path_evs,path_bandas,pathevR,pathbandasR,rutafig=rutafig)