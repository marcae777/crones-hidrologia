#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  Copyright 2018 MCANO <mario.cano@siata.gov.co>
import cprv1.cprv1 as cpr
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import os
import numpy as np

os.system('scp /home/nicolas/self_code/Crones/update_local_cron.log mcano@siata.gov.co:/var/www/mario')
os.system('scp /home/nicolas/update_local.log mcano@siata.gov.co:/var/www/mario')

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
    logging.basicConfig(filename = 'update_local.log',level=logging.INFO)
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
def update_all(self,df):
    for codigo in self.infost.index:
        obj = cpr.Nivel(codigo=codigo,user='sample_user',passwd='s@mple_p@ss')
        obj.table = 'hydro'
        obj.update_series(df[codigo],'nivel')
    
self = cpr.Nivel(codigo = 99,user='sample_user',passwd='s@mple_p@ss',SimuBasin=False)
df = pd.read_csv('/media/nicolas/maso/Mario/tres_horas.csv',index_col=0)
df.index = pd.to_datetime(df.index)
df.columns = np.array(df.columns,int)
df = df.iloc[-6:] #solo actualiza última hora
update_all(self,df)
os.system('scp /home/nicolas/update_local.log mcano@siata.gov.co:/var/www/mario')
os.system('scp /home/nicolas/self_code/Crones/update_local_cron.log mcano@siata.gov.co:/var/www/mario')
