#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  Copyright 2018 MCANO <mario.cano@siata.gov.co>
import matplotlib.dates as mdates
import cprv1.cprv1 as cpr
import pandas as pd
import numpy as np
import datetime
import time
import matplotlib.pyplot as plt
import os

self = cpr.Nivel(codigo=140,user='sample_user',passwd='s@mple_p@ss',SimuBasin=True)
end = self.round_time(datetime.datetime.now())
start = self.round_time(end - datetime.timedelta(days=2))
df = self.level_all(start,end)
df = df.resample('5min').mean()
dfr = self.risk_df(df)
dfr[dfr<2]=np.NaN
dfr = dfr.drop(260)
for codigo in dfr.index:
    try:
        self = cpr.Nivel(codigo=codigo,user='sample_user',passwd='s@mple_p@ss',SimuBasin=True)
        dates = dfr.loc[codigo].dropna().index
        self.radar_rain(dates[0],dates[-1])
        print codigo
        for date in dates:
            try:
                self.rain_report(date)
                print date
            except:
                pass
    except:
        pass
