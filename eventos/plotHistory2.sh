#!/bin/bash

date
appdir=`dirname $0`
logfile=$appdir/plotHistory2.log
lockfile=$appdir/plotHistory2.lck
pid=$$

echo $appdir

function plotHistory2 {
    python2 /media/nicolas/Home/Jupyter/Soraya/git/Alarmas/06_Crones/plotHistory2.py
}


(
        if flock -n 301; then
                cd $appdir
                plotHistory2
                echo $appdir $lockfile
                rm -f $lockfile
        else
            	echo "`date` [$pid] - Script is already executing. Exiting now." >> $logfile
        fi
) 301>$lockfile

exit 0