#!/bin/bash

date
appdir=`dirname $0`
logfile=$appdir/acumRad_3h.log
lockfile=$appdir/acumRad_3h.lck
pid=$$

echo $appdir

function acumRad_3h {

python /media/nicolas/Home/Jupyter/Soraya/git/Alarmas/06_Crones/acumRad_3h.py

}


(
        if flock -n 601; then
                cd $appdir
                acumRad_3h
                echo $appdir $lockfile
                rm -f $lockfile
        else
            	echo "`date` [$pid] - Script is already executing. Exiting now." >> $logfile
        fi
) 601>$lockfile

exit 0