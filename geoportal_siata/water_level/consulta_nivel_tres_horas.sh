#!/bin/bash

date
appdir=`dirname $0`
logfile=$appdir/NombreDelProceso.log
lockfile=$appdir/NombreDelProceso.lck
pid=$$

echo $appdir

function NombreDelProceso {

python /media/nicolas/Home/Jupyter/MarioLoco/repositories/CPR/crones/reportes_nivel_tres_horas.py

}


(
        if flock -n 201; then
                cd $appdir
                NombreDelProceso
                echo $appdir $lockfile
                rm -f $lockfile
        else
            	echo "`date` [$pid] - Script is already executing. Exiting now." >> $logfile
        fi
) 201>$lockfile

exit 0
