#!/bin/bash

if [ $# -eq 0 ]; then
	echo "please offer a parameter"
	return
fi

case $1 in
	start)
		python main.py
		echo "start detecting"
		;;
	create-env|c)
		if [ ! -d "./projet" ]; then
			virtualenv --system-site-packages projet || return
			. projet/bin/activate || return
			pip install -r requirements.txt || return
		else
			echo "environment projet exist already."
		fi
		;;
	push-to-server|p)
		scp main.py gongshang@47.93.27.152:~
		;;	
	*)
		echo "invalide"
		;;
esac
