#!/bin/bash

for i in {0..9}
do
	sleep 5
	python2.7 -u main_midvggm_tf.py NC_inc/run$i 20 > output$i.txt 2> errors$i.txt
done
