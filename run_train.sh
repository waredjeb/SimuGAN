#!/bin/bash

if [ ! -d $PWD"/results" ]; 
then
    mkdir $PWD/results && echo "Results' directory created in" $PWD
    
else
    echo "The results' directory already exists in" $PWD
fi

echo "####### JOB STARTING #######"
python gan_train.py -e 1000000 -b 64


