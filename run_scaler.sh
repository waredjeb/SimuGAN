#!/bin/bash

if [ ! -d $PWD"scaler" ]; 
then
    mkdir $PWD/scaler && echo "Scaler directory created in" $PWD
    
else
    echo "The scaler directory already exists in" $PWD
fi 

echo "####### Scaler job is starting #######"

python scaler.py
