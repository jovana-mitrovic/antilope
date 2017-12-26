#!/bin/bash

declare -a overlap=(1)
declare -a met=('adam')
declare -a lr=(0.001) #lr=(0.01 0.001 0.0001 0.00001)

declare -a sc=(0.2) #0.5 1 2)
declare -a p=(5 10 20) #(5 10 20 20 50 50 100 100 100)
declare -a n=(0 5 10) #5 10 10 20 10 20 50)
declare -a KLp=(-1 0 1) #(-1 -0.5 -0.1 0 0.1 0.5 1)

for j in "${overlap[@]}"
do
  for k in "${met[@]}"
  do
    for a in "${lr[@]}"
    do
#       for b in "${lrd[@]}"
#       do
       for c in "${sc[@]}"
        do
         for d in "${!p[@]}"
         do
          for e in "${KLp[@]}"
          do
             th core/train.lua --weightInit 'kernel' 'KLinear' --overlap $j --optimization $k --learningRate $a  --weightScale $c --numPos ${p[d]} --numNeg ${n[d]} --KLinearParam $e &
             done
           done
         done
#        done
     done
   done
done
