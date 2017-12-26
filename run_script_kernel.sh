#! usr/bin/bash

# for d in "${!p[@]}"
# do
#    for e in "${KLp[@]}"
#    do
#      th core/train.lua --numPos ${p[d]} --numNeg ${n[d]} --KLinearParam $e &
#    done
# done

#
declare -a p=(10 10 100 100) # 100 100 100 100)
declare -a n=(0 1 0 10) # 0 20 50 100)
declare -a w=(0.01 0.1 1 2 5) # 10 100 200 500 1000) #(0.1 0.5 2 1 5 10)
declare -a KLp=(-0.01 -0.05 -0.1 -0.5 0 0.5 1)


# declare -a p=(5 10 20 50 100 200) # 50 100 200) #20 100 100)
# declare -a n=(0) # 1 2 5 10 20 50 100 200) #2 0 10)
# declare -a dist=('uniform' 'normal')
# declare -a w=(0.01 0.1 1 2 5 10 100 200)    # (0.01 0.1 1 2 5 10 100 200 500 1000)
#
for t in "${n[@]}"
do
for d in "${p[@]}"
  do
     for bb in "${KLp[@]}"
     do
      for c in "${w[@]}"
      do
   CUDA_VISIBLE_DEVICES=3 th core/train_full_backprop.lua --numPos $d --numNeg $t --weightScale $c --KLinearParam $bb # --distribution $bb
   #--trainSize 10000 --end_path 'trainset_10000/'
     done
  done
done
done

# CUDA_VISIBLE_DEVICES=1 th core/train_full_backprop.lua --numPos 5 --numNeg 0 --weightScale 10 --distribution 'uniform'
#
# CUDA_VISIBLE_DEVICES=2 th core/train_full_backprop.lua --numPos 100 --numNeg 0 --weightScale 10 --distribution 'uniform'
#
# CUDA_VISIBLE_DEVICES=3 th core/train_full_backprop.lua --numPos 100 --numNeg 10 --weightScale 10 --distribution 'uniform'
#
# CUDA_VISIBLE_DEVICES=4 th core/train_full_backprop.lua --numPos 1000 --numNeg 100 --weightScale 100 --distribution 'uniform'

#
# weightScale = {0.2, 0.5, 1, 2, 5}
# numPos = {5,20,20, 100, 100}--, 10, 10, 20, 20, 20, 20, 50,50, 50, 50, 100, 100, 100, 100}
# numNeg = {0, 0,10, 0, 25,50}--, 0, 5, 0, 5, 10, 0, 5, 10, 25, 0, 5, 10, 25, 50}
#
#
