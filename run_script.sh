#! usr/bin/bash
declare -a types=('heuristic' 'xavier' 'xaviercaffe' 'kaiming')
# # # # declare -a dim=(998)
# # # # declare -a noise=(0)
# # # # # declare -a lr=(0.001) #lr=(0.01 0.001 0.0001 0.00001)
# # # # # declare -a lrd=(1e-5 1e-6 1e-7)
# # #
for i in "${types[@]}"
do
# # # #    for j in "${dim[@]}"
# # # #    do
# # # #      for k in "${noise[@]}"
# # # #      do
# # # #    #     for a in "${lr[@]}"
# # # #    #     do
# # # #          # for b in "${lrd[@]}"
# # # #          # do
          CUDA_VISIBLE_DEVICES=3 th core/train_full_backprop.lua --weightInit $i #--trainSize 10000 --end_path 'trainset_10000/' # --nDim $j --noiseFactor $k & # --overlap $j --optimization $k --learningRate $a & #--lrDecay $b &
# #          # done
# # # #        # done
# # # #      done
# # # #    done
done
