@echo off

echo ==============================================
echo  HW1b Part A: Transfer Learning
echo ==============================================

python main.py --task transfer --tl_option 1 --epoch 12 --batch_size 128 --learning_rate 1e-3 --num_workers 0 --seed 7
python main.py --task transfer --tl_option 2 --epoch 20 --batch_size 128 --learning_rate 1e-3 --num_workers 0 --seed 7

echo ==============================================
echo  HW1b Part B: Knowledge Distillation
echo ==============================================

python main.py --task distillation --epoch 20 --batch_size 128 --learning_rate 1e-3 --num_workers 0 --seed 7 --kd_temperature 4.0 --kd_alpha 0.7 --kd_smoothing 0.1

echo ==============================================
echo  All done!
echo ==============================================