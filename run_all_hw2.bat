@echo off
setlocal

echo ==============================================
echo  HW2 Full Pipeline
echo ==============================================
echo  This script runs Tasks 1-5 in dependency order.
echo.

echo ==============================================
echo  Task 1: Evaluate vanilla model on CIFAR-10-C
echo ==============================================
python main.py --task robustness --hw2_task task1 --batch_size 256 --num_workers 2 --seed 7
if errorlevel 1 goto :fail

echo ==============================================
echo  Task 2: Train AugMix model and evaluate on CIFAR-10-C
echo ==============================================
python main.py --task robustness --hw2_task task2 --batch_size 128 --num_workers 2 --epoch 30 --learning_rate 0.05 --weight_decay 5e-4 --augmix_lambda 12.0 --seed 7
if errorlevel 1 goto :fail

echo ==============================================
echo  Task 3: PGD attacks, Grad-CAM, and t-SNE
echo ==============================================
python main.py --task adversarial --batch_size 128 --num_workers 2 --pgd_steps 20 --linf_eps 0.0156862745 --l2_eps 0.25 --tsne_samples 1000 --seed 7
if errorlevel 1 goto :fail

echo ==============================================
echo  Task 4: Knowledge Distillation with AugMix Teacher
echo ==============================================
python main.py --task augmix_distillation --hw2_task task4 --batch_size 128 --num_workers 2 --epoch 30 --learning_rate 1e-3 --kd_temperature 4.0 --kd_alpha 0.7 --seed 7
if errorlevel 1 goto :fail

echo ==============================================
echo  Task 5: Adversarial Transferability
echo ==============================================
python main.py --task augmix_distillation --hw2_task task5 --batch_size 128 --num_workers 2 --seed 7
if errorlevel 1 goto :fail

echo ==============================================
echo  All HW2 tasks completed successfully.
echo  Results are saved under results\hw2\ and results\kd\
echo ==============================================
goto :eof

:fail
echo.
echo ==============================================
echo  The pipeline stopped because one command failed.
echo  Review the error output above before continuing.
echo ==============================================
exit /b 1
