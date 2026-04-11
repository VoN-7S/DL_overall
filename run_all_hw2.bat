@echo off
setlocal
cd /d "%~dp0"

set "START_FROM=%~1"
set "COMMON=--batch_size 128 --seed 7"
set "ROBUST_WORKERS=--num_workers 2"
set "KD_MATCH_HW1=--batch_size 128 --num_workers 0 --epoch 20 --learning_rate 1e-3 --weight_decay 0.0 --kd_temperature 4.0 --kd_alpha 0.7 --seed 7"

echo ==============================================
echo  HW2 Full Pipeline
echo ==============================================
echo  This script runs Tasks 1-5 in dependency order.
echo  Task 4 uses the same KD hyper-parameters as the saved HW1 runs
echo  so vanilla-teacher and AugMix-teacher students are comparable.
echo  Optional usage: run_all_hw2.bat task3  (runs Task 3 and after)
echo.

if /i "%START_FROM%"=="" goto :task1
if /i "%START_FROM%"=="task1" goto :task1
if /i "%START_FROM%"=="task2" goto :task2
if /i "%START_FROM%"=="task3" goto :task3
if /i "%START_FROM%"=="task4" goto :task4
if /i "%START_FROM%"=="task5" goto :task5
echo [ERROR] Invalid start task: %START_FROM%
echo Use one of: task1, task2, task3, task4, task5
exit /b 1

:task1
echo ==============================================
echo  Task 1: Evaluate vanilla model on CIFAR-10-C
echo ==============================================
python main.py --task robustness --hw2_task task1 --batch_size 256 %ROBUST_WORKERS% --seed 7
if errorlevel 1 goto :fail

:task2
echo ==============================================
echo  Task 2: Train AugMix model and evaluate on CIFAR-10-C
echo ==============================================
python main.py --task robustness --hw2_task task2 %COMMON% %ROBUST_WORKERS% --epoch 30 --learning_rate 0.05 --weight_decay 5e-4 --augmix_lambda 12.0
if errorlevel 1 goto :fail

:task3
echo ==============================================
echo  Task 3: PGD attacks, Grad-CAM, and t-SNE
echo ==============================================
python main.py --task adversarial %COMMON% %ROBUST_WORKERS% --pgd_steps 20 --linf_eps 0.0156862745 --l2_eps 0.25 --tsne_samples 1000
if errorlevel 1 goto :fail

:task4
echo ==============================================
echo  Task 4: Knowledge Distillation with AugMix Teacher
echo  Using HW1-matched KD settings for a fair teacher comparison
echo ==============================================
python main.py --task augmix_distillation --hw2_task task4 %KD_MATCH_HW1%
if errorlevel 1 goto :fail

:task5
echo ==============================================
echo  Task 5: Adversarial Transferability
echo  Evaluating both teachers on both student architectures
echo ==============================================
python main.py --task augmix_distillation --hw2_task task5 --batch_size 128 --num_workers 0 --seed 7
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
