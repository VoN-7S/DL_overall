@echo off

echo ==============================================
echo  HW2 Task 1: Evaluate vanilla model on CIFAR-10-C
echo ==============================================
python main.py --task robustness --hw2_task task1 --batch_size 256 --num_workers 2

echo ==============================================
echo  HW2 Task 2: Train AugMix model + CIFAR-10-C eval
echo ==============================================
python main.py --task robustness --hw2_task task2 --batch_size 128 --num_workers 2 --epochs 30 --learning_rate 0.05 --augmix_lambda 12.0 --seed 7

echo ==============================================
echo  HW2 Task 3: PGD attacks, Grad-CAM, t-SNE
echo ==============================================
python main.py --task adversarial --hw2_task task3 --batch_size 128 --num_workers 2 --pgd_steps 20 --tsne_samples 1000

echo ==============================================
echo  HW2 Task 4: KD with AugMix teacher
echo ==============================================
python main.py --task hw2_distillation --hw2_task task4 --batch_size 128 --num_workers 2 --epoch 30 --learning_rate 1e-3 --kd_temperature 4.0 --kd_alpha 0.7 --seed 7

echo ==============================================
echo  HW2 Task 5: Adversarial transferability
echo ==============================================
python main.py --task hw2_distillation --hw2_task task5 --batch_size 128 --num_workers 2

echo ==============================================
echo  All done! Results in results/hw2/
echo ==============================================