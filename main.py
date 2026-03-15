"""
main.py
-------
Unified entry point for all experiments in this repository.

Dispatch table
--------------
    --task mnist        -- MNIST MLP classification (HW1a)
    --task transfer     -- CIFAR-10 transfer learning (HW1b Part A)
    --task distillation -- CIFAR-10 knowledge distillation (HW1b Part B)

Usage
-----
    python main.py --task mnist --epoch 25 --mlp_hidden_layers 256 128
    python main.py --task transfer --tl_option both --epoch 30
    python main.py --task distillation --epoch 30
    python main.py --task distillation --kd_experiment 3 --kd_teacher_path results/kd/kd_resnet/model_no_ls.pth
"""

import ssl
import sys

ssl._create_default_https_context = ssl._create_unverified_context

from parameters import get_params


def main() -> None:
    """
    Parse arguments and dispatch to the appropriate experiment runner.

    Reads --task from the command line and calls the corresponding
    run function. Exits with error code 1 if the task is unknown.
    """
    params = get_params()

    if params.task == "mnist":
        from mnist import run_mnist
        run_mnist(params)

    elif params.task == "transfer":
        from transfer_learning import run_transfer
        run_transfer(params)

    elif params.task == "distillation":
        from knowledge_distillation import run_distillation
        run_distillation(params)

    else:
        print("Unknown task: " + params.task)
        print("Choose from: mnist, transfer, distillation")
        sys.exit(1)


if __name__ == "__main__":
    main()
