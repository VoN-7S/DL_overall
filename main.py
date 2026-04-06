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

    if params.task == "transfer":
        from transfer_learning import train_transfer
        train_transfer(params)

    elif params.task == "distillation":
        from knowledge_distillation import run_distillation
        run_distillation(params)

    elif params.task == "robustness":
        from augmix import run_robustness
        run_robustness(params)

    elif params.task == "adversarial":
        from adversarial import run_adversarial
        run_adversarial(params)

    elif params.task == "augmix_distillation":
        from knowledge_distillation import run_hw2_distillation
        run_hw2_distillation(params)

    else:
        print("Unknown task: " + params.task)
        print("Choose from: mnist, transfer, distillation, robustness, adversarial, augmix_distillation")
        sys.exit(1)


if __name__ == "__main__":
    main()
