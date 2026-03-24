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
