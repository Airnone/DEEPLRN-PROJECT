"""Allow running DEEPLRN workflows with ``python -m deeplrn <subcommand>``.

Recognized subcommands dispatch to the same entry points as the installed
console scripts (see ``pyproject.toml``):

    python -m deeplrn prepare     -> deeplrn.workflows.prepare_main
    python -m deeplrn split       -> deeplrn.splitting.main
    python -m deeplrn train       -> deeplrn.workflows.train_main
    python -m deeplrn infer       -> deeplrn.workflows.infer_main
    python -m deeplrn experiments -> deeplrn.experiments.experiments_main

With no recognized subcommand, falls back to the original preprocessing CLI
(``deeplrn.cli.main``) so existing invocations like
``python -m deeplrn --input report.pdf --output out/`` keep working.
"""

import sys

_SUBCOMMANDS = {"prepare", "split", "train", "infer", "experiments"}


def _dispatch(argv: list[str]) -> None:
    if argv and argv[0] in _SUBCOMMANDS:
        subcommand, rest = argv[0], argv[1:]
        if subcommand == "prepare":
            from deeplrn.workflows import prepare_main
            prepare_main(rest)
        elif subcommand == "split":
            from deeplrn.splitting import main as split_main
            split_main(rest)
        elif subcommand == "train":
            from deeplrn.workflows import train_main
            train_main(rest)
        elif subcommand == "infer":
            from deeplrn.workflows import infer_main
            infer_main(rest)
        elif subcommand == "experiments":
            from deeplrn.experiments import experiments_main
            experiments_main(rest)
        return

    from deeplrn.cli import main as preprocess_main
    preprocess_main(argv)


if __name__ == "__main__":
    _dispatch(sys.argv[1:])
