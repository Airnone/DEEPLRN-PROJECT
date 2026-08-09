import sys
from types import ModuleType

from deeplrn import __main__ as module_main


def test_module_entrypoint_dispatches_train(monkeypatch):
    calls = []
    workflows = ModuleType("deeplrn.workflows")
    workflows.train_main = lambda argv: calls.append(argv)
    monkeypatch.setitem(sys.modules, "deeplrn.workflows", workflows)

    module_main.main(["train", "--manifest", "split.json"])

    assert calls == [["--manifest", "split.json"]]
