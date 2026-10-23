"""Complaint → recall prediction pipeline.

Imports are lazy so that `python -m supervised_pipeline.<module>` runs each step cleanly on its own.
"""
import importlib

_EXPORTS = {
    "Config": "config",
    "load_data": "run", "run": "run", "train_and_evaluate": "run",
    "build_task": "tasks", "TASKS": "tasks",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        modname = _EXPORTS[name]
        module = importlib.import_module(f".{modname}", __name__)
        for k, m in _EXPORTS.items():    # cache every export of that module; this also replaces the
            if m == modname:             # 'run' submodule attribute with the run() function
                globals()[k] = getattr(module, k)
        return globals()[name]
    raise AttributeError(f"module 'supervised_pipeline' has no attribute '{name}'")
