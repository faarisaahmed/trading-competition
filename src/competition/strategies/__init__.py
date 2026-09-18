"""The competitors.

One module per team. Each is self-contained, deterministic, and free of any
LLM call at runtime -- the models designed these; the code trades them.
"""

from .base import Param, Params, Strategy, StrategyContext

__all__ = ["Strategy", "StrategyContext", "Param", "Params", "load_strategy"]


def load_strategy(spec: str):
    """Resolve a 'module.path:ClassName' spec to the class object."""
    from importlib import import_module

    if ":" not in spec:
        raise ValueError(f"strategy spec must be 'module:Class', got {spec!r}")
    mod_name, cls_name = spec.split(":", 1)
    try:
        mod = import_module(mod_name)
    except ImportError as e:
        raise ImportError(f"cannot import strategy module {mod_name!r}: {e}") from e
    try:
        cls = getattr(mod, cls_name)
    except AttributeError as e:
        raise ImportError(f"{mod_name!r} has no attribute {cls_name!r}") from e
    if not (isinstance(cls, type) and issubclass(cls, Strategy)):
        raise TypeError(f"{spec} is not a Strategy subclass")
    return cls
