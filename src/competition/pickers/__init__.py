"""Stock pickers -- one per team, each matching its team's philosophy.

The competition rule: "for round two each team is programmed a stock picker
to choose stocks to trade from, but this stock picker has to use a similar
approach to the actual team." So the momentum team's picker screens for
momentum; the pair trader's picker screens for cointegration; the RL agent's
picker is itself a bandit. A picker that disagreed with its strategy would be
a different entry wearing the same name.

Every picker is handed the *same* candidate pool and the same data. They
differ only in what they select from it.
"""

from .base import Picker, PickerContext, PickResult

__all__ = ["Picker", "PickerContext", "PickResult", "load_picker"]


def load_picker(spec: str):
    """Resolve a 'module.path:ClassName' spec to the class object."""
    from importlib import import_module

    if ":" not in spec:
        raise ValueError(f"picker spec must be 'module:Class', got {spec!r}")
    mod_name, cls_name = spec.split(":", 1)
    try:
        mod = import_module(mod_name)
    except ImportError as e:
        raise ImportError(f"cannot import picker module {mod_name!r}: {e}") from e
    try:
        cls = getattr(mod, cls_name)
    except AttributeError as e:
        raise ImportError(f"{mod_name!r} has no attribute {cls_name!r}") from e
    if not (isinstance(cls, type) and issubclass(cls, Picker)):
        raise TypeError(f"{spec} is not a Picker subclass")
    return cls
