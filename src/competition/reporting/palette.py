"""Chart colours and chrome, validated for colour-vision deficiency.

The eight categorical slots and both surfaces come from a palette whose
adjacent-pair separation was checked with a validator rather than by eye:

    light  worst adjacent CVD dE 9.1, normal-vision dE 19.6   (>=8 / >=15)
    dark   worst adjacent CVD dE 8.4, normal-vision dE 19.3

Three light-mode slots (aqua, yellow, magenta) sit below 3:1 contrast against
the light surface. The mitigation ("relief") is that this dashboard always
ships the standings **table** with values in text ink, and direct labels on the
chart -- so identity and magnitude are never carried by hue alone.

Two rules this module exists to enforce:

* **Colour follows the entity, not its rank.** Slots are assigned from the
  configured team order once, so a team keeps its colour when the standings
  reorder. A reader who learned "the Gambler is red" is never misled.
* **Never a ninth hue.** There are exactly eight scored teams and exactly eight
  slots. The benchmark is not a competitor, so it is drawn in muted ink as a
  reference line rather than being given a generated ninth colour.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

#: Fixed categorical order. Do not reorder: the sequence itself is the
#: CVD-safety mechanism -- adjacent pairs were validated in this order.
CATEGORICAL_LIGHT: tuple[str, ...] = (
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
)

CATEGORICAL_DARK: tuple[str, ...] = (
    "#3987e5",
    "#d95926",
    "#199e70",
    "#c98500",
    "#d55181",
    "#008300",
    "#9085e9",
    "#e66767",
)

LIGHT = {
    "surface": "#fcfcfb",
    "plane": "#f9f9f7",
    "text_primary": "#0b0b0b",
    "text_secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "border": "rgba(11,11,11,0.10)",
    "up": "#006300",
    "down": "#d03b3b",
    "good": "#0ca30c",
    "warning": "#fab219",
    "critical": "#d03b3b",
}

DARK = {
    "surface": "#1a1a19",
    "plane": "#0d0d0d",
    "text_primary": "#ffffff",
    "text_secondary": "#c3c2b7",
    "muted": "#898781",
    "grid": "#2c2c2a",
    "axis": "#383835",
    "border": "rgba(255,255,255,0.10)",
    "up": "#0ca30c",
    "down": "#d03b3b",
    "good": "#0ca30c",
    "warning": "#fab219",
    "critical": "#d03b3b",
}


def assign_slots(team_keys: Sequence[str]) -> dict[str, int]:
    """team_key -> categorical slot index, fixed by configured order.

    Keys past the eighth get slot -1, meaning "not a categorical series" --
    drawn in muted ink rather than a generated hue.
    """
    out: dict[str, int] = {}
    for i, key in enumerate(team_keys):
        out[key] = i if i < len(CATEGORICAL_LIGHT) else -1
    return out


def css_variables(slots: Mapping[str, int]) -> str:
    """The `:root`-scoped custom properties this dashboard reads by role.

    Dark values are declared under both the media query and the `data-theme`
    scope, so an explicit toggle wins over the OS setting in both directions.
    """
    def block(theme: Mapping[str, str], cats: Sequence[str]) -> str:
        lines = [f"    --{k.replace('_', '-')}: {v};" for k, v in theme.items()]
        for key, slot in sorted(slots.items()):
            colour = cats[slot] if 0 <= slot < len(cats) else theme["muted"]
            lines.append(f"    --team-{key.replace('_', '-')}: {colour};")
        return "\n".join(lines)

    return f""".viz-root {{
    color-scheme: light;
{block(LIGHT, CATEGORICAL_LIGHT)}
  }}
  @media (prefers-color-scheme: dark) {{
    :root:where(:not([data-theme="light"])) .viz-root {{
      color-scheme: dark;
{block(DARK, CATEGORICAL_DARK)}
    }}
  }}
  :root[data-theme="dark"] .viz-root {{
    color-scheme: dark;
{block(DARK, CATEGORICAL_DARK)}
  }}"""


def team_var(team_key: str) -> str:
    return f"var(--team-{team_key.replace('_', '-')})"
