"""Rendering competition state for humans.

Kept separate from the engine: the engine calls a *callback* to write a
dashboard, so nothing in the trading path ever imports the reporting layer and
a broken report can never affect a round.
"""

from .dashboard import DashboardData, build_dashboard_data, render_dashboard, write_dashboard

__all__ = ["render_dashboard", "write_dashboard", "build_dashboard_data", "DashboardData"]
