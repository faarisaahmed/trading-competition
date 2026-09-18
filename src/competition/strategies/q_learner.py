"""Team 3 -- Q-Learner: tabular reinforcement learning on a discretised tape.

Thesis
------
Don't encode a theory of markets; encode a *reward* and let the agent find
the policy. The agent has no concept of a stock, an earnings report or a
moving-average crossover. It sees a four-integer state, picks one of three
exposure levels, and gets paid in risk-normalised P&L. Over enough bars, the
Q-table becomes an empirical map of "in conditions like these, this much
exposure paid".

State (discretised so the table is small enough to actually fill)
-----------------------------------------------------------------
    trend   : bucketed per-bar regression slope        -> 5 levels
    dislocation : bucketed z-score vs the 20-bar mean  -> 5 levels
    vol     : bucketed ATR%                            -> 4 levels
    holding : which exposure level we are currently at -> 3 levels
                                                        = 300 states

Crucially the state is *symbol-agnostic*: every feature is scale-free (a
slope in percent, a z-score, an ATR ratio), so one shared Q-table transfers
from AAPL to a Round 3 mid-cap it has never seen. That is what makes an RL
entry viable in a three-week competition -- a per-symbol table would never
see enough data to leave its initialisation.

Actions
-------
Target portfolio weight from `exposure_actions` (default flat / 15% / 32%).
Discrete exposure, not discrete buy/sell, so the agent cannot accidentally
learn to churn.

Reward
------
    r = prev_exposure * bar_return / atr_pct        (risk-normalised P&L)
      - turnover_penalty * |exposure change|         (pay for trading)
      - drawdown_penalty * max(0, -position_drawdown)
    clipped to +/- reward_clip

Dividing by ATR% is what lets one table learn from both a sleepy utility and
a meme stock: a 1% move means something different in each, and the agent
should learn about *conditions*, not about volatility levels it will never
see again.

Learning schedule
-----------------
Epsilon-greedy, decaying `epsilon_start` -> `epsilon_end` over
`epsilon_decay_steps` updates. Optimistic initialisation nudges early
exploration. The table persists to disk between rounds (`persist_qtable`),
so the agent that shows up for Round 3 is the one that learned in Rounds 1
and 2 -- which is the whole point of entering an RL agent.

Two anti-churn measures, both necessary
---------------------------------------
An unconstrained tabular agent on 5-minute bars will flip exposure almost
every bar: epsilon-greedy exploration is random by construction, and if
trading is free in the reward then churning costs nothing. The first version
of this strategy turned over $583k on a $5,000 account in four days -- 117x
its bankroll -- and gave up 6.5% to the round trips. Two changes fix it:

* `turnover_penalty` is calibrated to the *real* cost. Reward is P&L / ATR%,
  so a round trip costing ~8bps of notional is worth about
  0.0008 / 0.0025 ~= 0.3 reward units per unit of exposure changed. The naive
  first guess of 0.02 was more than ten times too small to influence the
  policy at all; 0.35 makes the agent pay roughly what the spread costs it.
* `action_repeat` holds a chosen exposure for 3 bars before re-deciding --
  standard action-repeat (frame-skip) from RL practice. Learning still
  happens every bar: the reward for each bar is credited to the
  (state, action) pair actually in force, so no training signal is lost.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Mapping, Sequence
from typing import Any

from ..types import Bar, OrderIntent
from ..util import indicators as ind
from .base import Param, Strategy, StrategyContext


class QLearner(Strategy):
    DESCRIPTION = (
        "Tabular Q-learning over a scale-free (trend, dislocation, vol, holding) state "
        "choosing among discrete exposure levels; reward is ATR-normalised P&L minus "
        "turnover. Learns online and persists across rounds."
    )

    DEFAULT_TICK_SECONDS = 300

    PARAMS = (
        Param("tick_seconds", 300, "seconds between evaluations", minimum=5),
        Param("alpha", 0.10, "learning rate", minimum=1e-4, maximum=1.0),
        Param("gamma", 0.92, "discount factor", minimum=0.0, maximum=0.9999),
        Param("epsilon_start", 0.10, "initial exploration rate", minimum=0.0, maximum=1.0),
        Param("epsilon_end", 0.02, "floor exploration rate", minimum=0.0, maximum=1.0),
        Param("epsilon_decay_steps", 4000, "updates over which epsilon decays", minimum=1),
        Param("optimistic_init", 0.05, "initial Q value for unseen state-actions"),
        Param("trend_buckets", [-0.0006, -0.0002, 0.0002, 0.0006],
              "per-bar regression-slope bucket edges (5Min scale)"),
        Param("zscore_buckets", [-1.5, -0.5, 0.5, 1.5], "z-score bucket edges (scale-free)"),
        Param("vol_buckets", [0.0012, 0.0025, 0.005], "ATR% bucket edges (5Min scale)"),
        Param("exposure_actions", [0.0, 0.15, 0.32], "target weights the agent may choose"),
        Param("turnover_penalty", 0.35, "reward cost per unit of exposure change", minimum=0.0),
        Param("action_repeat", 3, "bars to hold a chosen exposure before re-deciding",
              minimum=1),
        Param("drawdown_penalty", 0.5, "reward cost per unit of position drawdown", minimum=0.0),
        Param("reward_clip", 3.0, "absolute reward clip", minimum=0.01),
        Param("max_positions", 3, "concurrent names", minimum=1),
        Param("atr_period", 14, "ATR period", minimum=2),
        Param("min_bars_required", 80, "warm-up bars before trading", minimum=20),
        Param("persist_qtable", True, "carry the learned table across rounds"),
    )

    def __init__(self, team, params=None):
        super().__init__(team, params)
        #: state_key -> list[float] of length len(exposure_actions)
        self.q: dict[str, list[float]] = {}
        self.steps: int = 0
        self.visits: dict[str, int] = {}
        self.total_reward: float = 0.0
        self.updates: int = 0

    # ------------------------------------------------------------------ #
    # state / action machinery
    # ------------------------------------------------------------------ #

    @property
    def n_actions(self) -> int:
        return len(self.p.exposure_actions)

    def features(self, bars: Sequence[Bar]) -> dict[str, float] | None:
        p = self.p
        if len(bars) < max(p.atr_period + 5, 40):
            return None
        closes = [b.close for b in bars]
        price = closes[-1]
        atr_pct = ind.atr_pct(bars, p.atr_period)
        if price <= 0 or atr_pct <= 0:
            return None
        # Close-to-close, NOT the bar's own open-to-close. The agent decides
        # at a bar close and holds until the next one, so the return that
        # rewards that decision is close(t)/close(t-1) - 1. Using the
        # intra-bar return would credit the agent for a move it was not
        # positioned for and silently drop every overnight gap.
        prev_close = closes[-2] if len(closes) >= 2 else price
        bar_return = (price / prev_close - 1.0) if prev_close > 0 else 0.0
        return {
            "slope": ind.linreg_slope(closes, 20),
            "z": ind.zscore(closes, 20),
            "atr_pct": atr_pct,
            "bar_return": bar_return,
            "price": price,
        }

    def nearest_action(self, weight: float) -> int:
        actions = list(self.p.exposure_actions)
        return min(range(len(actions)), key=lambda i: abs(actions[i] - weight))

    def state_key(self, f: Mapping[str, float], holding_idx: int) -> str:
        p = self.p
        t = bisect.bisect_right(list(p.trend_buckets), f["slope"])
        z = bisect.bisect_right(list(p.zscore_buckets), f["z"])
        v = bisect.bisect_right(list(p.vol_buckets), f["atr_pct"])
        return f"{t}|{z}|{v}|{holding_idx}"

    def q_row(self, key: str) -> list[float]:
        row = self.q.get(key)
        if row is None:
            row = [float(self.p.optimistic_init)] * self.n_actions
            self.q[key] = row
        return row

    @property
    def epsilon(self) -> float:
        p = self.p
        frac = min(self.steps / max(p.epsilon_decay_steps, 1), 1.0)
        return p.epsilon_start + (p.epsilon_end - p.epsilon_start) * frac

    def choose(self, key: str, rng, *, prefer: int | None = None) -> tuple[int, bool]:
        """Epsilon-greedy action. Returns (action_index, was_exploration).

        Ties break toward `prefer` -- the action currently in force. This
        matters far more than it looks: with optimistic initialisation every
        action in an unvisited state has *identical* value, so a random
        tie-break makes the agent re-roll its exposure at every decision
        point and pay the spread for the privilege. On a fresh table that was
        164x turnover in a week. When the agent is genuinely indifferent, the
        correct action is to do nothing.
        """
        row = self.q_row(key)
        if rng.random() < self.epsilon:
            return rng.randrange(self.n_actions), True
        best = max(row)
        ties = [i for i, v in enumerate(row) if v >= best - 1e-12]
        if len(ties) == 1:
            return ties[0], False
        if prefer is not None and prefer in ties:
            return prefer, False
        return rng.choice(ties), False

    # ------------------------------------------------------------------ #
    # learning
    # ------------------------------------------------------------------ #

    def reward(
        self,
        prev_exposure: float,
        bar_return: float,
        atr_pct: float,
        exposure_change: float,
        position_drawdown: float,
    ) -> float:
        p = self.p
        pnl = prev_exposure * bar_return / max(atr_pct, 1e-4)
        cost = p.turnover_penalty * abs(exposure_change)
        risk = p.drawdown_penalty * max(-position_drawdown, 0.0)
        r = pnl - cost - risk
        if not math.isfinite(r):
            return 0.0
        return max(min(r, p.reward_clip), -p.reward_clip)

    def update(self, key: str, action: int, reward: float, next_key: str) -> None:
        p = self.p
        row = self.q_row(key)
        nxt = self.q_row(next_key)
        target = reward + p.gamma * max(nxt)
        row[action] += p.alpha * (target - row[action])
        self.steps += 1
        self.updates += 1
        self.total_reward += reward
        self.visits[key] = self.visits.get(key, 0) + 1

    # ------------------------------------------------------------------ #
    # persistence
    # ------------------------------------------------------------------ #

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "steps": self.steps,
            "updates": self.updates,
            "total_reward": round(self.total_reward, 6),
            "n_actions": self.n_actions,
            "exposure_actions": list(self.p.exposure_actions),
            "q": {k: [round(v, 8) for v in row] for k, row in self.q.items()},
            "visits": dict(self.visits),
        }

    def load_state(self, blob: Mapping[str, Any]) -> None:
        if not blob:
            return
        stored_actions = list(blob.get("exposure_actions") or [])
        if stored_actions and stored_actions != list(self.p.exposure_actions):
            # The action space changed, so the old values index different
            # meanings. Refusing to load is the honest thing to do.
            self.log.warning(
                "q_learner: saved action space %s != configured %s; starting fresh",
                stored_actions, list(self.p.exposure_actions),
            )
            return
        raw = blob.get("q") or {}
        n = self.n_actions
        self.q = {
            str(k): [float(x) for x in v][:n] + [float(self.p.optimistic_init)] * max(n - len(v), 0)
            for k, v in raw.items()
            if isinstance(v, (list, tuple))
        }
        self.steps = int(blob.get("steps", 0))
        self.updates = int(blob.get("updates", 0))
        self.total_reward = float(blob.get("total_reward", 0.0))
        self.visits = {str(k): int(v) for k, v in (blob.get("visits") or {}).items()}
        self.log.info(
            "q_learner: loaded %d states, %d prior updates, epsilon now %.3f",
            len(self.q), self.steps, self.epsilon,
        )

    # ------------------------------------------------------------------ #
    # offline pre-training (run by `comp pretrain`)
    # ------------------------------------------------------------------ #

    def pretrain(
        self,
        bars_by_symbol: Mapping[str, Sequence[Bar]],
        *,
        passes: int = 3,
        rng=None,
        log_every: int = 0,
    ) -> dict[str, Any]:
        """Learn offline from historical bars, with no broker involved.

        Walks each symbol's series bar by bar, simulating the same
        state/action/reward loop `on_tick` runs live. Exposure is tracked as
        a notional weight; there is no cash constraint, because the point is
        to learn the *conditioning*, not to backtest a portfolio.
        """
        import random as _random

        rng = rng or _random.Random(0)
        p = self.p
        actions = list(p.exposure_actions)
        stats = {"symbols": 0, "bars": 0, "updates_before": self.updates}
        for _pass in range(max(passes, 1)):
            for series in bars_by_symbol.values():
                series = list(series)
                if len(series) < self.warmup_bars + 5:
                    continue
                stats["symbols"] += 1
                exposure = 0.0
                # The decision awaiting its reward: (state, action, the
                # exposure it put on, the turnover it cost to get there).
                pending: tuple[str, int, float, float] | None = None
                peak = cum = 0.0
                hold = 0
                last_action: int | None = None
                for i in range(self.warmup_bars, len(series)):
                    window = series[max(0, i - 300):i + 1]
                    f = self.features(window)
                    if f is None:
                        continue
                    stats["bars"] += 1
                    key = self.state_key(f, self.nearest_action(exposure))
                    if pending is not None:
                        pkey, paction, pexposure, pchange = pending
                        cum += pexposure * f["bar_return"]
                        peak = max(peak, cum)
                        r = self.reward(
                            prev_exposure=pexposure,
                            bar_return=f["bar_return"],
                            atr_pct=f["atr_pct"],
                            exposure_change=pchange,
                            position_drawdown=cum - peak,
                        )
                        self.update(pkey, paction, r, key)
                    # Mirror the live action-repeat so the table it learns is
                    # the table the live policy will actually consult.
                    if hold > 0 and last_action is not None:
                        action = last_action
                        hold -= 1
                    else:
                        action, _ = self.choose(key, rng, prefer=last_action)
                        hold = max(int(p.action_repeat) - 1, 0)
                    last_action = action
                    new_exposure = actions[action]
                    pending = (key, action, new_exposure, new_exposure - exposure)
                    exposure = new_exposure
                if log_every and stats["symbols"] % log_every == 0:
                    self.log.info(
                        "pretrain: %d symbols, %d bars, %d states",
                        stats["symbols"], stats["bars"], len(self.q),
                    )
        stats.update({
            "updates_after": self.updates,
            "states": len(self.q),
            "epsilon": round(self.epsilon, 4),
            "mean_reward": round(float(self.total_reward) / max(self.updates, 1), 6),
        })
        return stats

    # ------------------------------------------------------------------ #
    # live loop
    # ------------------------------------------------------------------ #

    def on_round_start(self, ctx: StrategyContext) -> None:
        ctx.state.setdefault("_symbols", {})
        ctx.log.info(
            "q_learner: %d states known, %d updates, epsilon %.3f, actions %s",
            len(self.q), self.steps, self.epsilon, list(self.p.exposure_actions),
        )

    def on_tick(self, ctx: StrategyContext) -> list[OrderIntent]:
        p = self.p
        actions = list(p.exposure_actions)
        # Learn once per bar, not once per tick: rewards must line up with a
        # completed bar return or the agent is being taught noise.
        if not self.sync_bar_clock(ctx):
            return []

        desired: dict[str, tuple[float, float, bool]] = {}   # sym -> (weight, q, explored)
        for sym in ctx.candidates(self.warmup_bars):
            f = self.features(ctx.bars(sym))
            if f is None:
                continue
            st = ctx.sym_state(sym)
            exposure = ctx.weight(sym)
            key = self.state_key(f, self.nearest_action(exposure))

            # --- credit assignment for the action taken on the last bar ----
            # The reward belongs to the exposure the agent *chose* last bar,
            # not to whatever the position drifted to since, so we replay the
            # stored target rather than re-reading the current weight.
            prev_key = st.get("q_state")
            prev_action = st.get("q_action")
            if prev_key is not None and prev_action is not None:
                prev_target = float(st.get("target_exposure", exposure))
                cum = float(st.get("cum_pnl", 0.0)) + prev_target * f["bar_return"]
                peak = max(float(st.get("peak_pnl", 0.0)), cum)
                r = self.reward(
                    prev_exposure=prev_target,
                    bar_return=f["bar_return"],
                    atr_pct=f["atr_pct"],
                    exposure_change=float(st.get("last_exposure_change", 0.0)),
                    position_drawdown=cum - peak,
                )
                self.update(str(prev_key), int(prev_action), r, key)
                st["cum_pnl"], st["peak_pnl"] = cum, peak
                st["last_reward"] = round(r, 5)

            # Action repeat: hold the last decision for `action_repeat` bars.
            # The reward is still credited every bar, to (current state, held
            # action) -- which is exactly the pair in force.
            holding = ctx.bar_clock() < int(st.get("hold_until", 0))
            if holding and st.get("q_action") is not None:
                action, explored = int(st["q_action"]), False
            else:
                current = st.get("q_action")
                action, explored = self.choose(
                    key, ctx.rng,
                    prefer=int(current) if current is not None
                    else self.nearest_action(exposure),
                )
                st["hold_until"] = ctx.bar_clock() + int(p.action_repeat)
            target = actions[action]
            st.update({
                "q_state": key,
                "q_action": action,
                "target_exposure": target,
                "last_exposure_change": target - float(st.get("target_exposure", exposure)),
                "q_value": round(self.q_row(key)[action], 5),
            })
            if target > 0:
                desired[sym] = (target, self.q_row(key)[action], explored)

        # --- position cap with hysteresis --------------------------------- #
        # Naively taking the top `max_positions` by Q value every bar is a
        # churn machine: with twelve candidates and a table whose values move
        # on every update, a different three win each bar and the other nine
        # get flattened. That defeated `action_repeat` entirely -- the
        # symbol-level policy said "hold" while the portfolio layer sold it,
        # then bought it back on the next bar.
        #
        # So incumbents keep their slots, and a challenger only displaces one
        # when its Q exceeds the weakest incumbent's by more than a round trip
        # actually costs. That threshold is the same turnover penalty the
        # reward function uses, applied at the portfolio level, so the two
        # layers finally agree with each other.
        switch_threshold = p.turnover_penalty * max(actions + [0.0]) * 2.0
        keep: dict[str, tuple[float, float, bool]] = {}
        incumbents = [s for s in desired if ctx.holds(s)]
        challengers = [s for s in desired if not ctx.holds(s)]
        for sym in sorted(incumbents, key=lambda x: -desired[x][1]):
            if len(keep) < p.max_positions:
                keep[sym] = desired[sym]
        for sym in sorted(challengers, key=lambda x: -desired[x][1]):
            if len(keep) < p.max_positions:
                keep[sym] = desired[sym]
                continue
            held_slots = {k: v for k, v in keep.items() if ctx.holds(k)}
            if not held_slots:
                break
            weakest = min(held_slots, key=lambda k: keep[k][1])
            if desired[sym][1] > keep[weakest][1] + switch_threshold:
                del keep[weakest]
                keep[sym] = desired[sym]
                ctx.sym_state(weakest)["q_action"] = 0
                ctx.sym_state(weakest)["target_exposure"] = 0.0

        # Anything the cap excluded is genuinely flat now, so record that in
        # its state -- otherwise it keeps "holding" an exposure it does not
        # have and asks to buy back in on the next bar.
        for sym in desired:
            if sym not in keep:
                st = ctx.sym_state(sym)
                st["q_action"] = 0
                st["target_exposure"] = 0.0
                st["hold_until"] = ctx.bar_clock() + int(p.action_repeat)

        weights = {sym: w for sym, (w, _q, _e) in keep.items()}
        intents = ctx.allocate(
            weights, tolerance=0.02, exit_others=True,
            reason=f"q-policy (eps {self.epsilon:.3f})",
        )
        for o in intents:
            sym = o.symbol
            q = keep.get(sym, (0.0, 0.0, False))
            o.reason = (
                f"q-policy w={q[0]:.2f} Q={q[1]:+.3f}"
                f"{' [explore]' if q[2] else ''} eps={self.epsilon:.3f}"
            )
            o.tag = "rl"
        if intents:
            ctx.log.debug(
                "q_learner bar %d: eps=%.3f states=%d -> %s",
                ctx.bar_clock(), self.epsilon, len(self.q),
                ", ".join(o.describe() for o in intents),
            )
        return intents

    def on_round_end(self, ctx: StrategyContext) -> None:
        ctx.log.info(
            "q_learner: finished with %d states, %d updates, mean reward %.4f, epsilon %.3f",
            len(self.q), self.updates, self.total_reward / max(self.updates, 1), self.epsilon,
        )

    # ------------------------------------------------------------------ #

    def policy_table(self, top: int = 20) -> str:
        """Most-visited states and what the agent learned to do in them."""
        rows = sorted(self.visits.items(), key=lambda kv: -kv[1])[:top]
        actions = list(self.p.exposure_actions)
        out = [
            f"q_learner policy ({len(self.q)} states, {self.updates} updates, "
            f"epsilon {self.epsilon:.3f})",
            "trend|zscore|vol|holding   visits   best action   Q values",
        ]
        for key, n in rows:
            row = self.q.get(key, [])
            if not row:
                continue
            best = max(range(len(row)), key=lambda i: row[i])
            qs = " ".join(f"{v:+.3f}" for v in row)
            out.append(f"{key:<24} {n:>7}   w={actions[best]:<8.2f}  [{qs}]")
        return "\n".join(out)
