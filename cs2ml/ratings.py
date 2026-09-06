"""Glicko-2 rating system (Glickman's algorithm) with daily RD decay.

Scale conversions: mu = (r - 1500) / 173.7178, phi = RD / 173.7178.
"""
from __future__ import annotations

import math

from . import config

_SCALE = 173.7178
_BASE_PHI = config.GLICKO_BASE_RD / _SCALE
_BASE_MU = (config.GLICKO_BASE_RATING - 1500.0) / _SCALE
_DAILY_C_PHI = config.GLICKO_DAILY_C / _SCALE
_TAU = config.GLICKO_TAU
_CONVERGENCE_TOL = 1e-6


class Glicko2Rating:
    __slots__ = ("mu", "phi", "vol", "last_day", "decay_day")

    def __init__(self) -> None:
        self.mu = _BASE_MU
        self.phi = _BASE_PHI
        self.vol = config.GLICKO_BASE_VOL
        self.last_day: float | None = None
        self.decay_day: float | None = None

    # -- public --

    def apply_idle(self, day: float) -> None:
        """Grow uncertainty for idle time. Idempotent for repeated calls:
        decay accumulates from the later of (last real match, last decay point)."""
        if self.last_day is None:
            self.last_day = day  # first observation: baseline, no decay
            return
        ref = self.decay_day if (self.decay_day is not None and self.decay_day > self.last_day) else self.last_day
        idle = day - ref
        if idle > 0:
            self.phi = math.sqrt(self.phi ** 2 + _DAILY_C_PHI ** 2 * idle)
            self.phi = min(self.phi, _BASE_PHI)
            self.decay_day = day

    @property
    def r(self) -> float:
        return self.mu * _SCALE + 1500.0

    @property
    def rd(self) -> float:
        return self.phi * _SCALE

    def win_prob(self, opponent: "Glicko2Rating") -> float:
        """P(this player beats opponent), Glicko-1 scale formula.

        Uses the OPPONENT's RD in g(): expected score depends on the opponent's
        rating deviation, not one's own.
        """
        g = 1.0 / math.sqrt(1.0 + 3.0 * opponent.phi ** 2 / math.pi ** 2)
        return 1.0 / (1.0 + math.exp(-g * (self.mu - opponent.mu)))


def update_pair(winner: Glicko2Rating, loser: Glicko2Rating) -> None:
    """Update both ratings for a single game (one vs one).

    Glicko-2 is a simultaneous update: both new ratings are computed from the
    same pre-match values. Snapshot both players first so the loser's step does
    not read the winner's already-updated mu/phi.
    """
    w_mu, w_phi = winner.mu, winner.phi
    l_mu, l_phi = loser.mu, loser.phi
    _update_player(winner, l_mu, l_phi, 1.0)
    _update_player(loser, w_mu, w_phi, 0.0)


def _update_player(player: Glicko2Rating, mu_j: float, phi_j: float, score: float) -> None:
    """Glicko-2 step 5 for a single opponent, using pre-match (mu_j, phi_j)."""
    mu, phi, vol = player.mu, player.phi, player.vol

    g = 1.0 / math.sqrt(1.0 + 3.0 * phi_j ** 2 / math.pi ** 2)
    e = 1.0 / (1.0 + math.exp(-g * (mu - mu_j)))

    v = 1.0 / (g ** 2 * e * (1.0 - e))
    delta = v * g * (score - e)

    # step 5.2-5.5: new volatility (Illinois algorithm on f(x))
    a = math.log(vol ** 2)
    tau = _TAU
    epsilon = _CONVERGENCE_TOL

    def f(x: float) -> float:
        ex = math.exp(x)
        num = ex * (delta ** 2 - phi ** 2 - v - ex)
        den = 2.0 * (phi ** 2 + v + ex) ** 2
        return num / den - (x - a) / tau ** 2

    big_a = a
    if delta ** 2 > phi ** 2 + v:
        big_b = math.log(delta ** 2 - phi ** 2 - v)
    else:
        k = 1.0
        while f(a - k * tau) < 0.0:
            k += 1.0
        big_b = a - k * tau
    fa, fb = f(big_a), f(big_b)
    while abs(big_b - big_a) > epsilon:
        big_c = big_a + (big_b - big_a) / 2.0
        fc = f(big_c)
        if fc * fb <= 0.0:
            big_a, fa = big_b, fb
        else:
            fa = fc
        big_b, fb = big_c, fc
    new_vol = math.exp(big_a / 2.0)

    phi_star = math.sqrt(phi ** 2 + new_vol ** 2)
    new_phi = 1.0 / math.sqrt(1.0 / phi_star ** 2 + 1.0 / v)
    new_mu = mu + new_phi ** 2 * g * (score - e)

    player.mu, player.phi, player.vol = new_mu, new_phi, new_vol
