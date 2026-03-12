"""
================================================================================
regime_strategy.py
================================================================================
Nautilus Trader Strategy that classifies every incoming TradeTick into one
of four market regimes and routes it to the appropriate sub-strategy.

REGIME MAP
──────────
┌─────────────────┬──────────────────────────────────────────────────────────┐
│ Regime          │ Signature                                                │
├─────────────────┼──────────────────────────────────────────────────────────┤
│ TRENDING        │ EMA slope significant, vol moderate, price inside bands  │
│ TRADING_RANGE   │ EMA slope flat, vol elevated, price inside bands,        │
│                 │ no clear directional bias                                │
│ BLACK_SWAN      │ Price breaches Bollinger Band + spike vol + spike volume │
│ UNDEFINED       │ Insufficient data or no regime signal dominant           │
└─────────────────┴──────────────────────────────────────────────────────────┘

INDICATOR TOOLKIT (pure Python / numpy — no Nautilus indicators required)
─────────────────────────────────────────────────────────────────────────────
• Bollinger Bands (adaptive width):
    Upper = μ + k(σ) · vol_scalar    Lower = μ − k(σ) · vol_scalar
    vol_scalar stretches the bands when realized vol rises, so an unusually
    large σ spike automatically widens the "normal" corridor, making genuine
    black-swan breaches harder to trigger from normal high-vol moves.

• Short-term Z-score:
    z = (last_price − μ_short) / σ_short
    Measures how many short-window standard deviations the price sits from
    its local mean.  Used for mean-reversion entry timing in TRADING_RANGE.

• EMA slope:
    We track two EMAs (fast + slow) and the rate-of-change of the fast EMA
    over `ema_slope_lookback` ticks.  A large positive slope → uptrend;
    large negative → downtrend; near-zero → stagnation / ranging.

• Realized volatility:
    Rolling standard deviation of log-returns over `tick_window` ticks.
    Used to scale the Bollinger Band k-multiplier and to classify regimes.

• Volume spike:
    Rolling mean + std of trade sizes.  A volume spike is declared when
    the current size > vol_mean + vol_spike_sigma · vol_std.

PLACEHOLDERS
─────────────
The three active regime strategies are stubbed:
  _execute_momentum_strategy()   ← TRENDING
  _execute_mean_reversion()      ← TRADING_RANGE
  _execute_defensive_strategy()  ← BLACK_SWAN
Fill these in when you are ready to place real orders via
  self.order_factory  /  self.submit_order().

DEPENDENCIES
─────────────
  pip install nautilus_trader numpy

USAGE (in run_backtest.py)
──────────────────────────
  from regime_strategy import FatTailConfig, FatTailStrategy

  config   = FatTailConfig(
      instrument_id       = instrument.id,
      tick_window         = 1000,
      total_position_size = 10.0,
  )
  strategy = FatTailStrategy(config=config)
  engine.add_strategy(strategy)
================================================================================
"""

from __future__ import annotations

# ─── Standard library ────────────────────────────────────────────────────────
import enum
import math
from collections import deque
from typing import Deque, Optional

# ─── Third-party ─────────────────────────────────────────────────────────────
import numpy as np

# ─── NautilusTrader ───────────────────────────────────────────────────────────
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import Strategy

# ─── Our math engine ─────────────────────────────────────────────────────────
# FatTailStrategy.py exports the position-sizing functions we need.
from FatTailStrategy import plan_crypto_entry


# ==============================================================================
# SECTION 1 — REGIME ENUM
# ==============================================================================

class RegimeState(enum.Enum):
    """
    The four possible market regimes this strategy recognises.

    UNDEFINED is intentionally the *starting* state: we never trade until
    enough ticks have accumulated to make an informed classification.
    """
    UNDEFINED      = "UNDEFINED"       # Warm-up period or ambiguous signal
    TRENDING       = "TRENDING"        # Directional momentum detectable
    TRADING_RANGE  = "TRADING_RANGE"   # Mean-reverting, elevated vol, no trend
    BLACK_SWAN     = "BLACK_SWAN"      # Extreme breach: price + vol + volume


# ==============================================================================
# SECTION 2 — STRATEGY CONFIG (Nautilus StrategyConfig dataclass)
# ==============================================================================

class FatTailConfig(StrategyConfig):
    """
    Configuration dataclass for FatTailStrategy.

    All fields have defaults so you can spin up the strategy with minimal
    boilerplate; override only what you need for experimentation.

    Parameters
    ──────────
    instrument_id : InstrumentId
        The instrument to subscribe to and trade (e.g. SOLUSDT.BINANCE).

    tick_window : int, default 1000
        Length of the main rolling window (ticks) used for:
          • Realized volatility (std of log-returns)
          • Bollinger Band mean / std
          • EMA warm-up

    short_window : int, default 50
        Shorter rolling window for the fast z-score calculation.
        Must be < tick_window.

    ema_fast_span : int, default 20
        Span (in ticks) for the fast exponential moving average.
        Used to compute EMA slope for trend detection.

    ema_slow_span : int, default 100
        Span (in ticks) for the slow EMA.
        Trend confirmation: fast EMA slope direction aligns with fast vs slow
        EMA crossover direction.

    ema_slope_lookback : int, default 20
        How many ticks back we look to measure the EMA rate-of-change.
        Larger values smooth the slope; smaller values react faster.

    bb_k : float, default 2.0
        Base Bollinger Band multiplier (number of std devs from the mean).
        The *actual* multiplier is bb_k × vol_scalar, so when volatility
        spikes the bands automatically widen.

    bb_vol_scale_cap : float, default 3.0
        Maximum multiplier on bb_k.  Prevents bands from expanding so wide
        that they can never be breached (which would disable black-swan
        detection).  Effective k is capped at bb_k × bb_vol_scale_cap.

    trend_slope_threshold : float, default 0.0002
        Minimum absolute EMA slope (as a fraction of price per tick) needed
        to declare a trend.  Below this threshold the EMA is "stagnating".
        Tune this to the typical tick-to-tick noise level of your asset.

    vol_trend_max : float, default 0.5
        Maximum realized vol (annualized log-return std, normalised) for
        TRENDING regime.  Above this the market is too choppy to trend-follow.

    vol_range_min : float, default 0.1
        Minimum realized vol for TRADING_RANGE.  Below this the market is
        too quiet to be a volatile range; it may be in a quiet drift.

    vol_spike_sigma : float, default 3.0
        Number of volume std devs above the mean required to call a
        "volume spike" (one of three conditions for BLACK_SWAN).

    zscore_range_threshold : float, default 1.0
        Price z-score magnitude below which the current price is considered
        "inside" the Bollinger Band corridor.  Values above this indicate
        a potential breach.

    total_position_size : float, default 10.0
        Total SOL to distribute across the fat-tail ladder rungs when a
        trade signal fires (passed to plan_crypto_entry / plan_crypto_entry).

    log_regime_changes : bool, default True
        If True, emit a log message every time the detected regime changes.
    """

    # ── Required ───────────────────────────────────────────────────────────
    instrument_id: InstrumentId

    # ── Rolling windows ────────────────────────────────────────────────────
    tick_window:         int   = 1_000
    short_window:        int   = 50
    ema_fast_span:       int   = 20
    ema_slow_span:       int   = 100
    ema_slope_lookback:  int   = 20

    # ── Bollinger Band parameters ──────────────────────────────────────────
    bb_k:               float = 2.0
    bb_vol_scale_cap:   float = 3.0

    # ── Trend detection thresholds ─────────────────────────────────────────
    trend_slope_threshold: float = 0.0002   # fraction of price per tick
    vol_trend_max:         float = 0.50     # normalised realized vol cap

    # ── Trading range thresholds ───────────────────────────────────────────
    vol_range_min: float = 0.10             # normalised realized vol floor

    # ── Black-swan detection ───────────────────────────────────────────────
    vol_spike_sigma: float = 3.0            # volume spike sensitivity

    # ── Z-score threshold ──────────────────────────────────────────────────
    zscore_range_threshold: float = 1.0

    # ── Position sizing ────────────────────────────────────────────────────
    total_position_size: float = 10.0

    # ── Diagnostics ────────────────────────────────────────────────────────
    log_regime_changes: bool = True


# ==============================================================================
# SECTION 3 — INDICATOR STATE (pure Python helper class)
# ==============================================================================

class _IndicatorBank:
    """
    Self-contained rolling-window indicator bank.

    Deliberately uses plain Python deques + numpy rather than the Nautilus
    built-in indicator framework, so the logic is transparent and testable
    without the engine running.

    Attributes (updated on every call to `update()`)
    ─────────────────────────────────────────────────
    last_price     : float          Most recently processed price.
    ema_fast       : float | None   Fast EMA value (None during warm-up).
    ema_slow       : float | None   Slow EMA value.
    ema_slope      : float | None   Fast EMA slope over `slope_lookback` ticks.
    realized_vol   : float | None   Rolling std of log-returns (normalised).
    bb_upper       : float | None   Adaptive upper Bollinger Band.
    bb_lower       : float | None   Adaptive lower Bollinger Band.
    bb_mean        : float | None   Rolling mean used for the bands.
    short_zscore   : float | None   Short-window z-score of price.
    vol_spike      : bool           True when current size > mean + k·std.
    is_ready       : bool           True once the main window is fully warmed up.
    """

    def __init__(
        self,
        tick_window:       int,
        short_window:      int,
        ema_fast_span:     int,
        ema_slow_span:     int,
        slope_lookback:    int,
        bb_k:              float,
        bb_vol_scale_cap:  float,
        vol_spike_sigma:   float,
    ) -> None:
        self._tick_window      = tick_window
        self._short_window     = short_window
        self._ema_fast_alpha   = 2.0 / (ema_fast_span + 1)
        self._ema_slow_alpha   = 2.0 / (ema_slow_span + 1)
        self._slope_lookback   = slope_lookback
        self._bb_k             = bb_k
        self._bb_vol_scale_cap = bb_vol_scale_cap
        self._vol_spike_sigma  = vol_spike_sigma

        # Raw storage
        self._prices:      Deque[float] = deque(maxlen=tick_window)
        self._log_returns: Deque[float] = deque(maxlen=tick_window)
        self._sizes:       Deque[float] = deque(maxlen=tick_window)
        self._ema_fast_history: Deque[float] = deque(maxlen=slope_lookback + 1)

        # Public state
        self.last_price:   float         = 0.0
        self.ema_fast:     Optional[float] = None
        self.ema_slow:     Optional[float] = None
        self.ema_slope:    Optional[float] = None
        self.realized_vol: Optional[float] = None
        self.bb_upper:     Optional[float] = None
        self.bb_lower:     Optional[float] = None
        self.bb_mean:      Optional[float] = None
        self.short_zscore: Optional[float] = None
        self.vol_spike:    bool           = False
        self.is_ready:     bool           = False

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def update(self, price: float, size: float) -> None:
        """
        Ingest one tick and recompute all indicators.

        Call this on every TradeTick before reading any indicator attribute.
        """
        # Log-return vs previous price (skip if this is the very first tick)
        if self._prices:
            prev = self._prices[-1]
            if prev > 0:
                self._log_returns.append(math.log(price / prev))

        self._prices.append(price)
        self._sizes.append(size)
        self.last_price = price

        # ── EMA update ─────────────────────────────────────────────────
        if self.ema_fast is None:
            self.ema_fast = price
            self.ema_slow = price
        else:
            self.ema_fast = (price - self.ema_fast) * self._ema_fast_alpha + self.ema_fast
            self.ema_slow = (price - self.ema_slow) * self._ema_slow_alpha + self.ema_slow

        self._ema_fast_history.append(self.ema_fast)

        # ── EMA slope (rate of change over slope_lookback ticks) ───────
        #   slope = (ema_fast_now - ema_fast_N_ticks_ago) / (N * price)
        #   Dividing by price makes it dimensionless (expressed as a
        #   fraction of current price per tick), which is comparable
        #   regardless of whether SOL is at $50 or $250.
        if len(self._ema_fast_history) >= self._slope_lookback + 1:
            ema_old   = self._ema_fast_history[0]
            raw_slope = (self.ema_fast - ema_old) / self._slope_lookback
            self.ema_slope = raw_slope / price if price != 0 else 0.0

        # ── Realized volatility (std of log-returns) ───────────────────
        if len(self._log_returns) >= 2:
            arr = np.asarray(self._log_returns, dtype=np.float64)
            self.realized_vol = float(np.std(arr, ddof=1))
        else:
            self.realized_vol = None

        # ── Adaptive Bollinger Bands ────────────────────────────────────
        #   The vol_scalar stretches the bands proportionally to realized
        #   vol.  We normalise realized_vol by dividing it by a baseline
        #   (its own long-run mean, approximated as an EMA of realized_vol
        #   values).  This makes the adaptive scaling self-calibrating.
        if len(self._prices) >= self._short_window:
            price_arr  = np.asarray(self._prices, dtype=np.float64)
            mu         = float(np.mean(price_arr))
            sigma      = float(np.std(price_arr, ddof=1))
            vol_scalar = self._compute_vol_scalar(sigma, mu)
            eff_k      = min(self._bb_k * vol_scalar, self._bb_k * self._bb_vol_scale_cap)

            self.bb_mean  = mu
            self.bb_upper = mu + eff_k * sigma
            self.bb_lower = mu - eff_k * sigma

        # ── Short-window Z-score ────────────────────────────────────────
        if len(self._prices) >= self._short_window:
            short_prices = list(self._prices)[-self._short_window:]
            arr          = np.asarray(short_prices, dtype=np.float64)
            mu_s         = float(np.mean(arr))
            sigma_s      = float(np.std(arr, ddof=1))
            self.short_zscore = (price - mu_s) / sigma_s if sigma_s > 0 else 0.0

        # ── Volume spike ────────────────────────────────────────────────
        if len(self._sizes) >= 2:
            size_arr   = np.asarray(self._sizes, dtype=np.float64)
            vol_mean   = float(np.mean(size_arr))
            vol_std    = float(np.std(size_arr, ddof=1))
            spike_thr  = vol_mean + self._vol_spike_sigma * vol_std
            self.vol_spike = size > spike_thr
        else:
            self.vol_spike = False

        # ── Readiness gate ──────────────────────────────────────────────
        self.is_ready = (
            len(self._prices) >= self._tick_window
            and self.ema_slope is not None
            and self.realized_vol is not None
            and self.bb_upper is not None
            and self.short_zscore is not None
        )

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    def _compute_vol_scalar(self, sigma: float, mu: float) -> float:
        """
        Return a multiplier > 1 when current volatility is elevated.

        We compute the coefficient of variation (CV = σ/μ) as a proxy for
        normalised volatility. A baseline CV of ~0.01 (1 %) is typical for
        a liquid crypto pair within a rolling 1,000-tick window.  When CV
        doubles to 0.02, the scalar returns 2×, widening the bands.

        The scalar is capped at `bb_vol_scale_cap` to prevent infinite
        stretching during sustained high-vol regimes.
        """
        if mu <= 0:
            return 1.0
        cv = sigma / mu          # dimensionless relative volatility
        baseline_cv = 0.01       # tune per asset; ~1 % is typical for SOL
        scalar = max(1.0, cv / baseline_cv)
        return min(scalar, self._bb_vol_scale_cap)


# ==============================================================================
# SECTION 4 — REGIME CLASSIFIER (pure function)
# ==============================================================================

def classify_regime(
    indicators:              _IndicatorBank,
    trend_slope_threshold:   float,
    vol_trend_max:           float,
    vol_range_min:           float,
    zscore_range_threshold:  float,
) -> RegimeState:
    """
    Classify the current market regime from the indicator snapshot.

    Decision tree
    ─────────────
    The tree is evaluated **top-to-bottom**; the first matching condition wins.

    1. UNDEFINED   — indicators not yet warmed up (is_ready == False)

    2. BLACK_SWAN  — ALL three must hold simultaneously:
         a. |short_zscore| > bb_k  (price broke past the adaptive Bollinger Band)
         b. realized_vol > vol_trend_max  (elevated vol)
         c. vol_spike == True  (unusual trade size)

    3. TRENDING    — ALL three:
         a. |ema_slope| >= trend_slope_threshold  (EMA is moving, not flat)
         b. realized_vol <= vol_trend_max          (not chaotic)
         c. |short_zscore| <= zscore_range_threshold  (price inside bands)

    4. TRADING_RANGE — ALL three:
         a. |ema_slope| < trend_slope_threshold   (EMA stagnating — no clear direction)
         b. realized_vol >= vol_range_min          (enough noise to mean-revert)
         c. |short_zscore| <= zscore_range_threshold  (price oscillates inside bands)

    5. UNDEFINED (fallback) — none of the above matched clearly.

    Why this ordering?
    ───────────────────
    BLACK_SWAN is checked before TRENDING because a true black-swan event can
    exhibit a strong directional EMA slope (panics trend hard).  We want the
    extreme breach + vol spike combination to override the trend signal.

    Parameters
    ──────────
    indicators              : _IndicatorBank — freshly updated indicator state.
    trend_slope_threshold   : float — minimum |slope| to call a trend.
    vol_trend_max           : float — max realized vol for trending regime.
    vol_range_min           : float — min realized vol for trading range.
    zscore_range_threshold  : float — max |z| to be "inside the bands".

    Returns
    ───────
    RegimeState enum value.
    """
    if not indicators.is_ready:
        return RegimeState.UNDEFINED

    slope    = abs(indicators.ema_slope)          # type: ignore[arg-type]
    rv       = indicators.realized_vol             # type: ignore[assignment]
    z        = abs(indicators.short_zscore)        # type: ignore[arg-type]
    bb_k_eff = 2.0  # matches FatTailConfig.bb_k default; ideally passed in

    # ── 1. BLACK_SWAN ──────────────────────────────────────────────────────
    # All three conditions must fire simultaneously.
    band_breach = z > bb_k_eff               # price outside adaptive BB
    high_vol    = rv > vol_trend_max         # realized vol is elevated
    if band_breach and high_vol and indicators.vol_spike:
        return RegimeState.BLACK_SWAN

    # ── 2. TRENDING ────────────────────────────────────────────────────────
    # EMA moving + moderate vol + price still inside bands
    ema_moving  = slope >= trend_slope_threshold
    vol_ok      = rv <= vol_trend_max
    inside_band = z <= zscore_range_threshold
    if ema_moving and vol_ok and inside_band:
        return RegimeState.TRENDING

    # ── 3. TRADING_RANGE ────────────────────────────────────────────────────
    # EMA flat (stagnating) + elevated vol + price oscillating inside bands
    ema_flat      = slope < trend_slope_threshold
    vol_elevated  = rv >= vol_range_min
    if ema_flat and vol_elevated and inside_band:
        return RegimeState.TRADING_RANGE

    # ── 4. UNDEFINED (fallback) ────────────────────────────────────────────
    return RegimeState.UNDEFINED


# ==============================================================================
# SECTION 5 — STRATEGY CLASS
# ==============================================================================

class FatTailStrategy(Strategy):
    """
    Nautilus Trader Strategy that:

      1. Accumulates TradeTick data into a rolling indicator bank.
      2. Classifies the market regime on every tick (post-warmup).
      3. Delegates to the appropriate regime sub-strategy placeholder.

    Sub-strategy dispatch
    ─────────────────────
    TRENDING      → _execute_momentum_strategy()
    TRADING_RANGE → _execute_mean_reversion()
    BLACK_SWAN    → _execute_defensive_strategy()
    UNDEFINED     → no action

    Regime-change logging
    ─────────────────────
    When `config.log_regime_changes` is True (the default), every regime
    transition is emitted as an INFO log so you can trace turning points in
    post-hoc analysis.

    Parameters
    ──────────
    config : FatTailConfig
        The strategy configuration dataclass (see FatTailConfig docstring).
    """

    def __init__(self, config: FatTailConfig) -> None:
        super().__init__(config=config)

        # Expose config attributes as local shortcuts
        self._instrument_id       = config.instrument_id
        self._total_position_size = config.total_position_size
        self._trend_threshold     = config.trend_slope_threshold
        self._vol_trend_max       = config.vol_trend_max
        self._vol_range_min       = config.vol_range_min
        self._zscore_threshold    = config.zscore_range_threshold
        self._log_regime_changes  = config.log_regime_changes

        # Current regime (starts UNDEFINED, changes as ticks arrive)
        self._regime: RegimeState = RegimeState.UNDEFINED

        # Tick counter for optional sparse diagnostics
        self._tick_count: int = 0

        # Indicator bank
        self._indicators = _IndicatorBank(
            tick_window      = config.tick_window,
            short_window     = config.short_window,
            ema_fast_span    = config.ema_fast_span,
            ema_slow_span    = config.ema_slow_span,
            slope_lookback   = config.ema_slope_lookback,
            bb_k             = config.bb_k,
            bb_vol_scale_cap = config.bb_vol_scale_cap,
            vol_spike_sigma  = config.vol_spike_sigma,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Nautilus lifecycle hooks
    # ──────────────────────────────────────────────────────────────────────

    def on_start(self) -> None:
        """
        Called by the engine when the strategy is started.

        Subscribe to TradeTick for our instrument.  We deliberately do NOT
        subscribe to bars or quotes: TradeTick gives us individual executed
        trades, which is exactly the high-fidelity feed we need to detect
        volume spikes and precise price breaches.
        """
        self.subscribe_trade_ticks(self._instrument_id)
        self.log.info(
            f"FatTailStrategy started — watching {self._instrument_id}. "
            f"Warmup: {self._indicators._tick_window} ticks required."
        )

    def on_stop(self) -> None:
        """
        Called when the strategy is stopped (end of backtest or live session).
        Unsubscribes from all data feeds and logs a final regime state.
        """
        self.unsubscribe_trade_ticks(self._instrument_id)
        self.log.info(
            f"FatTailStrategy stopped after {self._tick_count:,} ticks. "
            f"Final regime: {self._regime.value}"
        )

    def on_reset(self) -> None:
        """Reset mutable state so the strategy can be re-run cleanly."""
        self._regime      = RegimeState.UNDEFINED
        self._tick_count  = 0
        self._indicators  = _IndicatorBank(
            tick_window      = self.config.tick_window,
            short_window     = self.config.short_window,
            ema_fast_span    = self.config.ema_fast_span,
            ema_slow_span    = self.config.ema_slow_span,
            slope_lookback   = self.config.ema_slope_lookback,
            bb_k             = self.config.bb_k,
            bb_vol_scale_cap = self.config.bb_vol_scale_cap,
            vol_spike_sigma  = self.config.vol_spike_sigma,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Tick handler — core loop
    # ──────────────────────────────────────────────────────────────────────

    def on_trade_tick(self, tick: TradeTick) -> None:
        """
        Called by the engine on every incoming TradeTick.

        Execution flow
        ──────────────
        1. Update all rolling indicators with the new price and size.
        2. Classify the current regime.
        3. If the regime changed, log the transition.
        4. Dispatch to the appropriate sub-strategy.
        """
        self._tick_count += 1

        price = float(tick.price)
        size  = float(tick.size)

        # ── Step 1: update indicators ─────────────────────────────────
        self._indicators.update(price, size)

        # ── Step 2: classify regime ───────────────────────────────────
        new_regime = classify_regime(
            indicators             = self._indicators,
            trend_slope_threshold  = self._trend_threshold,
            vol_trend_max          = self._vol_trend_max,
            vol_range_min          = self._vol_range_min,
            zscore_range_threshold = self._zscore_threshold,
        )

        # ── Step 3: log regime transitions ────────────────────────────
        if new_regime != self._regime:
            if self._log_regime_changes:
                self.log.info(
                    f"[REGIME CHANGE] {self._regime.value} → {new_regime.value} "
                    f"| tick #{self._tick_count:,} "
                    f"| price={price:.4f} "
                    f"| rv={self._indicators.realized_vol:.6f} "
                    f"| slope={self._indicators.ema_slope:.8f} "
                    f"| z={self._indicators.short_zscore:.4f} "
                    f"| vol_spike={self._indicators.vol_spike}"
                )
            self._regime = new_regime

        # ── Step 4: dispatch ──────────────────────────────────────────
        if self._regime == RegimeState.TRENDING:
            self._execute_momentum_strategy(tick, price)

        elif self._regime == RegimeState.TRADING_RANGE:
            self._execute_mean_reversion(tick, price)

        elif self._regime == RegimeState.BLACK_SWAN:
            self._execute_defensive_strategy(tick, price)

        # RegimeState.UNDEFINED → no action until enough data available

    # ──────────────────────────────────────────────────────────────────────
    # SECTION 5A — TRENDING REGIME → Momentum strategy  [PLACEHOLDER]
    # ──────────────────────────────────────────────────────────────────────

    def _execute_momentum_strategy(self, tick: TradeTick, price: float) -> None:
        """
        [PLACEHOLDER] Momentum / trend-following strategy.

        WHEN THIS FIRES
        ───────────────
        The EMA slope is significant (above `trend_slope_threshold`) in either
        direction, realized volatility is moderate (below `vol_trend_max`),
        and the price is contained inside the adaptive Bollinger Bands.
        These are the ideal conditions for trend-following: the market has
        picked a direction and is not overextended yet.

        WHAT TO IMPLEMENT HERE
        ──────────────────────
        • Determine direction: `sign(self._indicators.ema_slope)`:
            +1 → uptrend  → buy / go long
            -1 → downtrend → sell / go short (if short-selling is enabled)

        • Size the entry using the fat-tail ladder from FatTailStrategy.py:
            entry_ladder = plan_crypto_entry(
                center_price        = price,
                extent_price        = price * (1 - 0.05),   # 5 % below current
                df                  = 3.0,
                total_position_size = self._total_position_size,
            )
            # Then loop over entry_ladder rows and submit limit orders:
            #   for _, row in entry_ladder.iterrows():
            #       order = self.order_factory.limit(
            #           instrument_id = self._instrument_id,
            #           order_side    = OrderSide.BUY,
            #           quantity      = Quantity.from_str(f"{row['position_size']:.2f}"),
            #           price         = Price.from_str(f"{row['price']:.2f}"),
            #       )
            #       self.submit_order(order)

        • Exit / stop logic:
            Use a trailing stop or exit when EMA slope reverses below threshold.

        Relevant indicators at call time
        ─────────────────────────────────
          self._indicators.ema_fast   — current fast EMA value
          self._indicators.ema_slope  — signed slope (+ = up, - = down)
          self._indicators.realized_vol — realized volatility
          self._indicators.short_zscore — price z-score vs short window
        """
        # ── STUB: log the signal, do not send orders yet ──────────────
        direction = "LONG" if (self._indicators.ema_slope or 0) > 0 else "SHORT"
        self.log.debug(
            f"[MOMENTUM] {direction} signal | price={price:.4f} "
            f"| slope={self._indicators.ema_slope:.8f}"
        )
        # TODO: implement order submission (see docstring above)

    # ──────────────────────────────────────────────────────────────────────
    # SECTION 5B — TRADING_RANGE REGIME → Z-score Mean Reversion  [PLACEHOLDER]
    # ──────────────────────────────────────────────────────────────────────

    def _execute_mean_reversion(self, tick: TradeTick, price: float) -> None:
        """
        [PLACEHOLDER] Z-score mean-reversion strategy for ranging markets.

        WHEN THIS FIRES
        ───────────────
        The EMA slope is below `trend_slope_threshold` (flat / stagnating),
        realized volatility is elevated (above `vol_range_min`), and the price
        is still inside the adaptive Bollinger Bands.  This is a noisy,
        directionless market — perfect for fading short-term extremes and
        reverting to the mean.

        WHAT TO IMPLEMENT HERE
        ──────────────────────
        • The short-window z-score (`self._indicators.short_zscore`) tells you
          how far price has stretched from its local mean:
            z < −zscore_range_threshold → price is low → BUY the dip
            z >  zscore_range_threshold → price is high → SELL the rip

        • Size using the fat-tail ladder — the distribution naturally puts more
          capital near the mean and less at the extremes:
            ladder = plan_crypto_entry(
                center_price        = self._indicators.bb_mean,
                extent_price        = self._indicators.bb_lower,   # buy side
                df                  = 4.0,   # moderately fat tails for ranging
                total_position_size = self._total_position_size,
            )

        • Exit: close the position when z-score reverts to 0 (price returns
          to the short-window mean).

        • Risk: place a stop just beyond the Bollinger Band to limit loss
          if the range breaks into a trend.

        Relevant indicators at call time
        ─────────────────────────────────
          self._indicators.short_zscore  — key entry signal
          self._indicators.bb_mean       — mean-reversion target
          self._indicators.bb_upper      — upper risk reference
          self._indicators.bb_lower      — lower risk reference
          self._indicators.realized_vol  — to scale stop distance
        """
        z    = self._indicators.short_zscore or 0.0
        side = "BUY" if z < 0 else "SELL"
        self.log.debug(
            f"[MEAN_REVERSION] {side} signal | price={price:.4f} "
            f"| z={z:.4f} "
            f"| bb_mean={self._indicators.bb_mean:.4f}"
        )
        # TODO: implement order submission (see docstring above)

    # ──────────────────────────────────────────────────────────────────────
    # SECTION 5C — BLACK_SWAN REGIME → Defensive strategy  [PLACEHOLDER]
    # ──────────────────────────────────────────────────────────────────────

    def _execute_defensive_strategy(self, tick: TradeTick, price: float) -> None:
        """
        [PLACEHOLDER] Defensive / risk-reduction strategy for black-swan events.

        WHEN THIS FIRES
        ───────────────
        All three conditions are simultaneously true:
          1. Price has breached an adaptive Bollinger Band (|z| > bb_k).
          2. Realized volatility is above `vol_trend_max` (extreme moves).
          3. A volume spike is detected (unusual trade size = liquidity event).

        This combination — band breach + elevated vol + volume spike — is the
        statistical fingerprint of a flash crash, liquidation cascade, or
        macro news shock.

        WHAT TO IMPLEMENT HERE
        ──────────────────────
        • PRIMARY: immediately flatten / reduce any open position.
            for position in self.cache.positions_open(instrument_id=self._instrument_id):
                self.close_position(position)

        • SECONDARY (optional — aggressive fade):
          Black-swan events often reverse sharply.  The fat-tail math was
          designed exactly for this: extreme levels get meaningful weight.
            recovery_ladder = plan_crypto_entry(
                center_price        = price,
                extent_price        = price * 0.85,   # buy 15 % below crash price
                df                  = 2.0,            # maximum fat tails (YOLO)
                total_position_size = self._total_position_size * 0.5,
            )
          Only do this with a fraction of your size and with hard stops.

        • CIRCUIT BREAKER (recommended):
          Implement a cooldown counter so that the strategy does not fire on
          every tick during a sustained panic.  Example:
            if self._swan_cooldown > 0:
                self._swan_cooldown -= 1
                return
            # ... submit defensive orders ...
            self._swan_cooldown = 200  # skip next 200 ticks

        Relevant indicators at call time
        ─────────────────────────────────
          self._indicators.short_zscore  — magnitude of the breach
          self._indicators.realized_vol  — severity of the vol spike
          self._indicators.vol_spike     — True (always True in this branch)
          self._indicators.bb_upper      — breached upper band reference
          self._indicators.bb_lower      — breached lower band reference
        """
        z   = self._indicators.short_zscore or 0.0
        rv  = self._indicators.realized_vol or 0.0
        bnd = "UPPER" if z > 0 else "LOWER"
        self.log.warning(
            f"[BLACK_SWAN] {bnd} band breached | price={price:.4f} "
            f"| z={z:.4f} "
            f"| realized_vol={rv:.6f} "
            f"| vol_spike=True — defensive mode active"
        )
        # TODO: implement position flattening / defensive order submission
        #       (see docstring above)

    # ──────────────────────────────────────────────────────────────────────
    # Diagnostic helpers (optional — useful in notebooks / custom reporting)
    # ──────────────────────────────────────────────────────────────────────

    @property
    def current_regime(self) -> RegimeState:
        """Read-only access to the most recently classified regime."""
        return self._regime

    @property
    def indicator_snapshot(self) -> dict:
        """
        Return a dict of the current indicator values.

        Handy for logging, unit tests, or building a real-time dashboard.

        Example
        ───────
        >>> snap = strategy.indicator_snapshot
        >>> print(snap)
        {'price': 135.22, 'ema_fast': 135.10, 'ema_slow': 134.90,
         'ema_slope': 0.000023, 'realized_vol': 0.0041, 'bb_upper': 136.14,
         'bb_lower': 134.22, 'short_zscore': 0.87, 'vol_spike': False,
         'is_ready': True, 'regime': 'TRENDING'}
        """
        ind = self._indicators
        return {
            "price":        ind.last_price,
            "ema_fast":     ind.ema_fast,
            "ema_slow":     ind.ema_slow,
            "ema_slope":    ind.ema_slope,
            "realized_vol": ind.realized_vol,
            "bb_upper":     ind.bb_upper,
            "bb_lower":     ind.bb_lower,
            "bb_mean":      ind.bb_mean,
            "short_zscore": ind.short_zscore,
            "vol_spike":    ind.vol_spike,
            "is_ready":     ind.is_ready,
            "regime":       self._regime.value,
        }
