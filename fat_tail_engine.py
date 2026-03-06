"""
================================================================================
fat_tail_engine.py
================================================================================
A systematic trading position-sizing library built on the Non-Central
Student's t-distribution (scipy.stats.nct).

WHY A T-DISTRIBUTION INSTEAD OF A NORMAL/GAUSSIAN CURVE?
---------------------------------------------------------
The standard Gaussian ("bell curve") assumes that extreme price moves are
nearly impossible — yet every trader knows that flash crashes, earnings gaps,
and liquidity voids happen far more often than the Gaussian predicts.

The Student's t-distribution solves this by adding "heavy tails": it assigns
meaningfully more probability mass to extreme outcomes.  We stack on top of
that the *non-centrality* parameter (nc), which lets us tilt the curve left
or right (i.e., skew our orders toward one side of the book).

CORE DISTRIBUTION: scipy.stats.nct(df, nc)
  • df  (Degrees of Freedom) — controls tail thickness.
        df = 2–4  → Extremely fat tails  (crypto, meme stocks)
        df = 5–15 → Moderately fat tails (equities, commodities)
        df > 30   → Approaches a standard Normal curve
  • nc  (Non-centrality parameter) — controls skew.
        nc < 0 → Skew distribution to the left  (more weight below center)
        nc = 0 → Symmetric around center price
        nc > 0 → Skew distribution to the right (more weight above center)

QUICK-START EXAMPLE:
    from fat_tail_engine import generate_fat_tail_levels, plan_crypto_entry

    # Generic engine call
    df_orders = generate_fat_tail_levels(
        center_price=30_000.0,   # BTC current price
        extent_price=25_000.0,   # Furthest buy level (below market)
        df=3.0,                  # Heavy tails for crypto
        num_steps=5,
        total_position_size=1.0  # 1 BTC total
    )

    # Or use an asset-class wrapper that pre-fills sensible defaults
    df_orders = plan_crypto_entry(
        center_price=30_000.0,
        extent_price=25_000.0,
        total_position_size=1.0
    )

Dependencies (install once):
    pip install scipy pandas numpy
================================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats


# ==============================================================================
# SECTION 1 — CORE MATHEMATICAL ENGINE
# ==============================================================================

def generate_fat_tail_levels(
    center_price: float,
    extent_price: float,
    df: float = 3.0,
    max_std: float = 4.0,
    num_steps: int = 5,
    skew: float = 0.0,
    total_position_size: float = 1.0,
    tick_size: float = 0.01,
) -> pd.DataFrame:
    """
    Generate a normalized DataFrame of price levels and their associated
    position-size weights using the Non-Central Student's t-distribution.

    The engine works by treating the journey from center_price to extent_price
    as a walk along the x-axis of the t-distribution.  Prices close to center
    receive weights from the peak of the PDF (lots of capital near the middle),
    while prices at the extreme end receive weights from the tails of the PDF
    (less capital, but still meaningful — that's the whole point!).

    Parameters
    ----------
    center_price : float
        The "anchor" price — typically the current market price or the mid of
        a range.  This maps to x = 0 (the center of the distribution).

    extent_price : float
        The furthest price level you are willing to work orders to.
        • If extent_price < center_price → ladder of BUY orders going DOWN
        • If extent_price > center_price → ladder of SELL orders going UP
        This maps to x = max_std on the t-distribution's x-axis.

    df : float, default 3.0
        Degrees of Freedom — the key "fat tail" dial.
        ┌──────────────┬──────────────────────────────────────────────────┐
        │ df value     │ Behaviour                                        │
        ├──────────────┼──────────────────────────────────────────────────┤
        │ 2 – 3        │ Extremely heavy tails; big weight on far levels  │
        │ 4 – 7        │ Moderately heavy tails (good for equities)       │
        │ 8 – 15       │ Light tails; resembles a slightly fat Gaussian   │
        │ > 30         │ Practically identical to a Normal distribution   │
        └──────────────┴──────────────────────────────────────────────────┘
        WARNING: df must be > 1 (the t-distribution is undefined at df ≤ 1).
        For the heaviest crypto setups, df = 2 is the practical minimum.

    max_std : float, default 4.0
        How many "standard deviations" (on the t-distribution's x-axis) the
        range center_price → extent_price represents.
        • Increase this to spread orders wider without changing df.
        • The default of 4.0 (vs. 3.0 for a normal Gaussian) takes advantage
          of the fatter tails that actually contain useful probability mass.

    num_steps : int, default 5
        Number of discrete price levels (order rungs) in the ladder.
        • Minimum sensible value: 2
        • Typical range: 3–10
        Each rung gets its own price + weight from the PDF.

    skew : float, default 0.0
        Tilts the weight distribution left or right.  Passed directly as the
        non-centrality parameter (nc) to scipy.stats.nct.
        • skew < 0 → heavier weight toward center_price (safer distribution)
        • skew = 0 → symmetric around center
        • skew > 0 → heavier weight toward extent_price (aggressive fade)
        Practical range: −3.0 to +3.0.  Beyond ±5 the distribution becomes
        very one-sided and the normalization will still work correctly, but the
        resulting orders may feel non-intuitive.

    total_position_size : float, default 1.0
        The total notional amount to be distributed across all rungs.
        Units are whatever you choose (BTC, shares, contracts, USD notional).
        Every rung's size will be a fraction of this number and they sum to
        exactly total_position_size after normalization.

    tick_size : float, default 0.01
        The minimum price increment for the asset.  All generated price levels
        are rounded to the nearest tick so orders can be submitted directly to
        an exchange without modification.
        • Equities: typically 0.01 (1 cent)
        • Crypto:   often 0.01, 0.10, 1.0, or 10.0 depending on the pair
        • Futures:  check the contract specification (e.g., 0.25 for ES)

    Returns
    -------
    pd.DataFrame with columns:
        price        — Exchange-ready price, rounded to tick_size
        raw_weight   — Unnormalized PDF value at that x-coordinate
        weight       — Normalized weight (sums to 1.0 across all rows)
        position_size — weight × total_position_size (sums to total_position_size)

    Raises
    ------
    ValueError
        • If center_price == extent_price (zero-width range)
        • If df <= 1 (distribution undefined)
        • If num_steps < 2 (need at least two rungs for a ladder)

    Example
    -------
    >>> df_orders = generate_fat_tail_levels(
    ...     center_price=100.0,
    ...     extent_price=90.0,   # Buy ladder going down 10 points
    ...     df=4.0,
    ...     num_steps=5,
    ...     total_position_size=1000.0,  # $1,000 total
    ...     tick_size=0.01,
    ... )
    >>> print(df_orders)
    """

    # ------------------------------------------------------------------
    # 1a. INPUT VALIDATION
    # We catch obviously wrong inputs early so errors are easy to trace.
    # ------------------------------------------------------------------
    if center_price == extent_price:
        raise ValueError(
            "center_price and extent_price cannot be the same. "
            "There is no range to distribute orders across."
        )
    if df <= 1:
        raise ValueError(
            f"df must be greater than 1 (received {df}). "
            "The t-distribution's variance is undefined at df ≤ 1, making "
            "it unsuitable for position sizing."
        )
    if num_steps < 2:
        raise ValueError(
            f"num_steps must be at least 2 (received {num_steps}). "
            "A single order is not a ladder."
        )
    if tick_size <= 0:
        raise ValueError(f"tick_size must be positive (received {tick_size}).")
    if total_position_size <= 0:
        raise ValueError(
            f"total_position_size must be positive (received {total_position_size})."
        )

    # ------------------------------------------------------------------
    # 1b. MAP PRICE RANGE → X-AXIS COORDINATES
    #
    # We need to translate real prices (e.g. $30,000) into the abstract
    # x-axis of the t-distribution (e.g. 0 → max_std).
    #
    # Strategy:
    #   • center_price  ↔  x = 0       (the mean of the distribution)
    #   • extent_price  ↔  x = max_std (the tail we're reaching for)
    #
    # np.linspace gives us evenly spaced x values from 0 to max_std.
    # We then convert those x values back to real prices using a simple
    # linear mapping:
    #
    #   price = center_price + (x / max_std) × (extent_price − center_price)
    #
    # The direction (up or down) falls out naturally from the sign of
    # (extent_price − center_price).
    # ------------------------------------------------------------------

    # Evenly spaced x coordinates from 0 (center) to max_std (extent)
    x_values = np.linspace(0, max_std, num_steps)

    # Convert x coordinates back to actual prices
    price_range = extent_price - center_price          # negative for buy ladders
    prices_raw = center_price + (x_values / max_std) * price_range

    # Round every price to the nearest valid tick
    # np.round can introduce tiny float errors, so we divide → round → multiply
    prices_ticked = np.round(prices_raw / tick_size) * tick_size

    # ------------------------------------------------------------------
    # 1c. CALCULATE RAW PDF WEIGHTS
    #
    # scipy.stats.nct.pdf(x, df, nc) gives the probability density of the
    # Non-Central t-distribution at each x point.
    #
    # Think of this as: "how likely is an extreme market move of this size?"
    # The higher the PDF value, the more capital sits at that level.
    #
    # The `skew` parameter maps directly to `nc` (non-centrality).
    # A positive skew shifts the hump of the curve rightward (toward extent),
    # putting more weight on farther-out levels.
    # ------------------------------------------------------------------

    raw_weights = stats.nct.pdf(x_values, df=df, nc=skew)

    # ------------------------------------------------------------------
    # 1d. NORMALIZE WEIGHTS TO SUM EXACTLY TO 1.0
    #
    # CRITICAL STEP — Why is this necessary?
    #
    # Unlike the full Normal distribution where ∫pdf dx from −∞ to +∞ = 1,
    # we are only using a *slice* of the t-distribution (x = 0 to max_std).
    # The remaining probability mass lives outside our range and is discarded.
    # Additionally, with heavy tails (low df), the absolute PDF values change
    # dramatically compared to a Gaussian.
    #
    # Normalizing ensures that our discrete weights *always* sum to 1.0,
    # regardless of df, max_std, skew, or num_steps.  Without this step, the
    # final position sizes would not equal total_position_size.
    # ------------------------------------------------------------------

    total_raw_weight = raw_weights.sum()

    if total_raw_weight == 0:
        # Edge case: if all PDF values are ~0 (e.g. extreme skew pushes
        # all mass outside our window), fall back to uniform distribution.
        normalized_weights = np.ones(num_steps) / num_steps
    else:
        normalized_weights = raw_weights / total_raw_weight

    # ------------------------------------------------------------------
    # 1e. SCALE TO TOTAL POSITION SIZE
    #
    # Multiply each normalized weight by the total notional amount.
    # The resulting sizes sum to exactly total_position_size.
    # ------------------------------------------------------------------

    position_sizes = normalized_weights * total_position_size

    # ------------------------------------------------------------------
    # 1f. ASSEMBLE OUTPUT DATAFRAME
    # ------------------------------------------------------------------

    result = pd.DataFrame({
        "price":         prices_ticked,
        "raw_weight":    raw_weights,
        "weight":        normalized_weights,
        "position_size": position_sizes,
    })

    # Label the direction so callers can inspect/log easily
    result.attrs["direction"] = "BUY" if extent_price < center_price else "SELL"
    result.attrs["center_price"] = center_price
    result.attrs["extent_price"] = extent_price
    result.attrs["df"] = df
    result.attrs["skew"] = skew
    result.attrs["total_position_size"] = total_position_size

    return result


# ==============================================================================
# SECTION 2 — ASSET-CLASS WRAPPERS
#
# These wrapper functions pre-fill `df` and other parameters with sensible
# defaults for specific asset classes, so traders don't need to remember the
# raw distribution parameters.  All underlying math is from the core engine
# above; these are just convenient entry-points.
# ==============================================================================

def plan_crypto_entry(
    center_price: float,
    extent_price: float,
    df: float = 3.0,
    num_steps: int = 7,
    skew: float = 0.0,
    total_position_size: float = 1.0,
    tick_size: float = 1.0,
    max_std: float = 4.0,
) -> pd.DataFrame:
    """
    Plan a DCA/ladder entry for a cryptocurrency pair.

    WHY df = 3.0 BY DEFAULT?
    ─────────────────────────
    Crypto markets are notorious for:
      • Flash crashes of 20–40% in minutes (e.g., the March 2020 COVID crash,
        May 2021 China ban, FTX collapse Nov 2022)
      • Thin, fragmented liquidity that magnifies wick depth
      • 24/7 trading with no circuit breakers

    A df of 3 gives a *much* heavier tail than a Gaussian and places
    meaningful order size at the very bottom of your ladder — exactly where
    you want to be filled during a sudden capitulation candle.

    Suggested df ranges for crypto:
      • df = 2   — Absolute YOLO mode; almost all weight at extreme levels
      • df = 3   — Default; good balance for BTC/ETH
      • df = 4–5 — If you prefer more weight near the current price

    Parameters (see generate_fat_tail_levels for full docs)
    -------------------------------------------------------
    center_price : float
        Current mid-market price of the crypto pair.
    extent_price : float
        The lowest price you are willing to buy (for a buy ladder).
    df : float, default 3.0
        Degrees of freedom.  Lower = fatter tails.
    num_steps : int, default 7
        Number of rungs.  Crypto ladders often benefit from more rungs
        because the bid-ask spread is wide and you want granular fills.
    skew : float, default 0.0
        Negative skew clusters orders near center (conservative).
        Positive skew spreads them toward the extreme (aggressive fade).
    total_position_size : float, default 1.0
        Total coins/contracts/USD to deploy across the ladder.
    tick_size : float, default 1.0
        Minimum price increment.  BTC/USD on most exchanges is $1.
        Adjust for altcoins (e.g., 0.0001 for a $0.50 token).
    max_std : float, default 4.0
        How many t-std-devs the range spans.

    Returns
    -------
    pd.DataFrame — see generate_fat_tail_levels for column definitions.

    Example
    -------
    >>> # Buy up to 0.5 BTC between $60,000 and $45,000
    >>> orders = plan_crypto_entry(
    ...     center_price=60_000,
    ...     extent_price=45_000,
    ...     total_position_size=0.5,
    ...     tick_size=1.0,
    ... )
    >>> print(orders)
    """
    return generate_fat_tail_levels(
        center_price=center_price,
        extent_price=extent_price,
        df=df,
        max_std=max_std,
        num_steps=num_steps,
        skew=skew,
        total_position_size=total_position_size,
        tick_size=tick_size,
    )


def plan_equity_entry(
    center_price: float,
    extent_price: float,
    df: float = 7.0,
    num_steps: int = 5,
    skew: float = 0.0,
    total_position_size: float = 100.0,
    tick_size: float = 0.01,
    max_std: float = 4.0,
) -> pd.DataFrame:
    """
    Plan a DCA/ladder entry for a stock or ETF.

    WHY df = 7.0 BY DEFAULT?
    ─────────────────────────
    Equities are less volatile than crypto on an intraday basis, but they
    experience heavy-tail events via:
      • Earnings surprises (stocks can gap 10–30% overnight)
      • Macro shocks (rate decisions, geopolitical events)
      • Sector contagion (bank runs, oil price spikes)

    df = 7 keeps tails noticeably fatter than Gaussian (capturing those
    earnings-day gaps) while still concentrating the bulk of capital near
    the current price — appropriate for a regulated market with circuit
    breakers and defined trading hours.

    Suggested df ranges for equities:
      • df = 4–5  — High-beta small-caps / meme stocks (YOLO tier)
      • df = 7    — Default; good for S&P 500 components
      • df = 10   — Blue-chip large-caps with low historical vol
      • df = 15+  — Near-Gaussian; use for ultra-defensive dividend payers

    Parameters (see generate_fat_tail_levels for full docs)
    -------------------------------------------------------
    center_price : float
        Current mid-market price of the stock.
    extent_price : float
        The lowest price you are willing to buy (or highest to short).
    df : float, default 7.0
        Degrees of freedom.
    num_steps : int, default 5
        Number of order rungs.
    skew : float, default 0.0
        Positive skew = more weight at the extreme (better avg fill price
        but requires stock to move further for full deployment).
    total_position_size : float, default 100.0
        Total shares to buy/sell across the ladder.
    tick_size : float, default 0.01
        US equities use $0.01 (1 cent) minimum.
    max_std : float, default 4.0
        How many t-std-devs the range spans.

    Returns
    -------
    pd.DataFrame — see generate_fat_tail_levels for column definitions.

    Example
    -------
    >>> # Buy 200 shares of NVDA between $900 and $750
    >>> orders = plan_equity_entry(
    ...     center_price=900.0,
    ...     extent_price=750.0,
    ...     df=7.0,
    ...     total_position_size=200,
    ...     tick_size=0.01,
    ... )
    >>> print(orders)
    """
    return generate_fat_tail_levels(
        center_price=center_price,
        extent_price=extent_price,
        df=df,
        max_std=max_std,
        num_steps=num_steps,
        skew=skew,
        total_position_size=total_position_size,
        tick_size=tick_size,
    )


def plan_option_hedge(
    center_strike: float,
    extent_strike: float,
    df: float = 4.0,
    num_steps: int = 6,
    skew: float = 1.5,
    total_contracts: float = 10.0,
    tick_size: float = 0.05,
    max_std: float = 4.0,
) -> pd.DataFrame:
    """
    Distribute option contracts across OTM strikes using the fat-tail curve
    to model the Volatility Smile / Skew observed in real options markets.

    THE VOLATILITY SMILE CONNECTION
    ────────────────────────────────
    In standard Black-Scholes theory, implied volatility (IV) should be flat
    across all strikes.  In reality it forms a "smile" or "skew" shape:
    OTM puts and calls trade at *higher* IV than ATM options.

    This is the market's own admission that extreme moves happen more often
    than a Gaussian predicts — i.e., the market prices in fat tails!

    Our nct distribution naturally replicates this shape.  By mapping:
      • center_strike  = at-the-money (ATM) strike   → x = 0  (peak of PDF)
      • extent_strike  = deep OTM strike              → x = max_std (tail)

    …we get a weight distribution that mirrors how professional hedgers think
    about scaling into OTM protection: buy most contracts close to ATM where
    gamma is high, taper off further OTM, but never completely ignore the
    deep OTM strikes (fat tail = Black Swan protection).

    WHY df = 4.0 AND skew = 1.5 BY DEFAULT?
    ──────────────────────────────────────────
    • df = 4 captures meaningful tail risk at deep OTM strikes, consistent
      with typical equity vol skew steepness.
    • skew = 1.5 (+nc) tilts the distribution toward the OTM end, reflecting
      the empirical observation that OTM puts/calls carry elevated IV.
    • Adjust skew negative (e.g., −1) to front-load contracts near ATM
      (gamma-focused hedging strategy).

    Parameters
    -------------------------------------------------------
    center_strike : float
        The ATM or near-ATM strike where your hedge is anchored.
    extent_strike : float
        The furthest OTM strike you want to include in the hedge.
        • For puts  (downside hedge): extent_strike < center_strike
        • For calls (upside  hedge): extent_strike > center_strike
    df : float, default 4.0
        Degrees of freedom (tail thickness / vol smile steepness).
    num_steps : int, default 6
        Number of OTM strikes to populate.
    skew : float, default 1.5
        Non-centrality; shifts weight toward OTM strikes to model vol skew.
    total_contracts : float, default 10.0
        Total option contracts to distribute.
    tick_size : float, default 0.05
        Minimum strike increment.  US equity options: $0.50 or $1.00 wide
        strikes → set tick_size=0.50 or 1.0 as appropriate.
    max_std : float, default 4.0
        How many t-std-devs the strike range spans.

    Returns
    -------
    pd.DataFrame — same columns as generate_fat_tail_levels.
        'position_size' = number of contracts to buy at each strike.

    Example
    -------
    >>> # Buy SPY puts from ATM ($450) to deep OTM ($400) — 10 contracts total
    >>> hedge = plan_option_hedge(
    ...     center_strike=450.0,
    ...     extent_strike=400.0,    # Put hedge going down
    ...     total_contracts=10.0,
    ...     tick_size=1.0,          # $1-wide strikes
    ... )
    >>> print(hedge)
    """
    # Options use `center_strike` / `extent_strike` terminology, but the
    # underlying math is identical — just rename for the call.
    result = generate_fat_tail_levels(
        center_price=center_strike,
        extent_price=extent_strike,
        df=df,
        max_std=max_std,
        num_steps=num_steps,
        skew=skew,
        total_position_size=total_contracts,
        tick_size=tick_size,
    )

    # Rename column for clarity in options context
    result = result.rename(columns={"position_size": "contracts"})
    result.attrs["instrument"] = "options"

    return result


# ==============================================================================
# SECTION 3 — UTILITY / DISPLAY HELPERS
# ==============================================================================

def summarize_orders(df: pd.DataFrame, instrument_label: str = "position_size") -> None:
    """
    Pretty-print a summary of the generated order ladder.

    Parameters
    ----------
    df : pd.DataFrame
        Output from any generate_* or plan_* function.
    instrument_label : str
        Column to display as the "size" column.  Use "contracts" for options,
        "position_size" for everything else.
    """
    direction = df.attrs.get("direction", "N/A")
    center    = df.attrs.get("center_price", "N/A")
    extent    = df.attrs.get("extent_price", "N/A")
    df_val    = df.attrs.get("df", "N/A")
    skew_val  = df.attrs.get("skew", "N/A")
    total     = df.attrs.get("total_position_size", "N/A")

    size_col = instrument_label if instrument_label in df.columns else "position_size"

    print("=" * 60)
    print(f"  FAT-TAIL ORDER LADDER SUMMARY")
    print("=" * 60)
    print(f"  Direction     : {direction}")
    print(f"  Center price  : {center}")
    print(f"  Extent price  : {extent}")
    print(f"  Degrees of freedom (df) : {df_val}  <- tail thickness dial")
    print(f"  Skew (nc)     : {skew_val}")
    print(f"  Total size    : {total}")
    print("-" * 60)
    print(f"  {'Price':>12}  {'Weight':>10}  {'Size':>14}")
    print(f"  {'-----':>12}  {'------':>10}  {'----':>14}")
    for _, row in df.iterrows():
        size_val = row.get(size_col, row.get("position_size", float("nan")))
        print(f"  {row['price']:>12.4f}  {row['weight']:>10.4f}  {size_val:>14.6f}")
    print("-" * 60)
    print(f"  Weight sum    : {df['weight'].sum():.6f}  (should be 1.000000)")
    if size_col in df.columns:
        print(f"  Size sum      : {df[size_col].sum():.6f}  (should match total)")
    print("=" * 60)


# ==============================================================================
# SECTION 4 — DEMO / QUICK-START
#
# Run this file directly (`python fat_tail_engine.py`) to see live output
# for all three asset-class wrappers.
# ==============================================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("  DEMO 1 - Crypto Entry (BTC, fat tails, df=3)")
    print("=" * 60)
    # Scenario: BTC is at $65,000.  We want to buy 0.5 BTC
    # if it dips to $50,000, using 6 rungs.
    # df=3 ensures we put real capital at $50–53k, not just scraps.
    btc_orders = plan_crypto_entry(
        center_price=65_000.0,
        extent_price=50_000.0,
        df=3.0,
        num_steps=6,
        total_position_size=0.5,  # 0.5 BTC total
        tick_size=1.0,            # $1 minimum price increment
    )
    summarize_orders(btc_orders)

    print("\n" + "=" * 60)
    print("  DEMO 2 - Equity Entry (NVDA, moderate tails, df=7)")
    print("=" * 60)
    # Scenario: NVDA at $900, willing to buy down to $750.
    # df=7 → heavier than Gaussian but not as extreme as crypto.
    nvda_orders = plan_equity_entry(
        center_price=900.0,
        extent_price=750.0,
        df=7.0,
        num_steps=5,
        total_position_size=100,  # 100 shares
        tick_size=0.01,
    )
    summarize_orders(nvda_orders)

    print("\n" + "=" * 60)
    print("  DEMO 3 - Options Hedge (SPY puts, vol smile, df=4)")
    print("=" * 60)
    # Scenario: SPY at $450 ATM.  Buy put protection down to $400.
    # df=4 + positive skew models the real-world equity vol skew
    # where OTM puts trade at elevated implied volatility.
    spy_hedge = plan_option_hedge(
        center_strike=450.0,
        extent_strike=400.0,   # Protective puts going downward
        df=4.0,
        num_steps=6,
        skew=1.5,              # Tilt toward OTM (models vol skew)
        total_contracts=10.0,
        tick_size=1.0,         # $1-wide strikes
    )
    summarize_orders(spy_hedge, instrument_label="contracts")

    print("\n" + "=" * 60)
    print("  DEMO 4 - Custom Engine Call (raw API, sell ladder)")
    print("=" * 60)
    # Scenario: We want to distribute 1,000 shares of sell orders
    # above the current price, using a very fat tail (df=2) to
    # ensure we capture any spike highs.
    sell_ladder = generate_fat_tail_levels(
        center_price=200.0,        # Current price
        extent_price=230.0,        # Highest sell target (above market)
        df=2.0,                    # Extremely fat tail — captures spike highs
        max_std=4.0,
        num_steps=5,
        skew=-1.0,                 # Slight left skew → more weight near $200
        total_position_size=1000,  # 1,000 shares
        tick_size=0.01,
    )
    summarize_orders(sell_ladder)
