"""
================================================================================
polymarket_iv.py
================================================================================
Standalone module that extracts implied volatility from Polymarket prediction
markets using Black-Scholes for binary (digital) options, and aggregates it
into a market sentiment score.

THE IDEA
────────
Polymarket "Yes" contracts are structurally identical to cash-or-nothing
binary call options: they pay $1 if the event occurs, $0 otherwise.  The
market price (e.g. $0.65) is the discounted risk-neutral probability.

By inverting the Black-Scholes digital call formula we can extract an
*implied volatility* (IV) from each contract's price.  High IV means the
market is uncertain; low IV means the outcome is considered near-certain.

Aggregating IV across many active markets gives a single "sentiment score"
that captures how much collective uncertainty exists on Polymarket at any
given moment.

BLACK-SCHOLES FOR BINARY CALLS
──────────────────────────────
  C_digital = e^{-rT} · N(d₂)

  d₂ = [ln(S/K) − ½σ²T] / (σ√T)

We set  S = K = 1  (binary contract: pays $1 or $0, always ATM relative
to itself) and  r = 0  (crypto, negligible risk-free rate), giving:

  d₂  = −½ σ √T
  N(d₂) = market_price
  σ  = −2 · N⁻¹(market_price) / √T

When the closed-form is numerically unstable (prices near 0 or 1), we
fall back to Brent's root-finding method on the full BS equation.

SENTIMENT SCORE
───────────────
  score ∈ [0, 1]
  0.0 = very calm / certain / complacent
  1.0 = extreme fear / uncertainty / volatility

The score is a volume-weighted mean of individual market IVs, passed
through a sigmoid-style normaliser calibrated to typical IV ranges.

USAGE (standalone)
──────────────────
  python polymarket_iv.py

DEPENDENCIES
────────────
  pip install requests scipy numpy

NOTE: This module does NOT import or depend on Nautilus Trader.
================================================================================
"""

from __future__ import annotations

import math
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import requests
from scipy import optimize
from scipy.stats import norm

# ─── Logging ──────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


# ==============================================================================
# SECTION 1 — POLYMARKET API CLIENT
# ==============================================================================

class PolymarketClient:
    """
    Thin wrapper around the Polymarket Gamma and CLOB APIs (read-only).

    Gamma API  — market metadata (titles, end dates, slugs, volumes)
        Base: https://gamma-api.polymarket.com

    CLOB API   — live pricing, order book, price history
        Base: https://clob.polymarket.com

    No authentication is required for the endpoints we use.
    """

    GAMMA_BASE = "https://gamma-api.polymarket.com"
    CLOB_BASE  = "https://clob.polymarket.com"

    def __init__(self, timeout: float = 15.0) -> None:
        self._session = requests.Session()
        self._session.headers.update({
            "Accept": "application/json",
            "User-Agent": "polymarket-iv-extractor/1.0",
        })
        self._timeout = timeout

    # ──────────────────────────────────────────────────────────────────────
    # Gamma API — market discovery
    # ──────────────────────────────────────────────────────────────────────

    def fetch_markets(
        self,
        limit: int = 50,
        active: bool = True,
        closed: bool = False,
        order: str = "volume24hr",
        ascending: bool = False,
        query: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch a list of markets from the Gamma API.

        Parameters
        ----------
        limit     : max number of markets to return (capped at 100 by API)
        active    : if True, only return markets that are currently active
        closed    : if True, include closed/resolved markets
        order     : field to sort by (e.g. 'volume24hr', 'liquidity')
        ascending : sort direction
        query     : optional search query (e.g., 'Solana')

        Returns
        -------
        List of market dicts, each containing:
          - question, slug, condition_id, end_date_iso
          - tokens: list of {token_id, outcome} dicts
          - volume, volume24hr, liquidity
          etc.
        """
        params: Dict[str, Any] = {
            "limit": min(limit, 100),
            "active": str(active).lower(),
            "closed": str(closed).lower(),
            "order": order,
            "ascending": str(ascending).lower(),
        }
        if query:
            # We use the /events endpoint for searching as it supports the 'q' param better,
            # or we can filter client-side if we just want a subset of top markets.
            # However, the /markets endpoint doesn't strictly support 'q', so we will
            # fetch a larger generic list or rely on the caller filtering.
            # But let's pass it anyway as 'slug' or rely on client-side filtering below.
            pass

        url = f"{self.GAMMA_BASE}/markets"
        try:
            if query:
                # API text search is unreliable, so we paginate and filter locally
                import re
                fetch_limit = min(limit * 50, 5000)  
                all_data = []
                for offset in range(0, fetch_limit, 100):
                    params["limit"] = 100
                    params["offset"] = offset
                    resp = self._session.get(url, params=params, timeout=self._timeout)
                    resp.raise_for_status()
                    page_data = resp.json()
                    if not page_data:
                        break
                    all_data.extend(page_data)
                
                # Relaxed Regex match for Solana
                # Match 'Solana' or 'SOL' (case insensitive). Try to avoid 'solve/solar' by requiring
                # non-alphabetical boundaries or the string ending.
                import re
                filtered_data = []
                for m in all_data:
                    text_to_search = (m.get("question", "") + " " + m.get("slug", "")).lower()
                    
                    # Match "solana", or "sol" followed by non-letter (like "sol/usd", "sol reach")
                    # or at string end. Use a very explicit pattern
                    if "solana" in text_to_search or re.search(r'\bsol\b|\bsol/', text_to_search):
                        filtered_data.append(m)
                
                # De-duplicate by condition_id in case API returned overlapping pages
                seen = set()
                unique_filtered = []
                for m in filtered_data:
                    cid = m.get("condition_id")
                    if cid and cid not in seen:
                        seen.add(cid)
                        unique_filtered.append(m)

                return unique_filtered[:limit]
            else:
                # Standard single-page fetch
                resp = self._session.get(url, params=params, timeout=self._timeout)
                resp.raise_for_status()
                return resp.json()
            
        except requests.RequestException as exc:
            logger.error("Failed to fetch markets from Gamma API: %s", exc)
            return []

    # ──────────────────────────────────────────────────────────────────────
    # CLOB API — live prices
    # ──────────────────────────────────────────────────────────────────────

    def fetch_midpoint(self, token_id: str) -> Optional[float]:
        """
        Get the current midpoint price for a token (Yes or No outcome).

        Returns a float in [0, 1] or None if the request fails.
        """
        url = f"{self.CLOB_BASE}/midpoint"
        params = {"token_id": token_id}
        try:
            resp = self._session.get(url, params=params, timeout=self._timeout)
            resp.raise_for_status()
            data = resp.json()
            return float(data.get("mid", 0))
        except (requests.RequestException, ValueError, KeyError) as exc:
            logger.warning("Failed to fetch midpoint for %s: %s", token_id, exc)
            return None

    def fetch_price(self, token_id: str) -> Optional[float]:
        """
        Get the last traded price for a token.

        Returns a float in [0, 1] or None if the request fails.
        """
        url = f"{self.CLOB_BASE}/price"
        params = {"token_id": token_id, "side": "buy"}
        try:
            resp = self._session.get(url, params=params, timeout=self._timeout)
            resp.raise_for_status()
            data = resp.json()
            return float(data.get("price", 0))
        except (requests.RequestException, ValueError, KeyError) as exc:
            logger.warning("Failed to fetch price for %s: %s", token_id, exc)
            return None

    def fetch_price_history(
        self,
        token_id: str,
        fidelity: int = 60,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch historical price data for a token.

        Parameters
        ----------
        token_id  : the CLOB token ID
        fidelity  : data granularity in minutes (default 60 = hourly)
        start_ts  : start Unix timestamp (default: 7 days ago)
        end_ts    : end Unix timestamp (default: now)

        Returns
        -------
        List of {"t": timestamp, "p": price} dicts.
        """
        now = int(time.time())
        if end_ts is None:
            end_ts = now
        if start_ts is None:
            start_ts = now - 7 * 86400  # 7 days ago

        url = f"{self.CLOB_BASE}/prices-history"
        params = {
            "market": token_id,
            "startTs": start_ts,
            "endTs": end_ts,
            "fidelity": fidelity,
        }
        try:
            resp = self._session.get(url, params=params, timeout=self._timeout)
            resp.raise_for_status()
            data = resp.json()
            # API returns {"history": [{"t": ..., "p": ...}, ...]}
            return data.get("history", [])
        except (requests.RequestException, ValueError) as exc:
            logger.warning("Failed to fetch price history for %s: %s", token_id, exc)
            return []


# ==============================================================================
# SECTION 2 — BLACK-SCHOLES IMPLIED VOLATILITY FOR BINARY OPTIONS
# ==============================================================================

class BlackScholesBinaryIV:
    """
    Extract implied volatility from binary (digital) option prices using
    the Black-Scholes framework.

    For a cash-or-nothing binary call with S=K (at-the-money) and r=0:

        C = N(d₂)
        d₂ = −½ σ √T

    Solving for σ:
        σ = −2 · N⁻¹(C) / √T

    where N⁻¹ is the inverse standard normal CDF (ppf).
    """

    # Price boundaries: we can't compute IV if the market is near 0 or 1
    PRICE_FLOOR = 0.01   # below this → outcome is "almost certain NO"
    PRICE_CEIL  = 0.99   # above this → outcome is "almost certain YES"

    # Time floor: avoid division-by-zero for nearly-expired markets
    MIN_T_YEARS = 1 / 365   # ~1 day minimum

    # IV bounds for the numerical solver
    IV_LOWER = 0.001   # 0.1%  annualized
    IV_UPPER = 20.0    # 2000% annualized (extreme, but possible)

    @staticmethod
    def bs_digital_call_price(sigma: float, T: float, r: float = 0.0) -> float:
        """
        Theoretical price of a cash-or-nothing binary call with S=K=1.

        Parameters
        ----------
        sigma : implied volatility (annualized)
        T     : time to expiry in years
        r     : risk-free rate (annualized, default 0)

        Returns
        -------
        Model price in [0, 1].
        """
        if sigma <= 0 or T <= 0:
            return 0.5  # undefined edge case, return ATM
        d2 = -0.5 * sigma * math.sqrt(T)
        return math.exp(-r * T) * norm.cdf(d2)

    @classmethod
    def extract_iv(
        cls,
        market_price: float,
        T: float,
        r: float = 0.0,
    ) -> Optional[float]:
        """
        Extract implied volatility from a binary option market price.

        Parameters
        ----------
        market_price : observed market price of the "Yes" contract (0, 1)
        T            : time to expiry in years
        r            : risk-free rate (annualized, default 0)

        Returns
        -------
        Implied volatility (annualized) as a float, or None if computation
        fails (e.g. price too close to 0 or 1, or T ≈ 0).

        Notes
        -----
        The function first attempts a closed-form solution.  If that yields
        an out-of-range result, it falls back to Brent's method on the
        full BS equation.
        """
        # ── Edge-case guards ──────────────────────────────────────────
        if market_price <= cls.PRICE_FLOOR or market_price >= cls.PRICE_CEIL:
            # Near-certain outcomes → IV is meaningless / undefined
            return None

        if T < cls.MIN_T_YEARS:
            # Almost expired → T is too small for a stable inversion
            return None

        sqrt_T = math.sqrt(T)

        # ── Fast path: closed-form ────────────────────────────────────
        # For S=K=1 and r=0:  σ = −2 · N⁻¹(price) / √T
        try:
            adjusted_price = market_price / math.exp(-r * T)
            # Clamp to avoid ppf(0) or ppf(1) = ±∞
            adjusted_price = max(0.001, min(0.999, adjusted_price))
            d2 = norm.ppf(adjusted_price)
            sigma_closed = -2.0 * d2 / sqrt_T

            if cls.IV_LOWER <= sigma_closed <= cls.IV_UPPER:
                return sigma_closed
        except (ValueError, OverflowError):
            pass  # fall through to numerical solver

        # ── Fallback: Brent's root-finding method ─────────────────────
        def objective(sigma: float) -> float:
            return cls.bs_digital_call_price(sigma, T, r) - market_price

        try:
            iv = optimize.brentq(
                objective,
                cls.IV_LOWER,
                cls.IV_UPPER,
                xtol=1e-8,
                maxiter=200,
            )
            return iv
        except (ValueError, RuntimeError) as exc:
            logger.debug("Brent solver failed for price=%.4f, T=%.4f: %s",
                         market_price, T, exc)
            return None


# ==============================================================================
# SECTION 3 — SENTIMENT SCORER
# ==============================================================================

@dataclass
class MarketIV:
    """Implied volatility result for a single Polymarket market."""
    question: str
    slug: str
    market_price: float           # "Yes" contract price in [0, 1]
    implied_vol: Optional[float]  # annualized IV, or None if not computable
    time_to_expiry_days: float    # remaining days until resolution
    volume_24h: float             # 24-hour volume in USD


@dataclass
class SentimentResult:
    """Aggregated sentiment score from multiple markets."""
    score: float                  # 0 = calm, 1 = extreme uncertainty
    weighted_mean_iv: float       # volume-weighted mean IV (annualized)
    num_markets_used: int         # number of markets with valid IV
    num_markets_fetched: int      # total markets fetched from API
    market_ivs: List[MarketIV] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class SentimentScorer:
    """
    Aggregate implied volatilities across Polymarket markets into a
    single sentiment score.

    The score is computed as:

      1. For each active market, extract IV from the "Yes" contract price.
      2. Compute a volume-weighted mean IV across all markets with valid IV.
      3. Pass the mean IV through a sigmoid normaliser:

             score = 1 − exp(−mean_iv / scale)

         where `scale` controls the sensitivity.  With the default scale
         of 1.5, typical IVs map to:

           IV = 0.3  → score ≈ 0.18  (calm)
           IV = 0.8  → score ≈ 0.41  (moderate uncertainty)
           IV = 1.5  → score ≈ 0.63  (elevated uncertainty)
           IV = 3.0  → score ≈ 0.86  (high fear)
           IV = 5.0  → score ≈ 0.96  (extreme)

    Parameters
    ----------
    iv_scale : float
        Sigmoid scale parameter.  Lower = more sensitive to low IVs.
    min_volume : float
        Minimum 24h volume (USD) for a market to be included.
    min_time_to_expiry_days : float
        Markets expiring sooner than this are excluded (IV is unstable).
    """

    def __init__(
        self,
        iv_scale: float = 1.5,
        min_volume: float = 1000.0,
        min_time_to_expiry_days: float = 1.0,
    ) -> None:
        self._iv_scale = iv_scale
        self._min_volume = min_volume
        self._min_ttl = min_time_to_expiry_days
        self._iv_engine = BlackScholesBinaryIV()

    def score(
        self,
        markets: List[Dict[str, Any]],
        prices: Optional[Dict[str, float]] = None,
    ) -> SentimentResult:
        """
        Compute sentiment score from a list of Polymarket market dicts.

        Parameters
        ----------
        markets : list of market dicts (from PolymarketClient.fetch_markets)
        prices  : optional dict of {token_id: price};  if not provided,
                  the scorer will attempt to get the price from market data
                  (the `outcomePrices` field or tokens' `price` field).

        Returns
        -------
        SentimentResult with the aggregate score and per-market breakdowns.
        """
        market_ivs: List[MarketIV] = []
        now = time.time()

        for mkt in markets:
            question = mkt.get("question", "Unknown")
            slug = mkt.get("slug", "")

            # ── Parse end date → time to expiry ───────────────────────
            end_date = mkt.get("end_date_iso") or mkt.get("endDate", "")
            T_days = self._parse_time_to_expiry(end_date, now)
            if T_days is None or T_days < self._min_ttl:
                continue
            T_years = T_days / 365.0

            # ── Get the "Yes" token price ─────────────────────────────
            yes_price = self._extract_yes_price(mkt, prices)
            if yes_price is None:
                continue

            # ── 24h volume ────────────────────────────────────────────
            vol_24h = float(mkt.get("volume24hr", 0) or 0)
            if vol_24h < self._min_volume:
                continue

            # ── Extract IV ────────────────────────────────────────────
            iv = self._iv_engine.extract_iv(yes_price, T_years)

            market_ivs.append(MarketIV(
                question=question,
                slug=slug,
                market_price=yes_price,
                implied_vol=iv,
                time_to_expiry_days=T_days,
                volume_24h=vol_24h,
            ))

        # ── Aggregate ─────────────────────────────────────────────────
        valid = [m for m in market_ivs if m.implied_vol is not None]

        if not valid:
            return SentimentResult(
                score=0.0,
                weighted_mean_iv=0.0,
                num_markets_used=0,
                num_markets_fetched=len(markets),
                market_ivs=market_ivs,
            )

        # Volume-weighted mean IV
        total_vol = sum(m.volume_24h for m in valid)
        if total_vol > 0:
            wmean_iv = sum(m.implied_vol * m.volume_24h for m in valid) / total_vol
        else:
            wmean_iv = np.mean([m.implied_vol for m in valid])

        # Sigmoid normalisation → [0, 1]
        score = 1.0 - math.exp(-wmean_iv / self._iv_scale)
        score = max(0.0, min(1.0, score))

        return SentimentResult(
            score=score,
            weighted_mean_iv=wmean_iv,
            num_markets_used=len(valid),
            num_markets_fetched=len(markets),
            market_ivs=market_ivs,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_time_to_expiry(end_date: str, now: float) -> Optional[float]:
        """Parse ISO-8601 end date and return days until expiry, or None."""
        if not end_date:
            return None
        try:
            from datetime import datetime, timezone
            # Handle various ISO formats
            end_date = end_date.replace("Z", "+00:00")
            dt = datetime.fromisoformat(end_date)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            expiry_ts = dt.timestamp()
            days = (expiry_ts - now) / 86400.0
            return days if days > 0 else None
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _extract_yes_price(
        market: Dict[str, Any],
        prices: Optional[Dict[str, float]],
    ) -> Optional[float]:
        """
        Extract the "Yes" outcome price from a market dict.

        Tries multiple data paths in the Gamma API response:
          1. External `prices` dict (if caller fetched from CLOB)
          2. `outcomePrices` field (JSON-encoded string like "[0.65, 0.35]")
          3. `tokens[0].price` (if the first token is "Yes")
        """
        tokens = market.get("tokens", [])

        # Strategy 1: use externally-provided prices
        if prices and tokens:
            for token in tokens:
                outcome = (token.get("outcome") or "").upper()
                tid = token.get("token_id", "")
                if outcome == "YES" and tid in prices:
                    return prices[tid]

        # Strategy 2: outcomePrices field (common in Gamma API response)
        outcome_prices = market.get("outcomePrices")
        if outcome_prices:
            try:
                import json
                parsed = json.loads(outcome_prices) if isinstance(outcome_prices, str) else outcome_prices
                if isinstance(parsed, list) and len(parsed) >= 1:
                    p = float(parsed[0])
                    if 0 < p < 1:
                        return p
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        # Strategy 3: token-level price
        if tokens:
            for token in tokens:
                outcome = (token.get("outcome") or "").upper()
                if outcome == "YES":
                    p = token.get("price")
                    if p is not None:
                        p = float(p)
                        if 0 < p < 1:
                            return p

        return None


# ==============================================================================
# SECTION 4 — CLI DEMO
# ==============================================================================

def _format_iv(iv: Optional[float]) -> str:
    """Format IV for display: percentage or 'N/A'."""
    if iv is None:
        return "  N/A   "
    return f"{iv * 100:7.1f}%"


def _sentiment_label(score: float) -> str:
    """Human-readable label for the sentiment score."""
    if score < 0.15:
        return "😴 Very Calm"
    elif score < 0.30:
        return "😌 Calm"
    elif score < 0.50:
        return "😐 Moderate"
    elif score < 0.70:
        return "😰 Elevated"
    elif score < 0.85:
        return "😨 High Fear"
    else:
        return "🔥 Extreme"


def main() -> None:
    """
    Fetch active Polymarket markets, compute implied volatility for each,
    and print an aggregate sentiment score.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    print()
    print("=" * 78)
    print("  SOLANA IMPLIED VOLATILITY → SENTIMENT SCORE")
    print("  Black-Scholes Binary Option IV Extractor (SOL Markets Only)")
    print("=" * 78)
    print()

    # ── Fetch markets ─────────────────────────────────────────────────
    client = PolymarketClient()
    logger.info("Fetching active Solana markets from Polymarket...")
    
    # We fetch a larger batch and filter by query="solana" internally
    # which now also matches "sol" automatically in the client regex
    solana_markets = client.fetch_markets(limit=200, active=True, query="solana")

    if not solana_markets:
        print("⚠  No Solana markets found currently active on Polymarket.")
        print("   This may be a network issue, API downtime, or no active SOL markets.")
        return

    logger.info("Filtered down to %d active Solana markets.", len(solana_markets))

    # ── Compute sentiment ─────────────────────────────────────────────
    # For crypto markets, IVs are naturally very high. We use a larger scale
    # so the score doesn't default to 1.0 instantly.
    scorer = SentimentScorer(
        iv_scale=5.0,  # Increased scale for crypto
        min_volume=100.0,  # Lower volume requirement as niche markets might have less vol
        min_time_to_expiry_days=0.5, # Reduced to catch near-term action
    )
    result = scorer.score(solana_markets)

    # ── Print per-market breakdown ────────────────────────────────────
    print("-" * 78)
    print(f"  {'Market':<40}  {'Price':>6}  {'IV':>8}  {'TTL':>6}  {'Vol24h':>10}")
    print("-" * 78)

    # Sort: valid IVs first (descending), then N/A
    sorted_ivs = sorted(
        result.market_ivs,
        key=lambda m: (m.implied_vol is None, -(m.implied_vol or 0)),
    )

    for m in sorted_ivs:
        q = m.question[:40] if len(m.question) > 40 else m.question
        ttl = f"{m.time_to_expiry_days:.0f}d"
        vol = f"${m.volume_24h:,.0f}"
        print(f"  {q:<40}  {m.market_price:>5.2f}¢  {_format_iv(m.implied_vol)}  {ttl:>6}  {vol:>10}")

    # ── Print aggregate score ─────────────────────────────────────────
    print()
    print("=" * 78)
    print(f"  SENTIMENT SCORE")
    print("-" * 78)
    print(f"  Markets fetched         : {result.num_markets_fetched}")
    print(f"  Markets with valid IV   : {result.num_markets_used}")
    print(f"  Weighted mean IV        : {result.weighted_mean_iv * 100:.1f}%  (annualized)")
    print(f"  Sentiment score         : {result.score:.4f}")
    print(f"  Label                   : {_sentiment_label(result.score)}")
    print("=" * 78)
    print()
    print("  Score interpretation:")
    print("    0.00 – 0.15  😴 Very calm (near-certain outcomes dominate)")
    print("    0.15 – 0.30  😌 Calm")
    print("    0.30 – 0.50  😐 Moderate uncertainty")
    print("    0.50 – 0.70  😰 Elevated uncertainty")
    print("    0.70 – 0.85  😨 High fear / volatility")
    print("    0.85 – 1.00  🔥 Extreme (rare — markets are deeply divided)")
    print()


if __name__ == "__main__":
    main()
