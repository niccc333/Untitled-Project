"""
================================================================================
run_backtest.py
================================================================================
Entry point for backtesting the FatTailStrategy against historical Solana
tick data using the NautilusTrader BacktestEngine.

ARCHITECTURE OVERVIEW
─────────────────────
The three files that make up this system are:

  FatTailStrategy.py  ← Core mathematics: Non-Central t-distribution
                          position sizing engine.  No Nautilus dependency.

  regime_strategy.py  ← Nautilus Actor/Strategy subclass (FatTailStrategy)
                          that consumes trade ticks, detects volatility
                          regimes, and calls the math engine to size orders.

  run_backtest.py     ← THIS FILE.  Wires everything together: creates the
                          simulated venue + account, loads tick data, runs
                          the strategy, and prints a P&L summary.

HOW TO RUN
──────────
  1. Place your tick data at  ./solana_ticks.csv
     Expected columns (case-sensitive):
       timestamp  — ISO 8601 string  e.g. "2024-01-15T12:34:56.789123Z"
                    OR Unix milliseconds as an integer / float
       price      — float, e.g.  98.34
       size       — float (quantity traded), e.g. 2.5

  2. Install dependencies:
       pip install nautilus_trader pandas

  3. Run:
       python run_backtest.py

IMPORTANT: `regime_strategy.py` must exist in the same directory and expose:
    • FatTailConfig  — a Nautilus StrategyConfig dataclass
    • FatTailStrategy — a Nautilus Strategy subclass accepting FatTailConfig
================================================================================
"""

from __future__ import annotations

# ─── Standard library ────────────────────────────────────────────────────────
import csv
import os
from datetime import datetime, timezone
from decimal import Decimal

# ─── Third-party ─────────────────────────────────────────────────────────────
try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False

# ─── NautilusTrader core ──────────────────────────────────────────────────────
from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.backtest.models import FillModel, LatencyModel
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.currencies import SOL, USDT
from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.enums import AccountType, AggressorSide, OmsType
from nautilus_trader.model.identifiers import InstrumentId, Symbol, TradeId, Venue
from nautilus_trader.model.instruments import CryptoPerpetual
from nautilus_trader.model.objects import Money, Price, Quantity
from nautilus_trader.test_kit.providers import TestInstrumentProvider   # optional helper
from nautilus_trader.core.datetime import dt_to_unix_nanos

# ─── Our custom strategy ──────────────────────────────────────────────────────
# regime_strategy.py must live alongside this file and export these two names.
from regime_strategy import FatTailConfig, FatTailStrategy


# ==============================================================================
# SECTION 1 — CONSTANTS
# ==============================================================================

# Tick data file.  Override via the SOLANA_TICKS env var if you want.
DEFAULT_TICK_FILE: str = os.environ.get(
    "SOLANA_TICKS", os.path.join(os.path.dirname(__file__), "solana_ticks.csv")
)

# Simulated venue name — must match what the strategy subscribes to.
VENUE_NAME: str = "BINANCE"

# Instrument identifier — "SOLUSDT.BINANCE"
INSTRUMENT_ID_STR: str = f"SOLUSDT.{VENUE_NAME}"

# Starting account balance in USDT
STARTING_BALANCE_USDT: float = 100_000.0

# Strategy parameters
TICK_WINDOW: int = 1_000       # rolling window of ticks used for vol regime
TOTAL_POSITION_SIZE: float = 10.0   # 10 SOL total across ladder rungs


# ==============================================================================
# SECTION 2 — INSTRUMENT DEFINITION
# ==============================================================================

def build_solusdt_instrument() -> CryptoPerpetual:
    """
    Create a Nautilus CryptoPerpetual instrument for SOLUSDT on BINANCE.

    Key parameters
    ──────────────
    • price_precision / price_increment
        SOL/USDT on Binance uses a tick size of $0.01.
        price_precision = 2  means prices are stored / compared to 2 d.p.
        price_increment  = Price("0.01")

    • size_precision / size_increment
        Binance allows SOL quantities down to 0.01 SOL.
        size_precision = 2 means quantities are rounded to 2 d.p.
        size_increment  = Quantity("0.01")

    • multiplier
        For spot / perp crypto the multiplier is 1 (not a leveraged future).

    Returns
    -------
    CryptoPerpetual — a Nautilus instrument ready to add to the engine.
    """
    instrument_id = InstrumentId(
        symbol=Symbol("SOLUSDT"),
        venue=Venue(VENUE_NAME),
    )

    return CryptoPerpetual(
        instrument_id=instrument_id,
        raw_symbol=Symbol("SOLUSDT"),
        base_currency=SOL,            # SOL is the base asset in the SOLUSDT pair
        quote_currency=USDT,
        settlement_currency=USDT,
        is_inverse=False,
        price_precision=2,
        size_precision=2,
        price_increment=Price.from_str("0.01"),
        size_increment=Quantity.from_str("0.01"),
        multiplier=Quantity.from_str("1"),
        lot_size=Quantity.from_str("0.01"),
        max_quantity=Quantity.from_str("9000000"),
        min_quantity=Quantity.from_str("0.01"),
        max_notional=None,
        min_notional=Money(Decimal("5"), USDT),   # Binance min order ~ $5
        max_price=Price.from_str("9999999.99"),
        min_price=Price.from_str("0.01"),
        margin_init=Decimal("0.05"),      # 5 % initial margin
        margin_maint=Decimal("0.025"),    # 2.5 % maintenance margin
        maker_fee=Decimal("0.0001"),      # 0.01 % maker fee
        taker_fee=Decimal("0.0004"),      # 0.04 % taker fee
        ts_event=0,
        ts_init=0,
    )


# ==============================================================================
# SECTION 3 — TICK DATA LOADER
# ==============================================================================

def _parse_timestamp(value: str | float | int) -> int:
    """
    Convert a flexible timestamp representation to Unix nanoseconds (int).

    Supported formats
    ─────────────────
    • ISO 8601 string  →  "2024-01-15T12:34:56.789123Z"
                          "2024-01-15 12:34:56.789"
    • Unix milliseconds (int or float, 13-digit era)
    • Unix seconds      (int or float, 10-digit era)

    Nautilus requires all timestamps as Unix nanoseconds for internal ordering
    and replay fidelity.  We use `dt_to_unix_nanos` from
    `nautilus_trader.core.datetime` which handles the datetime → nanosecond
    conversion correctly, including timezone-aware datetimes.

    Parameters
    ----------
    value : str | float | int
        Raw timestamp value from the CSV row.

    Returns
    -------
    int — Unix nanoseconds.

    Raises
    ------
    ValueError — if the value cannot be parsed as any of the above formats.
    """
    if isinstance(value, str):
        # Strip optional trailing 'Z' and try ISO 8601 parse
        value_clean = value.strip()
        if value_clean.endswith("Z"):
            value_clean = value_clean[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(value_clean)
        except ValueError:
            # Fall back: numpy / pandas can parse many more formats
            if _PANDAS_AVAILABLE:
                dt = pd.Timestamp(value_clean).to_pydatetime()
            else:
                raise ValueError(
                    f"Cannot parse timestamp string: '{value!r}'.  "
                    "Install pandas for broader format support."
                )
        # Make timezone-aware if naive (assume UTC)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt_to_unix_nanos(dt)

    # Numeric path ─────────────────────────────────────────────────────────
    numeric = float(value)
    if numeric > 1e12:
        # Likely milliseconds — convert to seconds first, then to nanoseconds
        return int(numeric * 1_000_000)   # ms → ns
    elif numeric > 1e9:
        # Already seconds — convert to nanoseconds
        return int(numeric * 1_000_000_000)
    else:
        raise ValueError(
            f"Numeric timestamp {value!r} is too small to be Unix ms or s. "
            "Expected a value in the Unix milliseconds or seconds era."
        )


def load_tick_data(file_path: str, instrument: CryptoPerpetual) -> list[TradeTick]:
    """
    Read a CSV of historical Solana trade ticks and return a list of Nautilus
    TradeTick objects sorted chronologically.

    Expected CSV columns (header row required)
    ──────────────────────────────────────────
    timestamp  — ISO 8601 string OR Unix milliseconds integer/float
    price      — float, e.g.  98.34
    size       — float, e.g.  2.50

    Any additional columns are silently ignored.

    WHY TradeTick and not Bar/Quote?
    ─────────────────────────────────
    The FatTailStrategy uses a rolling window of *individual trades* to
    estimate realized volatility (spread of tick-by-tick price deltas).
    Bars aggregate away the intra-bar tail behaviour we care about;
    quote ticks don't tell us what actually traded.  TradeTick gives us
    the highest-fidelity view of price discovery.

    Parameters
    ----------
    file_path : str
        Absolute or relative path to solana_ticks.csv.
    instrument : CryptoPerpetual
        The instrument object — needed to extract the InstrumentId and to
        validate price/quantity precision against the instrument spec.

    Returns
    -------
    list[TradeTick] — chronologically ordered, ready for engine.add_data().

    Raises
    ------
    FileNotFoundError — if the CSV is not found at file_path.
    KeyError          — if required columns are missing from the CSV header.
    ValueError        — if any row has an unparseable timestamp / price / size.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Tick data file not found: '{file_path}'\n"
            "Create solana_ticks.csv with columns: timestamp, price, size\n"
            "or export the path via the SOLANA_TICKS environment variable."
        )

    ticks: list[TradeTick] = []

    # ── Use pandas if available (vectorised and handles more date formats) ──
    if _PANDAS_AVAILABLE:
        df = pd.read_csv(file_path)
        required = {"timestamp", "price", "size"}
        missing = required - set(df.columns.str.lower())
        if missing:
            raise KeyError(
                f"CSV is missing required columns: {missing}.  "
                f"Found: {list(df.columns)}"
            )
        # Normalise column names to lowercase
        df.columns = df.columns.str.lower()

        for row_idx, row in df.iterrows():
            try:
                ts_nanos = _parse_timestamp(row["timestamp"])
                price    = Price.from_str(f"{float(row['price']):.2f}")
                quantity = Quantity.from_str(f"{float(row['size']):.2f}")
            except (ValueError, KeyError) as exc:
                raise ValueError(
                    f"Row {row_idx + 2}: failed to parse — {exc}"
                ) from exc

            tick = TradeTick(
                instrument_id=instrument.id,
                price=price,
                size=quantity,
                aggressor_side=AggressorSide.NO_AGGRESSOR,  # unknown from OHLCV data
                trade_id=TradeId(str(row_idx)),                # synthetic trade ID
                ts_event=ts_nanos,
                ts_init=ts_nanos,
            )
            ticks.append(tick)

    # ── Fallback: stdlib csv (no pandas) ───────────────────────────────────
    else:
        with open(file_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                raise ValueError("CSV file appears to be empty.")

            # Normalise header names
            fieldnames_lower = {f.lower() for f in reader.fieldnames}
            required = {"timestamp", "price", "size"}
            missing = required - fieldnames_lower
            if missing:
                raise KeyError(
                    f"CSV is missing required columns: {missing}.  "
                    f"Found: {list(reader.fieldnames)}"
                )

            for row_idx, raw_row in enumerate(reader):
                # Lowercase the keys for uniform access
                row = {k.lower(): v for k, v in raw_row.items()}
                try:
                    ts_nanos = _parse_timestamp(row["timestamp"])
                    price    = Price.from_str(f"{float(row['price']):.2f}")
                    quantity = Quantity.from_str(f"{float(row['size']):.2f}")
                except (ValueError, KeyError) as exc:
                    raise ValueError(
                        f"Row {row_idx + 2}: failed to parse — {exc}"
                    ) from exc

                tick = TradeTick(
                    instrument_id=instrument.id,
                    price=price,
                    size=quantity,
                    aggressor_side=AggressorSide.NO_AGGRESSOR,
                    trade_id=TradeId(str(row_idx)),
                    ts_event=ts_nanos,
                    ts_init=ts_nanos,
                )
                ticks.append(tick)

    # Nautilus requires data to be in ascending timestamp order.
    # Real exchange feeds are usually ordered, but sort defensively.
    ticks.sort(key=lambda t: t.ts_event)

    print(f"[INFO] Loaded {len(ticks):,} ticks from '{file_path}'")
    return ticks


# ==============================================================================
# SECTION 4 — BACKTEST ENGINE SETUP
# ==============================================================================

def build_engine() -> BacktestEngine:
    """
    Construct and return a configured Nautilus BacktestEngine.

    Engine configuration
    ────────────────────
    • bypass_logging=False     — keep full Nautilus logs for debugging.
      Set to True in production CI to suppress verbose output.
    • logging level=WARNING    — avoid INFO noise in the backtest run;
      change to "DEBUG" if you need step-by-step execution traces.
    • trader_id               — arbitrary identifier string used in logs.
    """
    config = BacktestEngineConfig(
        trader_id="BACKTEST-001",
        logging=LoggingConfig(
            log_level="WARNING",
            log_level_file="DEBUG",
            log_file_name="backtest_run",
        ),
    )
    return BacktestEngine(config=config)


def add_venue_to_engine(engine: BacktestEngine) -> None:
    """
    Register the simulated BINANCE venue with the engine.

    Venue configuration
    ───────────────────
    oms_type=OmsType.NETTING
        Solana is a crypto perpetual — positions net (long + short = flat).
        For a strategy that only goes long or only short this behaves
        identically to HEDGING, but NETTING is the industry-standard model
        for crypto perp venues.

    account_type=AccountType.MARGIN
        Margin accounts allow the strategy to hold open positions funded by
        the USDT balance.  Use CASH if you want a spot-only simulation.

    base_currency=USDT
        All P&L is denominated in and settled in USDT.

    starting_balances=[Money(100_000, USDT)]
        $100,000 USDT starting capital.

    fill_model & latency_model
        We use the default fill model (immediate fill at tick price) and no
        artificial latency.  Override these for more realistic simulation:
            FillModel(prob_fill_on_limit=0.5)  → 50% chance of limit fill
    """
    engine.add_venue(
        venue=Venue(VENUE_NAME),
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=USDT,
        starting_balances=[Money(Decimal(str(STARTING_BALANCE_USDT)), USDT)],
        fill_model=FillModel(),    # Default: immediate fill at market
        latency_model=LatencyModel(base_latency_nanos=0),
    )


# ==============================================================================
# SECTION 5 — MAIN ENTRY POINT
# ==============================================================================

def main() -> None:
    """
    Orchestrate the full backtest run.

    Steps
    ─────
    1. Build the BacktestEngine with logging config.
    2. Add the simulated BINANCE venue.
    3. Build and register the SOLUSDT instrument.
    4. Load tick data from solana_ticks.csv and add to the engine.
    5. Instantiate FatTailStrategy via FatTailConfig.
    6. Run the backtest.
    7. Print execution stats and P&L summary.
    """

    # ── 1. Engine ──────────────────────────────────────────────────────────
    print("[INFO] Building backtest engine …")
    engine = build_engine()

    # ── 2. Simulated venue ─────────────────────────────────────────────────
    print(f"[INFO] Adding simulated venue '{VENUE_NAME}' …")
    add_venue_to_engine(engine)

    # ── 3. Instrument ──────────────────────────────────────────────────────
    print("[INFO] Creating SOLUSDT instrument …")
    instrument = build_solusdt_instrument()
    engine.add_instrument(instrument)
    print(f"[INFO] Instrument registered: {instrument.id}")

    # ── 4. Historical tick data ────────────────────────────────────────────
    print(f"[INFO] Loading tick data from '{DEFAULT_TICK_FILE}' …")
    ticks = load_tick_data(DEFAULT_TICK_FILE, instrument)

    # engine.add_data() accepts any list of Nautilus Data objects.
    # Data must be sorted by ts_event; load_tick_data() ensures this.
    engine.add_data(ticks)
    print(f"[INFO] {len(ticks):,} TradeTick objects added to engine data catalog.")

    # ── 5. Strategy ────────────────────────────────────────────────────────
    print("[INFO] Configuring FatTailStrategy …")
    strategy_config = FatTailConfig(
        instrument_id=instrument.id,    # InstrumentId("SOLUSDT.BINANCE")
        tick_window=TICK_WINDOW,        # Rolling window for vol regime detection
        total_position_size=TOTAL_POSITION_SIZE,   # 10 SOL total across rungs
    )

    strategy = FatTailStrategy(config=strategy_config)
    engine.add_strategy(strategy)
    print(f"[INFO] Strategy '{strategy.id}' added.")

    # ── 6. Run ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Starting backtest …")
    print("=" * 60)
    engine.run()
    print("=" * 60)
    print("  Backtest complete.")
    print("=" * 60)

    # ── 7. Results ────────────────────────────────────────────────────────
    _print_results(engine, instrument)

    # Clean up resources (closes any internal threads/connections).
    engine.dispose()


def _print_results(engine: BacktestEngine, instrument: CryptoPerpetual) -> None:
    """
    Print a condensed post-run summary to stdout.

    What we report
    ──────────────
    • Account balances (final USDT balance vs starting capital)
    • Fill stats     (total orders submitted / filled / rejected)
    • Position info  (any residual open position at end of data)

    For a full tearsheet integrate with `nautilus_trader.analysis.statistics`
    or export engine.cache to a DataFrame for custom reporting.
    """
    print("\n" + "=" * 60)
    print("  BACKTEST RESULTS SUMMARY")
    print("=" * 60)

    # Account balance
    try:
        account = engine.portfolio.account(Venue(VENUE_NAME))
        if account is not None:
            final_balance = account.balance_total(USDT)
            pnl           = float(final_balance.as_decimal()) - STARTING_BALANCE_USDT
            pct_return    = (pnl / STARTING_BALANCE_USDT) * 100.0
            print(f"  Starting balance : {STARTING_BALANCE_USDT:>14,.2f} USDT")
            print(f"  Final balance    : {float(final_balance.as_decimal()):>14,.2f} USDT")
            print(f"  Net P&L          : {pnl:>+14,.2f} USDT  ({pct_return:+.2f} %)")
        else:
            print("  [WARN] Could not retrieve account — check venue name.")
    except Exception as exc:
        print(f"  [WARN] Account balance unavailable: {exc}")

    # Order / fill stats
    try:
        orders_total  = len(engine.cache.orders())
        orders_filled = len(engine.cache.orders_closed())
        print(f"\n  Orders submitted : {orders_total:>8,}")
        print(f"  Orders filled    : {orders_filled:>8,}")
    except Exception as exc:
        print(f"  [WARN] Order stats unavailable: {exc}")

    # Open positions at end of data
    try:
        positions = engine.cache.positions_open()
        if positions:
            print(f"\n  Open positions at end of data: {len(positions)}")
            for pos in positions:
                print(f"    {pos.instrument_id}  qty={pos.quantity}  "
                      f"avg_px={pos.avg_px_open:.4f}  upnl={pos.unrealized_pnl}")
        else:
            print("\n  No open positions at end of data (flat).")
    except Exception as exc:
        print(f"  [WARN] Position info unavailable: {exc}")

    print("=" * 60)


# ==============================================================================
# SECTION 6 — SCRIPT ENTRY GUARD
# ==============================================================================

if __name__ == "__main__":
    main()
