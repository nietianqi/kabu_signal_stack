"""kabu_signal_stack.py — Redesigned signal stack + VeighNa strategy wrapper.  [v3]

Architecture (4 layers):
  [VeighNa TickData]
      → KabuTickAdapter    (field mapping, bid/ask reversal switch, sanity check)
      → TickSnapshot       (clean L1/L2 snapshot, up to 5 price levels)
      → KabuSignalStack    (OBI / LOB-OFI / Tape-OFI / momentum / microprice tilt /
                            VWAP deviation / full-depth book imbalance)
      → KabuSignalStrategy (VeighNa CtaTemplate: state machine, order tracking, risk)

Verified against official sources (as of 2025):
  - TSE 呼値単位: JPX standard domestic equity tick size table (現物株式)
    ETF/REIT/TOPIX500 constituents may use different tables — verify before live use.
  - Trading hours: 09:00–11:30 (前場) / 12:30–15:30 (後場), orders accepted from 08:00
  - kabu Station API: BestBid/BestAsk field names may be reversed vs. industry convention.
    Set reverse_bid_ask=True if your VeighNa gateway does not normalise these fields.
  - Commission: kabu offers multiple fee plans (定額/都度). Default parameters use the
    ad-valorem 都度 model (片道 0.385% 税込, min ¥55). Always confirm against your
    actual account plan before live trading.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple

try:
    from vnpy.trader.constant import Direction, Status
    try:
        from vnpy_ctastrategy import CtaTemplate, StopOrder
        from vnpy.trader.object import OrderData, TradeData
    except Exception:
        from vnpy.app.cta_strategy import CtaTemplate, StopOrder  # type: ignore
        from vnpy.trader.object import OrderData, TradeData          # type: ignore
    _VNPY_AVAILABLE = True
except Exception:
    _VNPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Signal utilities (ZNormalizer, FlowFlipDetector)
# ---------------------------------------------------------------------------

class ZNormalizer:
    """Online rolling z-score normalizer.

    Accumulates a fixed-size window of values and normalises each new value
    to zero mean / unit variance.  Returns 0.0 during the initial warm-up
    period (< 10 samples) to avoid noisy early readings.

    Reference: signals.md §8
    """

    def __init__(self, lookback: int = 100) -> None:
        self._buf: Deque[float] = deque(maxlen=max(10, lookback))

    def normalize(self, value: float) -> float:
        self._buf.append(value)
        n = len(self._buf)
        if n < 10:
            return 0.0
        mean = sum(self._buf) / n
        var  = sum((x - mean) ** 2 for x in self._buf) / n
        std  = math.sqrt(max(var, 0.0))
        return 0.0 if std < 1e-10 else (value - mean) / std


class FlowFlipDetector:
    """Detect consecutive order-flow direction reversals (momentum exhaustion).

    When the tape aggressor direction flips ``flip_threshold`` or more times in
    a row (alternating buy/sell) it signals that momentum is exhausted and the
    current trend is likely to stall or reverse.

    Reference: signals.md §7
    """

    def __init__(self, flip_threshold: int = 3) -> None:
        self.threshold:     int  = flip_threshold
        self._consecutive:  int  = 0
        self._last_dir:     int  = 0
        self.flip_detected: bool = False

    def update(self, aggressor: int) -> bool:
        """Update with the latest aggressor direction (+1 / -1 / 0).

        Returns True when a flip of at least ``threshold`` reversals is detected.
        """
        if aggressor == 0:
            self.flip_detected = False
            return False
        if aggressor == self._last_dir:
            self._consecutive += 1
            self.flip_detected = False
        else:
            # Direction changed: check if the *previous* run was long enough
            self.flip_detected = self._consecutive >= self.threshold
            self._consecutive  = 1
            self._last_dir     = aggressor
        return self.flip_detected


# ---------------------------------------------------------------------------
# Trade record & PnL tracker
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    """Immutable record of one completed round-trip trade.

    Reference: risk.md §2
    """
    entry_price: float
    exit_price:  float
    volume:      float
    direction:   int         # +1 = LONG, -1 = SHORT
    commission:  float       # total round-trip commission (¥)
    entry_time:  datetime
    exit_time:   datetime
    exit_reason: str         # "TP" / "SL" / "TIMEOUT" / "SIGNAL_FLIP" / "TRAILING" / "FAST_LOSS"

    @property
    def pnl(self) -> float:
        return self.direction * (self.exit_price - self.entry_price) * self.volume - self.commission

    @property
    def hold_sec(self) -> float:
        return (self.exit_time - self.entry_time).total_seconds()

    @property
    def is_win(self) -> bool:
        return self.pnl > 0


class PnLTracker:
    """Accumulates TradeRecords and computes performance statistics.

    Reference: risk.md §2
    """

    def __init__(self) -> None:
        self.trades:     List[TradeRecord] = []
        self.cumulative: float = 0.0
        self._peak:      float = 0.0
        self.max_dd:     float = 0.0

    def record(self, trade: TradeRecord) -> None:
        self.trades.append(trade)
        self.cumulative += trade.pnl
        self._peak   = max(self._peak, self.cumulative)
        self.max_dd  = max(self.max_dd, self._peak - self.cumulative)

    def stats(self) -> dict:
        n = len(self.trades)
        if n == 0:
            return {}
        wins    = [t.pnl for t in self.trades if t.is_win]
        losses  = [t.pnl for t in self.trades if not t.is_win]
        all_pnl = [t.pnl for t in self.trades]
        mean_p  = sum(all_pnl) / n
        var_p   = sum((x - mean_p) ** 2 for x in all_pnl) / n
        std_p   = math.sqrt(max(var_p, 0.0))
        avg_w   = sum(wins)   / len(wins)   if wins   else 0.0
        avg_l   = sum(losses) / len(losses) if losses else 0.0
        by_reason = {
            r: sum(1 for t in self.trades if t.exit_reason == r)
            for r in ("TP", "SL", "TIMEOUT", "SIGNAL_FLIP", "TRAILING", "FAST_LOSS")
        }
        return {
            "n":             n,
            "win_rate":      len(wins) / n,
            "profit_factor": abs(avg_w / avg_l) if avg_l else 999.0,
            "expectancy":    mean_p,
            "sharpe_approx": mean_p / std_p * (252 ** 0.5) if std_p > 1e-10 else 0.0,
            "avg_hold_sec":  sum(t.hold_sec for t in self.trades) / n,
            "by_reason":     by_reason,
        }


# ---------------------------------------------------------------------------
# Order state machine
# ---------------------------------------------------------------------------

class OrderState(Enum):
    """Strategy-level order lifecycle state.

    Transitions:
      IDLE → PENDING_OPEN   : entry order sent
      PENDING_OPEN → OPEN   : on_trade (opening fill received)
      OPEN → PENDING_CLOSE  : exit order sent
      PENDING_CLOSE → IDLE  : on_trade (closing fill received)
      any → IDLE            : cancel_all + reset (emergency / cancelled order)
    """
    IDLE          = "idle"
    PENDING_OPEN  = "pending_open"
    OPEN          = "open"
    PENDING_CLOSE = "pending_close"


# ---------------------------------------------------------------------------
# TSE tick size (呼値単位) — standard domestic equity (現物株式)
# NOTE: ETF / REIT / TOPIX500 constituents may differ; verify with JPX circulars.
# ---------------------------------------------------------------------------

def get_tse_pricetick(price: float) -> float:
    """Return the minimum price increment for a standard TSE equity.

    Based on JPX 呼値単位 table for domestic equities (Prime / Standard / Growth markets).
    Always verify against the latest JPX announcement before live use.
    """
    if price <= 0:
        return 1.0
    if price < 3_000:
        return 1.0
    if price < 5_000:
        return 5.0
    if price < 30_000:
        return 10.0
    if price < 50_000:
        return 50.0
    if price < 100_000:
        return 100.0
    if price < 1_000_000:
        return 1_000.0
    return 10_000.0


# ---------------------------------------------------------------------------
# Adapter layer — field mapping + optional bid/ask reversal + sanity check
# ---------------------------------------------------------------------------

class KabuTickAdapter:
    """Extract and normalise a TickSnapshot from a VeighNa TickData object.

    kabu Station API field naming differs from financial industry convention:
      API field BestBid  → actual sell-side (ask) price
      API field BestAsk  → actual buy-side  (bid) price

    If your VeighNa gateway already corrects this mapping (e.g. the
    kabusapi gateway adapter swaps the fields on ingestion), leave
    ``reverse_bid_ask=False`` (default). If raw API field names pass
    through unchanged, set ``reverse_bid_ask=True``.

    This class is the single correction point; all downstream code
    receives normalised data where bid < ask.
    """

    def __init__(self, reverse_bid_ask: bool = False, max_spread_pct: float = 0.05) -> None:
        self.reverse_bid_ask = reverse_bid_ask
        self.max_spread_pct  = max_spread_pct

    def extract(self, tick, pricetick: float = 1.0) -> Optional["TickSnapshot"]:
        """Return a normalised TickSnapshot, or None if the tick fails sanity checks."""

        def _f(attr: str) -> float:
            return float(getattr(tick, attr, 0.0) or 0.0)

        # Read raw L1 fields
        raw_bid = _f("bid_price_1")
        raw_ask = _f("ask_price_1")

        # Apply optional field swap
        if self.reverse_bid_ask:
            bid1, ask1 = raw_ask, raw_bid
        else:
            bid1, ask1 = raw_bid, raw_ask

        # Sanity check: both prices positive, ordered, and spread within bounds
        # (spread_pct > max_spread_pct filters out 特別気配 / auction special quotes)
        if bid1 <= 0.0 or ask1 <= 0.0 or bid1 >= ask1:
            return None
        spread_pct = (ask1 - bid1) / max(bid1, 1e-9)
        if spread_pct > self.max_spread_pct:
            return None

        # Read up to 5 price levels for L2 book
        bids: List[Tuple[float, float]] = []
        asks: List[Tuple[float, float]] = []
        for i in range(1, 6):
            bp = _f(f"bid_price_{i}")
            bv = _f(f"bid_volume_{i}")
            ap = _f(f"ask_price_{i}")
            av = _f(f"ask_volume_{i}")
            # Apply reversal to L2 levels as well
            if self.reverse_bid_ask:
                bp, ap = ap, bp
                bv, av = av, bv
            if bp > 0.0 and bv > 0.0:
                bids.append((bp, bv))
            if ap > 0.0 and av > 0.0:
                asks.append((ap, av))

        # Ensure correct sort order
        bids.sort(key=lambda x: -x[0])   # descending (best bid first)
        asks.sort(key=lambda x:  x[0])   # ascending  (best ask first)

        dt = getattr(tick, "datetime", None)
        if not isinstance(dt, datetime):
            dt = datetime.now()

        return TickSnapshot(
            dt=dt,
            bid1=bid1,
            ask1=ask1,
            bid_vol1=bids[0][1] if bids else _f("bid_volume_1"),
            ask_vol1=asks[0][1] if asks else _f("ask_volume_1"),
            bids=bids,
            asks=asks,
            last_price=_f("last_price"),
            volume=_f("volume"),
            turnover=_f("turnover"),
            pricetick=max(1e-9, pricetick),
        )


# ---------------------------------------------------------------------------
# Signal configuration
# ---------------------------------------------------------------------------

@dataclass
class SignalConfig:
    # Gate: spread
    max_spread_ticks: float = 2.0

    # Gate: liquidity (minimum best-level size on each side)
    min_best_volume: int = 50

    # Gate: price velocity
    max_vol_ticks_per_sec: float = 3.0
    vol_window_sec: float = 2.0

    # Gate: trading hours (TSE standard session)
    enable_time_gate: bool = True
    trade_start_time: str = "09:00:10"
    morning_end_time: str = "11:30:00"
    afternoon_start_time: str = "12:30:00"
    trade_end_time: str = "15:25:00"
    event_cooldown_sec: float = 120.0

    # Gate: expected value (optional, disabled by default)
    ev_gate_enabled: bool = False
    ev_ratio_min: float = 1.2
    taker_cost_ratio: float = 0.5
    ev_use_separate_mode: bool = False
    ev_min_ticks: float = 0.1

    # Fill probability / maker-taker mode
    maker_preferred_spread_ticks: float = 2.0
    maker_adverse_sel_ratio: float = 0.3

    # Signal windows
    obi_levels: int = 5          # Weighted OBI: number of order book levels
    lob_ofi_window: int = 20
    tape_window_sec: float = 1.0
    momentum_window: int = 10

    # Per-signal entry thresholds (alpha-count voting)
    microprice_tilt_long: float = 0.3
    microprice_tilt_short: float = 0.3
    lob_ofi_long: float = 0.2
    lob_ofi_short: float = 0.2
    tape_ofi_long: float = 0.2
    tape_ofi_short: float = 0.2
    momentum_long: float = 0.5
    momentum_short: float = 0.5
    obi_long: float = 0.15
    obi_short: float = 0.15

    # Edge score weights
    w_microprice: float = 1.5
    w_lob_ofi: float = 1.5
    w_tape_ofi: float = 1.0
    w_momentum: float = 0.5
    w_obi: float = 1.0
    w_vwap: float = 0.8              # VWAP mean-reversion signal weight
    w_book_depth: float = 0.5        # Full-depth book imbalance weight

    # VWAP signal thresholds
    vwap_long: float = 0.2           # price below VWAP by this fraction → long alpha
    vwap_short: float = 0.2          # price above VWAP by this fraction → short alpha

    # Book depth signal thresholds
    book_depth_long: float = 0.15    # book_depth_ratio above this → long alpha
    book_depth_short: float = 0.15   # book_depth_ratio below -this → short alpha
    book_depth_levels: int = 10      # number of L2 levels to aggregate

    # Entry thresholds
    edge_score_long_threshold: float = 2.5
    edge_score_short_threshold: float = 2.5
    min_alpha_count: int = 2

    # Regime detection
    regime_window: int = 30
    noise_vol_threshold: float = 2.0
    regime_sample_interval: int = 5
    sign_autocorr_threshold: float = 0.3  # lag-1 sign autocorrelation threshold for TREND/REVERSION

    # Signal z-score normalization
    znorm_lookback: int = 100             # rolling window length for ZNormalizer

    # Flow flip detection
    flow_flip_threshold: int = 3          # consecutive direction reversals to declare a flip

    # Adapter
    reverse_bid_ask: bool = False         # Set True if gateway passes raw kabu field names
    max_spread_pct:  float = 0.05         # reject ticks where spread > 5% of bid (特別気配 filter)

    # Auto tick size
    auto_pricetick: bool = False


# ---------------------------------------------------------------------------
# Tick snapshot (normalised L1/L2)
# ---------------------------------------------------------------------------

@dataclass
class TickSnapshot:
    """Normalised market data snapshot for one tick.

    ``bids`` and ``asks`` are lists of (price, size) tuples:
      bids: descending price order (best bid first)
      asks: ascending price order  (best ask first)
    """
    dt: datetime
    bid1: float
    ask1: float
    bid_vol1: float
    ask_vol1: float
    bids: List[Tuple[float, float]]
    asks: List[Tuple[float, float]]
    last_price: float = 0.0
    volume: float = 0.0
    turnover: float = 0.0      # cumulative session turnover (¥) — used for VWAP calculation
    pricetick: float = 1.0

    @property
    def mid(self) -> float:
        return (self.bid1 + self.ask1) / 2.0

    @property
    def spread_ticks(self) -> float:
        if self.pricetick <= 0:
            return 999.0
        return max(0.0, (self.ask1 - self.bid1) / self.pricetick)

    @property
    def microprice(self) -> float:
        total = self.bid_vol1 + self.ask_vol1
        if total <= 0:
            return self.mid
        return (self.ask1 * self.bid_vol1 + self.bid1 * self.ask_vol1) / total

    @property
    def microprice_tilt(self) -> float:
        """Microprice displacement from mid, normalised by half-spread.

        Returns a value in approximately [-1, +1]:
          +1 → all volume on buy side (strong upward pressure)
          -1 → all volume on sell side (strong downward pressure)
        """
        half_spread = (self.ask1 - self.bid1) / 2.0
        if half_spread <= 0.0:
            return 0.0
        return (self.microprice - self.mid) / half_spread


# ---------------------------------------------------------------------------
# Module-level signal functions (pure, independently testable)
# ---------------------------------------------------------------------------

def calc_weighted_obi(
    bids: List[Tuple[float, float]],
    asks: List[Tuple[float, float]],
    levels: int = 5,
) -> float:
    """Weighted order book imbalance across multiple price levels.

    Returns a value in [-1, +1]:
      +1  → strong buy-side pressure (bids dominate)
      -1  → strong sell-side pressure (asks dominate)

    Weight decay: w_i = 1 / (1 + i), so best level has weight 1.0,
    second level 0.5, third 0.33, etc.
    """
    weights = [1.0 / (1.0 + i) for i in range(levels)]
    bid_vol = sum(w * s for w, (_, s) in zip(weights, bids[:levels]))
    ask_vol = sum(w * s for w, (_, s) in zip(weights, asks[:levels]))
    total = bid_vol + ask_vol
    return 0.0 if total <= 0.0 else (bid_vol - ask_vol) / total


def calc_lob_ofi_incremental(prev: TickSnapshot, curr: TickSnapshot) -> float:
    """Incremental LOB order flow imbalance (Cont, Kukanov & Stoikov 2014).

    Tracks changes at the best bid and ask levels only.
    Positive values indicate net buying pressure; negative values indicate
    net selling pressure.

    Uses O(1) incremental updates rather than full recomputation each tick.
    """
    # Bid-side contribution
    if curr.bid1 > prev.bid1:
        bid_ofi = curr.bid_vol1           # Higher bid appeared → buy pressure added
    elif curr.bid1 == prev.bid1:
        bid_ofi = curr.bid_vol1 - prev.bid_vol1
    else:
        bid_ofi = -prev.bid_vol1          # Bid level dropped → buy pressure removed

    # Ask-side contribution
    if curr.ask1 < prev.ask1:
        ask_ofi = -curr.ask_vol1          # Lower ask appeared → sell pressure added
    elif curr.ask1 == prev.ask1:
        ask_ofi = -(curr.ask_vol1 - prev.ask_vol1)
    else:
        ask_ofi = prev.ask_vol1           # Ask level rose → sell pressure removed

    return bid_ofi + ask_ofi


def calc_tape_aggressor(
    trade_price: float,
    bid: float,
    ask: float,
    prev_trade_price: Optional[float] = None,
) -> float:
    """Classify a trade as buyer-initiated (+1), seller-initiated (-1), or ambiguous (0).

    Uses the Lee-Ready (1991) algorithm:
      1. Quote rule: trade at or above ask → buy; at or below bid → sell.
      2. Tick rule (fallback for mid-spread prints): compare to previous trade.
    """
    if trade_price >= ask:
        return +1.0
    if trade_price <= bid:
        return -1.0
    if prev_trade_price is not None:
        if trade_price > prev_trade_price:
            return +1.0
        if trade_price < prev_trade_price:
            return -1.0
    return 0.0


def calc_book_depth_ratio(
    bids: List[Tuple[float, float]],
    asks: List[Tuple[float, float]],
    levels: int = 10,
) -> float:
    """Unweighted total book imbalance across all depth levels.

    Complements ``calc_weighted_obi`` (which decays weight with depth) by
    capturing large orders resting in deep levels — often a sign of
    institutional intent that doesn't yet appear at the touch.

    Returns a value in [-1, +1]:
      +1  → all visible depth is on the bid side (strong buying interest)
      -1  → all visible depth is on the ask side (strong selling interest)
    """
    bid_total = sum(s for _, s in bids[:levels])
    ask_total = sum(s for _, s in asks[:levels])
    total = bid_total + ask_total
    return 0.0 if total <= 0 else (bid_total - ask_total) / total


# ---------------------------------------------------------------------------
# Signal engine
# ---------------------------------------------------------------------------

class KabuSignalStack:
    """Multi-signal microstructure engine for TSE tick-by-tick scalping.

    Signals computed each tick:
      obi             — weighted multi-level order book imbalance
      microprice_tilt — microprice displacement from mid (half-spread normalised)
      lob_ofi         — incremental best-level order flow imbalance
      tape_ofi        — aggressor-classified volume imbalance (rolling window)
      momentum        — microprice relative to rolling mean

    Public interface:
      on_tick(tick)          → Optional[TickSnapshot]   (uses internal adapter)
      on_snapshot(snap)      → TickSnapshot             (for externally built snaps)
      can_open_long(skew)    → bool
      can_open_short(skew)   → bool
      should_exit_long()     → bool
      should_exit_short()    → bool
    """

    def __init__(self, config: SignalConfig, pricetick: float = 1.0) -> None:
        self.cfg = config
        self.pricetick = max(1e-9, float(pricetick or 1.0))
        self._adapter = KabuTickAdapter(config.reverse_bid_ask, config.max_spread_pct)
        self._prev_snap: Optional[TickSnapshot] = None
        self._tick_count: int = 0

        # Signal values
        self.obi: float = 0.0
        self.microprice_tilt: float = 0.0
        self.lob_ofi: float = 0.0
        self.tape_ofi: float = 0.0
        self.momentum: float = 0.0
        self.edge_score: float = 0.0
        self.regime: str = "REVERSION"
        self.adverse_selection: float = 0.0
        self.alpha_long_count: int = 0
        self.alpha_short_count: int = 0
        self.vwap_signal: float = 0.0       # +1 = price well below VWAP (buy), -1 = above VWAP (sell)
        self.book_depth_ratio: float = 0.0  # full-depth unweighted book imbalance
        self.flow_flip: bool = False         # True when momentum exhaustion flip is detected

        # VWAP cumulative accumulators (reset each session day)
        self._sess_vol: float = 0.0
        self._sess_turn: float = 0.0
        self._sess_date: str = ""

        # Signal z-score normalizers (one per signal to account for scale differences)
        _znorm_keys = ("obi", "lob_ofi", "tape_ofi", "momentum",
                       "microprice_tilt", "vwap_signal", "book_depth_ratio")
        self._znorm: Dict[str, ZNormalizer] = {
            k: ZNormalizer(config.znorm_lookback) for k in _znorm_keys
        }

        # Flow flip detector
        self._flow_flip_det = FlowFlipDetector(config.flow_flip_threshold)

        # Gate state
        self.g_spread_ok: bool = False
        self.g_liq_ok: bool = False
        self.g_vol_ok: bool = False
        self.g_time_ok: bool = True
        self.g_ev_ok: bool = True

        # Fill probability / maker-taker mode
        self.fill_prob_long: float = 1.0
        self.fill_prob_short: float = 1.0
        self.fill_mode_long: str = "TAKER"
        self.fill_mode_short: str = "TAKER"
        self.taker_ev_long: float = 0.0
        self.taker_ev_short: float = 0.0
        self.maker_ev_long: float = 0.0
        self.maker_ev_short: float = 0.0

        # Internal buffers
        self._mid_q: Deque[Tuple[datetime, float]] = deque(maxlen=512)
        self._mom_q: Deque[float] = deque(maxlen=max(2, config.momentum_window))
        self._lob_q: Deque[float] = deque(maxlen=max(2, config.lob_ofi_window))
        self._tape_q: Deque[Tuple[datetime, float]] = deque(maxlen=2048)
        self._regime_q: Deque[float] = deque(maxlen=max(5, config.regime_window))
        self._last_total_volume: float = 0.0
        self._cooldown_until: Optional[datetime] = None

        self._parse_time_settings()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def on_tick(self, tick) -> Optional[TickSnapshot]:
        """Process a raw VeighNa TickData. Returns the snapshot or None on invalid data."""
        if self.cfg.auto_pricetick:
            raw_last = float(getattr(tick, "last_price", 0.0) or 0.0)
            if raw_last > 0:
                self.pricetick = get_tse_pricetick(raw_last)

        snap = self._adapter.extract(tick, self.pricetick)
        if snap is None:
            return None
        return self.on_snapshot(snap)

    def on_snapshot(self, snap: TickSnapshot) -> TickSnapshot:
        """Process a pre-normalised TickSnapshot. Always returns the snapshot."""
        self._tick_count += 1
        now = snap.dt

        self._update_gate_spread(snap)
        self._update_gate_liq(snap)
        self._update_gate_vol(snap, now)
        self._update_gate_time(now)

        self._update_obi(snap)
        self._update_microprice(snap)
        self._update_lob_ofi(snap)
        self._update_tape_ofi(snap, now)
        self._update_momentum(snap)
        self._update_regime(snap)
        self._update_vwap_signal(snap)
        self._update_book_depth(snap)
        self._update_fill(snap)
        self._update_gate_ev(snap)
        self._compute_edge()

        self._prev_snap = snap
        return snap

    @property
    def all_gates_ok(self) -> bool:
        return (
            self.g_spread_ok
            and self.g_liq_ok
            and self.g_vol_ok
            and self.g_time_ok
            and self.g_ev_ok
        )

    def can_open_long(self, pos_skew: float = 0.0) -> bool:
        return (
            self.all_gates_ok
            and self.alpha_long_count >= self.cfg.min_alpha_count
            and self.edge_score >= self.cfg.edge_score_long_threshold
            and pos_skew < 1.0
        )

    def can_open_short(self, pos_skew: float = 0.0) -> bool:
        return (
            self.all_gates_ok
            and self.alpha_short_count >= self.cfg.min_alpha_count
            and self.edge_score <= -self.cfg.edge_score_short_threshold
            and pos_skew > -1.0
        )

    def should_exit_long(self) -> bool:
        """Signal-based long exit: edge reversal OR flow flip with selling pressure."""
        return self.edge_score <= -0.5 or (self.flow_flip and self.obi < 0)

    def should_exit_short(self) -> bool:
        """Signal-based short exit: edge reversal OR flow flip with buying pressure."""
        return self.edge_score >= 0.5 or (self.flow_flip and self.obi > 0)

    def update_pricetick(self, pricetick: float) -> None:
        if pricetick and pricetick > 0:
            self.pricetick = float(pricetick)

    def trigger_event_cooldown(self, now_dt: Optional[datetime] = None) -> None:
        now_dt = now_dt or datetime.now()
        self._cooldown_until = datetime.fromtimestamp(
            now_dt.timestamp() + self.cfg.event_cooldown_sec
        )

    def reset(self) -> None:
        self.__init__(self.cfg, self.pricetick)

    def summary(self) -> str:
        return (
            f"gates={int(self.all_gates_ok)} "
            f"edge={self.edge_score:+.2f} "
            f"obi={self.obi:+.2f} "
            f"alpha(L/S)={self.alpha_long_count}/{self.alpha_short_count} "
            f"regime={self.regime}"
        )

    def get_status_dict(self) -> dict:
        return {
            "sig_all_gates":      int(self.all_gates_ok),
            "sig_edge_score":     float(self.edge_score),
            "sig_regime":         self.regime,
            "sig_obi":            float(self.obi),
            "sig_alpha_long":     int(self.alpha_long_count),
            "sig_alpha_short":    int(self.alpha_short_count),
            "sig_adv_sel":        float(self.adverse_selection),
            "sig_fill_prob_long": float(self.fill_prob_long),
            "sig_fill_prob_short":float(self.fill_prob_short),
            "sig_fill_mode_long": self.fill_mode_long,
            "sig_taker_ev_long":  float(self.taker_ev_long),
            "sig_maker_ev_long":  float(self.maker_ev_long),
            "sig_vwap":           float(self.vwap_signal),
            "sig_book_depth":     float(self.book_depth_ratio),
            "sig_flow_flip":      int(self.flow_flip),
        }

    # ------------------------------------------------------------------
    # Gate updates
    # ------------------------------------------------------------------

    def _update_gate_spread(self, snap: TickSnapshot) -> None:
        self.g_spread_ok = snap.spread_ticks <= self.cfg.max_spread_ticks

    def _update_gate_liq(self, snap: TickSnapshot) -> None:
        self.g_liq_ok = (
            snap.bid_vol1 >= self.cfg.min_best_volume
            and snap.ask_vol1 >= self.cfg.min_best_volume
        )

    def _update_gate_vol(self, snap: TickSnapshot, now: datetime) -> None:
        self._mid_q.append((now, snap.mid))
        cutoff = now.timestamp() - max(0.2, self.cfg.vol_window_sec)
        while self._mid_q and self._mid_q[0][0].timestamp() < cutoff:
            self._mid_q.popleft()
        if len(self._mid_q) < 2:
            self.g_vol_ok = True
            return
        t0, m0 = self._mid_q[0]
        t1, m1 = self._mid_q[-1]
        dt_sec = max(1e-6, (t1 - t0).total_seconds())
        self.g_vol_ok = (
            abs(m1 - m0) / self.pricetick / dt_sec <= self.cfg.max_vol_ticks_per_sec
        )

    def _update_gate_time(self, now: datetime) -> None:
        if not self.cfg.enable_time_gate:
            self.g_time_ok = True
            return
        tm = now.time()
        in_morning   = self.trade_start <= tm <= self.morning_end
        in_afternoon = self.afternoon_start <= tm <= self.trade_end
        cooldown_ok  = self._cooldown_until is None or now >= self._cooldown_until
        self.g_time_ok = (in_morning or in_afternoon) and cooldown_ok

    def _update_gate_ev(self, snap: TickSnapshot) -> None:
        if not self.cfg.ev_gate_enabled:
            self.g_ev_ok = True
            return
        if self.cfg.ev_use_separate_mode:
            self.g_ev_ok = max(
                self.taker_ev_long, self.maker_ev_long,
                self.taker_ev_short, self.maker_ev_short,
            ) >= self.cfg.ev_min_ticks
            return
        half_sp = 0.5 * snap.spread_ticks
        denom = max(1e-6, self.cfg.taker_cost_ratio * half_sp)
        self.g_ev_ok = abs(self.microprice_tilt) / denom >= self.cfg.ev_ratio_min

    # ------------------------------------------------------------------
    # Signal updates
    # ------------------------------------------------------------------

    def _update_obi(self, snap: TickSnapshot) -> None:
        self.obi = calc_weighted_obi(snap.bids, snap.asks, self.cfg.obi_levels)

    def _update_microprice(self, snap: TickSnapshot) -> None:
        self.microprice_tilt = snap.microprice_tilt

    def _update_lob_ofi(self, snap: TickSnapshot) -> None:
        if self._prev_snap is None:
            self._lob_q.append(0.0)
            self.lob_ofi = 0.0
            return
        delta = calc_lob_ofi_incremental(self._prev_snap, snap)
        self._lob_q.append(delta)
        scale = max(1.0, sum(abs(x) for x in self._lob_q) / max(1, len(self._lob_q)))
        self.lob_ofi = sum(self._lob_q) / (scale * max(1, len(self._lob_q)))

    def _update_tape_ofi(self, snap: TickSnapshot, now: datetime) -> None:
        vol = max(0.0, snap.volume)
        if self._prev_snap is None:
            self._last_total_volume = vol
            return
        delta_vol = max(0.0, vol - self._last_total_volume)
        self._last_total_volume = vol
        if delta_vol <= 0:
            self._expire_tape(now)
            self._compute_tape()
            return
        prev_trade = self._prev_snap.last_price if self._prev_snap else None
        aggr = calc_tape_aggressor(snap.last_price, snap.bid1, snap.ask1, prev_trade)
        self._tape_q.append((now, aggr * delta_vol))
        self._expire_tape(now)
        self._compute_tape()
        # Update flow flip detector (convert float aggressor to int direction)
        self.flow_flip = self._flow_flip_det.update(int(round(aggr)))

    def _expire_tape(self, now: datetime) -> None:
        cutoff = now.timestamp() - max(0.1, self.cfg.tape_window_sec)
        while self._tape_q and self._tape_q[0][0].timestamp() < cutoff:
            self._tape_q.popleft()

    def _compute_tape(self) -> None:
        if not self._tape_q:
            self.tape_ofi = 0.0
            return
        buys  = sum(v for _, v in self._tape_q if v > 0)
        sells = -sum(v for _, v in self._tape_q if v < 0)
        total = buys + sells
        self.tape_ofi = 0.0 if total <= 0 else (buys - sells) / total

    def _update_momentum(self, snap: TickSnapshot) -> None:
        self._mom_q.append(snap.microprice)
        if len(self._mom_q) < 2:
            self.momentum = 0.0
            return
        mean = sum(self._mom_q) / len(self._mom_q)
        self.momentum = 0.0 if mean == 0 else (self._mom_q[-1] - mean) / mean

    def _update_regime(self, snap: TickSnapshot) -> None:
        """Detect market regime using lag-1 sign autocorrelation of returns.

        ``sign_autocorr > +threshold``  → TREND    (returns tend to continue)
        ``sign_autocorr < -threshold``  → REVERSION (returns tend to reverse)
        otherwise                        → NOISE

        Volatility gate (noise_vol_threshold) is applied first: if the
        price is moving too fast the signal-to-noise ratio is poor.
        """
        if self._tick_count % max(1, self.cfg.regime_sample_interval) != 0:
            return
        self._regime_q.append(snap.mid)
        if len(self._regime_q) < 5:
            self.regime = "REVERSION"
            return
        arr = list(self._regime_q)
        rets = [
            (arr[i] - arr[i - 1]) / arr[i - 1]
            for i in range(1, len(arr))
            if arr[i - 1] > 0
        ]
        if len(rets) < 3:
            self.regime = "REVERSION"
            return

        # Volatility gate — block noisy high-vol periods
        mean_ret = sum(rets) / len(rets)
        var = sum((r - mean_ret) ** 2 for r in rets) / len(rets)
        vol_ticks = math.sqrt(max(var, 0.0)) * snap.mid / max(self.pricetick, 1e-9)
        if vol_ticks >= self.cfg.noise_vol_threshold:
            self.regime = "NOISE"
            return

        # Lag-1 sign autocorrelation
        signs = [1 if r > 0 else (-1 if r < 0 else 0) for r in rets]
        n = len(signs)
        sign_autocorr = (
            sum(signs[i] * signs[i - 1] for i in range(1, n)) / max(1, n - 1)
        )
        thr = self.cfg.sign_autocorr_threshold
        if sign_autocorr >= thr:
            self.regime = "TREND"
        elif sign_autocorr <= -thr:
            self.regime = "REVERSION"
        else:
            self.regime = "NOISE"

    def _update_vwap_signal(self, snap: TickSnapshot) -> None:
        """Compute session VWAP from cumulative turnover/volume increments.

        Price below VWAP → positive signal (mean-reversion buy pressure).
        Price above VWAP → negative signal (sell pressure).
        Normalised to [-1, +1] using a ±3 tick window.
        VWAP accumulator resets each new calendar day.
        """
        date_str = snap.dt.strftime("%Y%m%d")
        if date_str != self._sess_date:
            # New session — reset accumulators
            self._sess_date = date_str
            self._sess_vol = 0.0
            self._sess_turn = 0.0

        if self._prev_snap is not None:
            delta_vol  = max(0.0, snap.volume   - self._prev_snap.volume)
            delta_turn = max(0.0, snap.turnover - self._prev_snap.turnover)
            self._sess_vol  += delta_vol
            self._sess_turn += delta_turn

        if self._sess_vol <= 0:
            self.vwap_signal = 0.0
            return

        vwap = self._sess_turn / self._sess_vol
        # Negative displacement = price below VWAP → positive (buy) signal
        raw_ticks = (vwap - snap.mid) / max(self.pricetick, 1e-9)
        self.vwap_signal = max(-1.0, min(1.0, raw_ticks / 3.0))

    def _update_book_depth(self, snap: TickSnapshot) -> None:
        """Compute full-depth unweighted book imbalance."""
        self.book_depth_ratio = calc_book_depth_ratio(
            snap.bids, snap.asks, self.cfg.book_depth_levels
        )

    def _update_fill(self, snap: TickSnapshot) -> None:
        spread = snap.spread_ticks
        self.adverse_selection = abs(self.microprice_tilt)
        base_prob = 1.0 / (1.0 + max(0.0, spread - 1.0))
        self.fill_prob_long = max(
            0.05, min(1.0, base_prob * (1.0 - 0.5 * self.adverse_selection))
        )
        self.fill_prob_short = self.fill_prob_long
        prefer_maker = (
            spread >= self.cfg.maker_preferred_spread_ticks
            and self.adverse_selection <= self.cfg.maker_adverse_sel_ratio
        )
        self.fill_mode_long = "MAKER" if prefer_maker else "TAKER"
        self.fill_mode_short = self.fill_mode_long
        half_sp = 0.5 * spread
        self.taker_ev_long  =  self.microprice_tilt - half_sp
        self.taker_ev_short = -self.microprice_tilt - half_sp
        maker_gross = half_sp - self.adverse_selection
        self.maker_ev_long  = maker_gross * self.fill_prob_long
        self.maker_ev_short = maker_gross * self.fill_prob_short

    def _compute_edge(self) -> None:
        """Compute weighted composite edge score.

        Each raw signal is first z-score normalised (online, rolling window)
        so that signals with different native scales (e.g. momentum ~0.0001
        vs obi ~1.0) contribute proportionally to their configured weights.
        Alpha-count thresholds are applied to the *raw* (un-normalised) values
        to preserve their intuitive meaning for the gate checks.
        """
        cfg = self.cfg
        zn = self._znorm   # alias

        # Z-normalise each signal
        z_tilt  = zn["microprice_tilt"].normalize(self.microprice_tilt)
        z_lofi  = zn["lob_ofi"].normalize(self.lob_ofi)
        z_tape  = zn["tape_ofi"].normalize(self.tape_ofi)
        z_mom   = zn["momentum"].normalize(self.momentum)
        z_obi   = zn["obi"].normalize(self.obi)
        z_vwap  = zn["vwap_signal"].normalize(self.vwap_signal)
        z_bdr   = zn["book_depth_ratio"].normalize(self.book_depth_ratio)

        self.edge_score = (
            cfg.w_microprice  * z_tilt
            + cfg.w_lob_ofi   * z_lofi
            + cfg.w_tape_ofi  * z_tape
            + cfg.w_momentum  * z_mom
            + cfg.w_obi       * z_obi
            + cfg.w_vwap      * z_vwap
            + cfg.w_book_depth * z_bdr
        )

        # Alpha-count uses raw values (threshold semantics defined in raw units)
        lc = sc = 0
        if self.microprice_tilt >= cfg.microprice_tilt_long:   lc += 1
        if self.microprice_tilt <= -cfg.microprice_tilt_short: sc += 1
        if self.lob_ofi >= cfg.lob_ofi_long:                   lc += 1
        if self.lob_ofi <= -cfg.lob_ofi_short:                 sc += 1
        if self.tape_ofi >= cfg.tape_ofi_long:                 lc += 1
        if self.tape_ofi <= -cfg.tape_ofi_short:               sc += 1
        if self.momentum >= cfg.momentum_long:                  lc += 1
        if self.momentum <= -cfg.momentum_short:                sc += 1
        if self.obi >= cfg.obi_long:                           lc += 1
        if self.obi <= -cfg.obi_short:                         sc += 1
        if self.vwap_signal >= cfg.vwap_long:                  lc += 1
        if self.vwap_signal <= -cfg.vwap_short:                sc += 1
        if self.book_depth_ratio >= cfg.book_depth_long:       lc += 1
        if self.book_depth_ratio <= -cfg.book_depth_short:     sc += 1
        self.alpha_long_count  = lc
        self.alpha_short_count = sc

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_time_settings(self) -> None:
        def _p(s: str, default: time) -> time:
            try:
                hh, mm, ss = [int(x) for x in s.split(":")]
                return time(hh, mm, ss)
            except Exception:
                return default

        self.trade_start     = _p(self.cfg.trade_start_time,     time(9,  0, 10))
        self.morning_end     = _p(self.cfg.morning_end_time,     time(11, 30,  0))
        self.afternoon_start = _p(self.cfg.afternoon_start_time, time(12, 30,  0))
        self.trade_end       = _p(self.cfg.trade_end_time,       time(15, 25,  0))


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def make_conservative_stack(pricetick: float = 1.0) -> KabuSignalStack:
    """Higher thresholds, more signal confirmation required, fewer trades."""
    return KabuSignalStack(
        SignalConfig(
            max_spread_ticks=1.5,
            min_best_volume=80,
            edge_score_long_threshold=3.0,
            edge_score_short_threshold=3.0,
            min_alpha_count=3,
        ),
        pricetick,
    )


def make_aggressive_stack(pricetick: float = 1.0) -> KabuSignalStack:
    """Lower thresholds, faster entry, more frequent signals."""
    return KabuSignalStack(
        SignalConfig(
            max_spread_ticks=2.0,
            min_best_volume=30,
            edge_score_long_threshold=1.5,
            edge_score_short_threshold=1.5,
            min_alpha_count=2,
        ),
        pricetick,
    )


# ---------------------------------------------------------------------------
# VeighNa strategy wrapper
# ---------------------------------------------------------------------------

if _VNPY_AVAILABLE:
    class KabuSignalStrategy(CtaTemplate):
        """VeighNa CTA strategy wrapping KabuSignalStack.

        Order lifecycle (state machine):
          IDLE → PENDING_OPEN → OPEN → PENDING_CLOSE → IDLE

        The state machine is the primary guard against duplicate orders:
        no new entry is allowed unless the state is IDLE.

        Exit triggers (evaluated each tick in OPEN state):
          1. Signal reversal  (edge_score crosses threshold)
          2. Take-profit      (pnl_ticks >= profit_ticks)
          3. Stop-loss        (pnl_ticks <= -loss_ticks)
          4. Timeout          (elapsed >= max_hold_seconds, unconditional)

        PnL uses actual fill prices from on_trade callbacks, not pre-order estimates.
        Commission uses kabu's ad-valorem model: max(commission_min, rate × value).
        """

        author = "KabuSignalStack v3"

        # --- Tunable parameters ---
        trade_volume: int = 100
        max_position: int = 300
        enable_long: bool = True
        enable_short: bool = False

        # Gate parameters (passed to SignalConfig on start)
        max_spread_ticks: float = 2.0
        min_best_volume: int = 50
        obi_levels: int = 3
        ev_gate_enabled: bool = False

        # Adapter
        reverse_bid_ask: bool = False   # Set True if kabu gateway passes raw field names
        auto_pricetick: bool = True

        # Commission — kabu ad-valorem model (都度手数料)
        # IMPORTANT: confirm against your actual account plan before live trading.
        commission_rate: float = 0.00385   # 片道 0.385% 税込
        commission_min: float = 55.0       # 最低手数料 ¥55 税込

        # Exit parameters
        profit_ticks: float = 3.0
        loss_ticks: float = 2.0
        max_hold_seconds: float = 30.0
        open_order_timeout_sec: float = 3.0
        close_order_timeout_sec: float = 2.0
        trailing_activate_ticks: float = 2.0
        trailing_drawdown_ticks: float = 1.2

        # Timing and confirmation
        strong_signal_threshold: float = 4.0
        weak_signal_confirm_ticks: int = 2
        entry_cooldown_sec: float = 0.4
        no_new_entry_after: str = "15:24:00"
        noise_regime_block: bool = True
        spread_edge_penalty: float = 0.4

        # Dynamic sizing
        lot_size: int = 100
        min_trade_volume: int = 100
        max_trade_volume: int = 300
        risk_scale_min: float = 0.4

        # Daily risk limits
        max_daily_loss: float = -50_000.0
        max_daily_drawdown: float = -30_000.0
        max_daily_trades: int = 80
        max_consecutive_losses: int = 4
        loss_cooldown_sec: float = 180.0
        max_order_req_per_sec: int = 5

        # Logging throttle interval
        log_interval_seconds: float = 10.0

        # --- v3 parameters ---
        # Execution quality
        maker_escape_timeout_sec: float = 1.5    # cancel stale MAKER order and re-enter as TAKER
        max_impact_pct: float = 0.5              # reject entry if vol > this × best-level size

        # Risk completeness
        fast_loss_ticks: float = 1.5             # fast-loss circuit breaker: ticks lost
        fast_loss_sec: float = 3.0               # fast-loss circuit breaker: time window (seconds)
        hot_open_guard: bool = True              # block entries during 09:00-09:02 and 12:30-12:32

        # --- v4 parameters ---
        # Signal normalisation & flow flip
        znorm_lookback: int = 100                # z-score normalization rolling window
        flow_flip_threshold: int = 3             # consecutive direction reversals to trigger flip

        # Separate open/close rate limits (close-first principle)
        open_min_interval_ms:  float = 100.0     # minimum ms between two consecutive open orders
        close_min_interval_ms: float = 50.0      # minimum ms between two consecutive close orders

        parameters = [
            "trade_volume", "max_position",
            "enable_long", "enable_short",
            "max_spread_ticks", "min_best_volume",
            "obi_levels", "ev_gate_enabled",
            "reverse_bid_ask", "auto_pricetick",
            "commission_rate", "commission_min",
            "profit_ticks", "loss_ticks", "max_hold_seconds",
            "open_order_timeout_sec", "close_order_timeout_sec",
            "trailing_activate_ticks", "trailing_drawdown_ticks",
            "strong_signal_threshold", "weak_signal_confirm_ticks",
            "entry_cooldown_sec", "no_new_entry_after",
            "noise_regime_block", "spread_edge_penalty",
            "lot_size", "min_trade_volume", "max_trade_volume", "risk_scale_min",
            "max_daily_loss", "max_daily_drawdown", "max_daily_trades",
            "max_consecutive_losses", "loss_cooldown_sec", "max_order_req_per_sec",
            "log_interval_seconds",
            # v3
            "maker_escape_timeout_sec", "max_impact_pct",
            "fast_loss_ticks", "fast_loss_sec", "hot_open_guard",
            # v4
            "znorm_lookback", "flow_flip_threshold",
            "open_min_interval_ms", "close_min_interval_ms",
        ]

        # --- UI-visible variables ---
        price_tick: float = 1.0
        order_state: str = "idle"
        sig_all_gates: int = 0
        sig_edge_score: float = 0.0
        sig_regime: str = "REVERSION"
        sig_obi: float = 0.0
        sig_alpha_long: int = 0
        sig_alpha_short: int = 0
        sig_adv_sel: float = 0.0
        sig_fill_mode_long: str = "TAKER"
        sig_taker_ev_long: float = 0.0
        sig_maker_ev_long: float = 0.0
        last_entry_price: float = 0.0   # Actual fill price from on_trade
        last_exit_price: float = 0.0    # Actual exit fill price from on_trade
        daily_pnl: float = 0.0
        daily_trades: int = 0
        consecutive_losses: int = 0
        daily_drawdown: float = 0.0
        risk_halt_reason: str = ""
        order_req_1s: int = 0
        total_trades: int = 0
        last_signal: str = ""
        sig_vwap: float = 0.0
        sig_book_depth: float = 0.0
        sig_flow_flip: int = 0
        unrealized_pnl: float = 0.0
        stat_win_rate: float = 0.0
        stat_pf: float = 0.0
        stat_avg_hold: float = 0.0

        variables = [
            "price_tick", "order_state",
            "sig_all_gates", "sig_edge_score", "sig_regime",
            "sig_obi", "sig_alpha_long", "sig_alpha_short",
            "sig_adv_sel", "sig_fill_mode_long",
            "sig_taker_ev_long", "sig_maker_ev_long",
            "last_entry_price", "last_exit_price",
            "daily_pnl", "daily_trades", "consecutive_losses", "daily_drawdown",
            "risk_halt_reason", "order_req_1s", "total_trades", "last_signal",
            "sig_vwap", "sig_book_depth", "sig_flow_flip", "unrealized_pnl",
            "stat_win_rate", "stat_pf", "stat_avg_hold",
        ]

        def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
            super().__init__(cta_engine, strategy_name, vt_symbol, setting)

            # Order state machine
            self._order_state: OrderState = OrderState.IDLE
            self._active_orderids: List[str] = []

            # Entry tracking — populated from actual fill in on_trade
            self._entry_fill_price: float = 0.0
            self._entry_fill_volume: float = 0.0
            self._entry_time: Optional[datetime] = None
            self._entry_direction: str = ""   # "LONG" | "SHORT"
            self._entry_mode: str = ""        # "MAKER" | "TAKER" (for MAKER escape logic)
            self._entry_volume: int = 0       # intended order volume (for MAKER escape re-entry)
            self._opened_total_volume: float = 0.0
            self._open_trade_gross_pnl: float = 0.0
            self._close_turnover: float = 0.0
            self._close_volume_accum: float = 0.0
            self._max_favorable_ticks: float = 0.0
            self._pending_exit_reason: str = ""   # set in _manage_exit, consumed in on_trade

            # v4: separate open/close rate limit timestamps
            self._last_open_order_ts:  Optional[datetime] = None
            self._last_close_order_ts: Optional[datetime] = None

            # v4: PnL tracker
            self._pnl_tracker = PnLTracker()

            # Risk
            self._is_trading_allowed: bool = True
            self._current_date: str = ""
            self._peak_daily_pnl: float = 0.0
            self._risk_cooldown_until: Optional[datetime] = None
            self._last_entry_attempt_dt: Optional[datetime] = None
            self._signal_pending_dir: int = 0
            self._signal_pending_count: int = 0
            self._state_since: Optional[datetime] = None
            self._order_req_ts: Deque[datetime] = deque()
            self._last_rate_limit_log_dt: Optional[datetime] = None

            # Throttle timers
            self._last_log_dt: Optional[datetime] = None
            self._last_put_dt: Optional[datetime] = None
            self._last_bad_tick_dt: Optional[datetime] = None

            # Signal engine (rebuilt with live parameters in on_start)
            self.sig = KabuSignalStack(SignalConfig(), 1.0)

        # ------------------------------------------------------------------
        # Lifecycle
        # ------------------------------------------------------------------

        def on_init(self) -> None:
            self.write_log(f"on_init {self.vt_symbol} vol={self.trade_volume}")
            self.load_tick(1)
            self.put_event()

        def on_start(self) -> None:
            # ----------------------------------------------------------------
            # Step 1: 强制订阅行情（解决合约未注册 / CTA engine 鸡生蛋问题）
            #
            # VeighNa CTA engine 的 init_strategy() 只有在合约已预注册时才会
            # 调用 gateway.subscribe()。若合约不在 KABU_STOCK_CONTRACTS，
            # 订阅静默失败，导致"委托失败，找不到合约"。
            #
            # 这里直接调用 main_engine.subscribe()，让 gateway 动态注册合约
            # 并建立 WebSocket 推送连接，无需 UI 手动输入股票代码。
            # ----------------------------------------------------------------
            self._force_subscribe()

            # Step 2: 读取 pricetick（合约已注册后才能取到正确值）
            pt = self.get_pricetick()
            if pt and pt > 0:
                self.price_tick = float(pt)

            # Step 3: 重建信号引擎
            cfg = SignalConfig(
                max_spread_ticks=self.max_spread_ticks,
                min_best_volume=self.min_best_volume,
                obi_levels=self.obi_levels,
                ev_gate_enabled=self.ev_gate_enabled,
                reverse_bid_ask=self.reverse_bid_ask,
                auto_pricetick=self.auto_pricetick,
                znorm_lookback=self.znorm_lookback,
                flow_flip_threshold=self.flow_flip_threshold,
            )
            self.sig = KabuSignalStack(cfg, self.price_tick)
            self._reset_position_state()
            self._parse_strategy_time_settings()
            self.write_log(
                f"on_start pricetick={self.price_tick} "
                f"reverse_bid_ask={self.reverse_bid_ask}"
            )
            self.put_event()

        def _force_subscribe(self) -> None:
            """强制向 gateway 发送订阅请求，确保合约注册和 WS 推送建立。

            解决路径：
              on_start → _force_subscribe
                → main_engine.subscribe(req, gateway_name)
                → gateway.subscribe(req)          ← 若合约未注册则动态创建
                → gateway.ws_api.subscribe(req)   ← 建立 WebSocket 推送

            优先使用已有合约的 gateway_name；若合约尚不存在则遍历所有已连接
            gateway 尝试订阅（适用于首次启动、新品种接入等场景）。
            """
            try:
                parts = self.vt_symbol.split(".", 1)
                if len(parts) != 2:
                    return
                symbol_str, exchange_str = parts
                # 使用已导入的 Direction/Status 所在 vnpy.trader.constant 模块
                from vnpy.trader.constant import Exchange as _Ex
                from vnpy.trader.object import SubscribeRequest as _SR
                ex = _Ex(exchange_str)
                req = _SR(symbol=symbol_str, exchange=ex)

                main_engine = self.cta_engine.main_engine

                # 优先从已有合约取 gateway_name（重启续用已知合约）
                contract = main_engine.get_contract(self.vt_symbol)
                if contract and getattr(contract, "gateway_name", None):
                    main_engine.subscribe(req, contract.gateway_name)
                    self.write_log(f"行情订阅: {self.vt_symbol} → {contract.gateway_name}")
                    return

                # 合约未注册时：逐一尝试已连接的 gateway
                gateways = getattr(main_engine, "gateways", {})
                for gw_name in gateways:
                    main_engine.subscribe(req, gw_name)
                    self.write_log(
                        f"行情订阅(动态注册): {self.vt_symbol} → {gw_name}"
                    )
                    return

                self.write_log(f"[WARN] 无可用 gateway，行情订阅失败: {self.vt_symbol}")

            except Exception as e:
                self.write_log(f"[WARN] _force_subscribe 异常: {e}")

        def on_stop(self) -> None:
            self._rl_cancel_all(datetime.now())
            self.write_log(self.sig.summary())
            self.write_log(
                f"on_stop daily_pnl={self.daily_pnl:.0f}JPY "
                f"total_trades={self.total_trades}"
            )
            self.put_event()

        # ------------------------------------------------------------------
        # Main tick handler
        # ------------------------------------------------------------------

        def on_tick(self, tick) -> None:
            # 1. Signal engine (also validates tick via adapter)
            snap = self.sig.on_tick(tick)
            if snap is None:
                self._maybe_log_bad_tick(tick)
                return

            tick_dt = snap.dt
            self._sync_sig_vars(tick_dt)
            self._update_unrealized_pnl(snap)
            self._check_date_reset(tick_dt)
            self._maybe_log(tick_dt)
            self._check_pending_timeouts(tick_dt, snap)

            # 2. State machine guard — if not IDLE, only manage open position
            if self._order_state != OrderState.IDLE:
                if self._order_state == OrderState.OPEN:
                    if not self._is_trading_allowed:
                        self._force_flatten(snap, tick_dt, reason="risk_halt")
                    else:
                        self._manage_exit(snap, tick_dt)
                self._throttled_put_event(tick_dt)
                return

            # 3. Risk gate
            if self._risk_cooldown_until and tick_dt < self._risk_cooldown_until:
                self._throttled_put_event(tick_dt)
                return

            if not self._is_trading_allowed or not self.sig.all_gates_ok:
                self._throttled_put_event(tick_dt)
                return

            if not self._entry_time_allowed(tick_dt):
                self._throttled_put_event(tick_dt)
                return

            if self._last_entry_attempt_dt:
                if (tick_dt - self._last_entry_attempt_dt).total_seconds() < self.entry_cooldown_sec:
                    self._throttled_put_event(tick_dt)
                    return

            # 4. Entry signals
            pos_skew = self.pos / max(self.max_position, 1)
            if self.enable_long and self.sig.can_open_long(pos_skew=pos_skew):
                vol = self._compute_order_volume(+1)
                if vol > 0 and self._entry_quality_ok(snap, +1, vol) and self._confirm_entry_signal(+1):
                    self._enter_long(snap, tick_dt, vol)
            elif self.enable_short and self.sig.can_open_short(pos_skew=pos_skew):
                vol = self._compute_order_volume(-1)
                if vol > 0 and self._entry_quality_ok(snap, -1, vol) and self._confirm_entry_signal(-1):
                    self._enter_short(snap, tick_dt, vol)
            else:
                self._signal_pending_dir = 0
                self._signal_pending_count = 0

            self._throttled_put_event(tick_dt)

        # ------------------------------------------------------------------
        # Order / trade callbacks
        # ------------------------------------------------------------------

        def on_trade(self, trade) -> None:
            trade_dt = getattr(trade, "datetime", None)
            if not isinstance(trade_dt, datetime):
                trade_dt = datetime.now()

            trade_price = float(trade.price)
            trade_volume = float(trade.volume)
            is_open_fill = (
                (self._entry_direction == "LONG" and trade.direction == Direction.LONG)
                or (self._entry_direction == "SHORT" and trade.direction == Direction.SHORT)
            )
            is_close_fill = (
                (self._entry_direction == "LONG" and trade.direction == Direction.SHORT)
                or (self._entry_direction == "SHORT" and trade.direction == Direction.LONG)
            )

            if self._order_state in (OrderState.PENDING_OPEN, OrderState.OPEN) and is_open_fill:
                old_vol = self._entry_fill_volume
                new_vol = old_vol + trade_volume
                if new_vol > 0:
                    self._entry_fill_price = (
                        self._entry_fill_price * old_vol + trade_price * trade_volume
                    ) / new_vol if old_vol > 0 else trade_price
                self._entry_fill_volume = new_vol
                self._opened_total_volume = new_vol
                self._entry_time = trade_dt
                self._max_favorable_ticks = 0.0
                self._order_state = OrderState.OPEN
                self._state_since = trade_dt
                self.last_entry_price = self._entry_fill_price
                self.order_state = OrderState.OPEN.value
                self.write_log(
                    f"FILL OPEN {self._entry_direction} "
                    f"avg={self._entry_fill_price:.1f} vol={self._entry_fill_volume:.0f}"
                )

            elif self._order_state in (OrderState.PENDING_CLOSE, OrderState.OPEN) and is_close_fill:
                realized_vol = min(trade_volume, max(self._entry_fill_volume, 0.0))
                if realized_vol <= 0:
                    realized_vol = trade_volume
                direction_sign = +1 if self._entry_direction == "LONG" else -1
                gross_pnl_part = direction_sign * (trade_price - self._entry_fill_price) * realized_vol
                self._open_trade_gross_pnl += gross_pnl_part
                self._close_turnover += trade_price * realized_vol
                self._close_volume_accum += realized_vol
                self._entry_fill_volume = max(0.0, self._entry_fill_volume - realized_vol)
                self.last_exit_price = trade_price

                if abs(self.pos) == 0 or self._entry_fill_volume <= 1e-9:
                    closed_volume = max(self._close_volume_accum, self._opened_total_volume, 0.0)
                    if closed_volume > 0:
                        close_avg = self._close_turnover / max(1e-9, self._close_volume_accum)
                    else:
                        close_avg = trade_price
                    commission_open = self._calc_commission(self._entry_fill_price, self._opened_total_volume)
                    commission_close = self._calc_commission(close_avg, closed_volume)
                    net_pnl = self._open_trade_gross_pnl - commission_open - commission_close
                    self.daily_pnl += net_pnl
                    self.daily_trades += 1
                    self.total_trades += 1
                    self._peak_daily_pnl = max(self._peak_daily_pnl, self.daily_pnl)
                    self.daily_drawdown = self.daily_pnl - self._peak_daily_pnl
                    if net_pnl < 0:
                        self.consecutive_losses += 1
                    else:
                        self.consecutive_losses = 0
                    self.write_log(
                        f"FILL CLOSE {self._entry_direction} "
                        f"exit={trade_price:.1f} pnl={net_pnl:+.0f}JPY "
                        f"daily={self.daily_pnl:.0f}JPY"
                    )
                    # v4: record trade for PnL statistics
                    entry_t = self._entry_time if self._entry_time else (trade_dt - timedelta(seconds=1))
                    rec = TradeRecord(
                        entry_price=self._entry_fill_price,
                        exit_price=close_avg,
                        volume=closed_volume,
                        direction=direction_sign,
                        commission=commission_open + commission_close,
                        entry_time=entry_t,
                        exit_time=trade_dt,
                        exit_reason=self._pending_exit_reason or "UNKNOWN",
                    )
                    self._pnl_tracker.record(rec)
                    self._check_daily_risk_halt(trade_dt)
                    self._reset_position_state()
                else:
                    # Partial close: keep monitoring remaining position.
                    self._order_state = OrderState.OPEN
                    self._state_since = trade_dt
                    self.order_state = OrderState.OPEN.value

            self.put_event()

        def on_order(self, order) -> None:
            # Detect cancellation or rejection and update state accordingly
            try:
                cancelled = order.status in (Status.CANCELLED, Status.REJECTED)
            except Exception:
                s = str(getattr(order, "status", "")).lower()
                cancelled = "cancel" in s or "reject" in s

            if cancelled and order.vt_orderid in self._active_orderids:
                self._active_orderids.remove(order.vt_orderid)
                if not self._active_orderids:
                    if self._order_state == OrderState.PENDING_OPEN:
                        # Entry order cancelled before fill → back to IDLE
                        self.write_log(f"Open order cancelled: {order.vt_orderid}")
                        self._reset_position_state()
                    elif self._order_state == OrderState.PENDING_CLOSE:
                        # Exit order cancelled → retry next tick
                        self.write_log(
                            f"Close order cancelled, will retry: {order.vt_orderid}"
                        )
                        self._order_state = OrderState.OPEN
                        self._state_since = datetime.now()
                        self.order_state = OrderState.OPEN.value

            self.put_event()

        def on_bar(self, bar) -> None:
            pass

        def on_stop_order(self, stop_order) -> None:
            self.put_event()

        # ------------------------------------------------------------------
        # Entry helpers
        # ------------------------------------------------------------------

        def _enter_long(self, snap: TickSnapshot, tick_dt: datetime, volume: int) -> None:
            # MAKER: post limit at bid (may not fill); TAKER: lift the ask immediately
            mode = self.sig.fill_mode_long
            order_price = snap.bid1 if mode == "MAKER" else snap.ask1
            ids = self._rl_buy(order_price, volume, tick_dt)
            if ids:
                self._active_orderids = list(ids)
                self._order_state = OrderState.PENDING_OPEN
                self._entry_direction = "LONG"
                self._entry_mode = mode
                self._entry_volume = volume
                self._entry_time = tick_dt
                self._state_since = tick_dt
                self._last_entry_attempt_dt = tick_dt
                self.order_state = OrderState.PENDING_OPEN.value
                self.last_signal = (
                    f"LONG/{mode} "
                    f"edge={self.sig.edge_score:+.2f}"
                )
                self.write_log(
                    f"OPEN LONG @ {order_price:.1f} x{volume} | "
                    f"{self.last_signal}"
                )

        def _enter_short(self, snap: TickSnapshot, tick_dt: datetime, volume: int) -> None:
            # MAKER: post limit at ask; TAKER: hit the bid immediately
            mode = self.sig.fill_mode_short
            order_price = snap.ask1 if mode == "MAKER" else snap.bid1
            ids = self._rl_short(order_price, volume, tick_dt)
            if ids:
                self._active_orderids = list(ids)
                self._order_state = OrderState.PENDING_OPEN
                self._entry_direction = "SHORT"
                self._entry_mode = mode
                self._entry_volume = volume
                self._entry_time = tick_dt
                self._state_since = tick_dt
                self._last_entry_attempt_dt = tick_dt
                self.order_state = OrderState.PENDING_OPEN.value
                self.last_signal = (
                    f"SHORT/{mode} "
                    f"edge={self.sig.edge_score:+.2f}"
                )
                self.write_log(
                    f"OPEN SHORT @ {order_price:.1f} x{volume} | "
                    f"{self.last_signal}"
                )

        # ------------------------------------------------------------------
        # Exit management
        # ------------------------------------------------------------------

        def _manage_exit(self, snap: TickSnapshot, tick_dt: datetime) -> None:
            if self._order_state != OrderState.OPEN or self._entry_time is None:
                return
            if self._active_orderids:
                # Wait for outstanding cancel/ack before sending another close order.
                return

            pt = max(self.price_tick, 1e-9)
            elapsed = (tick_dt - self._entry_time).total_seconds()

            # ----------------------------------------------------------------
            # C1: Fast loss circuit breaker — aggressive immediate exit if
            # the position loses fast_loss_ticks within fast_loss_sec of entry.
            # Indicates we entered into a strongly directional move against us.
            # ----------------------------------------------------------------
            if elapsed <= self.fast_loss_sec:
                if self.pos > 0:
                    fast_pnl = (snap.bid1 - self._entry_fill_price) / pt
                elif self.pos < 0:
                    fast_pnl = (self._entry_fill_price - snap.ask1) / pt
                else:
                    fast_pnl = 0.0
                if fast_pnl <= -self.fast_loss_ticks:
                    # Aggressive market-clearing exit (no offset)
                    self._pending_exit_reason = "FAST_LOSS"
                    if self.pos > 0:
                        ids = self._rl_sell(snap.bid1, abs(self.pos), tick_dt)
                    else:
                        ids = self._rl_cover(snap.ask1, abs(self.pos), tick_dt)
                    if ids:
                        self._active_orderids = list(ids)
                        self._order_state = OrderState.PENDING_CLOSE
                        self._state_since = tick_dt
                        self.order_state = OrderState.PENDING_CLOSE.value
                        self.write_log(
                            f"[FAST LOSS] {fast_pnl:.1f}ticks in {elapsed:.1f}s → aggressive exit"
                        )
                    return

            # ----------------------------------------------------------------
            # Normal exit logic with two-tier pricing (B2):
            #   • Urgent exits (stop-loss / timeout) → aggressive price (bid1 / ask1)
            #   • Take-profit / signal exits         → gentle limit (bid1-pt / ask1+pt)
            # ----------------------------------------------------------------
            if self.pos > 0:
                pnl_ticks = (snap.bid1 - self._entry_fill_price) / pt
                self._max_favorable_ticks = max(self._max_favorable_ticks, pnl_ticks)
                trailing_hit = (
                    self._max_favorable_ticks >= self.trailing_activate_ticks
                    and (self._max_favorable_ticks - pnl_ticks) >= self.trailing_drawdown_ticks
                )
                is_urgent = pnl_ticks <= -self.loss_ticks or elapsed >= self.max_hold_seconds
                should_exit = (
                    self.sig.should_exit_long()
                    or pnl_ticks >= self.profit_ticks
                    or is_urgent
                    or trailing_hit
                )
                if should_exit:
                    if pnl_ticks <= -self.loss_ticks:
                        self._pending_exit_reason = "SL"
                    elif elapsed >= self.max_hold_seconds:
                        self._pending_exit_reason = "TIMEOUT"
                    elif pnl_ticks >= self.profit_ticks:
                        self._pending_exit_reason = "TP"
                    elif trailing_hit:
                        self._pending_exit_reason = "TRAILING"
                    else:
                        self._pending_exit_reason = "SIGNAL_FLIP"
                    exit_price = snap.bid1 if is_urgent else snap.bid1 - pt
                    ids = self._rl_sell(exit_price, abs(self.pos), tick_dt)
                    if ids:
                        self._active_orderids = list(ids)
                        self._order_state = OrderState.PENDING_CLOSE
                        self._state_since = tick_dt
                        self.order_state = OrderState.PENDING_CLOSE.value

            elif self.pos < 0:
                pnl_ticks = (self._entry_fill_price - snap.ask1) / pt
                self._max_favorable_ticks = max(self._max_favorable_ticks, pnl_ticks)
                trailing_hit = (
                    self._max_favorable_ticks >= self.trailing_activate_ticks
                    and (self._max_favorable_ticks - pnl_ticks) >= self.trailing_drawdown_ticks
                )
                is_urgent = pnl_ticks <= -self.loss_ticks or elapsed >= self.max_hold_seconds
                should_exit = (
                    self.sig.should_exit_short()
                    or pnl_ticks >= self.profit_ticks
                    or is_urgent
                    or trailing_hit
                )
                if should_exit:
                    if pnl_ticks <= -self.loss_ticks:
                        self._pending_exit_reason = "SL"
                    elif elapsed >= self.max_hold_seconds:
                        self._pending_exit_reason = "TIMEOUT"
                    elif pnl_ticks >= self.profit_ticks:
                        self._pending_exit_reason = "TP"
                    elif trailing_hit:
                        self._pending_exit_reason = "TRAILING"
                    else:
                        self._pending_exit_reason = "SIGNAL_FLIP"
                    exit_price = snap.ask1 if is_urgent else snap.ask1 + pt
                    ids = self._rl_cover(exit_price, abs(self.pos), tick_dt)
                    if ids:
                        self._active_orderids = list(ids)
                        self._order_state = OrderState.PENDING_CLOSE
                        self._state_since = tick_dt
                        self.order_state = OrderState.PENDING_CLOSE.value

        def _force_flatten(self, snap: TickSnapshot, tick_dt: datetime, reason: str) -> None:
            """Immediate close path used by hard risk halt."""
            if self._active_orderids:
                return
            if self.pos > 0:
                ids = self._rl_sell(max(snap.bid1 - self.price_tick, self.price_tick), abs(self.pos), tick_dt)
            elif self.pos < 0:
                ids = self._rl_cover(snap.ask1 + self.price_tick, abs(self.pos), tick_dt)
            else:
                self._reset_position_state()
                return

            if ids:
                self._active_orderids = list(ids)
                self._order_state = OrderState.PENDING_CLOSE
                self._state_since = tick_dt
                self.order_state = OrderState.PENDING_CLOSE.value
                self.write_log(f"[FORCE FLAT] reason={reason}")

        # ------------------------------------------------------------------
        # Risk / PnL helpers
        # ------------------------------------------------------------------

        def _calc_commission(self, price: float, volume: float) -> float:
            """One-way commission: max(min_fee, rate × trade_value).

            IMPORTANT: verify commission_rate and commission_min against your
            actual kabu account plan (定額 vs 都度) before live trading.
            """
            trade_value = price * volume
            return max(self.commission_min, trade_value * self.commission_rate)

        def _check_daily_risk_halt(self, trade_dt=None) -> None:
            now_dt = trade_dt if isinstance(trade_dt, datetime) else datetime.now()

            if self.consecutive_losses >= self.max_consecutive_losses:
                self._risk_cooldown_until = now_dt + timedelta(seconds=self.loss_cooldown_sec)
                self.risk_halt_reason = (
                    f"consecutive_losses={self.consecutive_losses}, "
                    f"cooldown={self.loss_cooldown_sec:.0f}s"
                )
                self.write_log(f"[RISK COOL] {self.risk_halt_reason}")

            hard_halt_reason = ""
            if self.daily_pnl < self.max_daily_loss:
                hard_halt_reason = (
                    f"daily_pnl={self.daily_pnl:.0f}JPY < limit={self.max_daily_loss:.0f}JPY"
                )
            elif self.daily_drawdown < self.max_daily_drawdown:
                hard_halt_reason = (
                    f"daily_drawdown={self.daily_drawdown:.0f}JPY < limit={self.max_daily_drawdown:.0f}JPY"
                )
            elif self.daily_trades >= self.max_daily_trades:
                hard_halt_reason = (
                    f"daily_trades={self.daily_trades} >= limit={self.max_daily_trades}"
                )

            if hard_halt_reason:
                self._is_trading_allowed = False
                self.risk_halt_reason = hard_halt_reason
                self.sig.trigger_event_cooldown(now_dt)
                self.write_log(f"[RISK HALT] {hard_halt_reason}")

        def _check_date_reset(self, dt: datetime) -> None:
            date_str = dt.strftime("%Y%m%d")
            if date_str != self._current_date:
                self._current_date = date_str
                self.daily_pnl = 0.0
                self.daily_trades = 0
                self.consecutive_losses = 0
                self.daily_drawdown = 0.0
                self._peak_daily_pnl = 0.0
                self._risk_cooldown_until = None
                self.risk_halt_reason = ""
                self._order_req_ts.clear()
                self.order_req_1s = 0
                self._is_trading_allowed = True
                self.write_log(f"New trading day: {date_str}")

        def _check_pending_timeouts(self, now: datetime, snap: TickSnapshot) -> None:
            """Cancel stale pending orders and recover state safely.

            MAKER escape (B1): if a MAKER open order has been pending for
            ``maker_escape_timeout_sec`` without filling, cancel it and
            re-enter as a TAKER order (provided the signal is still valid).
            This avoids sitting in an unfilled limit order while the market
            moves against us (adverse selection).
            """
            if self._state_since is None:
                return

            elapsed = (now - self._state_since).total_seconds()

            # --- MAKER escape: runs before the full open timeout ---
            if (
                self._order_state == OrderState.PENDING_OPEN
                and self._entry_mode == "MAKER"
                and elapsed >= self.maker_escape_timeout_sec
                and elapsed < self.open_order_timeout_sec
            ):
                self.write_log(
                    f"MAKER escape {elapsed:.2f}s -> cancel and re-enter TAKER"
                )
                self._rl_cancel_all(now)
                self._active_orderids = []
                # Re-enter as TAKER only if signal is still valid
                ids = []
                vol = max(self.min_trade_volume, self._entry_volume or self.min_trade_volume)
                lot = max(1, self.lot_size)
                vol = (vol // lot) * lot
                if vol <= 0:
                    vol = lot
                if self._entry_direction == "LONG" and self.sig.can_open_long():
                    ids = self._rl_buy(snap.ask1, vol, now)
                elif self._entry_direction == "SHORT" and self.sig.can_open_short():
                    ids = self._rl_short(snap.bid1, vol, now)
                if ids:
                    self._active_orderids = list(ids)
                    self._entry_mode = "TAKER"
                    self._state_since = now
                else:
                    # Signal gone — abort entry
                    self._reset_position_state()
                return

            # --- Full open timeout ---
            if self._order_state == OrderState.PENDING_OPEN and elapsed >= self.open_order_timeout_sec:
                self.write_log(
                    f"OPEN timeout {elapsed:.2f}s -> send cancel and wait ack"
                )
                sent = self._rl_cancel_all(now)
                if sent:
                    self._state_since = now
                return

            # --- Close timeout: aggressive TAKER retry (bid1 / ask1, no offset) ---
            if self._order_state == OrderState.PENDING_CLOSE and elapsed >= self.close_order_timeout_sec:
                self.write_log(
                    f"CLOSE timeout {elapsed:.2f}s -> send cancel and retry after ack"
                )
                sent = self._rl_cancel_all(now)
                if sent:
                    # Move back to OPEN and wait for cancel update before new close.
                    self._order_state = OrderState.OPEN
                    self.order_state = OrderState.OPEN.value
                    self._state_since = now
                return

        def _parse_strategy_time_settings(self) -> None:
            def _p(s: str, default: time) -> time:
                try:
                    hh, mm, ss = [int(x) for x in s.split(":")]
                    return time(hh, mm, ss)
                except Exception:
                    return default

            self._no_new_entry_after_time = _p(self.no_new_entry_after, time(15, 24, 0))

        def _entry_time_allowed(self, dt: datetime) -> bool:
            """Return True if new entries are allowed at the given datetime.

            Guards applied:
            1. no_new_entry_after: no entries after configured cutoff (default 15:24).
            2. hot_open_guard: block entries during the first 2 minutes after
               market open/reopen (09:00-09:02 and 12:30-12:32) when the order
               book is thin, auction prints dominate, and signal quality is lowest.
            """
            tm = dt.time()
            if tm >= self._no_new_entry_after_time:
                return False
            if self.hot_open_guard:
                # Morning open guard
                if time(9, 0, 0) <= tm < time(9, 2, 0):
                    return False
                # Afternoon re-open guard
                if time(12, 30, 0) <= tm < time(12, 32, 0):
                    return False
            return True

        def _confirm_entry_signal(self, direction: int) -> bool:
            edge_abs = abs(self.sig.edge_score)
            if edge_abs >= self.strong_signal_threshold:
                self._signal_pending_dir = 0
                self._signal_pending_count = 0
                return True

            if direction != self._signal_pending_dir:
                self._signal_pending_dir = direction
                self._signal_pending_count = 1
                return False

            self._signal_pending_count += 1
            if self._signal_pending_count >= max(1, self.weak_signal_confirm_ticks):
                self._signal_pending_dir = 0
                self._signal_pending_count = 0
                return True
            return False

        def _compute_order_volume(self, direction: int) -> int:
            """Dynamic sizing from signal strength and current risk state."""
            base = max(self.min_trade_volume, self.trade_volume)
            edge_threshold = self.sig.cfg.edge_score_long_threshold if direction > 0 else self.sig.cfg.edge_score_short_threshold
            strength = abs(self.sig.edge_score) / max(1e-9, edge_threshold)
            strength_scale = min(1.8, max(0.6, strength))

            dd_limit = abs(min(-1.0, self.max_daily_drawdown))
            dd_ratio = min(1.0, abs(min(0.0, self.daily_drawdown)) / dd_limit)
            dd_scale = max(self.risk_scale_min, 1.0 - 0.6 * dd_ratio)
            loss_scale = max(self.risk_scale_min, 1.0 - 0.2 * self.consecutive_losses)

            raw = int(base * strength_scale * dd_scale * loss_scale)
            raw = max(self.min_trade_volume, min(self.max_trade_volume, raw))
            lot = max(1, self.lot_size)
            sized = (raw // lot) * lot
            return max(0, sized)

        def _entry_quality_ok(self, snap: TickSnapshot, direction: int, vol: int = 0) -> bool:
            """Check qualitative entry conditions beyond the signal gate.

            Args:
                snap:      current tick snapshot
                direction: +1 for long, -1 for short
                vol:       intended order volume (for market impact check)
            """
            if self.noise_regime_block and self.sig.regime == "NOISE":
                return False

            spread_extra = max(0.0, snap.spread_ticks - 1.0) * self.spread_edge_penalty
            if direction > 0:
                required = self.sig.cfg.edge_score_long_threshold + spread_extra
                if self.sig.edge_score < required:
                    return False
            else:
                required = self.sig.cfg.edge_score_short_threshold + spread_extra
                if self.sig.edge_score > -required:
                    return False

            # B3: Market impact check — reject if our order would consume more than
            # max_impact_pct of the best-level size (signals poor liquidity for our size).
            if vol > 0 and self.max_impact_pct > 0:
                best_vol = snap.ask_vol1 if direction > 0 else snap.bid_vol1
                if best_vol > 0 and vol / best_vol > self.max_impact_pct:
                    return False

            return True

        def _trim_order_req_window(self, now: datetime) -> None:
            cutoff = now - timedelta(seconds=1)
            while self._order_req_ts and self._order_req_ts[0] < cutoff:
                self._order_req_ts.popleft()
            self.order_req_1s = len(self._order_req_ts)

        def _can_send_order_req(self, now: datetime) -> bool:
            self._trim_order_req_window(now)
            if len(self._order_req_ts) < max(1, self.max_order_req_per_sec):
                return True
            if (
                self._last_rate_limit_log_dt is None
                or (now - self._last_rate_limit_log_dt).total_seconds() >= 1.0
            ):
                self._last_rate_limit_log_dt = now
                self.write_log(
                    f"[RATE LIMIT] order_req_1s={len(self._order_req_ts)} "
                    f"limit={self.max_order_req_per_sec}"
                )
            return False

        def _mark_order_req(self, now: datetime) -> None:
            self._order_req_ts.append(now)
            self.order_req_1s = len(self._order_req_ts)

        def _rl_buy(self, price: float, volume: int, now: datetime):
            # Open-order interval guard (separate from per-sec window)
            if self._last_open_order_ts is not None:
                if (now - self._last_open_order_ts).total_seconds() * 1000 < self.open_min_interval_ms:
                    return []
            if not self._can_send_order_req(now):
                return []
            self._last_open_order_ts = now
            self._mark_order_req(now)
            return self.buy(price, volume)

        def _rl_short(self, price: float, volume: int, now: datetime):
            if self._last_open_order_ts is not None:
                if (now - self._last_open_order_ts).total_seconds() * 1000 < self.open_min_interval_ms:
                    return []
            if not self._can_send_order_req(now):
                return []
            self._last_open_order_ts = now
            self._mark_order_req(now)
            return self.short(price, volume)

        def _rl_sell(self, price: float, volume: int, now: datetime):
            # Respect global request budget while keeping a short close interval.
            if self._last_close_order_ts is not None:
                if (now - self._last_close_order_ts).total_seconds() * 1000 < self.close_min_interval_ms:
                    return []
            if not self._can_send_order_req(now):
                return []
            self._last_close_order_ts = now
            self._mark_order_req(now)
            return self.sell(price, volume)

        def _rl_cover(self, price: float, volume: int, now: datetime):
            if self._last_close_order_ts is not None:
                if (now - self._last_close_order_ts).total_seconds() * 1000 < self.close_min_interval_ms:
                    return []
            if not self._can_send_order_req(now):
                return []
            self._last_close_order_ts = now
            self._mark_order_req(now)
            return self.cover(price, volume)

        def _rl_cancel_all(self, now: datetime) -> bool:
            if not self._can_send_order_req(now):
                return False
            self._mark_order_req(now)
            self.cancel_all()
            return True

        # ------------------------------------------------------------------
        # UI sync helpers
        # ------------------------------------------------------------------

        def _sync_sig_vars(self, now: Optional[datetime] = None) -> None:
            for k, v in self.sig.get_status_dict().items():
                if hasattr(self, k):
                    setattr(self, k, v)
            self._trim_order_req_window(now or datetime.now())
            self.order_state = self._order_state.value
            # v4: update PnL stat variables for UI display
            stats = self._pnl_tracker.stats()
            if stats:
                self.stat_win_rate = stats["win_rate"]
                self.stat_pf       = stats["profit_factor"]
                self.stat_avg_hold = stats["avg_hold_sec"]

        def _update_unrealized_pnl(self, snap: TickSnapshot) -> None:
            """Compute and update unrealized PnL for display in the UI variable.

            Only meaningful while the strategy holds an open position (OPEN state).
            Uses the current best bid/ask as the exit reference price (conservative:
            always assumes you have to cross the spread to exit).
            Includes a round-trip commission estimate so the displayed value
            approximates the realised P&L if you were to exit immediately.
            """
            if self._order_state == OrderState.OPEN and self._entry_fill_price > 0:
                if self._entry_direction == "LONG":
                    # Exit ref: sell at bid1
                    gross = (snap.bid1 - self._entry_fill_price) * self._entry_fill_volume
                elif self._entry_direction == "SHORT":
                    # Exit ref: cover at ask1
                    gross = (self._entry_fill_price - snap.ask1) * self._entry_fill_volume
                else:
                    gross = 0.0
                commission_est = (
                    self._calc_commission(self._entry_fill_price, self._entry_fill_volume)
                    + self._calc_commission(snap.mid, self._entry_fill_volume)
                )
                self.unrealized_pnl = gross - commission_est
            else:
                self.unrealized_pnl = 0.0

        def _maybe_log(self, dt: datetime) -> None:
            if (
                self._last_log_dt is None
                or (dt - self._last_log_dt).total_seconds() >= self.log_interval_seconds
            ):
                self._last_log_dt = dt
                self.write_log(
                    f"{self.sig.summary()} | "
                    f"pnl={self.daily_pnl:.0f}JPY dd={self.daily_drawdown:.0f} "
                    f"dtrades={self.daily_trades} total={self.total_trades} "
                    f"loss_streak={self.consecutive_losses}"
                )

        def _throttled_put_event(self, dt: datetime) -> None:
            if (
                self._last_put_dt is None
                or (dt - self._last_put_dt).total_seconds() >= 0.5
            ):
                self._last_put_dt = dt
                self.put_event()

        def _maybe_log_bad_tick(self, tick) -> None:
            dt = getattr(tick, "datetime", None)
            if not isinstance(dt, datetime):
                dt = datetime.now()
            if (
                self._last_bad_tick_dt is None
                or (dt - self._last_bad_tick_dt).total_seconds() >= 10.0
            ):
                self._last_bad_tick_dt = dt
                b = getattr(tick, "bid_price_1", "?")
                a = getattr(tick, "ask_price_1", "?")
                self.write_log(
                    f"Bad tick skipped (bid={b} ask={a}): "
                    f"check adapter sanity or reverse_bid_ask setting"
                )

        def _reset_position_state(self) -> None:
            self._order_state = OrderState.IDLE
            self._active_orderids = []
            self._entry_fill_price = 0.0
            self._entry_fill_volume = 0.0
            self._entry_time = None
            self._entry_direction = ""
            self._entry_mode = ""
            self._entry_volume = 0
            self._opened_total_volume = 0.0
            self._open_trade_gross_pnl = 0.0
            self._close_turnover = 0.0
            self._close_volume_accum = 0.0
            self._max_favorable_ticks = 0.0
            self._state_since = None
            self.order_state = OrderState.IDLE.value
            self.unrealized_pnl = 0.0
            self._pending_exit_reason = ""
            self._last_close_order_ts = None
