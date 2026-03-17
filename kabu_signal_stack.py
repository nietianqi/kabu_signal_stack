"""kabu_signal_stack.py 遯ｶ繝ｻRedesigned signal stack + VeighNa strategy wrapper.  [v3]

Architecture (4 layers):
  [VeighNa TickData]
      遶翫・KabuTickAdapter    (field mapping, bid/ask reversal switch, sanity check)
      遶翫・TickSnapshot       (clean L1/L2 snapshot, up to 5 price levels)
      遶翫・KabuSignalStack    (OBI / LOB-OFI / Tape-OFI / momentum / microprice tilt /
                            VWAP deviation / full-depth book imbalance)
      遶翫・KabuSignalStrategy (VeighNa CtaTemplate: state machine, order tracking, risk)

Verified against official sources (as of 2025):
  - TSE 陷ｻ・ｼ陋滂ｽ､陷雁・ｽｽ繝ｻ JPX standard domestic equity tick size table (霑ｴ・ｾ霑夲ｽｩ隴ｬ・ｪ陟代・
    ETF/REIT/TOPIX500 constituents may use different tables 遯ｶ繝ｻverify before live use.
  - Trading hours: 09:00遯ｶ繝ｻ1:30 (陷第ｦ奇｣ｰ・ｴ) / 12:30遯ｶ繝ｻ5:30 (陟墓ぁ・ｰ・ｴ), orders accepted from 08:00
  - kabu Station API: BestBid/BestAsk field names may be reversed vs. industry convention.
    Set reverse_bid_ask=True if your VeighNa gateway does not normalise these fields.
  - Commission: kabu offers multiple fee plans (陞ｳ螟撰ｽ｡繝ｻ鬩幢ｽｽ陟趣ｽｦ). Default parameters use the
    ad-valorem 鬩幢ｽｽ陟趣ｽｦ model (霑壹・・・0.385% 驕樊焔・ｾ・ｼ, min ・ゑｽ･55). Always confirm against your
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

    Reference: signals.md ・ゑｽｧ8
    """

    def __init__(self, lookback: int = 100) -> None:
        self._window = max(10, lookback)
        self._min_samples = min(50, self._window)
        self._buf: Deque[float] = deque(maxlen=self._window)
        self._sum_x = 0.0
        self._sum_x2 = 0.0

    def normalize(self, value: float) -> float:
        if len(self._buf) == self._window:
            old = self._buf[0]
            self._sum_x -= old
            self._sum_x2 -= old * old
        self._buf.append(value)
        self._sum_x += value
        self._sum_x2 += value * value
        n = len(self._buf)
        if n < self._min_samples:
            return 0.0
        mean = self._sum_x / n
        var  = max(0.0, self._sum_x2 / n - mean * mean)
        std  = math.sqrt(var)
        return 0.0 if std < 1e-10 else max(-4.0, min(4.0, (value - mean) / std))


class FlowFlipDetector:
    """Detect consecutive order-flow direction reversals (momentum exhaustion).

    When the tape aggressor direction flips ``flip_threshold`` or more times in
    a row (alternating buy/sell) it signals that momentum is exhausted and the
    current trend is likely to stall or reverse.

    Reference: signals.md ・ゑｽｧ7
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

    Reference: risk.md ・ゑｽｧ2
    """
    entry_price: float
    exit_price:  float
    volume:      float
    direction:   int         # +1 = LONG, -1 = SHORT
    commission:  float       # total round-trip commission (・ゑｽ･)
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

    Reference: risk.md ・ゑｽｧ2
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
      IDLE 遶翫・PENDING_OPEN   : entry order sent
      PENDING_OPEN 遶翫・OPEN   : on_trade (opening fill received)
      OPEN 遶翫・PENDING_CLOSE  : exit order sent
      PENDING_CLOSE 遶翫・IDLE  : on_trade (closing fill received)
      any 遶翫・IDLE            : cancel_all + reset (emergency / cancelled order)
    """
    IDLE          = "idle"
    PENDING_OPEN  = "pending_open"
    OPEN          = "open"
    PENDING_CLOSE = "pending_close"


class MarketState(Enum):
    """Micro-market condition used by execution/risk policy."""
    NORMAL = "NORMAL"
    QUEUE = "QUEUE"
    ABNORMAL = "ABNORMAL"


# kabu push quote signs that indicate non-normal quote state.
_SPECIAL_QUOTE_SIGNS = frozenset({"0102", "0103", "0107"})


# ---------------------------------------------------------------------------
# TSE tick size (陷ｻ・ｼ陋滂ｽ､陷雁・ｽｽ繝ｻ 遯ｶ繝ｻstandard domestic equity (霑ｴ・ｾ霑夲ｽｩ隴ｬ・ｪ陟代・
# NOTE: ETF / REIT / TOPIX500 constituents may differ; verify with JPX circulars.
# ---------------------------------------------------------------------------

def get_tse_pricetick(price: float) -> float:
    """Return the minimum price increment for a standard TSE equity.

    Based on JPX 陷ｻ・ｼ陋滂ｽ､陷雁・ｽｽ繝ｻtable for domestic equities (Prime / Standard / Growth markets).
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
# Adapter layer 遯ｶ繝ｻfield mapping + optional bid/ask reversal + sanity check
# ---------------------------------------------------------------------------

class KabuTickAdapter:
    """Extract and normalise a TickSnapshot from a VeighNa TickData object.

    kabu Station API field naming differs from financial industry convention:
      API field BestBid  遶翫・actual sell-side (ask) price
      API field BestAsk  遶翫・actual buy-side  (bid) price

    If your VeighNa gateway already corrects this mapping (e.g. the
    kabusapi gateway adapter swaps the fields on ingestion), leave
    ``reverse_bid_ask=False`` (default). If raw API field names pass
    through unchanged, set ``reverse_bid_ask=True``.

    This class is the single correction point; all downstream code
    receives normalised data where bid < ask.
    """

    def __init__(
        self,
        reverse_bid_ask: bool = False,
        max_spread_pct: float = 0.05,
        auto_fix_negative_spread: bool = True,
        auto_fix_negative_spread_max_ticks: float = 3.0,
    ) -> None:
        self.reverse_bid_ask = reverse_bid_ask
        self.max_spread_pct  = max_spread_pct
        self.auto_fix_negative_spread = auto_fix_negative_spread
        self.auto_fix_negative_spread_max_ticks = max(0.5, float(auto_fix_negative_spread_max_ticks))

    def extract(self, tick, pricetick: float = 1.0) -> Optional["TickSnapshot"]:
        """Return a normalised TickSnapshot, or None if the tick fails sanity checks."""

        def _f(attr: str) -> float:
            return float(getattr(tick, attr, 0.0) or 0.0)

        def _apply_swap(raw_bid: float, raw_ask: float, swap: bool) -> Tuple[float, float]:
            return (raw_ask, raw_bid) if swap else (raw_bid, raw_ask)

        # Read raw L1 fields
        raw_bid = _f("bid_price_1")
        raw_ask = _f("ask_price_1")

        # Apply configured field mapping first.
        use_swap = self.reverse_bid_ask
        bid1, ask1 = _apply_swap(raw_bid, raw_ask, use_swap)

        # Basic sanity: prices must be positive.
        if bid1 <= 0.0 or ask1 <= 0.0:
            return None

        # Auto-fix path (inspired by optimized_new):
        # if we see negative spread, try toggling side mapping for this tick.
        if bid1 >= ask1 and self.auto_fix_negative_spread:
            alt_swap = not use_swap
            alt_bid, alt_ask = _apply_swap(raw_bid, raw_ask, alt_swap)
            if alt_bid > 0.0 and alt_ask > alt_bid:
                spread_ticks = (alt_ask - alt_bid) / max(pricetick, 1e-9)
                if spread_ticks <= self.auto_fix_negative_spread_max_ticks:
                    use_swap = alt_swap
                    bid1, ask1 = alt_bid, alt_ask

        # Still invalid after auto-fix attempt.
        if bid1 >= ask1:
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
            # Apply final side mapping to L2 levels as well.
            if use_swap:
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

        bid_vol1 = _f("bid_volume_1")
        ask_vol1 = _f("ask_volume_1")
        if use_swap:
            bid_vol1, ask_vol1 = ask_vol1, bid_vol1

        return TickSnapshot(
            dt=dt,
            bid1=bid1,
            ask1=ask1,
            bid_vol1=bids[0][1] if bids else bid_vol1,
            ask_vol1=asks[0][1] if asks else ask_vol1,
            bids=bids,
            asks=asks,
            bid_sign=str(getattr(tick, "bid_sign", "") or getattr(tick, "BidSign", "") or ""),
            ask_sign=str(getattr(tick, "ask_sign", "") or getattr(tick, "AskSign", "") or ""),
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
    # v8 fix: lowered 2.0→1.0 so QUEUE mode (1-tick spread) qualifies as MAKER.
    # With strict_entry_advantage=True this was blocking ALL entries in QUEUE markets.
    maker_preferred_spread_ticks: float = 1.0
    maker_adverse_sel_ratio: float = 0.3

    # Signal windows
    obi_levels: int = 5          # Weighted OBI: number of order book levels
    lob_ofi_window: int = 20
    lob_ofi_depth: int = 5       # Multi-level LOB-OFI: number of levels
    lob_ofi_decay: float = 0.80  # Exponential decay weight per level
    tape_window_sec: float = 1.0
    momentum_window: int = 10

    # Whale pressure signal
    whale_qty_threshold: int = 1000   # Minimum trade size to count as whale
    w_whale: float = 0.5              # Weight in edge score composite
    whale_long: float = 0.3           # whale_ofi threshold for long alpha count
    whale_short: float = 0.3          # whale_ofi threshold for short alpha count

    # Trade lag gate
    # v8 fix: disabled by default (0.0). TSE stocks can go 10+ seconds without a print
    # in quiet afternoon periods, causing g_trade_lag_ok=False and blocking all entries.
    max_trade_lag_sec: float = 0.0    # 0=disabled; block entry if no trade for this many seconds

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
    vwap_long: float = 0.2           # price below VWAP by this fraction 遶翫・long alpha
    vwap_short: float = 0.2          # price above VWAP by this fraction 遶翫・short alpha

    # Book depth signal thresholds
    book_depth_long: float = 0.15    # book_depth_ratio above this 遶翫・long alpha
    book_depth_short: float = 0.15   # book_depth_ratio below -this 遶翫・short alpha
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
    max_spread_pct:  float = 0.05         # reject ticks where spread > 5% of bid (霑夲ｽｹ陋ｻ・･雎碁斡繝ｻ filter)
    auto_fix_negative_spread: bool = True
    auto_fix_negative_spread_max_ticks: float = 3.0

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
    bid_sign: str = ""
    ask_sign: str = ""
    last_price: float = 0.0
    volume: float = 0.0
    turnover: float = 0.0      # cumulative session turnover (・ゑｽ･) 遯ｶ繝ｻused for VWAP calculation
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
          +1 遶翫・all volume on buy side (strong upward pressure)
          -1 遶翫・all volume on sell side (strong downward pressure)
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
      +1  遶翫・strong buy-side pressure (bids dominate)
      -1  遶翫・strong sell-side pressure (asks dominate)

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
        bid_ofi = curr.bid_vol1           # Higher bid appeared 遶翫・buy pressure added
    elif curr.bid1 == prev.bid1:
        bid_ofi = curr.bid_vol1 - prev.bid_vol1
    else:
        bid_ofi = -prev.bid_vol1          # Bid level dropped 遶翫・buy pressure removed

    # Ask-side contribution
    if curr.ask1 < prev.ask1:
        ask_ofi = -curr.ask_vol1          # Lower ask appeared 遶翫・sell pressure added
    elif curr.ask1 == prev.ask1:
        ask_ofi = -(curr.ask_vol1 - prev.ask_vol1)
    else:
        ask_ofi = prev.ask_vol1           # Ask level rose 遶翫・sell pressure removed

    return bid_ofi + ask_ofi


def calc_tape_aggressor(
    trade_price: float,
    bid: float,
    ask: float,
    prev_trade_price: Optional[float] = None,
) -> float:
    """Classify a trade as buyer-initiated (+1), seller-initiated (-1), or ambiguous (0).

    Uses the Lee-Ready (1991) algorithm:
      1. Quote rule: trade at or above ask 遶翫・buy; at or below bid 遶翫・sell.
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
    capturing large orders resting in deep levels 遯ｶ繝ｻoften a sign of
    institutional intent that doesn't yet appear at the touch.

    Returns a value in [-1, +1]:
      +1  遶翫・all visible depth is on the bid side (strong buying interest)
      -1  遶翫・all visible depth is on the ask side (strong selling interest)
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
      obi             遯ｶ繝ｻweighted multi-level order book imbalance
      microprice_tilt 遯ｶ繝ｻmicroprice displacement from mid (half-spread normalised)
      lob_ofi         遯ｶ繝ｻincremental best-level order flow imbalance
      tape_ofi        遯ｶ繝ｻaggressor-classified volume imbalance (rolling window)
      momentum        遯ｶ繝ｻmicroprice relative to rolling mean

    Public interface:
      on_tick(tick)          遶翫・Optional[TickSnapshot]   (uses internal adapter)
      on_snapshot(snap)      遶翫・TickSnapshot             (for externally built snaps)
      can_open_long(skew)    遶翫・bool
      can_open_short(skew)   遶翫・bool
      should_exit_long()     遶翫・bool
      should_exit_short()    遶翫・bool
    """

    def __init__(self, config: SignalConfig, pricetick: float = 1.0) -> None:
        self.cfg = config
        self.pricetick = max(1e-9, float(pricetick or 1.0))
        self._adapter = KabuTickAdapter(
            reverse_bid_ask=config.reverse_bid_ask,
            max_spread_pct=config.max_spread_pct,
            auto_fix_negative_spread=config.auto_fix_negative_spread,
            auto_fix_negative_spread_max_ticks=config.auto_fix_negative_spread_max_ticks,
        )
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
        self.whale_ofi: float = 0.0          # whale pressure signal [-1, +1]

        # v7: multi-level LOB-OFI config
        self._lob_ofi_depth: int = max(1, config.lob_ofi_depth)
        self._lob_ofi_decay: float = max(0.0, min(1.0, config.lob_ofi_decay))

        # v7: whale pressure accumulators (rolling window)
        self._whale_buy_vol: float = 0.0
        self._whale_sell_vol: float = 0.0
        self._whale_q: Deque[Tuple[datetime, float]] = deque(maxlen=2048)

        # v7: trade lag tracking
        self._last_trade_price: float = 0.0
        self._last_trade_dt: Optional[datetime] = None
        self.g_trade_lag_ok: bool = True

        # VWAP cumulative accumulators (reset each session day)
        self._sess_vol: float = 0.0
        self._sess_turn: float = 0.0
        self._sess_date: str = ""

        # Signal z-score normalizers (one per signal to account for scale differences)
        _znorm_keys = ("obi", "lob_ofi", "tape_ofi", "momentum",
                       "microprice_tilt", "vwap_signal", "book_depth_ratio", "whale")
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
        self._update_gate_trade_lag(snap, now)

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
            and self.g_trade_lag_ok
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
            "sig_whale_ofi":      float(self.whale_ofi),
            "g_trade_lag_ok":     int(self.g_trade_lag_ok),
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
        prev = self._prev_snap
        score = 0.0
        decay = self._lob_ofi_decay
        for i in range(self._lob_ofi_depth):
            w = decay ** i
            bc = snap.bids[i] if i < len(snap.bids) else None
            bp = prev.bids[i] if i < len(prev.bids) else None
            if bc and not bp:
                bid_delta = float(bc[1])
            elif bp and not bc:
                bid_delta = -float(bp[1])
            elif bc and bp:
                if bc[0] > bp[0]:   bid_delta = float(bc[1])
                elif bc[0] < bp[0]: bid_delta = -float(bp[1])
                else:               bid_delta = float(bc[1] - bp[1])
            else:
                bid_delta = 0.0
            ac = snap.asks[i] if i < len(snap.asks) else None
            ap = prev.asks[i] if i < len(prev.asks) else None
            if ac and not ap:
                ask_delta = float(ac[1])
            elif ap and not ac:
                ask_delta = -float(ap[1])
            elif ac and ap:
                if ac[0] < ap[0]:   ask_delta = float(ac[1])
                elif ac[0] > ap[0]: ask_delta = -float(ap[1])
                else:               ask_delta = float(ac[1] - ap[1])
            else:
                ask_delta = 0.0
            score += w * (bid_delta - ask_delta)
        self._lob_q.append(score)
        scale = max(1.0, sum(abs(x) for x in self._lob_q) / max(1, len(self._lob_q)))
        self.lob_ofi = sum(self._lob_q) / (scale * max(1, len(self._lob_q)))

    def _update_tape_ofi(self, snap: TickSnapshot, now: datetime) -> None:
        vol = max(0.0, snap.volume)
        if self._prev_snap is None:
            self._last_total_volume = vol
            self.flow_flip = False
            return
        delta_vol = max(0.0, vol - self._last_total_volume)
        self._last_total_volume = vol
        if delta_vol <= 0:
            self._expire_tape(now)
            self._compute_tape()
            # No new prints: clear flip event so it doesn't stay sticky.
            self.flow_flip = self._flow_flip_det.update(0)
            return
        prev_trade = self._prev_snap.last_price if self._prev_snap else None
        aggr = calc_tape_aggressor(snap.last_price, snap.bid1, snap.ask1, prev_trade)
        self._tape_q.append((now, aggr * delta_vol))
        self._expire_tape(now)
        self._compute_tape()
        # Update flow flip detector (convert float aggressor to int direction)
        self.flow_flip = self._flow_flip_det.update(int(round(aggr)))
        # v7: update trade lag tracker
        if snap.last_price != self._last_trade_price and snap.last_price > 0:
            self._last_trade_price = snap.last_price
            self._last_trade_dt = now
        # v7: update whale pressure signal
        if delta_vol >= self.cfg.whale_qty_threshold:
            buy_v = delta_vol if aggr > 0 else 0.0
            sell_v = delta_vol if aggr < 0 else 0.0
            self._whale_buy_vol += buy_v
            self._whale_sell_vol += sell_v
            self._whale_q.append((now, buy_v - sell_v))
        self._expire_whale(now)
        self._compute_whale()

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

    def _expire_whale(self, now: datetime) -> None:
        cutoff = now.timestamp() - max(0.1, self.cfg.tape_window_sec)
        while self._whale_q and self._whale_q[0][0].timestamp() < cutoff:
            _, net = self._whale_q.popleft()
            if net > 0:
                self._whale_buy_vol -= net
            else:
                self._whale_sell_vol -= (-net)
        self._whale_buy_vol = max(0.0, self._whale_buy_vol)
        self._whale_sell_vol = max(0.0, self._whale_sell_vol)

    def _compute_whale(self) -> None:
        total = self._whale_buy_vol + self._whale_sell_vol
        self.whale_ofi = 0.0 if total <= 0 else (self._whale_buy_vol - self._whale_sell_vol) / total

    def _update_gate_trade_lag(self, snap: TickSnapshot, now: datetime) -> None:
        """Block entry when no trades have printed for max_trade_lag_sec (Tape-OFI unreliable)."""
        lag_sec = self.cfg.max_trade_lag_sec
        if lag_sec <= 0:
            self.g_trade_lag_ok = True
            return
        if self._last_trade_dt is None:
            self.g_trade_lag_ok = True
            return
        elapsed = (now - self._last_trade_dt).total_seconds()
        self.g_trade_lag_ok = elapsed < lag_sec

    def _update_momentum(self, snap: TickSnapshot) -> None:
        self._mom_q.append(snap.microprice)
        if len(self._mom_q) < 2:
            self.momentum = 0.0
            return
        mean = sum(self._mom_q) / len(self._mom_q)
        self.momentum = 0.0 if mean == 0 else (self._mom_q[-1] - mean) / mean

    def _update_regime(self, snap: TickSnapshot) -> None:
        """Detect market regime using lag-1 sign autocorrelation of returns.

        ``sign_autocorr > +threshold``  遶翫・TREND    (returns tend to continue)
        ``sign_autocorr < -threshold``  遶翫・REVERSION (returns tend to reverse)
        otherwise                        遶翫・NOISE

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

        # Volatility gate 遯ｶ繝ｻblock noisy high-vol periods
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

        Price below VWAP 遶翫・positive signal (mean-reversion buy pressure).
        Price above VWAP 遶翫・negative signal (sell pressure).
        Normalised to [-1, +1] using a ・ゑｽｱ3 tick window.
        VWAP accumulator resets each new calendar day.
        """
        date_str = snap.dt.strftime("%Y%m%d")
        if date_str != self._sess_date:
            # New session 遯ｶ繝ｻreset accumulators
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
        # Negative displacement = price below VWAP 遶翫・positive (buy) signal
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
        z_whale = zn["whale"].normalize(self.whale_ofi)

        self.edge_score = (
            cfg.w_microprice  * z_tilt
            + cfg.w_lob_ofi   * z_lofi
            + cfg.w_tape_ofi  * z_tape
            + cfg.w_momentum  * z_mom
            + cfg.w_obi       * z_obi
            + cfg.w_vwap      * z_vwap
            + cfg.w_book_depth * z_bdr
            + cfg.w_whale     * z_whale
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
        if self.whale_ofi >= cfg.whale_long:                   lc += 1
        if self.whale_ofi <= -cfg.whale_short:                 sc += 1
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


def make_strategy_preset_balanced() -> Dict[str, object]:
    """Balanced preset for liquid TSE names (first live tuning candidate)."""
    return {
        "trade_volume": 100,
        "max_position": 600,
        "enable_long": True,
        "enable_short": False,
        "reverse_bid_ask": False,
        "auto_pricetick": False,
        "max_spread_ticks": 2.0,
        "min_best_volume": 50,
        "strong_signal_threshold": 4.0,
        "weak_signal_confirm_ticks": 2,
        "entry_cooldown_sec": 0.4,
        "open_min_interval_ms": 100.0,
        "close_min_interval_ms": 50.0,
        "hold_if_loss": True,
        "strict_entry_advantage": True,
        "min_entry_edge_ticks": 1.0,
        "auto_tp_on_fill": True,
        "auto_tp_ticks": 2.0,
        "maker_escape_timeout_sec": 1.5,
        "smart_cancel_on_flip": True,
        "max_tick_stale_seconds": 5.0,
        "stale_quote_ms": 1200,
        "queue_spread_max_ticks": 1.0,
        "abnormal_max_spread_ticks": 6.0,
        "max_event_rate_hz": 160.0,
        "event_burst_min_events": 6,
        "state_window_ms": 3000,
        "jump_threshold_ticks": 4.0,
        "queue_min_top_qty": 300,
        "close_only_before_break_min": 5,
        "fair_value_beta": 0.75,
        "max_fair_shift_ticks": 3.0,
        "inventory_skew_ticks": 1.0,
        "max_fair_drift_ticks": 1.5,
    }


def make_strategy_preset_conservative() -> Dict[str, object]:
    """Conservative preset focused on fill quality and drawdown control."""
    p = make_strategy_preset_balanced()
    p.update(
        {
            "max_spread_ticks": 1.5,
            "min_best_volume": 100,
            "strong_signal_threshold": 4.8,
            "weak_signal_confirm_ticks": 3,
            "entry_cooldown_sec": 0.6,
            "open_min_interval_ms": 140.0,
            "max_trade_volume": 300,
            "risk_scale_min": 0.3,
            "max_daily_trades": 50,
            "loss_cooldown_sec": 240.0,
            "queue_min_top_qty": 500,
            "abnormal_max_spread_ticks": 5.0,
            "max_event_rate_hz": 120.0,
            "jump_threshold_ticks": 3.0,
            "fair_value_beta": 0.65,
            "max_fair_drift_ticks": 1.2,
        }
    )
    return p


def make_strategy_preset_aggressive() -> Dict[str, object]:
    """Aggressive preset for fast tape with strict emergency controls."""
    p = make_strategy_preset_balanced()
    p.update(
        {
            "max_spread_ticks": 2.5,
            "min_best_volume": 30,
            "strong_signal_threshold": 3.2,
            "weak_signal_confirm_ticks": 1,
            "entry_cooldown_sec": 0.2,
            "open_min_interval_ms": 70.0,
            "close_min_interval_ms": 30.0,
            "max_trade_volume": 600,
            "maker_escape_timeout_sec": 1.0,
            "max_event_rate_hz": 220.0,
            "abnormal_max_spread_ticks": 7.0,
            "jump_threshold_ticks": 5.0,
            "fair_value_beta": 0.9,
            "max_fair_shift_ticks": 4.0,
            "max_fair_drift_ticks": 2.0,
            "max_loss_per_trade_jpy": 12000.0,
        }
    )
    return p


# ---------------------------------------------------------------------------
# VeighNa strategy wrapper
# ---------------------------------------------------------------------------

if _VNPY_AVAILABLE:
    class KabuSignalStrategy(CtaTemplate):
        """VeighNa CTA strategy wrapping KabuSignalStack.

        Order lifecycle (state machine):
          IDLE 遶翫・PENDING_OPEN 遶翫・OPEN 遶翫・PENDING_CLOSE 遶翫・IDLE

        The state machine is the primary guard against duplicate orders:
        no new entry is allowed unless the state is IDLE.

        Exit triggers (evaluated each tick in OPEN state):
          1. Signal reversal  (edge_score crosses threshold)
          2. Take-profit      (pnl_ticks >= profit_ticks)
          3. Stop-loss        (pnl_ticks <= -loss_ticks)
          4. Timeout          (elapsed >= max_hold_seconds, unconditional)

        PnL uses actual fill prices from on_trade callbacks, not pre-order estimates.
        Commission uses kabu's ad-valorem model: max(commission_min, rate ・・・value).
        """

        author = "KabuSignalStack v3"

        # --- Tunable parameters ---
        trade_volume: int = 100
        max_position: int = 600
        enable_long: bool = True
        enable_short: bool = False

        # Gate parameters (passed to SignalConfig on start)
        max_spread_ticks: float = 2.0
        min_best_volume: int = 50
        obi_levels: int = 3
        ev_gate_enabled: bool = False

        # Adapter
        reverse_bid_ask: bool = False   # Set True if kabu gateway passes raw field names
        auto_fix_negative_spread: bool = True
        auto_fix_negative_spread_max_ticks: float = 3.0
        # ⚠️ 开启 TSE 表自动推断：kabu gateway 对部分股票 contract.pricetick=1.0 不可信。
        # auto_pricetick=True 时，每 tick 从 last_price 调用 get_tse_pricetick() 更新。
        # 同时也使 auto_fix_negative_spread 的 spread_ticks 计算更准确（如 4483 spread=5¥/5tick=1，不会被误过滤）
        auto_pricetick: bool = True

        # Commission 遯ｶ繝ｻkabu ad-valorem model (鬩幢ｽｽ陟趣ｽｦ隰・玄辟夊ｭ√・
        # IMPORTANT: confirm against your actual account plan before live trading.
        commission_rate: float = 0.00385   # 霑壹・・・0.385% 驕樊焔・ｾ・ｼ
        commission_min: float = 55.0       # 隴崢闖ｴ蜿也・隰ｨ・ｰ隴√・・ゑｽ･55 驕樊焔・ｾ・ｼ

        # Exit parameters
        profit_ticks: float = 3.0
        loss_ticks: float = 999.0            # v7: effectively disabled; use max_loss_per_trade_jpy
        max_hold_seconds: float = 30.0
        open_order_timeout_sec: float = 3.0
        close_order_timeout_sec: float = 5.0  # v7: wider timeout for non-TP close orders
        taker_exit_extra_ticks: float = 0.0
        trailing_activate_ticks: float = 2.0
        trailing_drawdown_ticks: float = 1.2

        # Timing and confirmation
        strong_signal_threshold: float = 4.0
        weak_signal_confirm_ticks: int = 2
        signal_confirm_expire_sec: float = 1.0
        entry_cooldown_sec: float = 0.4
        no_new_entry_after: str = "15:24:00"
        # v8 fix: disabled by default. Logs showed regime=NOISE 100% of the time for
        # low-pricetick stocks (e.g. 2000 JPY / 1 JPY tick), blocking all entries even
        # with edge=+9.85. Signal gates (alpha≥2, edge≥2.5) provide sufficient filtering.
        noise_regime_block: bool = False
        spread_edge_penalty: float = 0.4

        # Dynamic sizing
        lot_size: int = 100
        min_trade_volume: int = 100
        max_trade_volume: int = 600
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
        max_impact_pct: float = 0.5              # reject entry if vol > this ・・・best-level size

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

        # Hold-through-loss mode
        hold_if_loss: bool = True                # When True: losing positions are not closed by
                                                 # normal signal/timeout/trailing logic.

        # MAKER mode override
        force_maker_mode: bool = False           # When True: always enter at bid(long)/ask(short),
                                                 # exit at ask(long)/bid(short) 遯ｶ繝ｻcaptures the spread.
                                                 # When False: signal engine decides MAKER vs TAKER.

        # Entry edge discipline
        strict_entry_advantage: bool = True      # Only open when the quoted price is favorable.
        min_entry_edge_ticks: float = 2.0        # v7: must cover auto_tp_ticks=2 by default

        # Auto-TP on fill
        auto_tp_on_fill: bool = True             # When True: place limit TP close order immediately
                                                 # when open fill is confirmed (vs waiting for next tick)
        auto_tp_ticks: float = 2.0               # TP price offset in ticks from fill price
                                                 # LONG: fill + auto_tp_ticks*pt; SHORT: fill - auto_tp_ticks*pt
        auto_tp_retry_max: int = 15             # total retry count (fast + slow phases)
        auto_tp_retry_delay_sec: float = 0.3     # retry delay for fast phase
        auto_tp_fast_retries: int = 5            # first N retries use auto_tp_retry_delay_sec (fast)
        auto_tp_slow_retry_sec: float = 30.0     # interval for slow phase retries (after fast exhausted)

        # Feed stale watchdog
        max_tick_stale_seconds: float = 5.0      # 0 disables stale-feed watchdog
        stale_feed_force_flatten: bool = False   # True: stale feed triggers force flat (after cancels)

        # Smart cancel for pending open orders
        smart_cancel_on_flip: bool = True        # When True: cancel PENDING_OPEN order immediately
                                                 # if the entry signal reverses (no waiting for timeout)

        # --- v5 parameters ---
        # Per-trade JPY emergency stop (from reference strategy)
        max_loss_per_trade_jpy: float = 100_000.0  # v7: 10万円 hard floor; works with hold_if_loss=True

        # Verbose logging control (from reference strategy)
        verbose_log: bool = False                 # True: output detailed debug logs (rate-limit reasons, etc.)
        near_miss_log: bool = True               # True: log [NEAR♦] when signal reaches 60% of threshold (requires verbose_log)

        # --- v6 parameters (integrated from kabu_hft_new ideas) ---
        # Market state detector
        stale_quote_ms: int = 1200               # snapshot age > stale_quote_ms -> ABNORMAL
        queue_spread_max_ticks: float = 1.0      # one-tick regime -> QUEUE mode
        abnormal_max_spread_ticks: float = 6.0   # spread blowout threshold -> ABNORMAL
        max_event_rate_hz: float = 160.0         # burst threshold (with min events)
        event_burst_min_events: int = 6          # minimum samples to declare burst
        state_window_ms: int = 3000              # event-rate rolling window
        jump_threshold_ticks: float = 4.0        # mid jump threshold -> ABNORMAL
        queue_min_top_qty: int = 300             # QUEUE mode: retreat if best level is thin
        close_only_before_break_min: int = 5     # block new entries before lunch close window

        # Fair-value / reservation-price execution controls
        fair_value_beta: float = 0.75            # edge -> fair shift (ticks)
        max_fair_shift_ticks: float = 3.0        # cap fair shift in ticks
        inventory_skew_ticks: float = 1.0        # inventory penalty in ticks
        max_fair_drift_ticks: float = 1.5        # cancel pending entry when fair drift too far

        # --- v7 parameters ---
        # TP order persistence (hold_if_loss mode)
        tp_order_timeout_sec: float = 300.0      # TP limit order max wait time; 0=infinite
        # Entry quality: fair-value coverage beyond TP target
        min_fair_tp_coverage_ticks: float = 0.0  # extra safety margin (0=exact coverage required)

        parameters = [
            "trade_volume", "max_position",
            "enable_long", "enable_short",
            "max_spread_ticks", "min_best_volume",
            "obi_levels", "ev_gate_enabled",
            "reverse_bid_ask", "auto_fix_negative_spread", "auto_fix_negative_spread_max_ticks", "auto_pricetick",
            "commission_rate", "commission_min",
            "profit_ticks", "loss_ticks", "max_hold_seconds",
            "open_order_timeout_sec", "close_order_timeout_sec", "taker_exit_extra_ticks",
            "trailing_activate_ticks", "trailing_drawdown_ticks",
            "strong_signal_threshold", "weak_signal_confirm_ticks",
            "signal_confirm_expire_sec",
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
            "hold_if_loss",
            "force_maker_mode",
            "strict_entry_advantage", "min_entry_edge_ticks",
            "auto_tp_on_fill", "auto_tp_ticks",
            "auto_tp_retry_max", "auto_tp_retry_delay_sec",
            "auto_tp_fast_retries", "auto_tp_slow_retry_sec",
            "smart_cancel_on_flip",
            "max_tick_stale_seconds", "stale_feed_force_flatten",
            # v5
            "max_loss_per_trade_jpy",
            "verbose_log", "near_miss_log",
            # v6
            "stale_quote_ms", "queue_spread_max_ticks",
            "abnormal_max_spread_ticks", "max_event_rate_hz",
            "event_burst_min_events", "state_window_ms",
            "jump_threshold_ticks", "queue_min_top_qty",
            "close_only_before_break_min",
            "fair_value_beta", "max_fair_shift_ticks",
            "inventory_skew_ticks", "max_fair_drift_ticks",
            # v7
            "tp_order_timeout_sec", "min_fair_tp_coverage_ticks",
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
        sig_whale_ofi: float = 0.0
        g_trade_lag_ok: int = 1
        unrealized_pnl: float = 0.0
        stat_win_rate: float = 0.0
        stat_pf: float = 0.0
        stat_avg_hold: float = 0.0
        market_state: str = "NORMAL"
        market_reason: str = "init"
        market_event_rate_hz: float = 0.0
        market_jump_ticks: float = 0.0

        variables = [
            "price_tick", "order_state",
            "sig_all_gates", "sig_edge_score", "sig_regime",
            "sig_obi", "sig_alpha_long", "sig_alpha_short",
            "sig_adv_sel", "sig_fill_mode_long",
            "sig_taker_ev_long", "sig_maker_ev_long",
            "last_entry_price", "last_exit_price",
            "daily_pnl", "daily_trades", "consecutive_losses", "daily_drawdown",
            "risk_halt_reason", "order_req_1s", "total_trades", "last_signal",
            "sig_vwap", "sig_book_depth", "sig_flow_flip",
            "sig_whale_ofi", "g_trade_lag_ok", "unrealized_pnl",
            "stat_win_rate", "stat_pf", "stat_avg_hold",
            "market_state", "market_reason", "market_event_rate_hz", "market_jump_ticks",
        ]

        def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
            super().__init__(cta_engine, strategy_name, vt_symbol, setting)

            # Order state machine
            self._order_state: OrderState = OrderState.IDLE
            self._active_orderids: List[str] = []

            # Entry tracking 遯ｶ繝ｻpopulated from actual fill in on_trade
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
            self._maker_escape_pending: bool = False
            self._last_snap: Optional[TickSnapshot] = None
            self._cancel_waiting_ack: bool = False
            self._pending_open_orphan_since: Optional[datetime] = None
            self._pending_open_orphan_grace_sec: float = 1.0
            self._pending_tp_submit: bool = False
            self._pending_tp_price: float = 0.0
            self._pending_tp_volume: int = 0
            self._pending_tp_after: Optional[datetime] = None
            self._pending_tp_retries: int = 0

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
            self._signal_pending_dt: Optional[datetime] = None
            self._state_since: Optional[datetime] = None
            self._order_req_ts: Deque[datetime] = deque()
            self._last_rate_limit_log_dt: Optional[datetime] = None

            # Throttle timers
            self._last_log_dt: Optional[datetime] = None
            self._last_put_dt: Optional[datetime] = None
            self._last_bad_tick_dt: Optional[datetime] = None
            self._last_recv_time: Optional[datetime] = None
            self._tick_stale_state: bool = False
            self._stale_force_flatten_pending: bool = False
            self._market_state: MarketState = MarketState.NORMAL
            self._market_state_reason: str = "init"
            self._market_event_ts_ns: Deque[int] = deque(maxlen=4096)
            self._market_prev_mid: float = 0.0
            self._pending_entry_fair_price: float = 0.0
            self._pending_entry_alpha: float = 0.0

            # Signal engine (rebuilt with live parameters in on_start)
            self.sig = KabuSignalStack(SignalConfig(), 1.0)
            # Defensive default: allows on_tick to run safely even if on_start
            # has not parsed time settings yet.
            self._no_new_entry_after_time: time = time(15, 24, 0)

            # v5: periodic pricetick verification
            self._price_tick_verified: bool = False
            self._last_price_tick_check_dt: Optional[datetime] = None

            # v7: ATR-based volatility estimator
            self._atr: float = 0.0
            self._prev_mid_for_atr: float = 0.0

        # ------------------------------------------------------------------
        # Lifecycle
        # ------------------------------------------------------------------

        def on_init(self) -> None:
            self.write_log(f"on_init {self.vt_symbol} vol={self.trade_volume}")
            self.load_tick(1)
            self.put_event()

        def on_start(self) -> None:
            # Step 1: force subscribe once at start (helps after gateway reconnect).
            self._force_subscribe()

            # Step 2: load pricetick from contract if available.
            pt = self.get_pricetick()
            has_contract_pt = bool(pt and pt > 0)
            if has_contract_pt:
                self.price_tick = float(pt)

            if self.reverse_bid_ask:
                self.write_log(
                    "[WARN] reverse_bid_ask=True. With vnpy_kabu gateway this is often wrong. "
                    "If you see 'Bad tick skipped (bid>=ask)', set reverse_bid_ask=False."
                )

            # Step 3: rebuild signal engine from strategy parameters.
            cfg = SignalConfig(
                max_spread_ticks=self.max_spread_ticks,
                min_best_volume=self.min_best_volume,
                obi_levels=self.obi_levels,
                ev_gate_enabled=self.ev_gate_enabled,
                reverse_bid_ask=self.reverse_bid_ask,
                auto_fix_negative_spread=self.auto_fix_negative_spread,
                auto_fix_negative_spread_max_ticks=self.auto_fix_negative_spread_max_ticks,
                # If contract pricetick exists, disable heuristic auto inference.
                # Contract metadata is authoritative and avoids accidental 0.1/1.0 mismatch.
                auto_pricetick=(self.auto_pricetick and not has_contract_pt),
                znorm_lookback=self.znorm_lookback,
                flow_flip_threshold=self.flow_flip_threshold,
            )
            self.sig = KabuSignalStack(cfg, self.price_tick)
            if self.auto_pricetick and has_contract_pt:
                self.write_log(
                    f"[pricetick] 合约价位可用({self.price_tick})，关闭auto_pricetick推断以避免覆盖"
                )
            self._reset_position_state()
            self._market_state = MarketState.NORMAL
            self._market_state_reason = "on_start"
            self._market_event_ts_ns.clear()
            self._market_prev_mid = 0.0
            self._parse_strategy_time_settings()
            self.write_log(
                f"on_start pricetick={self.price_tick} "
                f"reverse_bid_ask={self.reverse_bid_ask}"
            )
            self.put_event()

        # ------------------------------------------------------------------
        # v5: Periodic pricetick verification & verbose logging helpers
        # ------------------------------------------------------------------

        def _periodic_verify_price_tick(self, now: datetime) -> None:
            """Re-read pricetick from contract every 30 s.

            Prevents the strategy from running with a stale fallback=1.0 when
            the kabu gateway loads contract data with a delay after on_start.
            Once successfully verified the check is skipped for efficiency.
            (Inspired by kabu_micro_edge_pro _periodic_verify_price_tick.)
            """
            if self._price_tick_verified:
                return
            if (self._last_price_tick_check_dt is not None and
                    (now - self._last_price_tick_check_dt).total_seconds() < 30.0):
                return
            self._last_price_tick_check_dt = now
            try:
                me = getattr(getattr(self, "cta_engine", None), "main_engine", None)
                if me is None:
                    return
                contract = me.get_contract(self.vt_symbol)
                if contract and hasattr(contract, "pricetick") and contract.pricetick > 0:
                    new_pt = float(contract.pricetick)
                    if abs(new_pt - self.price_tick) > 1e-9:
                        self.write_log(
                            f"[pricetick] 自动修正: {self.price_tick} → {new_pt} (合约延迟加载)"
                        )
                        self.price_tick = new_pt
                        self.sig.update_pricetick(new_pt)   # propagate to signal engine
                    # Contract pricetick is authoritative once available.
                    self.sig.cfg.auto_pricetick = False
                    self._price_tick_verified = True
            except Exception:
                pass  # silent fail; will retry in 30 s

        def _vlog(self, msg: str) -> None:
            """Verbose log: only emits when verbose_log=True.

            Use for high-frequency debug messages (rate-limit reasons, signal
            gate traces, etc.).  Critical events (fills, risk triggers) should
            always use write_log() directly.
            """
            if self.verbose_log:
                self.write_log(msg)

        def _force_subscribe(self) -> None:
            """Best-effort subscribe using the known contract gateway first."""
            try:
                parts = self.vt_symbol.split(".", 1)
                if len(parts) != 2:
                    return

                symbol_str, exchange_str = parts
                from vnpy.trader.constant import Exchange as _Ex
                from vnpy.trader.object import SubscribeRequest as _SR

                ex = _Ex(exchange_str)
                req = _SR(symbol=symbol_str, exchange=ex)
                main_engine = self.cta_engine.main_engine

                contract = main_engine.get_contract(self.vt_symbol)
                if contract and getattr(contract, "gateway_name", None):
                    main_engine.subscribe(req, contract.gateway_name)
                    self.write_log(f"Force subscribe {self.vt_symbol} -> {contract.gateway_name}")
                    return

                gateways = getattr(main_engine, "gateways", {})
                for gw_name in gateways:
                    main_engine.subscribe(req, gw_name)
                    self.write_log(f"Force subscribe fallback {self.vt_symbol} -> {gw_name}")
                    return

                self.write_log(f"[WARN] No gateway found for force_subscribe: {self.vt_symbol}")
            except Exception as e:
                self.write_log(f"[WARN] _force_subscribe error: {e}")

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
            self._last_snap = snap

            tick_dt = snap.dt
            self._last_recv_time = datetime.now()
            self._tick_stale_state = False
            self._stale_force_flatten_pending = False

            # v5: periodic pricetick verification (re-read every 30 s until confirmed)
            self._periodic_verify_price_tick(tick_dt)
            self._update_market_state(snap, tick_dt)

            # v7: ATR update (EMA of true range for volatility-based sizing)
            if self._prev_mid_for_atr > 0:
                move = abs(snap.mid - self._prev_mid_for_atr)
                tr = max(move, snap.ask1 - snap.bid1)
                self._atr = 0.05 * tr + 0.95 * (self._atr if self._atr > 0 else tr)
            self._prev_mid_for_atr = snap.mid

            self._sync_sig_vars(tick_dt)
            self._update_unrealized_pnl(snap)
            self._check_date_reset(tick_dt)
            self._maybe_log(tick_dt)
            self._check_pending_timeouts(tick_dt, snap)
            self._try_retry_pending_auto_tp(tick_dt)

            if self._market_state == MarketState.ABNORMAL:
                if (
                    self._order_state == OrderState.PENDING_OPEN
                    and self._active_orderids
                    and not self._cancel_waiting_ack
                ):
                    sent = self._rl_cancel_all(tick_dt)
                    if sent:
                        self._cancel_waiting_ack = True
                        self._state_since = tick_dt
                elif self._order_state == OrderState.OPEN:
                    self._force_flatten(snap, tick_dt, reason=f"abnormal_{self._market_state_reason}")
                    self._throttled_put_event(tick_dt)
                    return

            # 2. State machine guard 遯ｶ繝ｻif not IDLE, only manage open position
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

            if self._market_state == MarketState.ABNORMAL:
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
                if vol > 0 and self._entry_quality_ok(snap, +1, vol) and self._confirm_entry_signal(+1, tick_dt):
                    self._enter_long(snap, tick_dt, vol)
            elif self.enable_short and self.sig.can_open_short(pos_skew=pos_skew):
                vol = self._compute_order_volume(-1)
                if vol > 0 and self._entry_quality_ok(snap, -1, vol) and self._confirm_entry_signal(-1, tick_dt):
                    self._enter_short(snap, tick_dt, vol)
            else:
                self._signal_pending_dir = 0
                self._signal_pending_count = 0
                self._signal_pending_dt = None
                # [NEAR♦] near-miss diagnostic: show bottleneck when signal is ≥60% of threshold
                if self.near_miss_log and self.verbose_log and self.sig.all_gates_ok:
                    _edge = self.sig.edge_score
                    _eth_l = self.sig.cfg.edge_score_long_threshold
                    _eth_s = self.sig.cfg.edge_score_short_threshold
                    _al = self.sig.alpha_long_count
                    _as = self.sig.alpha_short_count
                    _amin = self.sig.cfg.min_alpha_count
                    _r_el = _edge / _eth_l if _eth_l > 0 else 0.0
                    _r_es = (-_edge) / _eth_s if _eth_s > 0 else 0.0
                    _r_al = _al / _amin if _amin > 0 else 0.0
                    _r_as = _as / _amin if _amin > 0 else 0.0
                    _ml = max(_r_el, _r_al)
                    _ms = max(_r_es, _r_as)
                    if max(_ml, _ms) >= 0.6:
                        if _ml >= _ms:
                            _w = "alpha" if _r_al < _r_el else "edge"
                            self._vlog(
                                f"[NEAR♦多] edge={_edge:+.2f}/{_eth_l:.2f}({_r_el*100:.0f}%) "
                                f"alpha={_al}/{_amin}({_r_al*100:.0f}%) → 瓶颈:{_w}"
                            )
                        else:
                            _w = "alpha" if _r_as < _r_es else "edge"
                            self._vlog(
                                f"[NEAR♦空] edge={_edge:+.2f}/{-_eth_s:.2f}({_r_es*100:.0f}%) "
                                f"alpha={_as}/{_amin}({_r_as*100:.0f}%) → 瓶颈:{_w}"
                            )

            self._throttled_put_event(tick_dt)

        def on_timer(self) -> None:
            if not getattr(self, "trading", True):
                return
            if self.max_tick_stale_seconds <= 0:
                return
            if self._last_recv_time is None:
                return

            now = datetime.now()
            stale_sec = (now - self._last_recv_time).total_seconds()
            if stale_sec <= float(self.max_tick_stale_seconds):
                return

            if not self._tick_stale_state:
                self._tick_stale_state = True
                self.write_log(f"[WATCHDOG] feed stale>{self.max_tick_stale_seconds:.1f}s")

            if self._active_orderids:
                if not self._cancel_waiting_ack:
                    self._rl_cancel_all(now)
                    self._cancel_waiting_ack = True
                if self.stale_feed_force_flatten and self.pos != 0:
                    self._stale_force_flatten_pending = True
                return

            if self.stale_feed_force_flatten and self.pos != 0 and self._last_snap is not None:
                self._stale_force_flatten_pending = True

            if self._stale_force_flatten_pending and self._last_snap is not None:
                self._force_flatten(self._last_snap, now, reason="feed_stale")

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
                self._maker_escape_pending = False
                self._pending_entry_fair_price = 0.0
                self._pending_entry_alpha = 0.0
                self.write_log(
                    f"FILL OPEN {self._entry_direction} "
                    f"avg={self._entry_fill_price:.1f} vol={self._entry_fill_volume:.0f}"
                )
                # Auto-TP is placed only when the entry order flow is fully settled.
                self._try_place_auto_tp(trade_dt)

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
                    # v4/v5: record trade first, then emit enhanced close summary
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
                    # Enhanced close summary (v5: emoji + hold time + running stats)
                    hold_sec = (trade_dt - entry_t).total_seconds()
                    _stats = self._pnl_tracker.stats()
                    _wr_str = f"{_stats.get('win_rate', 0.0) * 100:.1f}%" if _stats else "—"
                    _pf_str = f"{_stats.get('profit_factor', 0.0):.2f}" if _stats else "—"
                    _icon = "🟢盈" if net_pnl > 0 else ("🔴亏" if net_pnl < 0 else "⚪平")
                    _reason = self._pending_exit_reason or "?"
                    self.write_log(
                        f"[平仓{_icon}] {self._entry_direction} {_reason} "
                        f"entry={self._entry_fill_price:.1f}→exit={close_avg:.1f} "
                        f"{net_pnl:+.0f}¥ 持{hold_sec:.0f}s | "
                        f"今日:{self.daily_pnl:+.0f}¥({self.daily_trades}笔) "
                        f"胜率:{_wr_str} PF:{_pf_str}"
                    )
                    self._check_daily_risk_halt(trade_dt)
                    self._reset_position_state()
                else:
                    # Partial close: keep monitoring remaining position.
                    self._order_state = OrderState.OPEN
                    self._state_since = trade_dt
                    self.order_state = OrderState.OPEN.value

            self.put_event()

        def on_order(self, order) -> None:
            status = getattr(order, "status", None)
            vt_orderid = getattr(order, "vt_orderid", "")

            try:
                is_active = status in (Status.SUBMITTING, Status.NOTTRADED, Status.PARTTRADED)
                cancelled = status in (Status.CANCELLED, Status.REJECTED)
            except Exception:
                s = str(status).lower()
                is_active = any(x in s for x in ("submit", "nottraded", "parttraded", "partial", "pending"))
                cancelled = ("cancel" in s) or ("reject" in s)

            # Remove non-active orders (including all-traded) to avoid stale blockers.
            if vt_orderid in self._active_orderids and not is_active:
                self._active_orderids.remove(vt_orderid)

            if not self._active_orderids:
                self._cancel_waiting_ack = False
                if self._order_state == OrderState.PENDING_OPEN:
                    if cancelled and self._entry_fill_volume <= 0:
                        # Safe MAKER escape: retry only after cancel acknowledgement.
                        if self._maker_escape_pending and self._retry_maker_escape_as_taker(datetime.now()):
                            pass
                        else:
                            self.write_log(f"Open order cancelled: {vt_orderid}")
                            self._reset_position_state()
                    elif self._entry_fill_volume > 0:
                        # Order finished and at least partial fill exists -> open position state.
                        self._order_state = OrderState.OPEN
                        self._state_since = datetime.now()
                        self.order_state = OrderState.OPEN.value
                        self._maker_escape_pending = False
                        self._pending_open_orphan_since = None
                        self._stale_force_flatten_pending = False
                        self._try_place_auto_tp(datetime.now())
                elif self._order_state == OrderState.PENDING_CLOSE:
                    if cancelled:
                        # Exit order cancelled -> retry from OPEN state.
                        self.write_log(
                            f"Close order cancelled, will retry: {vt_orderid}"
                        )
                        was_tp = (self._pending_exit_reason == "TP")
                        self._order_state = OrderState.OPEN
                        self._state_since = datetime.now()
                        self.order_state = OrderState.OPEN.value
                        if was_tp:
                            self._try_place_auto_tp(datetime.now())
                elif self._order_state == OrderState.OPEN:
                    # Fallback for callback-order race:
                    # when open fill callback comes before final order-ack cleanup,
                    # auto TP can be skipped because _active_orderids was still non-empty.
                    # If order queue is now clear and we are genuinely in OPEN state,
                    # attempt auto TP once.
                    if self._entry_fill_volume > 0 and not self._pending_exit_reason:
                        self._try_place_auto_tp(datetime.now())

            self.put_event()

        def on_bar(self, bar) -> None:
            pass

        def on_stop_order(self, stop_order) -> None:
            self.put_event()

        # ------------------------------------------------------------------
        # Auto-TP helper 遯ｶ繝ｻcalled immediately after open fill in on_trade()
        # ------------------------------------------------------------------

        def _try_place_auto_tp(self, now: datetime) -> None:
            """Place auto-TP only when entry order flow is fully settled."""
            if not self.auto_tp_on_fill:
                return
            if self._order_state != OrderState.OPEN:
                return
            if self._entry_fill_volume <= 0:
                return
            if self._active_orderids:
                return
            self._place_auto_tp(now)

        def _schedule_auto_tp_retry(self, now: datetime, price: float, volume: int) -> None:
            if self.auto_tp_retry_max <= 0:
                self._clear_pending_auto_tp()
                return
            self._pending_tp_retries += 1
            if self._pending_tp_retries > self.auto_tp_retry_max:
                _slow_max = self.auto_tp_retry_max - self.auto_tp_fast_retries
                self.write_log(
                    f"[WARN] AUTO TP retry exhausted "
                    f"快速×{self.auto_tp_fast_retries}+慢速×{_slow_max}"
                )
                self._clear_pending_auto_tp()
                return
            self._pending_tp_submit = True
            self._pending_tp_price = float(price)
            self._pending_tp_volume = max(0, int(volume))
            if self._pending_tp_retries <= self.auto_tp_fast_retries:
                delay = max(0.05, self.auto_tp_retry_delay_sec)
                _phase = f"快速{self._pending_tp_retries}/{self.auto_tp_fast_retries}"
            else:
                delay = self.auto_tp_slow_retry_sec
                _sn = self._pending_tp_retries - self.auto_tp_fast_retries
                _sm = self.auto_tp_retry_max - self.auto_tp_fast_retries
                _phase = f"慢速{_sn}/{_sm}"
            self._pending_tp_after = now + timedelta(seconds=delay)
            self.write_log(
                f"[WARN] AUTO TP pending {_phase} after {delay:.1f}s "
                f"px={price:.1f} vol={volume}"
            )

        def _try_retry_pending_auto_tp(self, now: datetime) -> None:
            if not self._pending_tp_submit:
                return
            if not self.auto_tp_on_fill:
                self._clear_pending_auto_tp()
                return
            if self._order_state != OrderState.OPEN:
                return
            if self.pos == 0:
                self._clear_pending_auto_tp()
                return
            if self._active_orderids:
                return
            if self._pending_tp_after and now < self._pending_tp_after:
                return

            vol = int(round(abs(self._entry_fill_volume)))
            if vol <= 0:
                vol = int(abs(self._pending_tp_volume))
            if vol <= 0:
                self._clear_pending_auto_tp()
                return

            price = float(self._pending_tp_price)
            if price <= 0:
                pt = max(self.price_tick, 1e-9)
                if self._entry_direction == "LONG":
                    price = self._entry_fill_price + self.auto_tp_ticks * pt
                elif self._entry_direction == "SHORT":
                    price = self._entry_fill_price - self.auto_tp_ticks * pt
                else:
                    self._clear_pending_auto_tp()
                    return

            ids: List[str] = []
            if self._entry_direction == "LONG":
                ids = self._rl_sell(price, vol, now)
            elif self._entry_direction == "SHORT":
                ids = self._rl_cover(price, vol, now)
            else:
                self._clear_pending_auto_tp()
                return

            if ids:
                self._active_orderids = list(ids)
                self._order_state = OrderState.PENDING_CLOSE
                self._state_since = now
                self.order_state = OrderState.PENDING_CLOSE.value
                self._pending_exit_reason = "TP"
                _n = self._pending_tp_retries
                _phase = (
                    f"快速{_n}/{self.auto_tp_fast_retries}"
                    if _n <= self.auto_tp_fast_retries
                    else f"慢速{_n - self.auto_tp_fast_retries}/{self.auto_tp_retry_max - self.auto_tp_fast_retries}"
                )
                self._clear_pending_auto_tp()
                self.write_log(f"AUTO TP retry success [{_phase}] @ {price:.1f} x{vol}")
                return

            self._schedule_auto_tp_retry(now, price, vol)

        def _clear_pending_auto_tp(self) -> None:
            self._pending_tp_submit = False
            self._pending_tp_price = 0.0
            self._pending_tp_volume = 0
            self._pending_tp_after = None
            self._pending_tp_retries = 0

        def _place_auto_tp(self, now: datetime) -> None:
            """Place TP close order immediately from fill, then fallback to retry queue."""
            pt = max(self.price_tick, 1e-9)
            vol = int(round(abs(self._entry_fill_volume)))
            if vol <= 0:
                return

            ids: List[str] = []
            tp_price = 0.0
            if self._entry_direction == "LONG":
                tp_price = self._entry_fill_price + self.auto_tp_ticks * pt
                ids = self._rl_sell(tp_price, vol, now)
                price_str = f"{tp_price:.1f}"
            elif self._entry_direction == "SHORT":
                tp_price = self._entry_fill_price - self.auto_tp_ticks * pt
                ids = self._rl_cover(tp_price, vol, now)
                price_str = f"{tp_price:.1f}"
            else:
                return

            if ids:
                self._active_orderids = list(ids)
                self._order_state = OrderState.PENDING_CLOSE
                self._state_since = now
                self.order_state = OrderState.PENDING_CLOSE.value
                self._pending_exit_reason = "TP"
                self._clear_pending_auto_tp()
                self.write_log(
                    f"AUTO TP @ {price_str} x{vol} "
                    f"(fill={self._entry_fill_price:.1f} + {self.auto_tp_ticks:.1f}tick)"
                )
                return

            self._schedule_auto_tp_retry(now, tp_price, vol)

        # ------------------------------------------------------------------
        # Entry helpers
        # ------------------------------------------------------------------

        def _build_entry_plan(self, snap: TickSnapshot, direction: int) -> Tuple[str, float, float, float]:
            """Build entry mode, order price, fair value and reservation price."""
            pt = max(self.price_tick, 1e-9)
            fair_price, reservation = self._fair_and_reservation(snap)
            strict_mode = self.strict_entry_advantage or self.force_maker_mode
            if direction > 0:
                mode = "MAKER" if strict_mode else self.sig.fill_mode_long
                if mode == "MAKER":
                    price = snap.bid1
                    if self._market_state == MarketState.QUEUE and snap.bid_vol1 < max(1, self.queue_min_top_qty):
                        price = max(snap.bid1 - pt, pt)
                    elif reservation <= snap.bid1 - pt:
                        price = max(snap.bid1 - pt, pt)
                    elif (not self.strict_entry_advantage and abs(self.sig.edge_score) >= self.strong_signal_threshold
                          and snap.spread_ticks >= 2.0):
                        price = min(snap.bid1 + pt, snap.ask1 - pt)
                else:
                    price = snap.ask1
            else:
                mode = "MAKER" if strict_mode else self.sig.fill_mode_short
                if mode == "MAKER":
                    price = snap.ask1
                    if self._market_state == MarketState.QUEUE and snap.ask_vol1 < max(1, self.queue_min_top_qty):
                        price = snap.ask1 + pt
                    elif reservation >= snap.ask1 + pt:
                        price = snap.ask1 + pt
                    elif (not self.strict_entry_advantage and abs(self.sig.edge_score) >= self.strong_signal_threshold
                          and snap.spread_ticks >= 2.0):
                        price = max(snap.ask1 - pt, snap.bid1 + pt)
                else:
                    price = snap.bid1
            return mode, self._align_price(price), fair_price, reservation

        def _enter_long(self, snap: TickSnapshot, tick_dt: datetime, volume: int) -> None:
            mode, order_price, fair_price, _reservation = self._build_entry_plan(snap, +1)
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
                self._cancel_waiting_ack = False
                self._pending_open_orphan_since = None
                self._pending_entry_fair_price = fair_price
                self._pending_entry_alpha = self.sig.edge_score
                self._clear_pending_auto_tp()
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
            mode, order_price, fair_price, _reservation = self._build_entry_plan(snap, -1)
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
                self._cancel_waiting_ack = False
                self._pending_open_orphan_since = None
                self._pending_entry_fair_price = fair_price
                self._pending_entry_alpha = self.sig.edge_score
                self._clear_pending_auto_tp()
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
            extra = max(0.0, float(self.taker_exit_extra_ticks)) * pt
            elapsed = (tick_dt - self._entry_time).total_seconds()

            # ----------------------------------------------------------------
            # C0: JPY per-trade emergency stop (highest priority, overrides hold_if_loss)
            # Prevents unlimited loss when hold_if_loss=True or loss_ticks is wide.
            # Inspired by max_loss_per_trade in kabu_micro_edge_pro strategy.
            # ----------------------------------------------------------------
            if self.max_loss_per_trade_jpy > 0 and self._entry_fill_price > 0:
                direction_sign = +1 if self._entry_direction == "LONG" else -1
                ref_price = snap.bid1 if self._entry_direction == "LONG" else snap.ask1
                unrealized_jpy = (
                    direction_sign
                    * (ref_price - self._entry_fill_price)
                    * abs(self._entry_fill_volume)
                )
                if unrealized_jpy < -self.max_loss_per_trade_jpy:
                    self.write_log(
                        f"🚨 [EMER_LOSS] 单笔亏损 {unrealized_jpy:.0f}¥ "
                        f"< -{self.max_loss_per_trade_jpy:.0f}¥ → 紧急平仓"
                    )
                    self._pending_exit_reason = "EMER_LOSS"
                    self._clear_pending_auto_tp()
                    exit_price = snap.bid1 if self._entry_direction == "LONG" else snap.ask1
                    vol = int(round(abs(self._entry_fill_volume)))
                    if self._entry_direction == "LONG":
                        ids = self._rl_sell(exit_price, vol, tick_dt)
                    else:
                        ids = self._rl_cover(exit_price, vol, tick_dt)
                    if ids:
                        self._active_orderids = list(ids)
                        self._order_state = OrderState.PENDING_CLOSE
                        self._state_since = tick_dt
                        self.order_state = OrderState.PENDING_CLOSE.value
                    return

            # ----------------------------------------------------------------
            # C1: Fast loss circuit breaker 遯ｶ繝ｻaggressive immediate exit if
            # the position loses fast_loss_ticks within fast_loss_sec of entry.
            # Indicates we entered into a strongly directional move against us.
            # Disabled when hold_if_loss=True (never cut on fast adverse move).
            # ----------------------------------------------------------------
            if not self.hold_if_loss and elapsed <= self.fast_loss_sec:
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
                            f"[FAST LOSS] {fast_pnl:.1f}ticks in {elapsed:.1f}s 遶翫・aggressive exit"
                        )
                    return

            # ----------------------------------------------------------------
            # Normal exit logic with two-tier pricing (B2):
            #   遯ｶ・｢ Urgent exits (stop-loss / timeout) 遶翫・aggressive price (bid1 / ask1)
            #   遯ｶ・｢ Take-profit / signal exits         遶翫・gentle limit (bid1-pt / ask1+pt)
            # ----------------------------------------------------------------
            if self.pos > 0:
                pnl_ticks = (snap.bid1 - self._entry_fill_price) / pt
                self._max_favorable_ticks = max(self._max_favorable_ticks, pnl_ticks)
                trailing_hit = (
                    self._max_favorable_ticks >= self.trailing_activate_ticks
                    and (self._max_favorable_ticks - pnl_ticks) >= self.trailing_drawdown_ticks
                )
                loss_hold = self.hold_if_loss and pnl_ticks < 0
                sl_triggered = (pnl_ticks <= -self.loss_ticks) and not self.hold_if_loss
                timeout_triggered = (elapsed >= self.max_hold_seconds) and not loss_hold
                signal_exit = self.sig.should_exit_long() and not loss_hold
                trailing_exit = trailing_hit and not loss_hold
                is_urgent = sl_triggered or timeout_triggered
                should_exit = (
                    signal_exit
                    or pnl_ticks >= self.profit_ticks
                    or is_urgent
                    or trailing_exit
                )
                if should_exit:
                    if sl_triggered:
                        self._pending_exit_reason = "SL"
                    elif timeout_triggered:
                        self._pending_exit_reason = "TIMEOUT"
                    elif pnl_ticks >= self.profit_ticks:
                        self._pending_exit_reason = "TP"
                    elif trailing_exit:
                        self._pending_exit_reason = "TRAILING"
                    else:
                        self._pending_exit_reason = "SIGNAL_FLIP"
                    self._clear_pending_auto_tp()
                    if is_urgent:
                        # Aggressive taker: sell at best bid immediately
                        exit_price = snap.bid1
                    elif self._entry_mode == "MAKER":
                        # MAKER exit: post sell at ask 遶翫・capture spread (鬯ｮ蛟・ｽｻ・ｷ陷ｿ閧ｴ・ｸ繝ｻ
                        exit_price = snap.ask1
                    else:
                        # TAKER exit fallback: hit best bid (optional extra slip configurable)
                        exit_price = snap.bid1 - extra
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
                loss_hold = self.hold_if_loss and pnl_ticks < 0
                sl_triggered = (pnl_ticks <= -self.loss_ticks) and not self.hold_if_loss
                timeout_triggered = (elapsed >= self.max_hold_seconds) and not loss_hold
                signal_exit = self.sig.should_exit_short() and not loss_hold
                trailing_exit = trailing_hit and not loss_hold
                is_urgent = sl_triggered or timeout_triggered
                should_exit = (
                    signal_exit
                    or pnl_ticks >= self.profit_ticks
                    or is_urgent
                    or trailing_exit
                )
                if should_exit:
                    if sl_triggered:
                        self._pending_exit_reason = "SL"
                    elif timeout_triggered:
                        self._pending_exit_reason = "TIMEOUT"
                    elif pnl_ticks >= self.profit_ticks:
                        self._pending_exit_reason = "TP"
                    elif trailing_exit:
                        self._pending_exit_reason = "TRAILING"
                    else:
                        self._pending_exit_reason = "SIGNAL_FLIP"
                    self._clear_pending_auto_tp()
                    if is_urgent:
                        # Aggressive taker: cover at best ask immediately
                        exit_price = snap.ask1
                    elif self._entry_mode == "MAKER":
                        # MAKER exit: post buy at bid 遶翫・capture spread (闖ｴ諠ｹ・ｻ・ｷ陷ｿ閧ｴ・ｸ繝ｻ
                        exit_price = snap.bid1
                    else:
                        # TAKER exit fallback: lift best ask (optional extra slip configurable)
                        exit_price = snap.ask1 + extra
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
            self._clear_pending_auto_tp()
            pt = max(self.price_tick, 1e-9)
            extra = max(0.0, float(self.taker_exit_extra_ticks)) * pt
            if self.pos > 0:
                ids = self._rl_sell(max(snap.bid1 - extra, pt), abs(self.pos), tick_dt)
            elif self.pos < 0:
                ids = self._rl_cover(snap.ask1 + extra, abs(self.pos), tick_dt)
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
            """One-way commission: max(min_fee, rate ・・・trade_value).

            IMPORTANT: verify commission_rate and commission_min against your
            actual kabu account plan (陞ｳ螟撰ｽ｡繝ｻvs 鬩幢ｽｽ陟趣ｽｦ) before live trading.
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
                self._signal_pending_dir = 0
                self._signal_pending_count = 0
                self._signal_pending_dt = None
                self._market_event_ts_ns.clear()
                self._market_prev_mid = 0.0
                self._market_state = MarketState.NORMAL
                self._market_state_reason = "new_day"
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

            # Recovery guard: sometimes order callbacks leave PENDING_OPEN with
            # neither active order IDs nor any fill. Give it a short grace window,
            # then reset to avoid state-machine deadlock.
            if (
                self._order_state == OrderState.PENDING_OPEN
                and not self._active_orderids
                and self._entry_fill_volume <= 0
            ):
                if self._pending_open_orphan_since is None:
                    self._pending_open_orphan_since = now
                elif (
                    now - self._pending_open_orphan_since
                ).total_seconds() >= self._pending_open_orphan_grace_sec:
                    self.write_log(
                        "[RECOVER] orphan PENDING_OPEN (no active order/fill) -> reset"
                    )
                    self._reset_position_state()
                return
            else:
                self._pending_open_orphan_since = None

            # A cancel request has already been sent; wait for acknowledgement.
            if self._cancel_waiting_ack:
                if elapsed < 1.0:
                    return
                # Ack may be delayed/lost in some gateway paths; allow retry.
                self._cancel_waiting_ack = False

            # --- Smart cancel: if PENDING_OPEN and signal has flipped, cancel immediately ---
            # Avoids sitting in a stale limit order waiting for timeout when the signal is gone.
            if (
                self.smart_cancel_on_flip
                and self._order_state == OrderState.PENDING_OPEN
                and not self._maker_escape_pending
            ):
                signal_gone = (
                    (self._entry_direction == "LONG" and not self.sig.can_open_long())
                    or (self._entry_direction == "SHORT" and not self.sig.can_open_short())
                )
                alpha_flip = (
                    self._pending_entry_alpha != 0.0
                    and (self.sig.edge_score * self._pending_entry_alpha) < 0
                    and abs(self.sig.edge_score) >= max(0.5, self.strong_signal_threshold * 0.5)
                )
                fair_drift = False
                fair_drift_ticks = 0.0
                if self._pending_entry_fair_price > 0:
                    fair_now, _ = self._fair_and_reservation(snap)
                    fair_drift_ticks = abs(fair_now - self._pending_entry_fair_price) / max(self.price_tick, 1e-9)
                    fair_drift = fair_drift_ticks >= self.max_fair_drift_ticks
                if signal_gone or alpha_flip or fair_drift:
                    reason = "signal_flip" if signal_gone else ("alpha_flip" if alpha_flip else "fair_drift")
                    self.write_log(
                        f"[SMART CANCEL] {reason} during PENDING_OPEN ({self._entry_direction})"
                        + (f" drift={fair_drift_ticks:.2f}t" if fair_drift else "")
                    )
                    sent = self._rl_cancel_all(now)
                    if sent:
                        # Do NOT set _maker_escape_pending: signal is gone, just abort on ack.
                        self._cancel_waiting_ack = True
                        self._state_since = now
                    return

            # --- MAKER escape: runs before the full open timeout ---
            if (
                self._order_state == OrderState.PENDING_OPEN
                and self._entry_mode == "MAKER"
                and elapsed >= self.maker_escape_timeout_sec
                and elapsed < self.open_order_timeout_sec
            ):
                self.write_log(
                    f"MAKER escape {elapsed:.2f}s -> cancel and wait ack for TAKER retry"
                )
                sent = self._rl_cancel_all(now)
                if sent:
                    # Retry will be attempted in on_order after cancel acknowledgement.
                    self._cancel_waiting_ack = True
                    self._maker_escape_pending = True
                    self._state_since = now
                return
            # --- Full open timeout ---
            if self._order_state == OrderState.PENDING_OPEN and elapsed >= self.open_order_timeout_sec:
                self.write_log(
                    f"OPEN timeout {elapsed:.2f}s -> send cancel and wait ack"
                )
                sent = self._rl_cancel_all(now)
                if sent:
                    self._cancel_waiting_ack = True
                    self._state_since = now
                return

            # --- TP order exemption: when hold_if_loss=True, the TP limit order must
            # keep resting until filled or until tp_order_timeout_sec expires.
            # This prevents the 2-5s close_order_timeout from cancelling valid TP orders. ---
            if (
                self._order_state == OrderState.PENDING_CLOSE
                and self._pending_exit_reason == "TP"
                and self.hold_if_loss
            ):
                tp_timeout = getattr(self, "tp_order_timeout_sec", 300.0)
                if tp_timeout <= 0 or elapsed < tp_timeout:
                    return  # let TP limit order rest; do not cancel

            # --- Close timeout: aggressive TAKER retry (bid1 / ask1, no offset) ---
            if self._order_state == OrderState.PENDING_CLOSE and elapsed >= self.close_order_timeout_sec:
                self.write_log(
                    f"CLOSE timeout {elapsed:.2f}s -> send cancel and retry after ack"
                )
                sent = self._rl_cancel_all(now)
                if sent:
                    self._cancel_waiting_ack = True
                    # Move back to OPEN and wait for cancel update before new close.
                    self._order_state = OrderState.OPEN
                    self.order_state = OrderState.OPEN.value
                    self._state_since = now
                return

        def _retry_maker_escape_as_taker(self, now: datetime) -> bool:
            """Retry a cancelled MAKER entry as TAKER after cancel acknowledgement."""
            if not self._maker_escape_pending:
                return False

            # Consume the pending flag even if we decide not to retry.
            self._maker_escape_pending = False
            if self.strict_entry_advantage or self.force_maker_mode:
                return False
            snap = self._last_snap
            if snap is None:
                return False

            vol = max(self.min_trade_volume, self._entry_volume or self.min_trade_volume)
            lot = max(1, self.lot_size)
            vol = (vol // lot) * lot
            if vol <= 0:
                vol = lot

            ids = []
            if self._entry_direction == "LONG" and self.sig.can_open_long():
                ids = self._rl_buy(snap.ask1, vol, now)
            elif self._entry_direction == "SHORT" and self.sig.can_open_short():
                ids = self._rl_short(snap.bid1, vol, now)

            if not ids:
                return False

            self._active_orderids = list(ids)
            self._entry_mode = "TAKER"
            self._order_state = OrderState.PENDING_OPEN
            self.order_state = OrderState.PENDING_OPEN.value
            self._state_since = now
            self._last_entry_attempt_dt = now
            self._pending_open_orphan_since = None
            self._cancel_waiting_ack = False
            fair_price, _ = self._fair_and_reservation(snap)
            self._pending_entry_fair_price = fair_price
            self._pending_entry_alpha = self.sig.edge_score
            self.write_log(
                f"MAKER escape retry -> TAKER {self._entry_direction} @ "
                f"{(snap.ask1 if self._entry_direction == 'LONG' else snap.bid1):.1f} x{vol}"
            )
            return True

        def _update_market_state(self, snap: TickSnapshot, tick_dt: datetime) -> None:
            """Detect NORMAL/QUEUE/ABNORMAL state from spread/stale/rate/jump."""
            now_ns = int(datetime.now().timestamp() * 1_000_000_000)
            # Use local receive clock for event-rate estimation; quote timestamps
            # may be second-level granularity and would inflate burst detection.
            event_ts_ns = now_ns
            quote_ts_ns = int(tick_dt.timestamp() * 1_000_000_000) if isinstance(tick_dt, datetime) else now_ns
            if quote_ts_ns <= 0:
                quote_ts_ns = now_ns

            self._market_event_ts_ns.append(event_ts_ns)
            window_ns = max(250, int(self.state_window_ms)) * 1_000_000
            while self._market_event_ts_ns and event_ts_ns - self._market_event_ts_ns[0] > window_ns:
                self._market_event_ts_ns.popleft()

            if len(self._market_event_ts_ns) >= 2:
                duration_ns = max(1, event_ts_ns - self._market_event_ts_ns[0])
                event_rate_hz = (len(self._market_event_ts_ns) - 1) * 1_000_000_000 / duration_ns
            else:
                event_rate_hz = 0.0

            pt = max(self.price_tick, 1e-9)
            jump_ticks = 0.0
            if self._market_prev_mid > 0:
                jump_ticks = abs(snap.mid - self._market_prev_mid) / pt
            self._market_prev_mid = snap.mid

            # Guard against timezone mismatch in quote timestamp parsing.
            if abs(now_ns - quote_ts_ns) > 12 * 3600 * 1_000_000_000:
                stale_ms = 0.0
            else:
                stale_ms = max(0.0, (now_ns - quote_ts_ns) / 1_000_000)
            spread_ticks = snap.spread_ticks
            bid_sign = (snap.bid_sign or "").strip()
            ask_sign = (snap.ask_sign or "").strip()

            state = MarketState.NORMAL
            reason = "normal_flow"
            if stale_ms > max(1, int(self.stale_quote_ms)):
                state = MarketState.ABNORMAL
                reason = "stale_quote"
            elif bid_sign in _SPECIAL_QUOTE_SIGNS or ask_sign in _SPECIAL_QUOTE_SIGNS:
                state = MarketState.ABNORMAL
                reason = "special_quote_sign"
            elif spread_ticks >= self.abnormal_max_spread_ticks:
                state = MarketState.ABNORMAL
                reason = "spread_blowout"
            elif (
                len(self._market_event_ts_ns) >= max(2, self.event_burst_min_events)
                and event_rate_hz >= self.max_event_rate_hz
            ):
                state = MarketState.ABNORMAL
                reason = "event_burst"
            elif jump_ticks >= self.jump_threshold_ticks:
                state = MarketState.ABNORMAL
                reason = "price_jump"
            elif spread_ticks <= self.queue_spread_max_ticks + 1e-9:
                state = MarketState.QUEUE
                reason = "one_tick_queue"

            if state != self._market_state or reason != self._market_state_reason:
                self._vlog(f"[MARKET] {self._market_state.value}->{state.value} reason={reason}")

            self._market_state = state
            self._market_state_reason = reason
            self.market_state = state.value
            self.market_reason = reason
            self.market_event_rate_hz = float(event_rate_hz)
            self.market_jump_ticks = float(jump_ticks)

        def _fair_and_reservation(self, snap: TickSnapshot) -> Tuple[float, float]:
            """Compute fair value and inventory-skewed reservation price."""
            pt = max(self.price_tick, 1e-9)
            fair_shift_ticks = max(
                -self.max_fair_shift_ticks,
                min(self.max_fair_shift_ticks, self.fair_value_beta * self.sig.edge_score),
            )
            fair_price = snap.mid + fair_shift_ticks * pt
            inventory_ratio = 0.0
            if self.max_position > 0:
                inventory_ratio = max(-1.0, min(1.0, self.pos / float(self.max_position)))
            reservation_price = fair_price - self.inventory_skew_ticks * inventory_ratio * pt
            return fair_price, reservation_price

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
            cutoff = getattr(self, "_no_new_entry_after_time", time(15, 24, 0))
            if tm >= cutoff:
                return False
            cool_min = max(0, int(self.close_only_before_break_min))
            if cool_min > 0:
                m_close_only = (datetime(2000, 1, 1, 11, 30, 0) - timedelta(minutes=cool_min)).time()
                a_close_only = (datetime(2000, 1, 1, 15, 30, 0) - timedelta(minutes=cool_min)).time()
                if m_close_only <= tm < time(11, 30, 0):
                    return False
                if a_close_only <= tm <= time(15, 30, 0):
                    return False
            if self.hot_open_guard:
                # Morning open guard
                if time(9, 0, 0) <= tm < time(9, 2, 0):
                    return False
                # Afternoon re-open guard
                if time(12, 30, 0) <= tm < time(12, 32, 0):
                    return False
            return True

        def _confirm_entry_signal(self, direction: int, tick_dt: datetime) -> bool:
            if (
                self._signal_pending_dt is not None
                and (tick_dt - self._signal_pending_dt).total_seconds() > self.signal_confirm_expire_sec
            ):
                self._signal_pending_dir = 0
                self._signal_pending_count = 0
                self._signal_pending_dt = None

            edge_abs = abs(self.sig.edge_score)
            if edge_abs >= self.strong_signal_threshold:
                self._signal_pending_dir = 0
                self._signal_pending_count = 0
                self._signal_pending_dt = None
                return True

            if direction != self._signal_pending_dir:
                self._signal_pending_dir = direction
                self._signal_pending_count = 1
                self._signal_pending_dt = tick_dt
                return False

            self._signal_pending_count += 1
            self._signal_pending_dt = tick_dt
            if self._signal_pending_count >= max(1, self.weak_signal_confirm_ticks):
                self._signal_pending_dir = 0
                self._signal_pending_count = 0
                self._signal_pending_dt = None
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

            # v7: ATR volatility gate — halve size when ATR exceeds 2.5× normal
            snap = self._last_snap
            if snap is not None and self._atr > 0 and snap.mid > 0:
                pt = max(self.price_tick, 1e-9)
                atr_ticks = self._atr / pt
                if atr_ticks > 2.5:
                    raw = max(self.min_trade_volume, raw // 2)

            headroom = max(0, int(self.max_position) - abs(int(self.pos)))
            raw = min(raw, headroom)
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

            mode, order_price, fair_price, _reservation = self._build_entry_plan(snap, direction)
            pt = max(self.price_tick, 1e-9)
            edge_ticks = (
                (fair_price - order_price) / pt if direction > 0 else (order_price - fair_price) / pt
            )
            if self.strict_entry_advantage and mode != "MAKER":
                return False
            if edge_ticks < self.min_entry_edge_ticks:
                return False

            # B3: Market impact check 遯ｶ繝ｻreject if our order would consume more than
            # max_impact_pct of the best-level size (signals poor liquidity for our size).
            if vol > 0 and self.max_impact_pct > 0:
                best_vol = snap.ask_vol1 if direction > 0 else snap.bid_vol1
                if best_vol > 0 and vol / best_vol > self.max_impact_pct:
                    return False

            # v7: TP coverage check — fair value must cover TP target.
            # Ensures every entry price has a positive expected exit via auto-TP.
            if self.auto_tp_on_fill and self.auto_tp_ticks > 0:
                min_coverage = self.auto_tp_ticks + getattr(self, "min_fair_tp_coverage_ticks", 0.0)
                tp_coverage_ticks = (
                    (fair_price - order_price) / pt if direction > 0
                    else (order_price - fair_price) / pt
                )
                if tp_coverage_ticks < min_coverage:
                    self._vlog(
                        f"[ENTRY REJECT] fair_tp_coverage={tp_coverage_ticks:.2f} "
                        f"< required={min_coverage:.2f}"
                    )
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
                self._vlog(
                    f"[RATE LIMIT] order_req_1s={len(self._order_req_ts)} "
                    f"limit={self.max_order_req_per_sec}"
                )
            return False

        def _mark_order_req(self, now: datetime) -> None:
            self._order_req_ts.append(now)
            self.order_req_1s = len(self._order_req_ts)

        def _align_price(self, price: float) -> float:
            """Align price to current tick size to reduce broker rejects."""
            pt = max(self.price_tick, 1e-9)
            if price <= 0:
                return pt
            return max(pt, round(price / pt) * pt)

        def _rl_buy(self, price: float, volume: int, now: datetime):
            # Open-order interval guard (separate from per-sec window)
            if self._last_open_order_ts is not None:
                if (now - self._last_open_order_ts).total_seconds() * 1000 < self.open_min_interval_ms:
                    return []
            if not self._can_send_order_req(now):
                return []
            self._last_open_order_ts = now
            self._mark_order_req(now)
            return self.buy(self._align_price(price), volume)

        def _rl_short(self, price: float, volume: int, now: datetime):
            if self._last_open_order_ts is not None:
                if (now - self._last_open_order_ts).total_seconds() * 1000 < self.open_min_interval_ms:
                    return []
            if not self._can_send_order_req(now):
                return []
            self._last_open_order_ts = now
            self._mark_order_req(now)
            return self.short(self._align_price(price), volume)

        def _rl_sell(self, price: float, volume: int, now: datetime):
            # Respect global request budget while keeping a short close interval.
            if self._last_close_order_ts is not None:
                if (now - self._last_close_order_ts).total_seconds() * 1000 < self.close_min_interval_ms:
                    return []
            if not self._can_send_order_req(now):
                return []
            self._last_close_order_ts = now
            self._mark_order_req(now)
            return self.sell(self._align_price(price), volume)

        def _rl_cover(self, price: float, volume: int, now: datetime):
            if self._last_close_order_ts is not None:
                if (now - self._last_close_order_ts).total_seconds() * 1000 < self.close_min_interval_ms:
                    return []
            if not self._can_send_order_req(now):
                return []
            self._last_close_order_ts = now
            self._mark_order_req(now)
            return self.cover(self._align_price(price), volume)

        def _rl_cancel_all(self, now: datetime) -> bool:
            # Cancel is safety-critical: do not block it by order rate limiter.
            self._trim_order_req_window(now)
            if len(self._order_req_ts) >= max(1, self.max_order_req_per_sec):
                if (
                    self._last_rate_limit_log_dt is None
                    or (now - self._last_rate_limit_log_dt).total_seconds() >= 1.0
                ):
                    self._last_rate_limit_log_dt = now
                    self.write_log(
                        f"[RATE LIMIT] cancel_all bypass "
                        f"order_req_1s={len(self._order_req_ts)} limit={self.max_order_req_per_sec}"
                    )
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
            self.market_state = self._market_state.value
            self.market_reason = self._market_state_reason
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
                sig = self.sig
                gate_str = (
                    f"g[sp/liq/vol/tm/ev/lag]="
                    f"{int(sig.g_spread_ok)}/"
                    f"{int(sig.g_liq_ok)}/"
                    f"{int(sig.g_vol_ok)}/"
                    f"{int(sig.g_time_ok)}/"
                    f"{int(sig.g_ev_ok)}/"
                    f"{int(sig.g_trade_lag_ok)}"
                )
                self.write_log(
                    f"{self.sig.summary()} | "
                    f"mkt={self._market_state.value}:{self._market_state_reason} "
                    f"{gate_str} "
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
            self._maker_escape_pending = False
            self._cancel_waiting_ack = False
            self._pending_open_orphan_since = None
            self._signal_pending_dt = None
            self._pending_entry_fair_price = 0.0
            self._pending_entry_alpha = 0.0
            self._clear_pending_auto_tp()
            self._stale_force_flatten_pending = False
