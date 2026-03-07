"""kabu_signal_stack.py — Redesigned signal stack + VeighNa strategy wrapper.

Architecture (4 layers):
  [VeighNa TickData]
      → KabuTickAdapter    (field mapping, bid/ask reversal switch, sanity check)
      → TickSnapshot       (clean L1/L2 snapshot, up to 5 price levels)
      → KabuSignalStack    (OBI / LOB-OFI / Tape-OFI / momentum / microprice tilt)
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
from datetime import datetime, time
from enum import Enum
from typing import Deque, List, Optional, Tuple

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

    def __init__(self, reverse_bid_ask: bool = False) -> None:
        self.reverse_bid_ask = reverse_bid_ask

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

        # Sanity check: both prices must be positive and ordered correctly
        if bid1 <= 0.0 or ask1 <= 0.0 or bid1 >= ask1:
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

    # Entry thresholds
    edge_score_long_threshold: float = 2.5
    edge_score_short_threshold: float = 2.5
    min_alpha_count: int = 2

    # Regime detection
    regime_window: int = 30
    noise_vol_threshold: float = 2.0
    regime_sample_interval: int = 5

    # Adapter
    reverse_bid_ask: bool = False   # Set True if gateway passes raw kabu field names

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
        self._adapter = KabuTickAdapter(config.reverse_bid_ask)
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
        return self.edge_score <= -0.5

    def should_exit_short(self) -> bool:
        return self.edge_score >= 0.5

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
        mean_ret = sum(rets) / len(rets)
        var = sum((r - mean_ret) ** 2 for r in rets) / len(rets)
        vol_ticks = math.sqrt(max(var, 0.0)) * snap.mid / max(self.pricetick, 1e-9)
        if vol_ticks >= self.cfg.noise_vol_threshold:
            self.regime = "NOISE"
        elif abs(mean_ret) > 0.00005:
            self.regime = "TREND"
        else:
            self.regime = "REVERSION"

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
        cfg = self.cfg
        self.edge_score = (
            cfg.w_microprice * self.microprice_tilt
            + cfg.w_lob_ofi  * self.lob_ofi
            + cfg.w_tape_ofi * self.tape_ofi
            + cfg.w_momentum * self.momentum
            + cfg.w_obi      * self.obi
        )
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

        author = "KabuSignalStack v2"

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

        # Daily risk limit
        max_daily_loss: float = -50_000.0

        # Logging throttle interval
        log_interval_seconds: float = 10.0

        parameters = [
            "trade_volume", "max_position",
            "enable_long", "enable_short",
            "max_spread_ticks", "min_best_volume",
            "obi_levels", "ev_gate_enabled",
            "reverse_bid_ask", "auto_pricetick",
            "commission_rate", "commission_min",
            "profit_ticks", "loss_ticks", "max_hold_seconds",
            "max_daily_loss", "log_interval_seconds",
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
        total_trades: int = 0
        last_signal: str = ""

        variables = [
            "price_tick", "order_state",
            "sig_all_gates", "sig_edge_score", "sig_regime",
            "sig_obi", "sig_alpha_long", "sig_alpha_short",
            "sig_adv_sel", "sig_fill_mode_long",
            "sig_taker_ev_long", "sig_maker_ev_long",
            "last_entry_price", "last_exit_price",
            "daily_pnl", "total_trades", "last_signal",
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

            # Risk
            self._is_trading_allowed: bool = True
            self._current_date: str = ""

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
            pt = self.get_pricetick()
            if pt and pt > 0:
                self.price_tick = float(pt)
            cfg = SignalConfig(
                max_spread_ticks=self.max_spread_ticks,
                min_best_volume=self.min_best_volume,
                obi_levels=self.obi_levels,
                ev_gate_enabled=self.ev_gate_enabled,
                reverse_bid_ask=self.reverse_bid_ask,
                auto_pricetick=self.auto_pricetick,
            )
            self.sig = KabuSignalStack(cfg, self.price_tick)
            self._reset_position_state()
            self.write_log(
                f"on_start pricetick={self.price_tick} "
                f"reverse_bid_ask={self.reverse_bid_ask}"
            )
            self.put_event()

        def on_stop(self) -> None:
            self.cancel_all()
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
            self._sync_sig_vars()
            self._check_date_reset(tick_dt)
            self._maybe_log(tick_dt)

            # 2. State machine guard — if not IDLE, only manage open position
            if self._order_state != OrderState.IDLE:
                if self._order_state == OrderState.OPEN:
                    self._manage_exit(snap, tick_dt)
                self._throttled_put_event(tick_dt)
                return

            # 3. Risk gate
            if not self._is_trading_allowed or not self.sig.all_gates_ok:
                self._throttled_put_event(tick_dt)
                return

            # 4. Entry signals
            pos_skew = self.pos / max(self.max_position, 1)
            if self.enable_long and self.sig.can_open_long(pos_skew=pos_skew):
                self._enter_long(snap, tick_dt)
            elif self.enable_short and self.sig.can_open_short(pos_skew=pos_skew):
                self._enter_short(snap, tick_dt)

            self._throttled_put_event(tick_dt)

        # ------------------------------------------------------------------
        # Order / trade callbacks
        # ------------------------------------------------------------------

        def on_trade(self, trade) -> None:
            if self._order_state == OrderState.PENDING_OPEN:
                # Opening fill
                self._entry_fill_price = float(trade.price)
                self._entry_fill_volume = float(trade.volume)
                fill_dt = getattr(trade, "datetime", None)
                if isinstance(fill_dt, datetime):
                    self._entry_time = fill_dt
                self._order_state = OrderState.OPEN
                self.last_entry_price = self._entry_fill_price
                self.order_state = OrderState.OPEN.value
                self.write_log(
                    f"FILL OPEN {self._entry_direction} "
                    f"price={self._entry_fill_price:.1f} "
                    f"vol={self._entry_fill_volume:.0f}"
                )

            elif self._order_state == OrderState.PENDING_CLOSE:
                # Closing fill — compute net PnL from actual fill prices
                exit_price = float(trade.price)
                exit_volume = float(trade.volume)
                direction_sign = +1 if self._entry_direction == "LONG" else -1
                gross_pnl = (
                    direction_sign
                    * (exit_price - self._entry_fill_price)
                    * exit_volume
                )
                commission = (
                    self._calc_commission(exit_price, exit_volume)
                    + self._calc_commission(self._entry_fill_price, self._entry_fill_volume)
                )
                net_pnl = gross_pnl - commission
                self.daily_pnl += net_pnl
                self.total_trades += 1
                self.last_exit_price = exit_price
                self.write_log(
                    f"FILL CLOSE {self._entry_direction} "
                    f"exit={exit_price:.1f} "
                    f"pnl={net_pnl:+.0f}JPY "
                    f"daily={self.daily_pnl:.0f}JPY"
                )
                trade_dt = getattr(trade, "datetime", None)
                self._check_daily_risk_halt(trade_dt)
                self._reset_position_state()

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
                        self.order_state = OrderState.OPEN.value

            self.put_event()

        def on_bar(self, bar) -> None:
            pass

        def on_stop_order(self, stop_order) -> None:
            self.put_event()

        # ------------------------------------------------------------------
        # Entry helpers
        # ------------------------------------------------------------------

        def _enter_long(self, snap: TickSnapshot, tick_dt: datetime) -> None:
            # MAKER: post limit at bid (may not fill); TAKER: lift the ask immediately
            order_price = snap.bid1 if self.sig.fill_mode_long == "MAKER" else snap.ask1
            ids = self.buy(order_price, self.trade_volume)
            if ids:
                self._active_orderids = list(ids)
                self._order_state = OrderState.PENDING_OPEN
                self._entry_direction = "LONG"
                self._entry_time = tick_dt
                self.order_state = OrderState.PENDING_OPEN.value
                self.last_signal = (
                    f"LONG/{self.sig.fill_mode_long} "
                    f"edge={self.sig.edge_score:+.2f}"
                )
                self.write_log(
                    f"OPEN LONG @ {order_price:.1f} x{self.trade_volume} | "
                    f"{self.last_signal}"
                )

        def _enter_short(self, snap: TickSnapshot, tick_dt: datetime) -> None:
            # MAKER: post limit at ask; TAKER: hit the bid immediately
            order_price = snap.ask1 if self.sig.fill_mode_short == "MAKER" else snap.bid1
            ids = self.short(order_price, self.trade_volume)
            if ids:
                self._active_orderids = list(ids)
                self._order_state = OrderState.PENDING_OPEN
                self._entry_direction = "SHORT"
                self._entry_time = tick_dt
                self.order_state = OrderState.PENDING_OPEN.value
                self.last_signal = (
                    f"SHORT/{self.sig.fill_mode_short} "
                    f"edge={self.sig.edge_score:+.2f}"
                )
                self.write_log(
                    f"OPEN SHORT @ {order_price:.1f} x{self.trade_volume} | "
                    f"{self.last_signal}"
                )

        # ------------------------------------------------------------------
        # Exit management
        # ------------------------------------------------------------------

        def _manage_exit(self, snap: TickSnapshot, tick_dt: datetime) -> None:
            if self._order_state != OrderState.OPEN or self._entry_time is None:
                return

            pt = max(self.price_tick, 1e-9)
            elapsed = (tick_dt - self._entry_time).total_seconds()

            if self.pos > 0:
                pnl_ticks = (snap.bid1 - self._entry_fill_price) / pt
                should_exit = (
                    self.sig.should_exit_long()         # Signal reversal
                    or pnl_ticks >= self.profit_ticks   # Take-profit
                    or pnl_ticks <= -self.loss_ticks    # Stop-loss
                    or elapsed >= self.max_hold_seconds  # Timeout (unconditional)
                )
                if should_exit:
                    ids = self.sell(snap.bid1 - pt, abs(self.pos))
                    if ids:
                        self._active_orderids = list(ids)
                        self._order_state = OrderState.PENDING_CLOSE
                        self.order_state = OrderState.PENDING_CLOSE.value

            elif self.pos < 0:
                pnl_ticks = (self._entry_fill_price - snap.ask1) / pt
                should_exit = (
                    self.sig.should_exit_short()
                    or pnl_ticks >= self.profit_ticks
                    or pnl_ticks <= -self.loss_ticks
                    or elapsed >= self.max_hold_seconds  # Timeout (unconditional)
                )
                if should_exit:
                    ids = self.cover(snap.ask1 + pt, abs(self.pos))
                    if ids:
                        self._active_orderids = list(ids)
                        self._order_state = OrderState.PENDING_CLOSE
                        self.order_state = OrderState.PENDING_CLOSE.value

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
            if self.daily_pnl < self.max_daily_loss:
                self._is_trading_allowed = False
                now_dt = trade_dt if isinstance(trade_dt, datetime) else datetime.now()
                self.sig.trigger_event_cooldown(now_dt)
                self.write_log(
                    f"[RISK HALT] daily_pnl={self.daily_pnl:.0f}JPY "
                    f"< limit={self.max_daily_loss:.0f}JPY"
                )

        def _check_date_reset(self, dt: datetime) -> None:
            date_str = dt.strftime("%Y%m%d")
            if date_str != self._current_date:
                self._current_date = date_str
                self.daily_pnl = 0.0
                self._is_trading_allowed = True
                self.write_log(f"New trading day: {date_str}")

        # ------------------------------------------------------------------
        # UI sync helpers
        # ------------------------------------------------------------------

        def _sync_sig_vars(self) -> None:
            for k, v in self.sig.get_status_dict().items():
                if hasattr(self, k):
                    setattr(self, k, v)
            self.order_state = self._order_state.value

        def _maybe_log(self, dt: datetime) -> None:
            if (
                self._last_log_dt is None
                or (dt - self._last_log_dt).total_seconds() >= self.log_interval_seconds
            ):
                self._last_log_dt = dt
                self.write_log(
                    f"{self.sig.summary()} | "
                    f"pnl={self.daily_pnl:.0f}JPY trades={self.total_trades}"
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
            self.order_state = OrderState.IDLE.value
