# -*- coding: utf-8 -*-
"""
kabu_hft_vnpy_strategy.py
Self-contained VeighNa CtaTemplate strategy ported from kabu_hft_new.
No dependency on the kabu_hft package — all signal/state logic is inlined.

Features
--------
  - 5-signal microstructure engine: OBI / LOB-OFI / Tape-OFI / Micro-momentum / Microprice-tilt
  - Online z-score normalization (rolling window, O(1) update)
  - MarketState detection: NORMAL / QUEUE / ABNORMAL
  - Fair value + inventory skew pricing (ported from kabu_hft_new HFTStrategy)
  - QUEUE mode: queue at best bid/ask when spread <= 1 tick
  - Requote budget: sliding-window cap (max_requotes_per_minute)
  - MAKER / TAKER mode selection based on signal strength
  - auto_tp_on_fill: limit TP placed immediately on open fill
  - hold_if_loss: suppress SL/signal-flip exits
  - max_loss_per_trade_jpy: hard JPY floor (EMER_LOSS, overrides hold_if_loss)
  - verbose_log / _vlog(): gate debug logs behind flag
  - PnL tracker: win-rate / profit-factor / avg-hold per session
  - Periodic pricetick verification (re-read contract every 30 s)

kabu Station bid/ask convention
--------------------------------
  BidPrice field = actual market ask (reversed)
  AskPrice field = actual market bid (reversed)
  Set reverse_bid_ask=True (default) to correct this.
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple

from vnpy.trader.constant import Direction, Offset, Status
from vnpy.trader.object import OrderData, TickData, TradeData
from vnpy_ctastrategy import CtaTemplate

JST = timezone(timedelta(hours=9))


# ═════════════════════════════════════════════════════════════
# §1  Internal data types  (ported from kabu_hft.gateway)
# ═════════════════════════════════════════════════════════════

@dataclass
class _Level:
    price: float
    size: int


@dataclass
class _BoardSnapshot:
    ts_ns: int
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    bids: List[_Level]
    asks: List[_Level]
    prev_board: Optional["_BoardSnapshot"] = None

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def valid(self) -> bool:
        return self.bid > 0 and self.ask > 0 and self.bid < self.ask


@dataclass
class _TradePrint:
    ts_ns: int
    price: float
    size: int
    side: int   # +1 = buy, -1 = sell


@dataclass
class _SignalPacket:
    ts_ns: int
    obi_z: float
    lob_ofi_z: float
    tape_ofi_z: float
    micro_momentum_z: float
    microprice_tilt_z: float
    microprice: float
    mid: float
    composite: float


# ═════════════════════════════════════════════════════════════
# §2  Signal classes  (ported from kabu_hft.signals.microstructure)
# ═════════════════════════════════════════════════════════════

class _OnlineZScore:
    """Rolling z-score with O(1) update. Returns 0 until warm (>=10 samples). Clips to [-4, +4]."""
    __slots__ = ("window", "buf", "sum_x", "sum_x2")

    def __init__(self, window: int) -> None:
        self.window = window
        self.buf: deque = deque()
        self.sum_x = 0.0
        self.sum_x2 = 0.0

    def update(self, value: float) -> float:
        self.buf.append(value)
        self.sum_x += value
        self.sum_x2 += value * value
        if len(self.buf) > self.window:
            r = self.buf.popleft()
            self.sum_x -= r
            self.sum_x2 -= r * r
        n = len(self.buf)
        if n < 10:
            return 0.0
        mean = self.sum_x / n
        var = max(0.0, self.sum_x2 / n - mean * mean)
        if var <= 1e-12:
            return 0.0
        return max(-4.0, min(4.0, (value - mean) / math.sqrt(var)))


class _OBISignal:
    """Order Book Imbalance: depth-weighted bid vs ask pressure."""

    def __init__(self, depth: int, decay: float) -> None:
        self.weights = [decay ** i for i in range(depth)]

    def compute(self, snap: _BoardSnapshot) -> float:
        bw = sum(w * snap.bids[i].size for i, w in enumerate(self.weights) if i < len(snap.bids))
        aw = sum(w * snap.asks[i].size for i, w in enumerate(self.weights) if i < len(snap.asks))
        total = bw + aw
        return 0.0 if total <= 0 else (bw - aw) / total


class _LOBOFISignal:
    """LOB Order Flow Imbalance: change in depth at each level, decay-weighted."""

    def __init__(self, depth: int, decay: float) -> None:
        self.weights = [decay ** i for i in range(depth)]

    def compute(self, snap: _BoardSnapshot) -> float:
        prev = snap.prev_board
        if prev is None:
            return 0.0
        score = 0.0
        for i, w in enumerate(self.weights):
            score += w * (
                self._delta(snap.bids, prev.bids, i, is_bid=True)
                - self._delta(snap.asks, prev.asks, i, is_bid=False)
            )
        return score

    @staticmethod
    def _delta(curr: List[_Level], prev: List[_Level], i: int, *, is_bid: bool) -> float:
        hc, hp = i < len(curr), i < len(prev)
        if not hc and not hp:
            return 0.0
        if hc and not hp:
            return float(curr[i].size)
        if hp and not hc:
            return -float(prev[i].size)
        c, p = curr[i], prev[i]
        if is_bid:
            if c.price > p.price:
                return float(c.size)
            if c.price < p.price:
                return -float(p.size)
            return float(c.size - p.size)
        if c.price < p.price:
            return float(c.size)
        if c.price > p.price:
            return -float(p.size)
        return float(c.size - p.size)


class _TapeOFISignal:
    """Tape OFI: rolling buy/sell volume imbalance. Out-of-order trades are discarded."""

    def __init__(self, window_sec: int) -> None:
        self.window_ns = window_sec * 1_000_000_000
        self.events: deque = deque()
        self.buy_volume = 0
        self.sell_volume = 0
        self._last_ts_ns: int = 0

    def on_trade(self, trade: _TradePrint) -> float:
        if self._last_ts_ns > 0 and trade.ts_ns < self._last_ts_ns:
            return self.current  # discard out-of-order
        self._last_ts_ns = trade.ts_ns
        bv = trade.size if trade.side > 0 else 0
        sv = trade.size if trade.side < 0 else 0
        self.events.append((trade.ts_ns, bv, sv))
        self.buy_volume += bv
        self.sell_volume += sv
        self._trim(trade.ts_ns)
        return self.current

    def _trim(self, now_ns: int) -> None:
        while self.events and now_ns - self.events[0][0] > self.window_ns:
            _, bv, sv = self.events.popleft()
            self.buy_volume -= bv
            self.sell_volume -= sv

    @property
    def current(self) -> float:
        total = self.buy_volume + self.sell_volume
        return 0.0 if total <= 0 else (self.buy_volume - self.sell_volume) / total


class _MicropriceSignals:
    """Microprice, micro-momentum (EMA deviation), and microprice-tilt."""

    def __init__(self, ema_alpha: float, tick_size: float) -> None:
        self.ema_alpha = ema_alpha
        self.tick_size = max(tick_size, 1e-9)
        self.ema: Optional[float] = None

    def compute(self, snap: _BoardSnapshot) -> Tuple[float, float, float]:
        total = snap.bid_size + snap.ask_size
        if total <= 0:
            return snap.mid, 0.0, 0.0
        microprice = (snap.ask_size * snap.bid + snap.bid_size * snap.ask) / total
        if self.ema is None:
            self.ema = microprice
        else:
            self.ema = self.ema_alpha * microprice + (1.0 - self.ema_alpha) * self.ema
        micro_momentum = (microprice - self.ema) / self.tick_size
        half_spread = snap.spread / 2.0
        microprice_tilt = 0.0 if half_spread <= 0 else (microprice - snap.mid) / half_spread
        return microprice, micro_momentum, microprice_tilt


class _SignalStack:
    """Composite 5-signal engine. All signals are z-scored before weighting."""

    def __init__(
        self, *,
        obi_depth: int, obi_decay: float,
        lob_ofi_depth: int, lob_ofi_decay: float,
        tape_window_sec: int, mp_ema_alpha: float, tick_size: float,
        zscore_window: int,
        w_obi: float, w_lob_ofi: float, w_tape_ofi: float,
        w_micro_momentum: float, w_microprice_tilt: float,
    ) -> None:
        self.obi = _OBISignal(obi_depth, obi_decay)
        self.lob_ofi = _LOBOFISignal(lob_ofi_depth, lob_ofi_decay)
        self.tape = _TapeOFISignal(tape_window_sec)
        self.micro = _MicropriceSignals(mp_ema_alpha, tick_size)
        self.w_obi = w_obi
        self.w_lob_ofi = w_lob_ofi
        self.w_tape_ofi = w_tape_ofi
        self.w_micro_momentum = w_micro_momentum
        self.w_microprice_tilt = w_microprice_tilt
        self.z: Dict[str, _OnlineZScore] = {
            k: _OnlineZScore(zscore_window)
            for k in ("obi", "lob_ofi", "tape_ofi", "micro_momentum", "microprice_tilt")
        }
        self.last: Optional[_SignalPacket] = None

    def on_board(self, snap: _BoardSnapshot) -> _SignalPacket:
        obi_raw = self.obi.compute(snap)
        lob_ofi_raw = self.lob_ofi.compute(snap)
        tape_ofi_raw = self.tape.current
        microprice, mm_raw, mt_raw = self.micro.compute(snap)

        obi_z = self.z["obi"].update(obi_raw)
        lob_ofi_z = self.z["lob_ofi"].update(lob_ofi_raw)
        tape_ofi_z = self.z["tape_ofi"].update(tape_ofi_raw)
        mm_z = self.z["micro_momentum"].update(mm_raw)
        mt_z = self.z["microprice_tilt"].update(mt_raw)

        composite = (
            self.w_lob_ofi * lob_ofi_z
            + self.w_obi * obi_z
            + self.w_tape_ofi * tape_ofi_z
            + self.w_micro_momentum * mm_z
            + self.w_microprice_tilt * mt_z
        )
        pkt = _SignalPacket(
            ts_ns=snap.ts_ns,
            obi_z=obi_z, lob_ofi_z=lob_ofi_z, tape_ofi_z=tape_ofi_z,
            micro_momentum_z=mm_z, microprice_tilt_z=mt_z,
            microprice=microprice, mid=snap.mid, composite=composite,
        )
        self.last = pkt
        return pkt

    def on_trade(self, trade: _TradePrint) -> None:
        self.tape.on_trade(trade)


# ═════════════════════════════════════════════════════════════
# §3  Market state detector + requote budget
#     (ported from kabu_hft.core.market_state and execution.engine)
# ═════════════════════════════════════════════════════════════

class _MarketState(str, Enum):
    NORMAL = "NORMAL"
    QUEUE = "QUEUE"
    ABNORMAL = "ABNORMAL"


@dataclass
class _MarketStateView:
    state: _MarketState
    reason: str
    spread_ticks: float
    event_rate_hz: float


class _MarketStateDetector:
    """
    Classifies market condition on each tick.
    ABNORMAL: invalid quote / stale quote / spread blowout / event burst / price jump
    QUEUE:    spread <= queue_spread_max_ticks (1-tick market)
    NORMAL:   everything else
    """

    def __init__(
        self, *,
        tick_size: float,
        stale_quote_ms: int,
        queue_spread_max_ticks: float,
        abnormal_max_spread_ticks: float,
        max_event_rate_hz: float,
        event_burst_min_events: int,
        state_window_ms: int,
        jump_threshold_ticks: float,
    ) -> None:
        self.tick_size = max(tick_size, 1e-9)
        self.stale_quote_ms = max(stale_quote_ms, 1)
        self.queue_max_ticks = max(queue_spread_max_ticks, 0.0)
        self.abnormal_max_ticks = max(abnormal_max_spread_ticks, 0.5)
        self.max_event_rate_hz = max(max_event_rate_hz, 1.0)
        self.event_burst_min = max(event_burst_min_events, 2)
        self.window_ns = max(state_window_ms, 250) * 1_000_000
        self.jump_thr = max(jump_threshold_ticks, 0.5)
        self._event_times: deque = deque()
        self._prev_mid = 0.0

    def evaluate(self, snap: _BoardSnapshot, now_ns: Optional[int] = None) -> _MarketStateView:
        now = now_ns if now_ns is not None else time.time_ns()
        self._event_times.append(now)
        while self._event_times and now - self._event_times[0] > self.window_ns:
            self._event_times.popleft()

        spread_ticks = snap.spread / self.tick_size if snap.spread > 0 else 0.0
        stale_ms = max(0.0, (now - snap.ts_ns) / 1_000_000)
        jump_ticks = 0.0
        if self._prev_mid > 0.0:
            jump_ticks = abs(snap.mid - self._prev_mid) / self.tick_size
        self._prev_mid = snap.mid
        event_rate_hz = self._event_rate_hz(now)

        if not snap.valid:
            return _MarketStateView(_MarketState.ABNORMAL, "invalid_quote", spread_ticks, event_rate_hz)
        if stale_ms > self.stale_quote_ms:
            return _MarketStateView(_MarketState.ABNORMAL, "stale_quote", spread_ticks, event_rate_hz)
        if spread_ticks >= self.abnormal_max_ticks:
            return _MarketStateView(_MarketState.ABNORMAL, "spread_blowout", spread_ticks, event_rate_hz)
        if len(self._event_times) >= self.event_burst_min and event_rate_hz >= self.max_event_rate_hz:
            return _MarketStateView(_MarketState.ABNORMAL, "event_burst", spread_ticks, event_rate_hz)
        if jump_ticks >= self.jump_thr:
            return _MarketStateView(_MarketState.ABNORMAL, "price_jump", spread_ticks, event_rate_hz)
        if spread_ticks <= self.queue_max_ticks + 1e-9:
            return _MarketStateView(_MarketState.QUEUE, "one_tick_queue", spread_ticks, event_rate_hz)
        return _MarketStateView(_MarketState.NORMAL, "normal_flow", spread_ticks, event_rate_hz)

    def _event_rate_hz(self, now_ns: int) -> float:
        if len(self._event_times) < 2:
            return 0.0
        duration_ns = max(now_ns - self._event_times[0], 1)
        return (len(self._event_times) - 1) * 1_000_000_000 / duration_ns


class _RequoteBudget:
    """Sliding-window requote counter. allow() checks; consume() records usage after success."""

    def __init__(self, max_per_minute: int) -> None:
        self.max_per_minute = max_per_minute
        self._ts: deque = deque()

    def _trim(self, now_ns: int) -> None:
        window_ns = 60 * 1_000_000_000
        while self._ts and now_ns - self._ts[0] > window_ns:
            self._ts.popleft()

    def allow(self, now_ns: int) -> bool:
        self._trim(now_ns)
        return len(self._ts) < self.max_per_minute

    def consume(self, now_ns: int) -> None:
        self._ts.append(now_ns)


# ═════════════════════════════════════════════════════════════
# §4  PnL tracker  (ported from kabu_signal_stack pattern)
# ═════════════════════════════════════════════════════════════

@dataclass
class _TradeRecord:
    direction: str
    entry_price: float
    exit_price: float
    volume: float
    net_pnl: float
    hold_seconds: float
    reason: str
    entry_time: datetime


class _PnLTracker:
    def __init__(self) -> None:
        self._trades: List[_TradeRecord] = []

    def record(self, rec: _TradeRecord) -> None:
        self._trades.append(rec)

    def stats(self) -> dict:
        if not self._trades:
            return {}
        wins = [t for t in self._trades if t.net_pnl > 0]
        losses = [t for t in self._trades if t.net_pnl < 0]
        gp = sum(t.net_pnl for t in wins)
        gl = abs(sum(t.net_pnl for t in losses))
        return {
            "win_rate": len(wins) / len(self._trades),
            "profit_factor": gp / gl if gl > 0 else float("inf"),
            "avg_hold_seconds": sum(t.hold_seconds for t in self._trades) / len(self._trades),
            "total_trades": len(self._trades),
        }


# ═════════════════════════════════════════════════════════════
# §5  Enums
# ═════════════════════════════════════════════════════════════

class _OrderState(str, Enum):
    IDLE = "IDLE"
    PENDING_OPEN = "PENDING_OPEN"
    OPEN = "OPEN"
    PENDING_CLOSE = "PENDING_CLOSE"


# ═════════════════════════════════════════════════════════════
# §6  Main Strategy
# ═════════════════════════════════════════════════════════════

class KabuHFTVnpyStrategy(CtaTemplate):
    """
    Self-contained VeighNa HFT strategy ported from kabu_hft_new.
    Combines microstructure signals, market-state awareness, and inventory-skew pricing.
    """

    author = "kabu_hft_new port"

    # ── 合约 ────────────────────────────────────────────────
    price_tick: float = 1.0
    base_qty: int = 1
    max_qty: int = 1

    # ── 信号权重 ────────────────────────────────────────────
    obi_depth: int = 5
    obi_decay: float = 0.7
    lob_ofi_depth: int = 5
    lob_ofi_decay: float = 0.7
    tape_window_sec: int = 30
    mp_ema_alpha: float = 0.1
    zscore_window: int = 300
    w_lob_ofi: float = 0.30
    w_obi: float = 0.25
    w_tape_ofi: float = 0.20
    w_micro_momentum: float = 0.15
    w_microprice_tilt: float = 0.10

    # ── 入场/退场阈值 ────────────────────────────────────────
    entry_threshold: float = 1.5
    exit_threshold: float = 0.5
    strong_threshold: float = 2.5      # >= this → TAKER mode

    # ── 公允价值 + 库存偏斜 ──────────────────────────────────
    fair_value_beta: float = 0.5
    max_fair_shift_ticks: float = 3.0
    inventory_skew_ticks: float = 1.0
    max_inventory_qty: int = 1

    # ── MarketState 检测 ─────────────────────────────────────
    stale_quote_ms: int = 1200
    queue_spread_max_ticks: float = 1.0
    abnormal_max_spread_ticks: float = 6.0
    max_event_rate_hz: float = 160.0
    event_burst_min_events: int = 6
    state_window_ms: int = 3000
    jump_threshold_ticks: float = 4.0

    # ── QUEUE 模式 ───────────────────────────────────────────
    queue_min_top_qty: int = 200

    # ── Requote 预算 ─────────────────────────────────────────
    max_requotes_per_minute: int = 20

    # ── 持仓管理 ─────────────────────────────────────────────
    max_hold_seconds: int = 60
    loss_ticks: float = 4.0
    profit_ticks: float = 4.0
    trailing_ticks: float = 2.0
    max_loss_per_trade_jpy: float = 0.0   # 0 = disabled

    # ── 执行 ─────────────────────────────────────────────────
    force_maker_mode: bool = False
    auto_tp_on_fill: bool = True
    hold_if_loss: bool = False
    smart_cancel_on_flip: bool = True

    # ── 限流 ─────────────────────────────────────────────────
    open_min_interval_ms: int = 100
    close_min_interval_ms: int = 50
    max_order_req_per_sec: int = 8
    cooling_seconds: int = 5

    # ── 日志 ─────────────────────────────────────────────────
    verbose_log: bool = False
    log_interval_seconds: int = 60

    # ── 行情保护 ─────────────────────────────────────────────
    max_tick_stale_seconds: float = 5.0
    stale_feed_force_flatten: bool = True
    reverse_bid_ask: bool = True         # kabu Station bid/ask direction is reversed

    # ── 风险 ─────────────────────────────────────────────────
    daily_loss_limit: float = 5000.0

    # VeighNa display variables
    order_state: str = _OrderState.IDLE.value
    pos: int = 0
    daily_pnl: float = 0.0
    daily_trades: int = 0
    composite_signal: float = 0.0
    market_state_str: str = "NORMAL"

    parameters = [
        "price_tick", "base_qty", "max_qty",
        "obi_depth", "obi_decay", "lob_ofi_depth", "lob_ofi_decay",
        "tape_window_sec", "mp_ema_alpha", "zscore_window",
        "w_lob_ofi", "w_obi", "w_tape_ofi", "w_micro_momentum", "w_microprice_tilt",
        "entry_threshold", "exit_threshold", "strong_threshold",
        "fair_value_beta", "max_fair_shift_ticks", "inventory_skew_ticks", "max_inventory_qty",
        "stale_quote_ms", "queue_spread_max_ticks", "abnormal_max_spread_ticks",
        "max_event_rate_hz", "event_burst_min_events", "state_window_ms", "jump_threshold_ticks",
        "queue_min_top_qty", "max_requotes_per_minute",
        "max_hold_seconds", "loss_ticks", "profit_ticks", "trailing_ticks",
        "max_loss_per_trade_jpy",
        "force_maker_mode", "auto_tp_on_fill", "hold_if_loss", "smart_cancel_on_flip",
        "open_min_interval_ms", "close_min_interval_ms",
        "max_order_req_per_sec", "cooling_seconds",
        "verbose_log", "log_interval_seconds",
        "max_tick_stale_seconds", "stale_feed_force_flatten", "reverse_bid_ask",
        "daily_loss_limit",
    ]

    variables = [
        "order_state", "pos", "daily_pnl", "daily_trades",
        "composite_signal", "market_state_str",
    ]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)

        # Signal / state engines (initialised lazily in _init_engine)
        self._sig: Optional[_SignalStack] = None
        self._msd: Optional[_MarketStateDetector] = None
        self._requote_budget: Optional[_RequoteBudget] = None

        # Order state machine
        self._order_state: _OrderState = _OrderState.IDLE
        self._entry_direction: str = ""        # "LONG" | "SHORT"
        self._entry_fill_price: float = 0.0
        self._entry_fill_volume: float = 0.0
        self._entry_time: Optional[datetime] = None
        self._state_since: Optional[datetime] = None

        # Inventory tracking (for skew calculation)
        self._inv_side: int = 0                # +1 long, -1 short, 0 flat
        self._inv_qty: int = 0

        # Active orders
        self._active_orderids: List[str] = []
        self._auto_tp_orderid: str = ""

        # Last processed snapshot
        self._last_snap: Optional[_BoardSnapshot] = None
        self._last_mkt: _MarketStateView = _MarketStateView(
            _MarketState.NORMAL, "init", 0.0, 0.0
        )

        # Rate limiting
        self._last_open_dt: Optional[datetime] = None
        self._last_close_dt: Optional[datetime] = None
        self._order_req_ts: deque = deque()
        self._cooling_until: Optional[datetime] = None

        # Pricetick auto-verification (every 30 s until confirmed)
        self._price_tick_verified: bool = False
        self._last_price_tick_check_dt: Optional[datetime] = None

        # Exit tracking
        self._pending_exit_reason: str = ""
        self._peak_pnl_jpy: float = 0.0

        # Session PnL
        self._pnl_tracker = _PnLTracker()
        self._daily_pnl_raw: float = 0.0
        self._daily_trades_raw: int = 0

        # Periodic status log
        self._last_log_dt: Optional[datetime] = None

    # ─────────────────────────────────────────────────────────
    # VeighNa lifecycle
    # ─────────────────────────────────────────────────────────

    def on_init(self) -> None:
        self.write_log("[on_init] KabuHFTVnpyStrategy initializing")
        self._init_engine()
        self.load_bar(1)

    def on_start(self) -> None:
        self.write_log("[on_start] strategy started")
        self._reset_position_state()

    def on_stop(self) -> None:
        self.write_log("[on_stop] strategy stopped")

    def _init_engine(self) -> None:
        tick = max(self.price_tick, 1e-9)
        self._sig = _SignalStack(
            obi_depth=self.obi_depth, obi_decay=self.obi_decay,
            lob_ofi_depth=self.lob_ofi_depth, lob_ofi_decay=self.lob_ofi_decay,
            tape_window_sec=self.tape_window_sec, mp_ema_alpha=self.mp_ema_alpha,
            tick_size=tick, zscore_window=self.zscore_window,
            w_obi=self.w_obi, w_lob_ofi=self.w_lob_ofi, w_tape_ofi=self.w_tape_ofi,
            w_micro_momentum=self.w_micro_momentum, w_microprice_tilt=self.w_microprice_tilt,
        )
        self._msd = _MarketStateDetector(
            tick_size=tick, stale_quote_ms=self.stale_quote_ms,
            queue_spread_max_ticks=self.queue_spread_max_ticks,
            abnormal_max_spread_ticks=self.abnormal_max_spread_ticks,
            max_event_rate_hz=self.max_event_rate_hz,
            event_burst_min_events=self.event_burst_min_events,
            state_window_ms=self.state_window_ms,
            jump_threshold_ticks=self.jump_threshold_ticks,
        )
        self._requote_budget = _RequoteBudget(self.max_requotes_per_minute)

    # ─────────────────────────────────────────────────────────
    # Main tick handler
    # ─────────────────────────────────────────────────────────

    def on_tick(self, tick: TickData) -> None:
        if not self.trading:
            return
        if self._sig is None:
            self._init_engine()

        tick_dt = tick.datetime.astimezone(JST) if tick.datetime else datetime.now(JST)
        now_ns = int(tick_dt.timestamp() * 1_000_000_000)

        # 1. Pricetick auto-verification
        self._periodic_verify_price_tick(tick_dt)

        # 2. Build internal snapshot (handles bid/ask reversal)
        snap = self._tick_to_snapshot(tick, now_ns)
        if snap is None:
            return

        # 3. Market state
        mkt = self._msd.evaluate(snap, now_ns)
        if mkt.state.value != self.market_state_str:
            self.write_log(
                f"[MKTSTATE] {self.market_state_str} → {mkt.state.value} "
                f"reason={mkt.reason} spread={mkt.spread_ticks:.1f}T "
                f"rate={mkt.event_rate_hz:.0f}Hz"
            )
        self.market_state_str = mkt.state.value
        self._last_mkt = mkt

        # 4. Signals
        pkt = self._sig.on_board(snap)
        composite = pkt.composite
        self.composite_signal = round(composite, 3)

        # 5. Fair value + inventory skew
        fair_price, reservation = self._fair_and_reservation(snap, composite)

        # 6. Cooling guard
        if self._cooling_until and tick_dt < self._cooling_until:
            if self._order_state == _OrderState.IDLE:
                self._last_snap = snap
                return

        # 7. State dispatch
        if self._order_state == _OrderState.IDLE:
            if mkt.state != _MarketState.ABNORMAL:
                self._manage_entry(snap, composite, fair_price, reservation, tick_dt, mkt)
        elif self._order_state == _OrderState.PENDING_OPEN:
            self._manage_pending_open(snap, composite, tick_dt, mkt, reservation)
        elif self._order_state == _OrderState.OPEN:
            self._manage_exit(snap, composite, tick_dt, mkt)
        elif self._order_state == _OrderState.PENDING_CLOSE:
            self._manage_pending_close(tick_dt)

        # 8. Periodic status log
        if self._last_log_dt is None or \
                (tick_dt - self._last_log_dt).total_seconds() >= self.log_interval_seconds:
            self._log_status(tick_dt, snap, composite)
            self._last_log_dt = tick_dt

        self._last_snap = snap

    # ─────────────────────────────────────────────────────────
    # Tick → internal snapshot adapter
    # ─────────────────────────────────────────────────────────

    def _tick_to_snapshot(self, tick: TickData, now_ns: int) -> Optional[_BoardSnapshot]:
        if self.reverse_bid_ask:
            # kabu Station naming convention: BidPrice = actual ask, AskPrice = actual bid
            real_bid = tick.ask_price_1
            real_ask = tick.bid_price_1
            real_bid_sz = int(tick.ask_volume_1 or 0)
            real_ask_sz = int(tick.bid_volume_1 or 0)
        else:
            real_bid = tick.bid_price_1
            real_ask = tick.ask_price_1
            real_bid_sz = int(tick.bid_volume_1 or 0)
            real_ask_sz = int(tick.ask_volume_1 or 0)

        if real_bid <= 0 or real_ask <= 0 or real_bid >= real_ask:
            return None

        bids: List[_Level] = []
        asks: List[_Level] = []
        for i in range(1, 6):
            if self.reverse_bid_ask:
                bp = getattr(tick, f"ask_price_{i}", 0.0) or 0.0
                bs = int(getattr(tick, f"ask_volume_{i}", 0) or 0)
                ap = getattr(tick, f"bid_price_{i}", 0.0) or 0.0
                as_ = int(getattr(tick, f"bid_volume_{i}", 0) or 0)
            else:
                bp = getattr(tick, f"bid_price_{i}", 0.0) or 0.0
                bs = int(getattr(tick, f"bid_volume_{i}", 0) or 0)
                ap = getattr(tick, f"ask_price_{i}", 0.0) or 0.0
                as_ = int(getattr(tick, f"ask_volume_{i}", 0) or 0)
            if bp > 0:
                bids.append(_Level(price=bp, size=bs))
            if ap > 0:
                asks.append(_Level(price=ap, size=as_))

        return _BoardSnapshot(
            ts_ns=now_ns,
            bid=real_bid, ask=real_ask,
            bid_size=real_bid_sz, ask_size=real_ask_sz,
            bids=bids, asks=asks,
            prev_board=self._last_snap,
        )

    # ─────────────────────────────────────────────────────────
    # Fair value + inventory skew  (ported from HFTStrategy._fair_and_reservation)
    # ─────────────────────────────────────────────────────────

    def _fair_and_reservation(self, snap: _BoardSnapshot, composite: float) -> Tuple[float, float]:
        tick = max(self.price_tick, 1e-9)
        fair_shift = max(
            -self.max_fair_shift_ticks,
            min(self.max_fair_shift_ticks, self.fair_value_beta * composite),
        )
        fair_price = snap.mid + fair_shift * tick

        signed_inv = self._inv_side * self._inv_qty if self._inv_side != 0 else 0
        inv_ratio = signed_inv / max(self.max_inventory_qty, 1)
        skew_mult = 1.5 if abs(inv_ratio) >= 0.66 else 1.0
        skew = self.inventory_skew_ticks * skew_mult * inv_ratio
        reservation = fair_price - skew * tick
        return fair_price, reservation

    # ─────────────────────────────────────────────────────────
    # Entry management
    # ─────────────────────────────────────────────────────────

    def _manage_entry(
        self,
        snap: _BoardSnapshot,
        composite: float,
        fair_price: float,
        reservation: float,
        now: datetime,
        mkt: _MarketStateView,
    ) -> None:
        if abs(composite) < self.entry_threshold:
            return
        if self.daily_loss_limit > 0 and self._daily_pnl_raw < -self.daily_loss_limit:
            self._vlog(f"[SKIP] daily_loss_limit {self._daily_pnl_raw:.0f}¥")
            return
        if self._last_open_dt:
            if (now - self._last_open_dt).total_seconds() * 1000 < self.open_min_interval_ms:
                return

        direction = "LONG" if composite > 0 else "SHORT"
        is_maker, entry_price = self._select_entry_price(snap, composite, reservation, mkt, direction)

        ids = (self._rl_buy(entry_price, self.base_qty, now) if direction == "LONG"
               else self._rl_short(entry_price, self.base_qty, now))

        if ids:
            self._active_orderids = list(ids)
            self._entry_direction = direction
            self._order_state = _OrderState.PENDING_OPEN
            self._state_since = now
            self.order_state = _OrderState.PENDING_OPEN.value
            self._last_open_dt = now
            self._vlog(
                f"[OPEN] {direction} α={composite:.3f} price={entry_price:.1f} "
                f"maker={is_maker} fair={fair_price:.1f} res={reservation:.1f} "
                f"mkt={mkt.state.value}"
            )

    def _select_entry_price(
        self,
        snap: _BoardSnapshot,
        composite: float,
        reservation: float,
        mkt: _MarketStateView,
        direction: str,
    ) -> Tuple[bool, float]:
        """Returns (is_maker, price)."""
        tick = self.price_tick
        is_strong = abs(composite) >= self.strong_threshold

        # QUEUE mode: queue at best level; retreat to TAKER if queue is thin
        if mkt.state == _MarketState.QUEUE:
            if direction == "LONG":
                if snap.bid_size < self.queue_min_top_qty:
                    return False, snap.ask    # queue too thin → TAKER
                return True, snap.bid
            else:
                if snap.ask_size < self.queue_min_top_qty:
                    return False, snap.bid
                return True, snap.ask

        # Force maker: always post at bid/ask
        if self.force_maker_mode:
            return True, (snap.bid if direction == "LONG" else snap.ask)

        # Strong signal → TAKER (cross spread)
        if is_strong:
            return False, (snap.ask if direction == "LONG" else snap.bid)

        # Normal MAKER: reservation-adjusted
        if direction == "LONG":
            price = snap.bid - tick if reservation <= snap.bid - tick else snap.bid
            return True, price
        else:
            price = snap.ask + tick if reservation >= snap.ask + tick else snap.ask
            return True, price

    def _manage_pending_open(
        self,
        snap: _BoardSnapshot,
        composite: float,
        now: datetime,
        mkt: _MarketStateView,
        reservation: float,
    ) -> None:
        if self._state_since is None:
            return
        pending_ms = (now - self._state_since).total_seconds() * 1000

        # Timeout → cancel + cooling
        if pending_ms > 2000:
            self._cancel_all()
            self._set_idle_cooling(now, "pending_timeout")
            return

        # Signal flip → cancel (smart_cancel_on_flip)
        if self.smart_cancel_on_flip:
            flipped = (
                (self._entry_direction == "LONG" and composite < -self.exit_threshold) or
                (self._entry_direction == "SHORT" and composite > self.exit_threshold)
            )
            if flipped:
                self._cancel_all()
                self._set_idle_cooling(now, "signal_flip")
                return

        # ABNORMAL while pending → cancel
        if mkt.state == _MarketState.ABNORMAL:
            self._cancel_all()
            self._set_idle_cooling(now, f"abnormal_{mkt.reason}")
            return

        # Requote if reservation price drifted by >= 1 tick (cancel + re-enter next tick)
        now_ns = int(now.timestamp() * 1_000_000_000)
        if self._requote_budget and self._requote_budget.allow(now_ns):
            _, desired_price = self._select_entry_price(
                snap, composite, reservation, mkt, self._entry_direction
            )
            # We cancel; next on_tick will re-enter at new price
            # (can't cheaply inspect working order price in VeighNa without tracking it)
            # Only requote if we've waited at least open_min_interval_ms
            if pending_ms > self.open_min_interval_ms:
                self._cancel_all()
                self._requote_budget.consume(now_ns)
                self._set_idle_cooling(now, "requote")
                self._cooling_until = now + timedelta(milliseconds=50)  # short cool

    def _manage_pending_close(self, now: datetime) -> None:
        if self._state_since is None:
            return
        pending_ms = (now - self._state_since).total_seconds() * 1000
        if pending_ms > 3000:
            # Force re-exit on next tick
            self._cancel_all()
            self._order_state = _OrderState.OPEN
            self.order_state = _OrderState.OPEN.value
            self._state_since = now

    # ─────────────────────────────────────────────────────────
    # Exit management  (C0–C5 priority chain)
    # ─────────────────────────────────────────────────────────

    def _manage_exit(
        self,
        snap: _BoardSnapshot,
        composite: float,
        now: datetime,
        mkt: _MarketStateView,
    ) -> None:
        if self._entry_fill_price <= 0:
            return

        tick = self.price_tick
        dir_sign = +1 if self._entry_direction == "LONG" else -1
        ref_price = snap.bid if self._entry_direction == "LONG" else snap.ask
        vol = abs(self._entry_fill_volume)
        unrealized = dir_sign * (ref_price - self._entry_fill_price) * vol

        # C0: JPY emergency stop (overrides hold_if_loss)
        if self.max_loss_per_trade_jpy > 0 and unrealized < -self.max_loss_per_trade_jpy:
            self.write_log(
                f"🚨 [EMER_LOSS] unrealized={unrealized:.0f}¥ "
                f"< -{self.max_loss_per_trade_jpy:.0f}¥ → 紧急平仓"
            )
            self._send_exit(snap, force_taker=True, reason="EMER_LOSS", now=now)
            return

        # C1: ABNORMAL → force TAKER exit
        if mkt.state == _MarketState.ABNORMAL:
            self._send_exit(snap, force_taker=True, reason=f"ABNORMAL_{mkt.reason}", now=now)
            return

        # C2: fast loss (skipped if hold_if_loss)
        loss_threshold = self.loss_ticks * tick * vol
        if not self.hold_if_loss and unrealized < -loss_threshold:
            self._send_exit(snap, force_taker=False, reason="STOP_LOSS", now=now)
            return

        # C3: auto TP order already live — let it fill
        if self._auto_tp_orderid:
            return

        # C4: trailing stop
        if unrealized > 0:
            self._peak_pnl_jpy = max(self._peak_pnl_jpy, unrealized)
            profit_threshold = self.profit_ticks * tick * vol
            trailing_threshold = self.trailing_ticks * tick * vol
            if (self._peak_pnl_jpy > profit_threshold and
                    (self._peak_pnl_jpy - unrealized) > trailing_threshold):
                self._send_exit(snap, force_taker=False, reason="TRAILING", now=now)
                return

        # C5: timeout
        if self._entry_time:
            hold_s = (now - self._entry_time).total_seconds()
            if hold_s > self.max_hold_seconds:
                self._send_exit(snap, force_taker=False, reason="TIMEOUT", now=now)
                return

        # Signal reversal (skipped if hold_if_loss)
        if not self.hold_if_loss:
            sig_rev = (
                (self._entry_direction == "LONG" and composite < -self.exit_threshold) or
                (self._entry_direction == "SHORT" and composite > self.exit_threshold)
            )
            if sig_rev:
                self._send_exit(snap, force_taker=False, reason="SIGNAL_FLIP", now=now)

    def _send_exit(
        self,
        snap: _BoardSnapshot,
        *,
        force_taker: bool,
        reason: str,
        now: datetime,
    ) -> None:
        if self._order_state != _OrderState.OPEN:
            return
        if self._last_close_dt:
            if (now - self._last_close_dt).total_seconds() * 1000 < self.close_min_interval_ms:
                return

        vol = int(round(abs(self._entry_fill_volume)))
        if self._entry_direction == "LONG":
            exit_price = snap.bid if force_taker else snap.ask
            ids = self._rl_sell(exit_price, vol, now)
        else:
            exit_price = snap.ask if force_taker else snap.bid
            ids = self._rl_cover(exit_price, vol, now)

        if ids:
            self._active_orderids = list(ids)
            self._pending_exit_reason = reason
            self._order_state = _OrderState.PENDING_CLOSE
            self._state_since = now
            self.order_state = _OrderState.PENDING_CLOSE.value
            self._last_close_dt = now
            self._vlog(f"[EXIT] {reason} taker={force_taker} price={exit_price:.1f}")

    # ─────────────────────────────────────────────────────────
    # VeighNa callbacks
    # ─────────────────────────────────────────────────────────

    def on_order(self, order: OrderData) -> None:
        vt_orderid = order.vt_orderid

        if order.status in (Status.CANCELLED, Status.REJECTED):
            if vt_orderid in self._active_orderids:
                self._active_orderids.remove(vt_orderid)
                if not self._active_orderids:
                    if self._order_state == _OrderState.PENDING_OPEN:
                        self._reset_position_state()
                        self._vlog(f"[ORDER] open order cancelled id={vt_orderid}")
            if vt_orderid == self._auto_tp_orderid:
                self._auto_tp_orderid = ""

        if order.status == Status.ALLTRADED:
            if vt_orderid in self._active_orderids:
                self._active_orderids.remove(vt_orderid)

    def on_trade(self, trade: TradeData) -> None:
        trade_dt = trade.datetime.astimezone(JST) if trade.datetime else datetime.now(JST)
        is_long = trade.direction == Direction.LONG

        # ── Entry fill ──
        if self._order_state == _OrderState.PENDING_OPEN:
            self._entry_fill_price = trade.price
            self._entry_fill_volume = float(trade.volume)
            self._entry_time = trade_dt
            self._inv_side = +1 if is_long else -1
            self._inv_qty = int(trade.volume)
            self._order_state = _OrderState.OPEN
            self.order_state = _OrderState.OPEN.value
            self._state_since = trade_dt
            self._peak_pnl_jpy = 0.0
            self.pos = self._inv_qty if is_long else -self._inv_qty
            self.write_log(
                f"[开仓] {'LONG' if is_long else 'SHORT'} "
                f"price={trade.price:.1f} vol={trade.volume}"
            )
            # Place auto-TP limit order
            if self.auto_tp_on_fill and self._last_snap:
                tick = self.price_tick
                if is_long:
                    tp_price = self._entry_fill_price + self.profit_ticks * tick
                    tp_ids = self._rl_sell(tp_price, int(trade.volume), trade_dt)
                else:
                    tp_price = self._entry_fill_price - self.profit_ticks * tick
                    tp_ids = self._rl_cover(tp_price, int(trade.volume), trade_dt)
                if tp_ids:
                    self._auto_tp_orderid = tp_ids[0]
            return

        # ── Exit fill ──
        if self._order_state in (_OrderState.PENDING_CLOSE, _OrderState.OPEN):
            close_price = trade.price
            entry_t = self._entry_time if self._entry_time else trade_dt - timedelta(seconds=1)
            hold_s = (trade_dt - entry_t).total_seconds()
            dir_sign = +1 if self._entry_direction == "LONG" else -1
            net_pnl = dir_sign * (close_price - self._entry_fill_price) * abs(self._entry_fill_volume)
            self._daily_pnl_raw += net_pnl
            self._daily_trades_raw += 1
            self.daily_pnl = round(self._daily_pnl_raw, 0)
            self.daily_trades = self._daily_trades_raw

            rec = _TradeRecord(
                direction=self._entry_direction,
                entry_price=self._entry_fill_price,
                exit_price=close_price,
                volume=abs(self._entry_fill_volume),
                net_pnl=net_pnl,
                hold_seconds=hold_s,
                reason=self._pending_exit_reason or "?",
                entry_time=entry_t,
            )
            self._pnl_tracker.record(rec)

            stats = self._pnl_tracker.stats()
            wr_str = f"{stats.get('win_rate', 0.0) * 100:.1f}%" if stats else "—"
            pf_str = f"{stats.get('profit_factor', 0.0):.2f}" if stats else "—"
            icon = "🟢盈" if net_pnl > 0 else ("🔴亏" if net_pnl < 0 else "⚪平")
            reason = self._pending_exit_reason or "?"
            self.write_log(
                f"[平仓{icon}] {self._entry_direction} {reason} "
                f"entry={self._entry_fill_price:.1f}→exit={close_price:.1f} "
                f"{net_pnl:+.0f}¥ 持{hold_s:.0f}s | "
                f"今日:{self._daily_pnl_raw:+.0f}¥({self._daily_trades_raw}笔) "
                f"胜率:{wr_str} PF:{pf_str}"
            )

            # Cancel dangling auto-TP
            if self._auto_tp_orderid:
                self.cancel_order(self._auto_tp_orderid)
                self._auto_tp_orderid = ""

            self._reset_position_state()
            self._cooling_until = trade_dt + timedelta(seconds=self.cooling_seconds)

    # ─────────────────────────────────────────────────────────
    # Rate-limited order helpers
    # ─────────────────────────────────────────────────────────

    def _can_send_order(self, now: datetime) -> bool:
        ts = now.timestamp()
        cutoff = ts - 1.0
        while self._order_req_ts and self._order_req_ts[0] < cutoff:
            self._order_req_ts.popleft()
        if len(self._order_req_ts) >= self.max_order_req_per_sec:
            self._vlog(
                f"[RATE LIMIT] {len(self._order_req_ts)}/{self.max_order_req_per_sec} req/s"
            )
            return False
        self._order_req_ts.append(ts)
        return True

    def _rl_buy(self, price: float, vol: int, now: datetime) -> List[str]:
        if not self._can_send_order(now):
            return []
        oid = self.buy(price, vol)
        return [oid] if oid else []

    def _rl_short(self, price: float, vol: int, now: datetime) -> List[str]:
        if not self._can_send_order(now):
            return []
        oid = self.short(price, vol)
        return [oid] if oid else []

    def _rl_sell(self, price: float, vol: int, now: datetime) -> List[str]:
        if not self._can_send_order(now):
            return []
        oid = self.sell(price, vol)
        return [oid] if oid else []

    def _rl_cover(self, price: float, vol: int, now: datetime) -> List[str]:
        if not self._can_send_order(now):
            return []
        oid = self.cover(price, vol)
        return [oid] if oid else []

    def _cancel_all(self) -> None:
        for oid in list(self._active_orderids):
            self.cancel_order(oid)
        self._active_orderids.clear()

    # ─────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────

    def _reset_position_state(self) -> None:
        self._order_state = _OrderState.IDLE
        self.order_state = _OrderState.IDLE.value
        self._entry_direction = ""
        self._entry_fill_price = 0.0
        self._entry_fill_volume = 0.0
        self._entry_time = None
        self._state_since = None
        self._inv_side = 0
        self._inv_qty = 0
        self._active_orderids.clear()
        self._auto_tp_orderid = ""
        self._pending_exit_reason = ""
        self._peak_pnl_jpy = 0.0
        self.pos = 0

    def _set_idle_cooling(self, now: datetime, reason: str) -> None:
        self._vlog(f"[CANCEL→IDLE] reason={reason} cooling={self.cooling_seconds}s")
        self._reset_position_state()
        self._cooling_until = now + timedelta(seconds=self.cooling_seconds)

    def _periodic_verify_price_tick(self, now: datetime) -> None:
        """Re-read pricetick from contract every 30 s until confirmed."""
        if self._price_tick_verified:
            return
        if self._last_price_tick_check_dt and \
                (now - self._last_price_tick_check_dt).total_seconds() < 30.0:
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
                    self.write_log(f"[pricetick] 自动修正: {self.price_tick} → {new_pt}")
                    self.price_tick = new_pt
                    if self._sig:
                        self._sig.micro.tick_size = max(new_pt, 1e-9)
                self._price_tick_verified = True
        except Exception:
            pass

    def _vlog(self, msg: str) -> None:
        """Verbose log: only emits when verbose_log=True."""
        if self.verbose_log:
            self.write_log(msg)

    def _log_status(self, now: datetime, snap: _BoardSnapshot, composite: float) -> None:
        stats = self._pnl_tracker.stats()
        wr = f"{stats.get('win_rate', 0.0) * 100:.1f}%" if stats else "—"
        pf = f"{stats.get('profit_factor', 0.0):.2f}" if stats else "—"
        self.write_log(
            f"[STATUS] state={self._order_state.value} mkt={self.market_state_str} "
            f"bid={snap.bid:.1f} ask={snap.ask:.1f} "
            f"α={composite:.3f} "
            f"pnl={self._daily_pnl_raw:+.0f}¥({self._daily_trades_raw}笔) "
            f"wr={wr} pf={pf}"
        )
