"""
kabu_pairs_arbitrage_strategy.py

高频统计套利策略（配对交易）
在两只高度相关的东证股票之间执行统计套利：
  1. 计算对数价差的滚动 z-score（O(1) 在线更新）
  2. 当价差偏离 entry_z_score 且 LOB 方向一致时入场
  3. 当价差回归 exit_z_score 时出场（或触发紧急止损/超时）
  4. 同时提交两条腿的限价委托，并监测腿不同步情况

关键设计：单 CtaTemplate 管理双股票
  - B 股 tick 通过 symbol_strategy_map 注册后自动路由到 on_tick()
  - B 股委托通过 _submit_leg_b() 直接调用 CtaEngine 内部接口
  - 防止 self.pos 被 B 腿成交污染（在 on_trade() 中反向修正）

推荐测试股票对：8306(三菱UFJ) vs 8316(三井住友)  相关系数约 0.90
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date, time, timedelta
from collections import deque
import math
from typing import Deque, Dict, List, Optional, Set

from vnpy.trader.object import TickData, TradeData, OrderData, SubscribeRequest
from vnpy.trader.constant import Direction, Offset, OrderType, Exchange

try:
    from vnpy_ctastrategy import CtaTemplate, StopOrder
except Exception:
    from vnpy.app.cta_strategy import CtaTemplate, StopOrder


# ─────────────────────────────────────────────
#  委托元数据
# ─────────────────────────────────────────────

@dataclass
class _PairOrderMeta:
    """配对套利委托元数据"""
    created_dt: datetime
    purpose: str        # "ENTRY_A", "ENTRY_B", "EXIT_A", "EXIT_B", "EMERGENCY_A", "EMERGENCY_B"
    direction: Direction
    price: float
    volume: int
    is_b_leg: bool = False


# ─────────────────────────────────────────────
#  策略主体
# ─────────────────────────────────────────────

class KabuPairsArbitrageStrategy(CtaTemplate):
    """
    高频配对套利策略（Kabu TSE）

    入场逻辑（5层门控）：
      Gate 1: 交易时段
      Gate 2: 出场后冷却期
      Gate 3: 下单最小间隔
      Gate 4: |z-score| > entry_z_score（价差偏离阈值）
      Gate 5: LOB 盘口量比方向一致 + 确认计数

    出场逻辑（4个条件，优先级从高到低）：
      1. |z| >= emergency_z_score → 紧急平仓
      2. unrealized_pnl < max_loss_per_trade → 止损
      3. 持仓时间 >= max_hold_seconds → 超时平仓
      4. z 回归 exit_z_score → 正常平仓
    """

    author = "KabuPairs"

    # ─── 合约参数 ───
    symbol_b: str = "8316"          # B 股票代码（A = vt_symbol 主合约）
    exchange_b: str = "TSE"         # B 股票交易所
    hedge_ratio: float = 1.0        # 对冲比例 η（spread = log(A) - η·log(B)）
    volume_a: int = 100             # A 每次交易量（股）
    volume_b: int = 100             # B 每次交易量（股）

    # ─── 价差 / z-score ───
    spread_window: int = 60         # 滚动统计窗口（ticks）
    entry_z_score: float = 2.0      # 入场 z 值阈值
    exit_z_score: float = 0.5       # 出场 z 值（回归目标）
    emergency_z_score: float = 4.0  # 紧急止损 z 值
    min_std_threshold: float = 0.0002  # 最低波动率门限（低于此不交易）
    warmup_min_bars: int = 15       # 暖机最少 bar 数（< 此值不发信号）

    # ─── LOB 门控 ───
    lob_gate_enabled: bool = True
    lob_min_score: float = 0.15     # 两股 LOB 最低方向一致性阈值 [0,1]
    confirm_ticks: int = 1          # 信号确认 tick 数

    # ─── 滑点 ───
    entry_slip_ticks: int = 1       # 入场超价 ticks（主动成交）
    exit_slip_ticks: int = 1        # 出场超价 ticks

    # ─── 风控 ───
    max_loss_per_trade: float = -3000.0   # 单笔最大亏损（日元）
    max_daily_loss: float = -30000.0      # 日内最大亏损（日元）
    cooldown_seconds: float = 3.0         # 出场后冷却时间（秒）
    leg_timeout_seconds: float = 2.0      # 腿不同步超时（秒）
    max_hold_seconds: float = 30.0        # 最长持仓时间（秒）
    min_order_interval_ms: int = 200      # 最小下单间隔（ms）
    staleness_ms: float = 500.0           # tick 过期时间（ms）

    # ─── 交易时间过滤 ───
    trade_start_time: str = "09:00:10"
    trade_end_time: str = "15:25:00"
    morning_end_time: str = "11:30:00"
    afternoon_start_time: str = "12:30:00"
    enable_time_filter: bool = True

    # ─── VeighNa 框架要求 ───
    parameters = [
        "symbol_b", "exchange_b", "hedge_ratio",
        "volume_a", "volume_b",
        "spread_window", "entry_z_score", "exit_z_score",
        "emergency_z_score", "min_std_threshold", "warmup_min_bars",
        "lob_gate_enabled", "lob_min_score", "confirm_ticks",
        "entry_slip_ticks", "exit_slip_ticks",
        "max_loss_per_trade", "max_daily_loss",
        "cooldown_seconds", "leg_timeout_seconds", "max_hold_seconds",
        "min_order_interval_ms", "staleness_ms",
        "trade_start_time", "trade_end_time",
        "morning_end_time", "afternoon_start_time", "enable_time_filter",
    ]

    variables = [
        "pos", "_pos_b",
        "_current_z", "_current_spread",
        "_pair_direction", "_daily_pnl", "_is_trading_allowed",
    ]

    # ═══════════════════════════════════════════
    #  初始化
    # ═══════════════════════════════════════════

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)

        # ─── B 股信息 ───
        self._vt_symbol_b: str = ""
        self._pricetick_a: float = 1.0
        self._pricetick_b: float = 1.0
        self._contract_b_ready: bool = False

        # ─── Tick 缓存 ───
        self._tick_a: Optional[TickData] = None
        self._tick_b: Optional[TickData] = None

        # ─── 价差滚动统计（O(1) 在线更新） ───
        self._spread_history: Deque[float] = deque(maxlen=self.spread_window)
        self._spread_sum: float = 0.0
        self._spread_sq_sum: float = 0.0
        self._current_spread: float = 0.0
        self._current_z: float = 0.0

        # ─── LOB 盘口量比 ───
        self._lob_a: float = 0.0
        self._lob_b: float = 0.0

        # ─── 持仓状态 ───
        self._pos_b: float = 0.0
        self._pair_direction: Optional[str] = None   # "LONG_A_SHORT_B" / "SHORT_A_LONG_B"
        self._entry_spread: float = 0.0
        self._entry_time: Optional[datetime] = None
        self._entry_price_a: float = 0.0
        self._entry_price_b: float = 0.0
        self._closing_in_progress: bool = False      # 防止重复平仓

        # ─── 信号确认计数器 ───
        self._confirm_long_a_short_b: int = 0
        self._confirm_short_a_long_b: int = 0

        # ─── 腿状态跟踪 ───
        self._leg_a_filled: bool = False
        self._leg_b_filled: bool = False
        self._leg_a_fill_vol: float = 0.0
        self._leg_b_fill_vol: float = 0.0
        self._leg_submit_time: Optional[datetime] = None

        # ─── 委托跟踪 ───
        self._active_orders_a: Dict[str, _PairOrderMeta] = {}
        self._active_orders_b: Dict[str, _PairOrderMeta] = {}
        self._b_orderid_set: Set[str] = set()        # 所有B腿 vt_orderid（防pos污染）

        # ─── 风控 ───
        self._daily_pnl: float = 0.0
        self._daily_date: Optional[date] = None
        self._is_trading_allowed: bool = True
        self._unrealized_pnl: float = 0.0
        self._last_exit_dt: Optional[datetime] = None
        self._last_order_dt: Optional[datetime] = None

        # ─── 交易时间解析（默认值，on_start 会重新解析） ───
        self._trade_start_time_obj: time = time(9, 0, 10)
        self._trade_end_time_obj: time = time(15, 25, 0)
        self._morning_end_time_obj: time = time(11, 30, 0)
        self._afternoon_start_time_obj: time = time(12, 30, 0)

    # ═══════════════════════════════════════════
    #  VeighNa 生命周期
    # ═══════════════════════════════════════════

    def on_start(self) -> None:
        """策略启动：订阅 B 股，注册 tick 路由"""
        self.write_log("=" * 60)
        self.write_log(f"配对套利策略启动: A={self.vt_symbol}, B={self.symbol_b}.{self.exchange_b}")

        # 解析交易时间字符串
        self._parse_time_settings()

        # 构建 B 股 vt_symbol
        self._vt_symbol_b = f"{self.symbol_b}.{self.exchange_b}"

        # 重新初始化 deque（防止参数更改后 maxlen 不一致）
        self._spread_history = deque(maxlen=self.spread_window)
        self._spread_sum = 0.0
        self._spread_sq_sum = 0.0

        main_engine = self.cta_engine.main_engine

        # ── Step 1: 加载 A 合约信息 ──
        try:
            contract_a = main_engine.get_contract(self.vt_symbol)
            if contract_a and getattr(contract_a, "pricetick", 0) > 0:
                self._pricetick_a = float(contract_a.pricetick)
                self.write_log(f"✅ A合约已加载: {self.vt_symbol}, pricetick={self._pricetick_a}")
            else:
                self.write_log(f"⚠️  A合约 {self.vt_symbol} 未找到，请检查行情订阅")
        except Exception as e:
            self.write_log(f"⚠️  A合约加载异常: {e}")

        # ── Step 2: 加载并订阅 B 合约 ──
        try:
            contract_b = main_engine.get_contract(self._vt_symbol_b)
            if contract_b:
                if getattr(contract_b, "pricetick", 0) > 0:
                    self._pricetick_b = float(contract_b.pricetick)
                gateway_b = getattr(contract_b, "gateway_name", None)
                if gateway_b:
                    symbol_part, exch_part = self._vt_symbol_b.split(".")
                    req = SubscribeRequest(
                        symbol=symbol_part,
                        exchange=Exchange(exch_part)
                    )
                    main_engine.subscribe(req, gateway_b)
                    self._contract_b_ready = True
                    self.write_log(
                        f"✅ B合约已订阅: {self._vt_symbol_b}, pricetick={self._pricetick_b}"
                    )
                else:
                    self.write_log(f"⚠️  B合约 {self._vt_symbol_b} 无 gateway_name")
            else:
                self.write_log(
                    f"⚠️  B合约 {self._vt_symbol_b} 未找到 - 开仓将被禁用\n"
                    f"   建议：先在行情界面手动订阅 {self._vt_symbol_b}，再重启策略"
                )
        except Exception as e:
            self.write_log(f"❌ B合约订阅异常: {e}")

        # ── Step 3: 注册 B 股 tick 路由 ──
        # 将本策略实例注册到 symbol_strategy_map[B]，
        # 使得 CtaEngine 收到 B 的 tick 时也会调用本策略的 on_tick()
        try:
            self.cta_engine.symbol_strategy_map[self._vt_symbol_b].append(self)
            self.write_log(f"✅ B股tick路由已注册: {self._vt_symbol_b}")
        except Exception as e:
            self.write_log(f"❌ B股tick路由注册失败: {e}")

        self.put_event()
        self.write_log("=" * 60)

    def on_stop(self) -> None:
        """策略停止：撤单，注销 B 股 tick 路由"""
        self.write_log("策略停止: 撤单")
        self.cancel_all()

        # 注销 B 股 tick 路由，防止本策略停止后仍收到 B 股 tick
        if self._vt_symbol_b:
            try:
                lst = self.cta_engine.symbol_strategy_map.get(self._vt_symbol_b, [])
                if self in lst:
                    lst.remove(self)
                    self.write_log(f"✅ B股tick路由已注销: {self._vt_symbol_b}")
            except Exception as e:
                self.write_log(f"⚠️  B股tick路由注销异常: {e}")

        self.put_event()

    # ═══════════════════════════════════════════
    #  市场数据处理
    # ═══════════════════════════════════════════

    def on_tick(self, tick: TickData) -> None:
        """Tick 分发器：区分 A / B 股，更新缓存后触发主循环"""
        if not self.trading:
            return

        if tick.vt_symbol == self.vt_symbol:
            self._tick_a = tick
        elif tick.vt_symbol == self._vt_symbol_b:
            self._tick_b = tick
        else:
            return

        # 统一处理 tick.datetime（去掉时区 → naive datetime）
        now_dt = tick.datetime
        if now_dt is None:
            now_dt = datetime.now()
        elif getattr(now_dt, "tzinfo", None) is not None:
            now_dt = now_dt.replace(tzinfo=None)

        self._on_market_update(now_dt)

    # ═══════════════════════════════════════════
    #  核心编排
    # ═══════════════════════════════════════════

    def _on_market_update(self, now_dt: datetime) -> None:
        """市场数据更新后的核心处理逻辑"""

        # Step 1: 检查两只股票 tick 是否都新鲜
        if not self._both_ticks_fresh(now_dt):
            return

        # Step 2: 更新价差 z-score（O(1)）
        self._update_spread()

        # Step 3: 更新 LOB 盘口量比
        self._lob_a = self._compute_lob(self._tick_a)
        self._lob_b = self._compute_lob(self._tick_b)

        # Step 4: 每日重置检查
        self._check_new_day(now_dt)

        # Step 5: 风控 kill-switch
        if not self._is_trading_allowed:
            self._emergency_close_all("daily_risk_stop")
            return

        # Step 6: 更新浮动盈亏（有持仓时）
        if self._has_open_position():
            self._update_unrealized_pnl()

        # Step 7: 腿不同步超时处理
        self._check_leg_timeout(now_dt)

        # Step 8: 出场 or 入场逻辑
        if self._has_open_position():
            self._check_exit(now_dt)
        elif not self._active_orders_a and not self._active_orders_b:
            self._check_entry(now_dt)

    # ═══════════════════════════════════════════
    #  价差 z-score（O(1) 滚动更新）
    # ═══════════════════════════════════════════

    def _update_spread(self) -> None:
        """O(1) 滚动价差 z-score 更新"""
        mid_a = self._get_mid(self._tick_a)
        mid_b = self._get_mid(self._tick_b)

        # 防止 log(0) 或 log(负数)
        if mid_a <= 0 or mid_b <= 0:
            return

        # 对数价差：spread = log(A) - η·log(B)
        spread = math.log(mid_a) - self.hedge_ratio * math.log(mid_b)
        self._current_spread = spread

        # O(1) 滚动更新：append 前先减去最旧值（若 deque 已满）
        if len(self._spread_history) == self.spread_window:
            old = self._spread_history[0]
            self._spread_sum -= old
            self._spread_sq_sum -= old * old

        self._spread_history.append(spread)
        self._spread_sum += spread
        self._spread_sq_sum += spread * spread

        n = len(self._spread_history)

        # 暖机期：样本不足时不发信号
        if n < max(self.warmup_min_bars, self.spread_window // 4):
            self._current_z = 0.0
            return

        mean = self._spread_sum / n
        variance = max(0.0, self._spread_sq_sum / n - mean * mean)
        std = math.sqrt(variance)

        # 波动率过低时不交易（价差无意义）
        if std < self.min_std_threshold:
            self._current_z = 0.0
            return

        self._current_z = (spread - mean) / std

    @staticmethod
    def _get_mid(tick: TickData) -> float:
        """计算中间价 (bid1+ask1)/2，fallback 到 last_price"""
        bid = float(tick.bid_price_1 or 0)
        ask = float(tick.ask_price_1 or 0)
        if bid > 0 and ask > 0:
            return (bid + ask) / 2.0
        last = float(getattr(tick, "last_price", 0) or 0)
        return last

    @staticmethod
    def _compute_lob(tick: TickData) -> float:
        """LOB 盘口量比: (bid_vol1 - ask_vol1) / (bid_vol1 + ask_vol1) → [-1, +1]"""
        bv = float(tick.bid_volume_1 or 0)
        av = float(tick.ask_volume_1 or 0)
        total = bv + av
        if total <= 0:
            return 0.0
        return (bv - av) / total

    # ═══════════════════════════════════════════
    #  辅助工具
    # ═══════════════════════════════════════════

    def _both_ticks_fresh(self, now_dt: datetime) -> bool:
        """检查 A / B 两股 tick 是否都在 staleness_ms 以内"""
        if self._tick_a is None or self._tick_b is None:
            return False
        threshold = timedelta(milliseconds=self.staleness_ms)
        for tick in (self._tick_a, self._tick_b):
            dt = tick.datetime
            if dt is None:
                return False
            if getattr(dt, "tzinfo", None) is not None:
                dt = dt.replace(tzinfo=None)
            if (now_dt - dt) > threshold:
                return False
        return True

    def _check_new_day(self, now_dt: datetime) -> None:
        """每日重置日内风控状态"""
        today = now_dt.date()
        if self._daily_date is None:
            self._daily_date = today
            return
        if today != self._daily_date:
            self.write_log(
                f"[日结] {self._daily_date} | 日内盈亏={self._daily_pnl:+.0f}¥"
            )
            self._daily_date = today
            self._daily_pnl = 0.0
            self._is_trading_allowed = True
            # 重置价差历史（跨日价差统计无意义）
            self._spread_history.clear()
            self._spread_sum = 0.0
            self._spread_sq_sum = 0.0
            self._current_z = 0.0

    def _parse_time_settings(self) -> None:
        """解析交易时间字符串到 time 对象"""
        def _parse(s: str, default: time) -> time:
            try:
                parts = s.split(":")
                return time(int(parts[0]), int(parts[1]), int(parts[2]) if len(parts) > 2 else 0)
            except Exception:
                return default

        self._trade_start_time_obj = _parse(self.trade_start_time, time(9, 0, 10))
        self._trade_end_time_obj = _parse(self.trade_end_time, time(15, 25, 0))
        self._morning_end_time_obj = _parse(self.morning_end_time, time(11, 30, 0))
        self._afternoon_start_time_obj = _parse(self.afternoon_start_time, time(12, 30, 0))

    def _is_trading_time(self, dt: datetime) -> bool:
        """检查是否在允许交易的时段内"""
        if not self.enable_time_filter:
            return True
        t = dt.time()
        if getattr(t, "tzinfo", None) is not None:
            t = t.replace(tzinfo=None)
        if t < self._trade_start_time_obj or t > self._trade_end_time_obj:
            return False
        if self._morning_end_time_obj <= t < self._afternoon_start_time_obj:
            return False
        return True

    def _in_cooldown(self, now_dt: datetime) -> bool:
        """检查是否在出场后的冷却期内"""
        if self._last_exit_dt is None:
            return False
        elapsed = (now_dt - self._last_exit_dt).total_seconds()
        return elapsed < self.cooldown_seconds

    def _has_open_position(self) -> bool:
        """检查是否有未平仓位（A 或 B 有任意持仓）"""
        return abs(self.pos) > 0 or abs(self._pos_b) > 0

    @staticmethod
    def _round_price(price: float, pricetick: float) -> float:
        """将价格对齐到最小变动单位"""
        if pricetick <= 0:
            return price
        return round(round(price / pricetick) * pricetick, 8)

    def _update_unrealized_pnl(self) -> None:
        """更新浮动盈亏（日元，未计手续费）"""
        if not self._has_open_position():
            self._unrealized_pnl = 0.0
            return
        mid_a = self._get_mid(self._tick_a) if self._tick_a else self._entry_price_a
        mid_b = self._get_mid(self._tick_b) if self._tick_b else self._entry_price_b
        pnl_a = (mid_a - self._entry_price_a) * self.pos
        pnl_b = (mid_b - self._entry_price_b) * self._pos_b
        self._unrealized_pnl = pnl_a + pnl_b

    # ═══════════════════════════════════════════
    #  入场逻辑
    # ═══════════════════════════════════════════

    def _check_entry(self, now_dt: datetime) -> None:
        """入场信号检查（5层门控）"""
        if not self._contract_b_ready:
            return

        # ── Gate 1: 交易时段 ──
        if not self._is_trading_time(now_dt):
            self._confirm_long_a_short_b = 0
            self._confirm_short_a_long_b = 0
            return

        # ── Gate 2: 出场后冷却期 ──
        if self._in_cooldown(now_dt):
            return

        # ── Gate 3: 下单最小间隔 ──
        if self._last_order_dt is not None:
            elapsed_ms = (now_dt - self._last_order_dt).total_seconds() * 1000
            if elapsed_ms < self.min_order_interval_ms:
                return

        # ── Gate 4: 暖机期（z-score 未收敛） ──
        n = len(self._spread_history)
        if n < max(self.warmup_min_bars, self.spread_window // 4):
            return

        z = self._current_z
        abs_z = abs(z)

        # ── Gate 3: z-score 超过入场阈值 ──
        if abs_z < self.entry_z_score:
            # 无信号，重置确认计数
            self._confirm_long_a_short_b = 0
            self._confirm_short_a_long_b = 0
            return

        # ── Gate 4+5: LOB 方向一致性 + 确认计数 ──
        lob_a = self._lob_a
        lob_b = self._lob_b

        if z < -self.entry_z_score:
            # A 相对便宜 → 买 A / 卖 B
            # LOB: A 买盘旺(lob_a>0)，B 卖盘旺(lob_b<0)
            lob_ok = (not self.lob_gate_enabled) or (
                lob_a >= self.lob_min_score and lob_b <= -self.lob_min_score
            )
            if lob_ok:
                self._confirm_long_a_short_b += 1
                self._confirm_short_a_long_b = 0
            else:
                self._confirm_long_a_short_b = 0
                return

            if self._confirm_long_a_short_b >= self.confirm_ticks:
                self._confirm_long_a_short_b = 0
                self._submit_entry("LONG_A_SHORT_B", now_dt)

        elif z > self.entry_z_score:
            # A 相对贵 → 卖 A / 买 B
            lob_ok = (not self.lob_gate_enabled) or (
                lob_a <= -self.lob_min_score and lob_b >= self.lob_min_score
            )
            if lob_ok:
                self._confirm_short_a_long_b += 1
                self._confirm_long_a_short_b = 0
            else:
                self._confirm_short_a_long_b = 0
                return

            if self._confirm_short_a_long_b >= self.confirm_ticks:
                self._confirm_short_a_long_b = 0
                self._submit_entry("SHORT_A_LONG_B", now_dt)

    def _submit_entry(self, direction: str, now_dt: datetime) -> None:
        """同时提交两腿入场委托"""
        tick_a = self._tick_a
        tick_b = self._tick_b
        pt_a = self._pricetick_a
        pt_b = self._pricetick_b

        if direction == "LONG_A_SHORT_B":
            # A: 买入（激进，用 ask + slip）
            ask_a = float(tick_a.ask_price_1 or 0)
            if ask_a <= 0:
                self.write_log("⚠️ [ENTRY] A股ask1=0，跳过入场")
                return
            price_a = self._round_price(ask_a + self.entry_slip_ticks * pt_a, pt_a)
            dir_a, offset_a = Direction.LONG, Offset.OPEN

            # B: 卖出做空（激进，用 bid - slip）
            bid_b = float(tick_b.bid_price_1 or 0)
            if bid_b <= 0:
                self.write_log("⚠️ [ENTRY] B股bid1=0，跳过入场")
                return
            price_b = self._round_price(bid_b - self.entry_slip_ticks * pt_b, pt_b)
            dir_b, offset_b = Direction.SHORT, Offset.OPEN

        else:  # SHORT_A_LONG_B
            # A: 卖出做空（激进，用 bid - slip）
            bid_a = float(tick_a.bid_price_1 or 0)
            if bid_a <= 0:
                self.write_log("⚠️ [ENTRY] A股bid1=0，跳过入场")
                return
            price_a = self._round_price(bid_a - self.entry_slip_ticks * pt_a, pt_a)
            dir_a, offset_a = Direction.SHORT, Offset.OPEN

            # B: 买入（激进，用 ask + slip）
            ask_b = float(tick_b.ask_price_1 or 0)
            if ask_b <= 0:
                self.write_log("⚠️ [ENTRY] B股ask1=0，跳过入场")
                return
            price_b = self._round_price(ask_b + self.entry_slip_ticks * pt_b, pt_b)
            dir_b, offset_b = Direction.LONG, Offset.OPEN

        self.write_log(
            f"📡 [入场] {direction} | z={self._current_z:.3f} | "
            f"A={price_a:.2f}({dir_a.value}) B={price_b:.2f}({dir_b.value}) | "
            f"LOB: a={self._lob_a:.2f} b={self._lob_b:.2f}"
        )

        # 先提交 A 腿
        ids_a = self._submit_leg_a(dir_a, offset_a, price_a, self.volume_a, "ENTRY_A")
        if not ids_a:
            self.write_log("⚠️ [ENTRY] A腿提交失败，放弃入场")
            return

        # 再提交 B 腿
        ids_b = self._submit_leg_b(dir_b, offset_b, price_b, self.volume_b, "ENTRY_B")
        if not ids_b:
            self.write_log("⚠️ [ENTRY] B腿提交失败，撤销A腿")
            for oid in ids_a:
                self.cancel_order(oid)
            for oid in ids_a:
                self._active_orders_a.pop(oid, None)
            return

        # 记录入场状态
        self._pair_direction = direction
        self._entry_spread = self._current_spread
        self._leg_submit_time = now_dt
        self._leg_a_filled = False
        self._leg_b_filled = False
        self._leg_a_fill_vol = 0.0
        self._leg_b_fill_vol = 0.0
        self._last_order_dt = now_dt
        self._closing_in_progress = False

        self.write_log(f"✅ [入场] A腿={ids_a}, B腿={ids_b}")

    # ─────────── A 腿委托（使用 CtaTemplate 内置 buy/sell/short/cover） ───────────

    def _submit_leg_a(
        self,
        direction: Direction,
        offset: Offset,
        price: float,
        volume: int,
        purpose: str
    ) -> List[str]:
        """提交 A 腿委托（使用 CtaTemplate 内置接口）"""
        try:
            if direction == Direction.LONG:
                ids = self.buy(price, volume) if offset == Offset.OPEN else self.cover(price, volume)
            else:
                ids = self.short(price, volume) if offset == Offset.OPEN else self.sell(price, volume)

            if not ids:
                return []
            if isinstance(ids, str):
                ids = [ids]
            ids = [str(x) for x in ids]

            now = datetime.now()
            for oid in ids:
                self._active_orders_a[oid] = _PairOrderMeta(
                    created_dt=now,
                    purpose=purpose,
                    direction=direction,
                    price=price,
                    volume=volume,
                    is_b_leg=False
                )
            return ids

        except Exception as e:
            self.write_log(f"❌ A腿下单异常: {e}")
            return []

    # ─────────── B 腿委托（直接调用 CtaEngine 内部，复制 send_server_order） ───────────

    def _submit_leg_b(
        self,
        direction: Direction,
        offset: Offset,
        price: float,
        volume: int,
        purpose: str
    ) -> List[str]:
        """
        提交 B 腿委托（复制 CtaEngine.send_server_order 的实现）

        关键：手动注册委托到 orderid_strategy_map，
        使得 on_order / on_trade 能正确路由到本策略。
        并将 vt_orderid 加入 _b_orderid_set，防止 self.pos 被污染。
        """
        try:
            from vnpy.trader.object import OrderRequest

            main_engine = self.cta_engine.main_engine
            contract_b = main_engine.get_contract(self._vt_symbol_b)
            if contract_b is None:
                self.write_log(f"❌ B合约 {self._vt_symbol_b} 未找到，无法下单")
                return []

            gateway_name = getattr(contract_b, "gateway_name", "")

            # 构建委托请求（与 send_server_order 相同结构）
            original_req = OrderRequest(
                symbol=contract_b.symbol,
                exchange=contract_b.exchange,
                direction=direction,
                offset=offset,
                type=OrderType.LIMIT,
                price=price,
                volume=volume,
                reference=f"CtaStrategy_{self.strategy_name}"
            )

            # 通过 offset converter 转换（处理 kabu 融券 HoldID 转换）
            req_list = main_engine.convert_order_request(
                original_req,
                gateway_name,
                False,   # lock
                False    # net
            )

            vt_orderids: List[str] = []
            now = datetime.now()

            for req in req_list:
                vt_orderid = main_engine.send_order(req, gateway_name)
                if not vt_orderid:
                    continue

                vt_orderids.append(vt_orderid)

                # 通知 OrderConverter 更新（部分 VeighNa 版本需要）
                try:
                    main_engine.update_order_request(req, vt_orderid, gateway_name)
                except AttributeError:
                    pass

                # ── 关键：注册到 CtaEngine 路由表 ──
                # 使得 on_order / on_trade 能路由到本策略
                self.cta_engine.orderid_strategy_map[vt_orderid] = self
                self.cta_engine.strategy_orderid_map[self.strategy_name].add(vt_orderid)

                # ── 标记为 B 腿（防止 self.pos 被污染） ──
                self._b_orderid_set.add(vt_orderid)
                self._active_orders_b[vt_orderid] = _PairOrderMeta(
                    created_dt=now,
                    purpose=purpose,
                    direction=direction,
                    price=price,
                    volume=volume,
                    is_b_leg=True
                )

            if not vt_orderids:
                self.write_log(
                    f"⚠️ B腿委托提交返回空（kabu可能拒单）: {purpose} {direction.value} px={price:.2f}"
                )
            return vt_orderids

        except Exception as e:
            self.write_log(f"❌ B腿下单异常: {e}")
            return []

    # ═══════════════════════════════════════════
    #  委托 / 成交回调
    # ═══════════════════════════════════════════

    def on_order(self, order: OrderData) -> None:
        """委托状态更新回调"""
        oid = order.vt_orderid
        is_b = oid in self._b_orderid_set

        # 委托结束 → 从活跃委托表移除
        if not order.is_active():
            if is_b:
                self._active_orders_b.pop(oid, None)
            else:
                self._active_orders_a.pop(oid, None)

        try:
            traded = float(getattr(order, "traded", 0) or 0)
            leg_tag = "B" if is_b else "A"
            self.write_log(
                f"[ORDER-{leg_tag}] {oid[-8:]} {order.status.value} "
                f"px={float(order.price):.2f} "
                f"traded={traded:.0f}/{int(order.volume or 0)}"
            )
        except Exception:
            pass

        self.put_event()

    def on_trade(self, trade: TradeData) -> None:
        """
        成交回调

        关键：若为 B 腿成交，CtaEngine 已错误修改了 self.pos，
        需立即反向修正，改为更新 self._pos_b。

        CtaEngine 的 pos 更新规则：
          Direction.LONG  → self.pos += volume
          Direction.SHORT → self.pos -= volume
        """
        is_b = trade.vt_orderid in self._b_orderid_set
        vol = abs(float(trade.volume or 0))
        if vol <= 0:
            return

        offset = getattr(trade, "offset", None)
        leg_tag = "B" if is_b else "A"

        self.write_log(
            f"[TRADE-{leg_tag}] {trade.direction.value} {offset} "
            f"px={float(trade.price):.2f} vol={vol:.0f}"
        )

        if is_b:
            # ── pos 反污染：撤销 CtaEngine 对 self.pos 的错误修改 ──
            if trade.direction == Direction.LONG:
                self.pos -= vol       # CtaEngine 加了，我们减回来
                self._pos_b += vol    # 更新 B 腿持仓
            else:
                self.pos += vol       # CtaEngine 减了，我们加回来
                self._pos_b -= vol

            # 记录 B 腿入场成交
            if offset == Offset.OPEN:
                self._leg_b_filled = True
                self._leg_b_fill_vol += vol
                # 加权平均成交价
                if self._entry_price_b == 0:
                    self._entry_price_b = float(trade.price)
                else:
                    prev_vol = self._leg_b_fill_vol - vol
                    self._entry_price_b = (
                        (self._entry_price_b * prev_vol + float(trade.price) * vol)
                        / self._leg_b_fill_vol
                    )
            else:
                # B 腿平仓成交
                if abs(self._pos_b) < 1e-9:
                    self._pos_b = 0.0

        else:
            # ── A 腿成交：pos 已被 CtaEngine 正确更新 ──
            if offset == Offset.OPEN:
                self._leg_a_filled = True
                self._leg_a_fill_vol += vol
                # 记录入场时间（首次成交）
                if self._entry_time is None:
                    dt = trade.datetime
                    if dt is not None and getattr(dt, "tzinfo", None) is not None:
                        dt = dt.replace(tzinfo=None)
                    self._entry_time = dt
                # 加权平均成交价
                if self._entry_price_a == 0:
                    self._entry_price_a = float(trade.price)
                else:
                    prev_vol = self._leg_a_fill_vol - vol
                    self._entry_price_a = (
                        (self._entry_price_a * prev_vol + float(trade.price) * vol)
                        / self._leg_a_fill_vol
                    )
            else:
                # A 腿平仓成交
                if abs(self.pos) < 1e-9:
                    self.pos = 0

        # ── 两腿都成交时打印入场完成日志 ──
        if (self._leg_a_filled and self._leg_b_filled
                and self._entry_time is not None
                and self._pair_direction is not None):
            self.write_log(
                f"✅ [入场完成] {self._pair_direction} | "
                f"entryA={self._entry_price_a:.2f} entryB={self._entry_price_b:.2f} | "
                f"spread={self._entry_spread:.6f}"
            )

        # ── 检查是否已全部平仓 ──
        if not self._has_open_position() and self._pair_direction is not None:
            self._on_position_closed()

        self.put_event()

    def _on_position_closed(self) -> None:
        """仓位完全关闭时的收尾工作"""
        pnl = self._unrealized_pnl
        self._daily_pnl += pnl
        self.write_log(
            f"💰 [平仓完成] {self._pair_direction} | "
            f"pnl≈{pnl:+.0f}¥ | 日内累计={self._daily_pnl:+.0f}¥"
        )

        # 检查日内亏损上限
        if self._daily_pnl < self.max_daily_loss:
            self._is_trading_allowed = False
            self.write_log(
                f"🛑 [日内止损] 日内亏损{self._daily_pnl:+.0f}¥ < "
                f"限额{self.max_daily_loss:.0f}¥，今日停止交易"
            )

        self._reset_position_state()

    def _reset_position_state(self) -> None:
        """重置仓位相关状态，进入冷却期"""
        self._pair_direction = None
        self._entry_spread = 0.0
        self._entry_time = None
        self._entry_price_a = 0.0
        self._entry_price_b = 0.0
        self._leg_a_filled = False
        self._leg_b_filled = False
        self._leg_a_fill_vol = 0.0
        self._leg_b_fill_vol = 0.0
        self._leg_submit_time = None
        self._unrealized_pnl = 0.0
        self._closing_in_progress = False
        self._last_exit_dt = datetime.now()  # 进入冷却期

    # ═══════════════════════════════════════════
    #  出场逻辑
    # ═══════════════════════════════════════════

    def _check_exit(self, now_dt: datetime) -> None:
        """出场条件检查（4个条件，按优先级）"""
        if self._closing_in_progress:
            return

        z = self._current_z

        # ── 条件1: z-score 紧急暴走 ──
        if abs(z) >= self.emergency_z_score:
            self.write_log(
                f"🚨 [EMERGENCY] z={z:.3f} >= {self.emergency_z_score} → 紧急平仓"
            )
            self._emergency_close_all("emergency_z")
            return

        # ── 条件2: 单笔最大亏损 ──
        self._update_unrealized_pnl()
        if self._unrealized_pnl < self.max_loss_per_trade:
            self.write_log(
                f"🚨 [MAX_LOSS] pnl={self._unrealized_pnl:+.0f}¥ < "
                f"{self.max_loss_per_trade:.0f}¥ → 止损平仓"
            )
            self._emergency_close_all("max_loss")
            return

        # ── 条件3: 最长持仓超时 ──
        if self._entry_time is not None:
            hold_secs = (now_dt - self._entry_time).total_seconds()
            if hold_secs >= self.max_hold_seconds:
                self.write_log(
                    f"⏰ [TIME_EXIT] 持仓{hold_secs:.1f}s >= {self.max_hold_seconds}s → 超时平仓"
                )
                self._close_both_legs("time_exit")
                return

        # ── 条件4: z 回归正常范围 ──
        if self._pair_direction == "LONG_A_SHORT_B":
            # 当初 z 极负（A 便宜），等待 z 回升
            if z >= -self.exit_z_score:
                self.write_log(
                    f"✅ [Z_REVERSION] z={z:.3f} ≥ -{self.exit_z_score} → 回归平仓"
                )
                self._close_both_legs("z_reversion")
        elif self._pair_direction == "SHORT_A_LONG_B":
            # 当初 z 极正（A 贵），等待 z 下降
            if z <= self.exit_z_score:
                self.write_log(
                    f"✅ [Z_REVERSION] z={z:.3f} ≤ {self.exit_z_score} → 回归平仓"
                )
                self._close_both_legs("z_reversion")

    def _close_both_legs(self, reason: str) -> None:
        """正常平仓两腿（限价单，使用 exit_slip_ticks）"""
        if self._closing_in_progress:
            return
        self._closing_in_progress = True

        self.write_log(
            f"📤 [CLOSE-{reason}] pos_a={self.pos:.0f} pos_b={self._pos_b:.0f}"
        )

        # 撤销所有待成交委托
        self.cancel_all()

        tick_a = self._tick_a
        tick_b = self._tick_b
        pt_a = self._pricetick_a
        pt_b = self._pricetick_b

        ok_a = True
        ok_b = True

        # ── 平 A 腿 ──
        if abs(self.pos) > 0:
            vol_a = abs(int(self.pos))
            if self.pos > 0:
                # A 多头 → 限价卖出
                bid_a = float(tick_a.bid_price_1 or 0) if tick_a else 0
                if bid_a <= 0:
                    self.write_log("⚠️ A股bid1=0，无法平仓A腿")
                    self._closing_in_progress = False
                    return
                price_a = self._round_price(bid_a - self.exit_slip_ticks * pt_a, pt_a)
                ids = self._submit_leg_a(
                    Direction.SHORT, Offset.CLOSE, price_a, vol_a, f"EXIT_{reason}_A"
                )
            else:
                # A 空头 → 限价买入
                ask_a = float(tick_a.ask_price_1 or 0) if tick_a else 0
                if ask_a <= 0:
                    self.write_log("⚠️ A股ask1=0，无法平仓A腿")
                    self._closing_in_progress = False
                    return
                price_a = self._round_price(ask_a + self.exit_slip_ticks * pt_a, pt_a)
                ids = self._submit_leg_a(
                    Direction.LONG, Offset.CLOSE, price_a, vol_a, f"EXIT_{reason}_A"
                )
            if not ids:
                ok_a = False
                self.write_log("⚠️ A腿平仓单提交失败")

        # ── 平 B 腿 ──
        if abs(self._pos_b) > 0:
            vol_b = abs(int(self._pos_b))
            if self._pos_b > 0:
                # B 多头 → 限价卖出
                bid_b = float(tick_b.bid_price_1 or 0) if tick_b else 0
                if bid_b <= 0:
                    self.write_log("⚠️ B股bid1=0，无法平仓B腿")
                    self._closing_in_progress = False
                    return
                price_b = self._round_price(bid_b - self.exit_slip_ticks * pt_b, pt_b)
                ids_b = self._submit_leg_b(
                    Direction.SHORT, Offset.CLOSE, price_b, vol_b, f"EXIT_{reason}_B"
                )
            else:
                # B 空头 → 限价买入
                ask_b = float(tick_b.ask_price_1 or 0) if tick_b else 0
                if ask_b <= 0:
                    self.write_log("⚠️ B股ask1=0，无法平仓B腿")
                    self._closing_in_progress = False
                    return
                price_b = self._round_price(ask_b + self.exit_slip_ticks * pt_b, pt_b)
                ids_b = self._submit_leg_b(
                    Direction.LONG, Offset.CLOSE, price_b, vol_b, f"EXIT_{reason}_B"
                )
            if not ids_b:
                ok_b = False
                self.write_log("⚠️ B腿平仓单提交失败")

        # 两腿都失败 → 重置标志，允许下次重试
        if not ok_a and not ok_b:
            self._closing_in_progress = False

    def _emergency_close_all(self, reason: str) -> None:
        """紧急平仓（使用更激进的滑点，当前复用 _close_both_legs）"""
        self._close_both_legs(reason)

    # ═══════════════════════════════════════════
    #  腿不同步超时处理
    # ═══════════════════════════════════════════

    def _check_leg_timeout(self, now_dt: datetime) -> None:
        """检查腿不同步超时并单边平仓"""
        if self._leg_submit_time is None:
            return

        # 两腿都成交且有持仓 → 无需超时处理
        if self._has_open_position() and self._leg_a_filled and self._leg_b_filled:
            return

        # 未开始入场也无需处理
        if not self._leg_a_filled and not self._leg_b_filled and not self._has_open_position():
            return

        elapsed = (now_dt - self._leg_submit_time).total_seconds()
        if elapsed < self.leg_timeout_seconds:
            return

        self.write_log(
            f"⚠️ [LEG_TIMEOUT] {elapsed:.1f}s | "
            f"A_filled={self._leg_a_filled}({self._leg_a_fill_vol:.0f}) "
            f"B_filled={self._leg_b_filled}({self._leg_b_fill_vol:.0f})"
        )

        # 撤销所有待成交委托
        self.cancel_all()

        if self._leg_a_filled and not self._leg_b_filled:
            # 只有 A 成交 → 紧急平仓 A
            self.write_log("⚡ [TIMEOUT] 只有A腿成交，紧急平仓A腿")
            self._emergency_close_a()
            self._leg_submit_time = None

        elif self._leg_b_filled and not self._leg_a_filled:
            # 只有 B 成交 → 紧急平仓 B
            self.write_log("⚡ [TIMEOUT] 只有B腿成交，紧急平仓B腿")
            self._emergency_close_b()
            self._leg_submit_time = None

        elif not self._leg_a_filled and not self._leg_b_filled:
            # 两腿都未成交 → 重置状态
            self.write_log("⚡ [TIMEOUT] 两腿均未成交，重置状态")
            self._reset_position_state()

    def _emergency_close_a(self) -> None:
        """紧急单边平仓 A 腿"""
        if abs(self.pos) < 1e-9 or self._tick_a is None:
            return
        pt_a = self._pricetick_a
        tick_a = self._tick_a
        vol_a = abs(int(self.pos))

        if self.pos > 0:
            bid = float(tick_a.bid_price_1 or 0)
            if bid <= 0:
                return
            price = self._round_price(bid - self.exit_slip_ticks * pt_a, pt_a)
            self._submit_leg_a(Direction.SHORT, Offset.CLOSE, price, vol_a, "EMERGENCY_A")
        else:
            ask = float(tick_a.ask_price_1 or 0)
            if ask <= 0:
                return
            price = self._round_price(ask + self.exit_slip_ticks * pt_a, pt_a)
            self._submit_leg_a(Direction.LONG, Offset.CLOSE, price, vol_a, "EMERGENCY_A")

    def _emergency_close_b(self) -> None:
        """紧急单边平仓 B 腿"""
        if abs(self._pos_b) < 1e-9 or self._tick_b is None:
            return
        pt_b = self._pricetick_b
        tick_b = self._tick_b
        vol_b = abs(int(self._pos_b))

        if self._pos_b > 0:
            bid = float(tick_b.bid_price_1 or 0)
            if bid <= 0:
                return
            price = self._round_price(bid - self.exit_slip_ticks * pt_b, pt_b)
            self._submit_leg_b(Direction.SHORT, Offset.CLOSE, price, vol_b, "EMERGENCY_B")
        else:
            ask = float(tick_b.ask_price_1 or 0)
            if ask <= 0:
                return
            price = self._round_price(ask + self.exit_slip_ticks * pt_b, pt_b)
            self._submit_leg_b(Direction.LONG, Offset.CLOSE, price, vol_b, "EMERGENCY_B")

    # ═══════════════════════════════════════════
    #  Bar 回调（本策略不使用）
    # ═══════════════════════════════════════════

    def on_bar(self, bar) -> None:
        """配对套利策略仅使用 Tick 数据，忽略 Bar"""
        pass
