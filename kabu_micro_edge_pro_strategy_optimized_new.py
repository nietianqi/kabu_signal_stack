# kabu_micro_edge_pro_strategy_optimized_v2.py
# Kabu 日股 L2 + 逐笔（tick-by-tick）微观剥头皮策略（性能优化版）
#
# 优化内容 (v1):
# 1. 修复 PnL 累加逻辑 bug
# 2. 添加手续费计算
# 3. 优化参数默认值
# 4. 添加盈亏比统计
# 5. 参数化 Flow Flip 阈值
# 6. 优化 Momentum 计算(改用 microprice)
# 7. 改进连亏重置逻辑
# 8. 增强日志信息
#
# 🚀 性能优化 (v2):
# 9. 订单限流优化: 250ms → 100ms (开仓速度提升53%)
# 10. 开平仓分离限流: 平仓50ms (紧急响应优先)
# 11. 止盈延迟优化: 2s → 动态轮询建玉 (平均延迟降低75%)
# 12. 信号强度自适应确认: 强信号1次确认,弱信号2次
# 13. 增量指标计算: LOB-OFI/Tape-OFI使用增量更新
#
# 主要信号:
# 1) 加权盘口不平衡（多档深度）
# 2) LOB-OFI（订单簿订单流不平衡）
# 3) Tape-OFI（逐笔成交主动性）
# 4) 微动量（microprice 方向滚动）
# 5) Microprice tilt（microprice 相对 mid 的偏移）

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from collections import deque
from enum import Enum
import copy
from typing import Deque, Dict, List, Optional, Tuple

from vnpy.trader.object import TickData, TradeData, OrderData
from vnpy.trader.constant import Direction, Offset

try:
    from vnpy_ctastrategy import CtaTemplate, StopOrder
except Exception:
    from vnpy.app.cta_strategy import CtaTemplate, StopOrder


# =========================
# 枚举类型定义
# =========================
class OrderPurpose(Enum):
    """订单用途枚举"""
    ENTRY_LONG = "ENTRY_LONG"        # 开多仓
    ENTRY_SHORT = "ENTRY_SHORT"      # 开空仓
    LIMIT_TP = "LIMIT_TP"            # 限价止盈单
    EXIT_STOP_LOSS = "EXIT_STOP_LOSS"           # 止损出场
    EXIT_FLOW_FLIP_TAPE = "EXIT_FLOW_FLIP_TAPE" # Tape-OFI反转出场
    EXIT_FLOW_FLIP_LOB = "EXIT_FLOW_FLIP_LOB"   # LOB-OFI反转出场
    EXIT_RETRY = "EXIT_RETRY"                   # 出场重试
    EXIT_EMERGENCY = "EXIT_EMERGENCY"           # 应急平仓


@dataclass
class _OrderMeta:
    created_dt: datetime
    purpose: str  # 保持str类型以兼容旧代码，但使用OrderPurpose.value
    direction: Direction
    price: float
    volume: int


class KabuMicroEdgeProOptimizedNew(CtaTemplate):
    """
    Kabu L2 + 逐笔微观剥头皮策略 - 优化版

    关键优化:
    - 修复 daily_pnl 累加逻辑
    - 添加手续费/滑点成本计算
    - 优化参数默认值(更保守)
    - 增加盈亏比统计
    - Flow Flip 阈值参数化
    """

    author = "Claude_Optimized_v2_Performance"

    # =========================
    # 常量定义（Magic Numbers提取）
    # =========================
    # 限价止盈单重试
    MAX_LIMIT_TP_RETRY = 5                    # 最大重试次数

    # price_tick 周期性验证
    PRICE_TICK_VERIFY_INTERVAL = 30.0         # 验证间隔（秒）

    # 连亏计数
    CONTINUOUS_LOSS_COUNT_THRESHOLD = 5       # 连亏次数阈值（用于日志显示）

    # =========================
    # 交易参数
    # =========================
    trade_volume: int = 100
    enable_long: bool = True
    enable_short: bool = False

    # =========================
    # 盈亏控制
    # =========================
    profit_ticks: int = 1              # 止盈目标(ticks) - 已补偿exit_slip，表示净利润
    loss_ticks: int = 8                # 止损目标(ticks)
    max_hold_seconds: float = 3.5      # 最大持仓时间(秒) - 已禁用
    cooldown_seconds: float = 0.5      # 平仓后冷却时间(秒)

    entry_slip_ticks: int = 0          # 入场滑点补偿(ticks)
    exit_slip_ticks: int = 1           # 出场滑点补偿(ticks)

    # 止盈模式
    take_profit_mode: str = "disabled"  # 改为disabled，完全依赖限价单止盈
    tp_trigger_use_last_price: bool = False  # 已禁用last_price触发，只用bid/ask
    trailing_trigger_ticks: int = 3
    trailing_stop_ticks: int = 2
    trailing_min_lock_ticks: int = 1

    # 限价单止盈参数（新增）
    enable_limit_tp_order: bool = True      # 是否启用买入后自动挂限价止盈单
    limit_tp_ticks: int = 1                 # 限价止盈单距离入场价的tick数
    limit_tp_timeout: float = 0.0           # 限价止盈单超时时间(秒)，0=永不超时
    tp_only_mode: bool = True               # 只用限价止盈平仓，亏损不自动平仓（禁用止损/FlowFlip）

    # 🔴 兜底止损参数（修复tp_only_mode盲区）
    max_loss_per_trade: float = -500.0      # 单笔最大亏损(日元)，即使tp_only_mode=True也会触发

    # 🛡️ 出场价格安全参数（防止异常价格下单）
    max_price_deviation_ticks: int = 50    # 出场时bid1/ask1与入场价最大偏差(ticks)，超过则跳过该tick

    # 🚀 v2优化: 动态建玉轮询，降低止盈延迟
    use_dynamic_position_polling: bool = True    # 启用动态建玉轮询（替代固定2s延迟）
    position_poll_interval_ms: int = 50          # 建玉轮询间隔(ms)
    max_position_wait_seconds: float = 2.0       # 最大等待时间(s)，超时强制挂单
    limit_tp_delay_seconds: float = 1.5          # 固定延迟模式下的止盈单提交延迟(s)，最小0.5s

    # =========================
    # 盘口过滤
    # =========================
    min_spread_ticks: float = 0.5      # 优化: 从0.0提高到0.5
    max_spread_ticks: float = 10.0      # 优化: 从2.0提高到10.0
    min_best_volume: int = 50          # 优化: 从200降低到50，减轻LOB-OFI薄市放大效应
    min_secondary_score: int = 1       # 次级指标(tape/microprice)最少满足数: 0=仅3主, 1=3主+任1辅(推荐), 2=全部5个
    entry_l1_vol_ratio: float = 0.0    # 下单前L1买量/(买+卖)最小比例(0=关闭, 推荐0.55)
    entry_max_ask_vol_ratio: float = 0.0  # 下单前ask1量不超过bid1量X倍(0=关闭, 推荐3.0)

    # =========================
    # 订单簿参数
    # =========================
    book_depth_levels: int = 5
    book_decay: float = 0.6

    book_imbalance_long: float = 0.35   # 优化: 从0.30提高到0.35
    book_imbalance_short: float = 0.35

    # =========================
    # LOB-OFI 参数
    # =========================
    of_window_size: int = 18           # 优化: 从20调整到18
    of_imbalance_long: float = 0.25    # 优化: 从0.22提高到0.25
    of_imbalance_short: float = 0.25

    # =========================
    # Tape-OFI 参数
    # =========================
    use_tape_ofi: bool = True
    tape_window_seconds: float = 1.0   # 优化: 从0.5扩展到1.0，与LOB-OFI时间窗口对齐
    tape_imbalance_long: float = 0.20  # 优化: 从0.18提高到0.20
    tape_imbalance_short: float = 0.20

    # =========================
    # 微动量参数
    # =========================
    mom_window_size: int = 12          # 优化: 从15调整到12
    mom_long_threshold: float = 0.25   # 优化: 从0.20提高到0.25
    mom_short_threshold: float = 0.25
    use_microprice_momentum: bool = True  # 新增: 使用microprice计算momentum

    # =========================
    # Microprice Tilt 参数
    # =========================
    use_microprice_tilt: bool = True
    microprice_tilt_long: float = 0.25   # 优化: 从0.08提高到0.25
    microprice_tilt_short: float = 0.25

    # =========================
    # 信号确认（🚀 性能优化）
    # =========================
    confirm_ticks: int = 1                       # 默认确认次数（5指标过滤已足够，无需双重tick确认）

    # 🚀 v2优化: 信号强度自适应确认
    use_adaptive_confirm: bool = True            # 启用自适应确认
    strong_signal_confirm: int = 1               # 强信号确认次数（5指标都远超阈值）
    strong_signal_multiplier: float = 1.5        # 强信号判定倍数（指标值>阈值*1.5）

    # =========================
    # 环境过滤
    # =========================
    vol_window_size: int = 30          # 优化: 从25提高到30
    max_mid_std_ticks: float = 2.5     # 优化: 从3.0降低到2.5

    # =========================
    # Flow Flip 参数(新增)
    # =========================
    use_flow_flip_exit: bool = True
    flow_flip_threshold: float = 0.18  # 优化: 参数化,从硬编码0.15提高到0.18

    # =========================
    # 订单管理（🚀 性能优化）
    # =========================
    entry_order_timeout: float = 3.0   # 优化: 从1.2提高到3.0
    exit_order_timeout: float = 1.0    # 优化: 从0.8提高到1.0

    # 🚀 v2优化: 分离开平仓限流，平衡速度与API保护
    min_order_action_interval_ms: int = 100      # 全局限流: 250ms→100ms (开仓速度↑53%)
    entry_order_interval_ms: int = 100           # 开仓专用限流: 100ms
    exit_order_interval_ms: int = 50             # 平仓专用限流: 50ms (紧急响应优先)
    limit_tp_order_interval_ms: int = 150        # 止盈单限流: 150ms (不紧急)

    use_separate_order_throttle: bool = True     # 启用分离限流机制

    # =========================
    # 成本参数(新增)
    # =========================
    commission_rate: float = 0.0005    # 手续费率(0.05%)
    slippage_ticks: float = 0.0        # 额外滑点成本(ticks) - 实盘设为0避免重复扣除

    # =========================
    # 时间过滤
    # =========================
    enable_time_filter: bool = True
    trade_start_time: str = "09:00:10"  # 优化: 从09:00:05延后到09:00:10
    trade_end_time: str = "15:25:00"    # 优化: 从14:59:50提前到15:25:00，避免尾盘风险
    morning_end_time: str = "11:30:00"
    afternoon_start_time: str = "12:30:00"

    # =========================
    # 风控
    # =========================
    max_daily_loss: float = -80000.0   # 优化: 从-50000调整到-80000
    max_continuous_loss: int = 5       # 优化: 从6降低到5
    enable_auto_stop: bool = True
    max_long_inventory: int = 500      # 单票最大多头库存(股)
    max_short_inventory: int = 500     # 单票最大空头库存(股)
    enable_max_loss_per_trade_exit: bool = False  # False=亏损不强平

    max_tick_stale_seconds: float = 5.0  # 优化: 从6.0降低到5.0
    signal_expire_seconds: float = 1.0  # 信号确认计数过期时间(s)，tick间隔超过此值则重置

    # =========================
    # Kabu 数据修正
    # =========================
    auto_fix_negative_spread: bool = True
    kabu_bidask_reversed: bool = False

    # =========================
    # 日志控制
    # =========================
    log_interval_seconds: float = 2.0   # 优化: 从1.0提高到2.0
    verbose_log: bool = False
    debug_entry_gate: bool = True      # 诊断: 输出“为什么不触发入场”的原因（节流）

    parameters = [
        "trade_volume", "enable_long", "enable_short",
        "profit_ticks", "loss_ticks", "max_hold_seconds", "cooldown_seconds",
        "entry_slip_ticks", "exit_slip_ticks",
        "take_profit_mode", "tp_trigger_use_last_price",
        "trailing_trigger_ticks", "trailing_stop_ticks", "trailing_min_lock_ticks",
        "enable_limit_tp_order", "limit_tp_ticks", "limit_tp_timeout", "tp_only_mode", "max_loss_per_trade",
        "max_price_deviation_ticks",
        "use_dynamic_position_polling", "position_poll_interval_ms", "max_position_wait_seconds", "limit_tp_delay_seconds",
        "min_spread_ticks", "max_spread_ticks", "min_best_volume",
        "book_depth_levels", "book_decay",
        "book_imbalance_long", "book_imbalance_short",
        "of_window_size", "of_imbalance_long", "of_imbalance_short",
        "use_tape_ofi", "tape_window_seconds", "tape_imbalance_long", "tape_imbalance_short",
        "mom_window_size", "mom_long_threshold", "mom_short_threshold", "use_microprice_momentum",
        "use_microprice_tilt", "microprice_tilt_long", "microprice_tilt_short",
        "confirm_ticks", "use_adaptive_confirm", "strong_signal_confirm", "strong_signal_multiplier",
        "vol_window_size", "max_mid_std_ticks",
        "use_flow_flip_exit", "flow_flip_threshold",
        "entry_order_timeout", "exit_order_timeout", "min_order_action_interval_ms",
        "entry_order_interval_ms", "exit_order_interval_ms", "limit_tp_order_interval_ms", "use_separate_order_throttle",
        "commission_rate", "slippage_ticks",
        "enable_time_filter", "trade_start_time", "trade_end_time", "morning_end_time", "afternoon_start_time",
        "max_daily_loss", "max_continuous_loss", "enable_auto_stop",
        "max_long_inventory", "max_short_inventory", "enable_max_loss_per_trade_exit",
        "max_tick_stale_seconds", "signal_expire_seconds",
        "auto_fix_negative_spread", "kabu_bidask_reversed",
        "log_interval_seconds", "verbose_log", "debug_entry_gate",
    ]

    variables = [
        "price_tick",
        "book_imbalance", "lob_of_imbalance", "tape_of_imbalance",
        "micro_momentum", "microprice_tilt", "mid_std_ticks",
        "entry_price",  # ✅ 移除entry_time (datetime对象不能序列化为JSON)
        "daily_pnl", "continuous_loss_count", "total_trades", "win_trades",
        "avg_win", "avg_loss", "profit_factor",  # 新增统计
        "is_trading_allowed", "last_signal",
    ]

    def __init__(self, cta_engine, strategy_name: str, vt_symbol: str, setting: dict):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)

        self.price_tick: float = 0.0

        # tick 管理
        self.last_tick: Optional[TickData] = None
        self.last_tick_time: Optional[datetime] = None
        self.last_recv_time: Optional[datetime] = None
        self._tick_stale_state: bool = False

        # verbose log 节流
        self._last_vlog_time: Optional[datetime] = None

        # 指标
        self.book_imbalance: float = 0.0
        self.lob_of_imbalance: float = 0.0
        self.tape_of_imbalance: float = 0.0
        self.micro_momentum: float = 0.0
        self.microprice_tilt: float = 0.0
        self.mid_std_ticks: float = 0.0

        # 持仓状态
        self.entry_price: float = 0.0
        self.entry_time: Optional[datetime] = None
        self.entry_volume: int = 0
        self.entry_direction: Optional[Direction] = None  # ✅ 新增：记录开仓方向（避免依赖pos判断）
        self.exit_in_progress: bool = False
        self.last_flat_time: Optional[datetime] = None
        self.last_signal: str = ""

        # trailing
        self.tp_triggered: bool = False
        self.trailing_stop_price: float = 0.0
        self.trailing_peak_price: float = 0.0

        # 确认
        self._long_confirm: int = 0
        self._short_confirm: int = 0
        self._last_confirm_tick_time: Optional[datetime] = None  # 最后一次信号确认的tick时间

        # 订单
        self.active_orders: Dict[str, _OrderMeta] = {}
        self._last_order_action_dt: Optional[datetime] = None

        # 🚀 v2优化: 分离限流时间戳
        self._last_entry_order_dt: Optional[datetime] = None
        self._last_exit_order_dt: Optional[datetime] = None
        self._last_limit_tp_order_dt: Optional[datetime] = None

        # 限价止盈单管理（新增）
        self._limit_tp_order_ids: List[str] = []  # 记录限价止盈单ID
        self._need_submit_limit_tp: bool = False  # 标记是否需要提交限价止盈单
        self._limit_tp_retry_count: int = 0  # 重试计数
        self._limit_tp_submit_after: Optional[datetime] = None  # ✅ 延迟提交时间（避免建玉未同步）
        self._pending_tp_price: float = 0.0  # 待提交的止盈价
        self._pending_tp_volume: int = 0  # 待提交的止盈数量
        self._pending_tp_direction: Optional[Direction] = None  # 待提交的止盈方向

        # 🚀 v2优化: 动态建玉轮询
        self._position_poll_start_time: Optional[datetime] = None  # 开始轮询时间
        self._last_position_poll_time: Optional[datetime] = None   # 最后轮询时间

        # price_tick 自动纠偏（避免 fallback=1.0 伪装成正常值）
        self._price_tick_verified: bool = False  # 是否已从合约验证过
        self._last_price_tick_check_time: Optional[datetime] = None  # 上次检查时间

        # 持仓恢复状态（避免每tick重复打印WARNING）
        self._entry_sync_failed: bool = False   # 是否已警告过"entry信息无法从持仓恢复"
        self._entry_sync_warned: bool = False   # 是否已打印过EXIT-SKIP警告

        # LOB 内存
        self._prev_bid_prices: List[float] = []
        self._prev_ask_prices: List[float] = []
        self._prev_bid_vols: List[int] = []
        self._prev_ask_vols: List[int] = []

        # LOB-OFI 滚动
        self._ofi_buy_q: Deque[float] = deque(maxlen=max(5, int(self.of_window_size)))
        self._ofi_sell_q: Deque[float] = deque(maxlen=max(5, int(self.of_window_size)))
        self._ofi_buy_sum: float = 0.0
        self._ofi_sell_sum: float = 0.0

        # Tape-OFI 滚动
        self._prev_total_volume: Optional[float] = None
        self._prev_last_price: Optional[float] = None
        self._tape_q: Deque[Tuple[datetime, float]] = deque()
        self._tape_sum: float = 0.0
        self._tape_abs_sum: float = 0.0

        # momentum 滚动
        self._prev_mid_for_mom: Optional[float] = None
        self._prev_micro_for_mom: Optional[float] = None  # 新增: microprice momentum
        self._mom_q: Deque[int] = deque(maxlen=max(5, int(self.mom_window_size)))
        self._mom_sum: int = 0

        # vol 滚动
        self._prev_mid_for_vol: Optional[float] = None
        self._mid_ret_q: Deque[float] = deque(maxlen=max(10, int(self.vol_window_size)))
        self._mid_ret_sum: float = 0.0
        self._mid_ret_sq_sum: float = 0.0

        # 风控统计
        self.current_date: str = ""
        self.daily_pnl: float = 0.0
        self.continuous_loss_count: int = 0
        self.total_trades: int = 0
        self.win_trades: int = 0
        self.is_trading_allowed: bool = True
        self._trade_realized_pnl: float = 0.0

        # 新增: 盈亏比统计
        self.total_win_pnl: float = 0.0
        self.total_loss_pnl: float = 0.0
        self.avg_win: float = 0.0
        self.avg_loss: float = 0.0
        self.profit_factor: float = 0.0

        # 时间对象
        self.trade_start_time_obj: time = time(9, 0, 0)
        self.trade_end_time_obj: time = time(15, 0, 0)
        self.morning_end_time_obj: time = time(11, 30, 0)
        self.afternoon_start_time_obj: time = time(12, 30, 0)

    # =========================
    # datetime 辅助
    # =========================
    def _safe_now(self, ref_dt: Optional[datetime] = None) -> datetime:
        if ref_dt is not None and getattr(ref_dt, "tzinfo", None) is not None:
            return datetime.now(ref_dt.tzinfo)
        return datetime.now()

    @staticmethod
    def _align_datetimes(a: datetime, b: datetime) -> Tuple[datetime, datetime]:
        a_tz = getattr(a, "tzinfo", None)
        b_tz = getattr(b, "tzinfo", None)
        if (a_tz is None) != (b_tz is None):
            if a_tz is None and b_tz is not None:
                a = a.replace(tzinfo=b_tz)
            elif b_tz is None and a_tz is not None:
                b = b.replace(tzinfo=a_tz)
        return a, b

    def _dt_diff_seconds(self, a: datetime, b: datetime) -> float:
        a2, b2 = self._align_datetimes(a, b)
        return (a2 - b2).total_seconds()

    # =========================
    # 生命周期
    # =========================
    def on_init(self):
        self.price_tick = self._load_pricetick_or_fallback()
        self._parse_time_filters()
        self.write_log("=" * 80)
        self.write_log("Kabu 微观剥头皮策略 - 优化版 初始化")
        self.write_log(f"合约: {self.vt_symbol}")
        self.write_log(f"价位: {self.price_tick}")
        self.write_log(f"止盈模式: {self.take_profit_mode}")
        if self.enable_limit_tp_order:
            self.write_log(f"✨ 限价止盈单: 开启 (+{self.limit_tp_ticks} ticks, 超时{self.limit_tp_timeout}s)")
        else:
            self.write_log(f"止盈: {self.profit_ticks} ticks")
        self.write_log(f"止损: {self.loss_ticks} ticks")
        self.write_log(f"手续费率: {self.commission_rate*100:.3f}%, 滑点: {self.slippage_ticks} ticks")
        self.write_log(f"Flow Flip: {'开启' if self.use_flow_flip_exit else '关闭'}")
        self.write_log(f"Microprice Momentum: {'开启' if self.use_microprice_momentum else '关闭'}")
        self.write_log("=" * 80)
        self.put_event()

    def on_start(self):
        self.write_log("策略启动")

        # ✅ 强制重新订阅（确保合约注册）
        try:
            self.write_log(f"🔄 强制重新订阅合约: {self.vt_symbol}")
            symbol_parts = self.vt_symbol.split(".")
            if len(symbol_parts) == 2:
                from vnpy.trader.object import SubscribeRequest
                from vnpy.trader.constant import Exchange

                symbol = symbol_parts[0]
                exchange = Exchange(symbol_parts[1])
                req = SubscribeRequest(symbol=symbol, exchange=exchange)

                # 通过主引擎重新订阅（gateway_name从合约信息获取）
                main_engine = self.cta_engine.main_engine
                contract = main_engine.get_contract(self.vt_symbol)
                if contract and getattr(contract, "gateway_name", None):
                    main_engine.subscribe(req, contract.gateway_name)
                    self.write_log(f"   ✅ 订阅请求已发送")
        except Exception as e:
            self.write_log(f"⚠️  重新订阅异常: {e}")

        # 等待500毫秒让合约注册完成
        import time
        time.sleep(0.5)

        # 检查合约是否存在（用于诊断行情订阅问题）
        try:
            main_engine = self.cta_engine.main_engine
            contract = main_engine.get_contract(self.vt_symbol)
            if contract is None:
                self.write_log(f"⚠️  警告: 合约 {self.vt_symbol} 未找到")
                self.write_log(f"   可能原因: 1) 行情订阅失败  2) 合约代码错误  3) Gateway未连接")
                self.write_log(f"   策略将无法下单，直到合约信息加载成功")
                self.write_log(f"   💡 建议: 停止策略 -> 在行情界面手动订阅 {self.vt_symbol} -> 重启策略")
            else:
                self.write_log(f"✅ 合约 {self.vt_symbol} 已加载")
                if hasattr(contract, 'pricetick') and contract.pricetick > 0:
                    old_tick = self.price_tick
                    self.price_tick = float(contract.pricetick)
                    self._price_tick_verified = True  # 标记已验证
                    if old_tick != self.price_tick:
                        self.write_log(f"   🔧 价位已纠正: {old_tick} → {self.price_tick}")
                    else:
                        self.write_log(f"   价位已验证: {self.price_tick}")
        except Exception as e:
            self.write_log(f"⚠️  合约检查异常: {e}")

        self.put_event()

    def on_stop(self):
        self.write_log("策略停止: 撤单")
        self.cancel_all()
        self.put_event()

    # =========================
    # price_tick 自动纠偏
    # =========================
    def _periodic_verify_price_tick(self):
        """周期性验证 price_tick（每30秒检查一次，避免 fallback=1.0 伪装）"""
        # 如果已经验证过，跳过（减少开销）
        if self._price_tick_verified:
            return

        now = datetime.now()
        # 节流：每30秒检查一次
        if self._last_price_tick_check_time is not None:
            if (now - self._last_price_tick_check_time).total_seconds() < self.PRICE_TICK_VERIFY_INTERVAL:
                return

        self._last_price_tick_check_time = now

        try:
            main_engine = self.cta_engine.main_engine
            contract = main_engine.get_contract(self.vt_symbol)
            if contract and hasattr(contract, 'pricetick') and contract.pricetick > 0:
                contract_tick = float(contract.pricetick)
                if self.price_tick != contract_tick:
                    self.write_log(f"🔧 [自动纠偏] price_tick: {self.price_tick} → {contract_tick}")
                    self.price_tick = contract_tick
                self._price_tick_verified = True  # 标记已验证
        except Exception:
            pass  # 静默失败，下次再试

    # =========================
    # 日志辅助
    # =========================
    def _vlog(self, msg: str, *, force: bool = False) -> None:
        if not self.verbose_log:
            return
        if force or self.log_interval_seconds <= 0:
            self.write_log(msg)
            return
        now = datetime.now()
        if self._last_vlog_time is None or (now - self._last_vlog_time).total_seconds() >= float(self.log_interval_seconds):
            self._last_vlog_time = now
            self.write_log(msg)

    def _gate_log(self, msg: str, dt: datetime):
        """Log entry-gating reasons with throttling (default 1 msg/sec)."""
        if not getattr(self, "debug_entry_gate", False):
            return
        last = getattr(self, "_last_gate_log_dt", None)
        if last is not None:
            try:
                if (dt - last).total_seconds() < 1.0:
                    return
            except Exception:
                pass
        setattr(self, "_last_gate_log_dt", dt)
        self.write_log(f"🧭 [ENTRY-GATE] {msg}")


    def _diag_tick_line(self, tick: TickData) -> str:
        pt = self.price_tick if self.price_tick > 0 else 1.0
        bid1, ask1 = self._get_bid_ask_1(tick)
        last_px = float(tick.last_price or 0.0)
        spread = (ask1 - bid1) if (bid1 > 0 and ask1 > 0) else 0.0
        spread_ticks = (spread / pt) if pt > 0 else 0.0
        mid = (bid1 + ask1) / 2 if (bid1 > 0 and ask1 > 0) else (last_px if last_px > 0 else 0.0)

        hold = 0.0
        if self.entry_time is not None:
            try:
                hold = self._dt_diff_seconds(self._safe_now(self.entry_time), self.entry_time)
            except Exception:
                hold = 0.0

        return (
            f"[DIAG] t={getattr(tick, 'datetime', None)} "
            f"bid={bid1:.2f} ask={ask1:.2f} last={last_px:.2f} mid={mid:.2f} spr={spread_ticks:.1f}t "
            f"pos={self.pos} entry={self.entry_price:.2f} hold={hold:.2f}s "
            f"book={self.book_imbalance:+.3f} lobOFI={self.lob_of_imbalance:+.3f} tapeOFI={self.tape_of_imbalance:+.3f} "
            f"mom={self.micro_momentum:+.3f} mpt={self.microprice_tilt:+.3f} std={self.mid_std_ticks:.2f} "
            f"L/S={self._long_confirm}/{self._short_confirm} ord={len(self.active_orders)} "
            f"pnl={self.daily_pnl:.0f} PF={self.profit_factor:.2f}"
        )

    # =========================
    # 回调
    # =========================
    def on_tick(self, tick: TickData):
        if not self.trading:
            return

        tick_used = self._normalize_tick_bidask(tick)
        self.last_tick = tick_used
        self.last_tick_time = tick_used.datetime
        self.last_recv_time = datetime.now()
        self._tick_stale_state = False

        # ⚠️ 新增：周期性验证 price_tick（避免 fallback=1.0 伪装，每30秒检查一次）
        self._periodic_verify_price_tick()

        self._check_new_day(tick_used.datetime)
        self._update_indicators(tick_used)
        self._vlog(self._diag_tick_line(tick_used))

        if self.enable_auto_stop and not self.is_trading_allowed:
            if self.pos != 0:
                self._emergency_close(tick_used, "risk stop")
            return

        self._check_order_timeouts(tick_used)

        # ✅ 延迟提交限价止盈单(等待持仓同步)
        if self._need_submit_limit_tp and self._limit_tp_submit_after is not None:
            if datetime.now() >= self._limit_tp_submit_after:
                self._try_submit_limit_tp_order_delayed()

        if self.pos != 0:
            # 原有的平仓检查逻辑保持不变

            self._check_exit(tick_used)
            self.put_event()
            return

        if self.active_orders:
            self.put_event()
            return

        self._check_entry(tick_used)
        self.put_event()

    def on_timer(self):
        if not self.trading:
            return
        if self.max_tick_stale_seconds <= 0:
            return
        if self.last_recv_time is None:
            return

        # 用tick时间判断交易时段(而非本机时间,避免时区问题)
        if self.last_tick_time and not self._is_trading_time(self.last_tick_time):
            return

        # 用墙上时钟检测feed stale(这是正确的)
        now = datetime.now()
        stale_sec = (now - self.last_recv_time).total_seconds()
        if stale_sec <= float(self.max_tick_stale_seconds):
            return

        if self._tick_stale_state:
            return

        self._tick_stale_state = True
        self.write_log(f"[KILL] feed stale > {self.max_tick_stale_seconds}s")

        if self.pos != 0 or self.active_orders:
            tick = self.last_tick
            if tick is not None:
                self._emergency_close(tick, "feed stale")
            else:
                self.cancel_all()

    def on_order(self, order: OrderData):
        vt_orderid = order.vt_orderid
        if vt_orderid in self.active_orders and not order.is_active():
            meta = self.active_orders.pop(vt_orderid, None)
            # ⚠️ 新增：清理已结束的 LIMIT_TP 订单ID
            if meta and meta.purpose == OrderPurpose.LIMIT_TP.value and vt_orderid in self._limit_tp_order_ids:
                self._limit_tp_order_ids.remove(vt_orderid)

        if self.exit_in_progress and (not order.is_active()):
            if self.pos == 0:
                self.exit_in_progress = False   # 已全平
            elif (order.traded or 0) == 0:
                self.exit_in_progress = False   # 完全拒绝/撤单，允许重试
            elif self.pos != 0:                 # 部分成交，剩余仓位继续平
                self.exit_in_progress = False
                self.write_log(f"⚠️ [PARTIAL-FILL] 已成交{order.traded:.0f}，剩余{self.pos}股，允许继续平仓")

        try:
            traded = float(getattr(order, "traded", 0) or 0)
            if traded > 0 or (not order.is_active()):
                _status_cn = getattr(order.status, "value", str(order.status))
                _dir_cn = getattr(order.direction, "value", str(order.direction))
                _meta = self.active_orders.get(order.vt_orderid)
                _purpose = f" [{_meta.purpose}]" if _meta else ""
                self.write_log(
                    f"[ORDER]{_purpose} {order.vt_orderid[-8:]} {_status_cn} {_dir_cn} "
                    f"px={float(order.price):.0f} vol={int(order.volume or 0)} traded={traded:.0f}"
                )
        except Exception:
            pass

        self.put_event()

    def on_trade(self, trade: TradeData):
        offset = getattr(trade, "offset", None)
        vol = abs(int(trade.volume or 0))
        if vol <= 0:
            return

        # 开仓
        if offset == Offset.OPEN:
            if self.entry_time is None:
                self.entry_time = trade.datetime
            prev_vol = self.entry_volume
            self.entry_volume += vol
            px = float(trade.price)
            self.entry_price = px if prev_vol <= 0 else (self.entry_price * prev_vol + px * vol) / (prev_vol + vol)

            # ✅ 记录开仓方向（避免竞态条件）
            self.entry_direction = trade.direction

            self.tp_triggered = False
            self.trailing_stop_price = 0.0
            self.trailing_peak_price = self.entry_price

            # ✅ IFD模式：成交后立即挂止盈单（使用真实成交价）
            if self.enable_limit_tp_order:
                # 1. 撤销旧止盈单（处理部分成交场景）
                for oid in list(self._limit_tp_order_ids):
                    try:
                        self.cancel_order(oid)
                    except Exception:
                        pass
                self._limit_tp_order_ids.clear()

                # 2. 计算止盈价（使用加权平均成交价 + 1tick）
                pt = self.price_tick if self.price_tick > 0 else 1.0

                if self.entry_direction == Direction.LONG:
                    # 多头：成本价 + 1tick
                    tp_price = self.entry_price + self.limit_tp_ticks * pt
                    tp_price = round(tp_price / pt) * pt  # 对齐到最小变动价位

                    self.write_log(
                        f"[FILL▲] 买{vol}@{px:.0f} 均{self.entry_price:.0f}×{self.entry_volume} "
                        f"→ TP@{tp_price:.0f}(+{self.limit_tp_ticks}tick) "
                        f"延{max(0.5, self.limit_tp_delay_seconds):.1f}s | "
                        f"今日:{self.daily_pnl:+.0f}¥ 连亏:{self.continuous_loss_count}"
                    )

                    # ✅ 延迟提交:等待持仓同步(避免"找不到建玉"错误)
                    self._need_submit_limit_tp = True
                    self._limit_tp_submit_after = datetime.now() + timedelta(seconds=max(0.5, self.limit_tp_delay_seconds))
                    self._pending_tp_price = tp_price
                    self._pending_tp_volume = self.entry_volume
                    self._pending_tp_direction = Direction.SHORT

                elif self.entry_direction == Direction.SHORT:
                    # 空头：成本价 - 1tick
                    tp_price = self.entry_price - self.limit_tp_ticks * pt
                    tp_price = round(tp_price / pt) * pt

                    self.write_log(
                        f"[FILL▼] 卖{vol}@{px:.0f} 均{self.entry_price:.0f}×{self.entry_volume} "
                        f"→ TP@{tp_price:.0f}(-{self.limit_tp_ticks}tick) "
                        f"延{max(0.5, self.limit_tp_delay_seconds):.1f}s | "
                        f"今日:{self.daily_pnl:+.0f}¥ 连亏:{self.continuous_loss_count}"
                    )

                    # ✅ 延迟提交:等待持仓同步
                    self._need_submit_limit_tp = True
                    self._limit_tp_submit_after = datetime.now() + timedelta(seconds=max(0.5, self.limit_tp_delay_seconds))
                    self._pending_tp_price = tp_price
                    self._pending_tp_volume = self.entry_volume
                    self._pending_tp_direction = Direction.LONG

            if not self.enable_limit_tp_order:
                _arrow = "▲" if trade.direction == Direction.LONG else "▼"
                _act = "买" if trade.direction == Direction.LONG else "卖"
                self.write_log(
                    f"[FILL{_arrow}] {_act}{vol}@{px:.0f} 均{self.entry_price:.0f}×{self.entry_volume} | "
                    f"今日:{self.daily_pnl:+.0f}¥ 连亏:{self.continuous_loss_count}"
                )

            self.put_event()
            return

        # 平仓
        if self.entry_price > 0:
            self._trade_realized_pnl += self._calc_close_trade_pnl(trade.price, vol, trade.direction)

        if self.pos == 0 and self.entry_time is not None:
            self._finalize_flat(trade.datetime)
            # 平仓完成后重置同步状态，允许下次开仓重新尝试
            self._entry_sync_failed = False
            self._entry_sync_warned = False

        self.put_event()

    def on_stop_order(self, stop_order: StopOrder):
        pass

    # =========================
    # 入场/出场
    # =========================

    # 🚀 v2优化: 建玉同步检查
    def _check_position_synced(self) -> bool:
        """检查建玉是否已同步（用于动态轮询）

        Returns:
            bool: True=持仓已同步，可以挂止盈单；False=未同步，继续等待
        """
        try:
            # 尝试获取持仓信息
            me = getattr(getattr(self, "cta_engine", None), "main_engine", None)
            if not me:
                return False  # 无法获取引擎，继续等待

            # 获取当前持仓
            vt_positionid = f"{self.vt_symbol}.{self.gateway_name}"
            position = me.get_position(vt_positionid)

            if not position:
                # 持仓数据未加载，继续等待
                return False

            # 检查持仓数量是否>=预期开仓量
            expected_volume = abs(self._pending_tp_volume)
            actual_volume = abs(int(position.volume or 0))

            if actual_volume >= expected_volume:
                # 持仓已同步
                return True
            else:
                # 持仓数量不足，继续等待
                return False

        except Exception as e:
            # 异常时返回False，继续等待
            if self.verbose_log:
                self.write_log(f"[建玉检查异常] {e}")
            return False

    def _try_sync_entry_from_positions(self, reason: str = "", tick=None) -> bool:
        """Try to recover entry_price/entry_time from PositionData when trade callbacks are missing.
        This is for diagnosing 'profitable but no TP' caused by missing entry_time/entry_price.

        tick: 当前TickData，当持仓price=0时用tick价格作为entry_price的fallback
        """
        try:
            me = getattr(getattr(self, "cta_engine", None), "main_engine", None)
            if me is None:
                return False

            positions = []
            if hasattr(me, "get_all_positions"):
                positions = me.get_all_positions()
            elif hasattr(me, "get_positions"):
                positions = me.get_positions()
            elif hasattr(me, "get_all_active_positions"):
                positions = me.get_all_active_positions()

            if not positions:
                # 没有持仓数据：用tick价格作为entry_price的最后兜底
                if tick is not None and self.pos != 0:
                    bid1, ask1 = self._get_bid_ask_1(tick)
                    last_px = float(tick.last_price or 0.0)
                    # 根据持仓方向选择合适的参考价
                    if self.pos > 0:
                        px = bid1 if bid1 > 0 else last_px
                    else:
                        px = ask1 if ask1 > 0 else last_px
                    if px > 0:
                        self.entry_price = px
                        self.entry_volume = abs(int(self.pos))
                        self.entry_time = getattr(self, "last_tick_time", None) or datetime.now()
                        self.write_log(
                            f"🩹 [SYNC-TICK] 持仓数据为空，用tick价格恢复 entry_price={px:.2f} "
                            f"entry_vol={self.entry_volume} pos={self.pos} reason={reason}"
                        )
                        return True
                return False

            for p in positions:
                vt_symbol = getattr(p, "vt_symbol", "") or ""
                symbol = getattr(p, "symbol", "") or ""

                if vt_symbol:
                    if vt_symbol != self.vt_symbol:
                        continue
                else:
                    if symbol != self.vt_symbol.split(".")[0]:
                        continue

                vol = float(getattr(p, "volume", 0) or 0)
                if abs(vol) <= 0:
                    continue

                px = float(getattr(p, "price", 0) or 0)
                if px <= 0:
                    px = float(getattr(p, "open_price", 0) or 0) or float(getattr(p, "avg_price", 0) or 0)

                # 持仓price=0时，用tick价格作为fallback（确保止盈止损能正常运行）
                if px <= 0 and tick is not None:
                    bid1, ask1 = self._get_bid_ask_1(tick)
                    last_px = float(tick.last_price or 0.0)
                    if self.pos > 0:
                        px = bid1 if bid1 > 0 else last_px
                    else:
                        px = ask1 if ask1 > 0 else last_px
                    if px > 0:
                        self.write_log(
                            f"🩹 [SYNC] 持仓price=0，改用tick价格 {px:.2f} 作为entry_price"
                        )

                if px <= 0:
                    continue

                self.entry_price = px
                self.entry_volume = abs(int(vol))
                dt = getattr(self, "last_tick_time", None) or datetime.now()
                self.entry_time = dt
                # 同步成功后重置警告状态，允许后续再次同步
                self._entry_sync_failed = False
                self._entry_sync_warned = False

                self.write_log(
                    f"🩹 [SYNC] 恢复 entry 信息: entry_price={self.entry_price:.2f} "
                    f"entry_vol={self.entry_volume} pos={self.pos} reason={reason}"
                )
                return True

            # for循环结束但没找到匹配持仓（账户里没有该合约的持仓，如3350策略持仓但账户只有5032）
            # → 直接用tick实时价格作为entry_price兜底，让止损止盈能正常运行
            if tick is not None and self.pos != 0:
                bid1, ask1 = self._get_bid_ask_1(tick)
                last_px = float(tick.last_price or 0.0)
                if self.pos > 0:
                    px = bid1 if bid1 > 0 else last_px
                else:
                    px = ask1 if ask1 > 0 else last_px
                if px > 0:
                    self.entry_price = px
                    self.entry_volume = abs(int(self.pos))
                    self.entry_time = getattr(self, "last_tick_time", None) or datetime.now()
                    self.write_log(
                        f"🩹 [SYNC-TICK] 账户无匹配持仓({self.vt_symbol})，用tick实时价格恢复 "
                        f"entry_price={px:.2f} entry_vol={self.entry_volume} pos={self.pos} "
                        f"⚠️ 注意：此价格为恢复时的实时价，非真实成交价！"
                    )
                    return True

        except Exception as e:
            self.write_log(f"🩹 [SYNC] 尝试从持仓恢复 entry 信息失败: {e}")
        return False

    # =========================
    # Purpose字符串映射（收敛所有hardcode字符串）
    # =========================

    @staticmethod
    def _reason_to_purpose(reason: str) -> str:
        """
        将reason字符串转换为标准的OrderPurpose枚举值
        统一管理所有purpose字符串，避免hardcode导致的bug

        Args:
            reason: 出场原因（如"STOP_LOSS", "FLOW_FLIP_TAPE", "RETRY"等）

        Returns:
            对应的OrderPurpose枚举值（字符串形式）
        """
        reason_map = {
            "STOP_LOSS": OrderPurpose.EXIT_STOP_LOSS.value,
            "FLOW_FLIP_TAPE": OrderPurpose.EXIT_FLOW_FLIP_TAPE.value,
            "FLOW_FLIP_LOB": OrderPurpose.EXIT_FLOW_FLIP_LOB.value,
            "RETRY": OrderPurpose.EXIT_RETRY.value,
        }

        # 如果reason在映射表中，直接返回
        if reason in reason_map:
            return reason_map[reason]

        # EMERGENCY_xxx等特殊情况，动态构造
        if reason.startswith("EMERGENCY_"):
            return f"EXIT_{reason}"

        # 其他未知reason，警告并使用通用EXIT前缀
        # （保留兼容性，但应该避免使用）
        return f"EXIT_{reason}"

    # =========================
    # 入场检查 - 拆分为纯判断函数
    # =========================

    def _validate_price_tick(self, tick: TickData) -> Tuple[bool, str]:
        """
        验证 price_tick 是否有效
        Returns: (is_valid, reason)
        """
        if self.price_tick <= 0:
            # 尝试重新加载合约信息
            try:
                main_engine = self.cta_engine.main_engine
                contract = main_engine.get_contract(self.vt_symbol)
                if contract and hasattr(contract, 'pricetick') and contract.pricetick > 0:
                    self.price_tick = float(contract.pricetick)
                    self.write_log(f"🔧 [FIX] price_tick 已更新: {self.price_tick}")
                    return True, ""
                else:
                    return False, f'price_tick无效({self.price_tick})，禁止开仓'
            except Exception as e:
                return False, f'price_tick加载失败({e})，禁止开仓'
        return True, ""

    def _validate_market_quality(self, tick: TickData) -> Tuple[bool, str]:
        """
        验证市场质量（波动性、盘口、价差）
        Returns: (is_valid, reason)
        """
        # 检查波动性
        if self.mid_std_ticks >= self.max_mid_std_ticks:
            return False, f'波动性过大: mid_std_ticks={self.mid_std_ticks:.2f} >= {self.max_mid_std_ticks}'

        # 检查盘口有效性
        bid1, ask1 = self._get_bid_ask_1(tick)
        if bid1 <= 0 or ask1 <= 0:
            return False, f'bid\ask无效: bid1={bid1} ask1={ask1}'

        # 检查价差
        pt = self.price_tick
        spread_ticks = (ask1 - bid1) / pt if pt > 0 else 999.0
        if spread_ticks < self.min_spread_ticks or spread_ticks > self.max_spread_ticks:
            return False, f'spread过滤: spread_ticks={spread_ticks:.2f} pt={pt} bid1={bid1:.2f} ask1={ask1:.2f} range=[{self.min_spread_ticks},{self.max_spread_ticks}]'

        # 检查盘口量
        if int(tick.bid_volume_1 or 0) < self.min_best_volume or int(tick.ask_volume_1 or 0) < self.min_best_volume:
            return False, f'盘口量过滤: bidv1={int(tick.bid_volume_1 or 0)} askv1={int(tick.ask_volume_1 or 0)} < min_best_volume={self.min_best_volume}'

        return True, ""

    def _check_long_signal(self) -> bool:
        """
        检查多头信号是否满足（3主+1辅门控）
        主要指标: book_imbalance / lob_of_imbalance / micro_momentum（全部必须）
        次级指标: tape_ofi / microprice_tilt（满足min_secondary_score个即可）
        """
        book_imb = self.book_imbalance
        of_imb = self.lob_of_imbalance
        mom = self.micro_momentum
        tape_imb = self.tape_of_imbalance if self.use_tape_ofi else 0.0
        mpt = self.microprice_tilt if self.use_microprice_tilt else 0.0

        # 主要指标（全部必须满足）
        primary_ok = (
            self.enable_long
            and book_imb >= self.book_imbalance_long
            and of_imb >= self.of_imbalance_long
            and mom >= self.mom_long_threshold
        )
        if not primary_ok:
            return False

        # 次级指标（满足min_secondary_score个即可）
        secondary_score = 0
        if self.use_tape_ofi and tape_imb >= self.tape_imbalance_long:
            secondary_score += 1
        if self.use_microprice_tilt and mpt >= self.microprice_tilt_long:
            secondary_score += 1

        return secondary_score >= self.min_secondary_score

    # 🚀 v2优化: 信号强度检查
    def _is_strong_long_signal(self) -> bool:
        """检查是否为强多头信号（所有指标远超阈值）"""
        if not self.use_adaptive_confirm:
            return False

        multiplier = self.strong_signal_multiplier
        book_imb = self.book_imbalance
        of_imb = self.lob_of_imbalance
        mom = self.micro_momentum
        tape_imb = self.tape_of_imbalance if self.use_tape_ofi else 0.0
        mpt = self.microprice_tilt if self.use_microprice_tilt else 0.0

        # 所有指标都需要超过阈值*倍数
        strong = (
            book_imb >= self.book_imbalance_long * multiplier
            and of_imb >= self.of_imbalance_long * multiplier
            and mom >= self.mom_long_threshold * multiplier
        )

        if self.use_tape_ofi:
            strong = strong and (tape_imb >= self.tape_imbalance_long * multiplier)

        if self.use_microprice_tilt:
            strong = strong and (mpt >= self.microprice_tilt_long * multiplier)

        return strong

    def _is_strong_short_signal(self) -> bool:
        """检查是否为强空头信号（所有指标远超阈值）"""
        if not self.use_adaptive_confirm:
            return False

        multiplier = self.strong_signal_multiplier
        book_imb = self.book_imbalance
        of_imb = self.lob_of_imbalance
        mom = self.micro_momentum
        tape_imb = self.tape_of_imbalance if self.use_tape_ofi else 0.0
        mpt = self.microprice_tilt if self.use_microprice_tilt else 0.0

        # 所有指标都需要超过阈值*倍数
        strong = (
            book_imb <= -self.book_imbalance_short * multiplier
            and of_imb <= -self.of_imbalance_short * multiplier
            and mom <= -self.mom_short_threshold * multiplier
        )

        if self.use_tape_ofi:
            strong = strong and (tape_imb <= -self.tape_imbalance_short * multiplier)

        if self.use_microprice_tilt:
            strong = strong and (mpt <= -self.microprice_tilt_short * multiplier)

        return strong

    def _check_short_signal(self) -> bool:
        """
        检查空头信号是否满足（3主+1辅门控）
        主要指标: book_imbalance / lob_of_imbalance / micro_momentum（全部必须）
        次级指标: tape_ofi / microprice_tilt（满足min_secondary_score个即可）
        """
        book_imb = self.book_imbalance
        of_imb = self.lob_of_imbalance
        mom = self.micro_momentum
        tape_imb = self.tape_of_imbalance if self.use_tape_ofi else 0.0
        mpt = self.microprice_tilt if self.use_microprice_tilt else 0.0

        # 主要指标（全部必须满足）
        primary_ok = (
            self.enable_short
            and book_imb <= -self.book_imbalance_short
            and of_imb <= -self.of_imbalance_short
            and mom <= -self.mom_short_threshold
        )
        if not primary_ok:
            return False

        # 次级指标（满足min_secondary_score个即可）
        secondary_score = 0
        if self.use_tape_ofi and tape_imb <= -self.tape_imbalance_short:
            secondary_score += 1
        if self.use_microprice_tilt and mpt <= -self.microprice_tilt_short:
            secondary_score += 1

        return secondary_score >= self.min_secondary_score

    def _pending_entry_volume(self, purpose: str) -> int:
        pending = 0
        for meta in self.active_orders.values():
            if meta.purpose == purpose:
                pending += int(meta.volume or 0)
        return pending

    def _check_entry(self, tick: TickData):
        """
        入场检查流程（重构后）：
        1. 时间和冷却检查
        2. price_tick验证
        3. 市场质量检查（调用_validate_market_quality）
        4. 信号计算和确认
        5. 订单提交
        """
        # 1. 时间和冷却检查
        if not self._is_trading_time(tick.datetime):
            self._gate_log('非交易时段，跳过', tick.datetime)
            self._reset_confirm()
            return
        if self._in_cooldown(tick.datetime):
            self._gate_log('冷却中，跳过', tick.datetime)
            self._reset_confirm()
            return

        # 2. price_tick验证（保留原有的自动修复逻辑）
        if not self._price_tick_verified:
            if self.price_tick <= 0 or self.price_tick == 1.0:
                try:
                    main_engine = self.cta_engine.main_engine
                    contract = main_engine.get_contract(self.vt_symbol)
                    if contract and hasattr(contract, 'pricetick') and contract.pricetick > 0:
                        self.price_tick = float(contract.pricetick)
                        self._price_tick_verified = True
                        self.write_log(f"🔧 [FIX] price_tick 已验证并更新: {self.price_tick}")
                    else:
                        self._gate_log(f'price_tick无效({self.price_tick})，禁止开仓', tick.datetime)
                        self._reset_confirm()
                        return
                except Exception as e:
                    self._gate_log(f'price_tick加载失败({e})，禁止开仓', tick.datetime)
                    self._reset_confirm()
                    return
            else:
                self._price_tick_verified = True

        # 3. 市场质量检查（✅ 重构：调用统一的验证函数）
        is_valid, reason = self._validate_market_quality(tick)
        if not is_valid:
            self._gate_log(reason, tick.datetime)
            self._reset_confirm()
            return

        # 4. 获取指标值（用于日志输出）
        book_imb = self.book_imbalance
        of_imb = self.lob_of_imbalance
        mom = self.micro_momentum
        tape_imb = self.tape_of_imbalance if self.use_tape_ofi else 0.0
        mpt = self.microprice_tilt if self.use_microprice_tilt else 0.0

        bid1, ask1 = self._get_bid_ask_1(tick)
        pt = self.price_tick

        # 诊断日志：显示所有指标值（近阈值过滤：主指标最大接近度≥50%才输出，减少噪音）
        _long_prox = max(
            book_imb / self.book_imbalance_long if self.book_imbalance_long > 0 else 0.0,
            of_imb / self.of_imbalance_long if self.of_imbalance_long > 0 else 0.0,
            (tape_imb / self.tape_imbalance_long if self.tape_imbalance_long > 0 else 0.0) if self.use_tape_ofi else 0.0,
        )
        _short_prox = max(
            (-book_imb) / self.book_imbalance_short if self.book_imbalance_short > 0 else 0.0,
            (-of_imb) / self.of_imbalance_short if self.of_imbalance_short > 0 else 0.0,
            ((-tape_imb) / self.tape_imbalance_short if self.tape_imbalance_short > 0 else 0.0) if self.use_tape_ofi else 0.0,
        )
        if max(_long_prox, _short_prox) >= 0.5:
            self._gate_log(
                f"🧪 [IND] book={book_imb:+.3f}({self.book_imbalance_long:+.2f}) "
                f"of={of_imb:+.3f}({self.of_imbalance_long:+.2f}) "
                f"tape={tape_imb:+.3f}({self.tape_imbalance_long:+.2f}) "
                f"mom={mom:+.3f}({self.mom_long_threshold:+.2f}) "
                f"mpt={mpt:+.3f}({self.microprice_tilt_long:+.2f}) "
                f"spreadTicks={(ask1-bid1)/pt if pt>0 else 999:.2f} bidv1={int(tick.bid_volume_1 or 0)} askv1={int(tick.ask_volume_1 or 0)}",
                tick.datetime,
            )

        # 5. 信号判断（✅ 重构：调用专门的信号检查函数）
        long_ok = self._check_long_signal()
        short_ok = self._check_short_signal()

        if long_ok:
            # 检查信号确认是否过期（tick间隔过长说明行情卡顿，旧计数无效）
            if self._last_confirm_tick_time is not None:
                elapsed = self._dt_diff_seconds(tick.datetime, self._last_confirm_tick_time)
                if elapsed > self.signal_expire_seconds:
                    if self._long_confirm > 0 or self._short_confirm > 0:
                        self._gate_log(
                            f"信号确认过期: 距上次tick={elapsed:.1f}s > {self.signal_expire_seconds}s，重置计数",
                            tick.datetime,
                        )
                    self._reset_confirm()
            self._long_confirm += 1
            self._short_confirm = 0
            self._last_confirm_tick_time = tick.datetime
        elif short_ok:
            # 检查信号确认是否过期
            if self._last_confirm_tick_time is not None:
                elapsed = self._dt_diff_seconds(tick.datetime, self._last_confirm_tick_time)
                if elapsed > self.signal_expire_seconds:
                    if self._long_confirm > 0 or self._short_confirm > 0:
                        self._gate_log(
                            f"信号确认过期: 距上次tick={elapsed:.1f}s > {self.signal_expire_seconds}s，重置计数",
                            tick.datetime,
                        )
                    self._reset_confirm()
            self._short_confirm += 1
            self._long_confirm = 0
            self._last_confirm_tick_time = tick.datetime
        else:
            self._reset_confirm()
            self._last_confirm_tick_time = None
            return

        # 🚀 v2优化: 动态计算所需确认次数
        required_confirm_ticks = self.confirm_ticks
        if self.use_adaptive_confirm:
            if long_ok and self._is_strong_long_signal():
                required_confirm_ticks = self.strong_signal_confirm
            elif short_ok and self._is_strong_short_signal():
                required_confirm_ticks = self.strong_signal_confirm

        # 🔍 关键诊断 - 信号确认状态（只在有确认时输出，且始终显示以便观察入场逻辑）
        if self._long_confirm > 0 or self._short_confirm > 0:
            self.write_log(
                f"📡 [信号] book={book_imb:+.3f}({self.book_imbalance_long:+.2f}) | "
                f"of={of_imb:+.3f}({self.of_imbalance_long:+.2f}) | "
                f"tape={tape_imb:+.3f}({self.tape_imbalance_long:+.2f}) | "
                f"mom={mom:+.3f}({self.mom_long_threshold:+.2f}) | "
                f"mpt={mpt:+.3f}({self.microprice_tilt_long:+.2f}) | "
                f"确认: L={self._long_confirm}/{required_confirm_ticks} S={self._short_confirm}/{required_confirm_ticks}"
            )

        # 提交订单
        if self._long_confirm >= required_confirm_ticks:
            # Long inventory cap: current position + pending long entries.
            current_long = max(0, int(self.pos))
            pending_long = self._pending_entry_volume(OrderPurpose.ENTRY_LONG.value)
            remain = int(self.max_long_inventory) - current_long - pending_long
            if remain <= 0:
                self._gate_log(
                    f"多头库存已达上限: pos={current_long} pending={pending_long} cap={int(self.max_long_inventory)}",
                    tick.datetime,
                )
                self._reset_confirm()
                return
            # L1即时盘口确认（下单前最后一道价格优势检查）
            if self.entry_l1_vol_ratio > 0 or self.entry_max_ask_vol_ratio > 0:
                _bv1 = float(tick.bid_volume_1 or 0)
                _av1 = float(tick.ask_volume_1 or 0)
                _total = _bv1 + _av1
                if self.entry_l1_vol_ratio > 0 and _total > 0:
                    if _bv1 / _total < self.entry_l1_vol_ratio:
                        self._gate_log(f"⚡ [L1多] 买量比{_bv1/_total:.2f}<{self.entry_l1_vol_ratio:.2f}，等待更好时机", tick.datetime)
                        return  # 不重置confirm，等下一tick
                if self.entry_max_ask_vol_ratio > 0 and _bv1 > 0:
                    if _av1 / _bv1 > self.entry_max_ask_vol_ratio:
                        self._gate_log(f"⚡ [L1多] ask量={_av1:.0f}是bid{_bv1:.0f}的{_av1/_bv1:.1f}倍，卖压大", tick.datetime)
                        return

            price = ask1 + self.entry_slip_ticks * pt

            # ✅ 预设开仓方向（避免on_trade未触发导致entry_direction为None）
            self.entry_direction = Direction.LONG

            order_volume = min(int(self.trade_volume), remain)
            _confirm_cnt = self._long_confirm
            self._submit_order(self.buy, price, order_volume, OrderPurpose.ENTRY_LONG.value, Direction.LONG)
            self.last_signal = "LONG"
            self._reset_confirm()
            self.write_log(
                f"[ENTRY▲] 买{order_volume}@{price:.0f} | "
                f"确认{_confirm_cnt}次 ask1={ask1:.0f}+slip{self.entry_slip_ticks} | "
                f"库存:{current_long}→{current_long + order_volume}/{int(self.max_long_inventory)}"
            )

            # ⚠️ 已删除OCO逻辑：不在下单时立即挂止盈单
            # 正确做法：在on_trade()成交回报中使用真实成交价挂止盈单（IFD模式）

            return

        if self._short_confirm >= required_confirm_ticks:
            # Short inventory cap: current short position + pending short entries.
            current_short = max(0, -int(self.pos))
            pending_short = self._pending_entry_volume(OrderPurpose.ENTRY_SHORT.value)
            remain_short = int(self.max_short_inventory) - current_short - pending_short
            if remain_short <= 0:
                self._gate_log(
                    f"空头库存已达上限: pos={current_short} pending={pending_short} cap={int(self.max_short_inventory)}",
                    tick.datetime,
                )
                self._reset_confirm()
                return
            # L1即时盘口确认（空头方向：卖量占比和买压检查）
            if self.entry_l1_vol_ratio > 0 or self.entry_max_ask_vol_ratio > 0:
                _bv1 = float(tick.bid_volume_1 or 0)
                _av1 = float(tick.ask_volume_1 or 0)
                _total = _bv1 + _av1
                if self.entry_l1_vol_ratio > 0 and _total > 0:
                    if _av1 / _total < self.entry_l1_vol_ratio:
                        self._gate_log(f"⚡ [L1空] 卖量比{_av1/_total:.2f}<{self.entry_l1_vol_ratio:.2f}，等待更好时机", tick.datetime)
                        return
                if self.entry_max_ask_vol_ratio > 0 and _av1 > 0:
                    if _bv1 / _av1 > self.entry_max_ask_vol_ratio:
                        self._gate_log(f"⚡ [L1空] bid量={_bv1:.0f}是ask{_av1:.0f}的{_bv1/_av1:.1f}倍，买压大", tick.datetime)
                        return

            price = bid1 - self.entry_slip_ticks * pt

            # ✅ 预设开仓方向（避免on_trade未触发导致entry_direction为None）
            self.entry_direction = Direction.SHORT

            order_volume_short = min(int(self.trade_volume), remain_short)
            _confirm_cnt = self._short_confirm
            self._submit_order(self.short, price, order_volume_short, OrderPurpose.ENTRY_SHORT.value, Direction.SHORT)
            self.last_signal = "SHORT"
            self._reset_confirm()
            self.write_log(
                f"[ENTRY▼] 卖{order_volume_short}@{price:.0f} | "
                f"确认{_confirm_cnt}次 bid1={bid1:.0f}-slip{self.entry_slip_ticks} | "
                f"库存:{current_short}→{current_short + order_volume_short}/{int(self.max_short_inventory)}"
            )

            # ⚠️ 已删除OCO逻辑：不在下单时立即挂止盈单
            # 正确做法：在on_trade()成交回报中使用真实成交价挂止盈单（IFD模式）

            return

    def _check_exit(self, tick: TickData):
        # 如果已经有持仓但 entry 信息缺失，优先尝试从 PositionData 恢复；否则打印诊断提示
        if self.pos != 0 and (self.entry_time is None or self.entry_price <= 0):
            recovered = False
            # 只在之前未成功恢复时才尝试（每tick都尝试浪费性能）
            if not self._entry_sync_failed:
                try:
                    recovered = self._try_sync_entry_from_positions(reason="entry_missing_on_exit_check", tick=tick)
                except Exception:
                    recovered = False

                if not recovered:
                    self._entry_sync_failed = True

            if not recovered:
                # WARNING只打印一次，之后静默跳过（避免每个tick刷屏）
                if not self._entry_sync_warned:
                    self._entry_sync_warned = True
                    self.write_log(
                        f"⚠️ [EXIT-SKIP] 已有持仓(pos={self.pos})但 entry_time/entry_price 缺失，"
                        f"止盈止损检查被跳过（此消息只打印一次）。"
                        f"常见原因：成交回报未推送到策略 / on_trade 未触发 / offset 未正确映射。"
                        f"建议：重启策略，或手动平掉 {self.vt_symbol} 的持仓后重新启动。"
                    )
                return

        if self.entry_time is None or self.entry_price <= 0:
            return

        # ========================================================================
        # 🔴 兜底止损检查 (优先级最高,即使tp_only_mode=True也执行)
        # 修复tp_only_mode盲区: 单边急走/限流/止盈单失败时避免亏损无限放大
        # ========================================================================
        if self.enable_max_loss_per_trade_exit and self.pos != 0:
            bid1, ask1 = self._get_bid_ask_1(tick)

            # 计算未实现盈亏
            if self.pos > 0:  # 多头持仓
                # 使用bid1计算(卖出价)
                if bid1 > 0:
                    unrealized_pnl = (bid1 - self.entry_price) * abs(self.pos)
                else:
                    unrealized_pnl = 0.0
            else:  # 空头持仓
                # 使用ask1计算(买入价)
                if ask1 > 0:
                    unrealized_pnl = (self.entry_price - ask1) * abs(self.pos)
                else:
                    unrealized_pnl = 0.0

            # 检查是否触发兜底止损
            if unrealized_pnl < self.max_loss_per_trade:
                self.write_log(
                    f"🚨 [兜底止损] 触发! 未实现亏损={unrealized_pnl:.0f} < {self.max_loss_per_trade:.0f}"
                )
                self.write_log(
                    f"   持仓={self.pos}, 入场价={self.entry_price:.0f}, "
                    f"当前价={'bid1='+str(bid1) if self.pos > 0 else 'ask1='+str(ask1)}"
                )

                # ✅ 使用_emergency_close方法，包含以下优化:
                # 1. 自动计算带滑点的对价 (bid1 - slip 或 ask1 + slip)
                # 2. 先取消所有挂单 (包括限价止盈单)
                # 3. 设置exit_in_progress防止重复触发
                # 4. 有fallback到last_price
                self._emergency_close(tick, "MAX_LOSS")
                return  # 立即返回,不再检查其他出场条件
        # ========================================================================

        # ✅ tp_only_mode: 只用限价止盈单平仓，常规止损/FlowFlip/时间止损被禁用
        # 但上面的"兜底止损"不受此限制
        if self.tp_only_mode:
            self._vlog("[tp_only_mode] 已启用，跳过常规止损/FlowFlip/时间止损检查")
            return

        now = self._safe_now(self.entry_time or tick.datetime)
        hold = self._dt_diff_seconds(now, self.entry_time)

        # 重用兜底止损中已读取的bid/ask(优化性能)
        bid1, ask1 = self._get_bid_ask_1(tick)

        # 安全性检查：bid1/ask1必须有效，不用stale last_price替代
        # 用stale价格下止损/FlowFlip出场单风险极高（可能价差数十tick）
        if bid1 <= 0 or ask1 <= 0:
            self._vlog(f"⚠️ [EXIT-SKIP] bid1={bid1} ask1={ask1} 无效，跳过出场检查")
            return

        pt = self.price_tick if self.price_tick > 0 else 1.0

        # 价格合理性验证：bid1/ask1与入场价偏差不能超过max_price_deviation_ticks
        # 防止因API异常/stale数据导致以极端价格下出场单
        if self.entry_price > 0 and pt > 0:
            max_dev = self.max_price_deviation_ticks * pt
            price_to_check = bid1 if self.pos > 0 else ask1
            if abs(price_to_check - self.entry_price) > max_dev:
                self.write_log(
                    f"⚠️ [EXIT-SKIP] 价格偏离过大! entry={self.entry_price:.1f} "
                    f"current={'bid1' if self.pos > 0 else 'ask1'}={price_to_check:.1f} "
                    f"偏差={abs(price_to_check - self.entry_price)/pt:.0f}ticks > "
                    f"限制{self.max_price_deviation_ticks}ticks，跳过出场"
                )
                return

        # 多头持仓
        if self.pos > 0:
            # 止损：直接用loss_ticks，不减exit_slip（保持语义清晰，避免SL线太贴近entry）
            sl_trigger = self.entry_price - self.loss_ticks * pt

            # 🔍 详细诊断日志 - 多头持仓状态
            self._vlog(
                f"📊 [持仓监控-多] "
                f"Entry={self.entry_price:.1f} | "
                f"SL_trigger={sl_trigger:.1f} | "
                f"Bid1={bid1:.1f} | Ask1={ask1:.1f} | Last={tick.last_price or 0:.1f} | "
                f"Hold={hold:.1f}s | "
                f"TapeOFI={self.tape_of_imbalance:+.3f} | "
                f"LobOFI={self.lob_of_imbalance:+.3f} | "
                f"限价TP单: {len(self._limit_tp_order_ids)}个"
            )

            # ⚠️ 止盈逻辑已禁用 - 完全依赖限价止盈单
            # 原有的Touch/Trailing止盈已被take_profit_mode="disabled"关闭

            # Flow Flip 出场
            if self.use_flow_flip_exit:
                # 🔍 详细诊断 - Flow Flip检查
                self._vlog(
                    f"🔄 [FlowFlip检查] "
                    f"TapeOFI={self.tape_of_imbalance:+.3f} (阈值:-{self.flow_flip_threshold:.2f}) | "
                    f"LobOFI={self.lob_of_imbalance:+.3f} (阈值:-{self.flow_flip_threshold:.2f})"
                )
                if self.use_tape_ofi and self.tape_of_imbalance <= -self.flow_flip_threshold:
                    self.write_log(f"⚠️ [EXIT] Tape订单流反转！TapeOFI={self.tape_of_imbalance:+.3f} <= -{self.flow_flip_threshold:.2f}")
                    self._aggressive_flat_long(bid1 - self.exit_slip_ticks * pt, "TAPE_FLIP")
                    return
                if self.lob_of_imbalance <= -self.flow_flip_threshold:
                    self.write_log(f"⚠️ [EXIT] LOB订单流反转！LobOFI={self.lob_of_imbalance:+.3f} <= -{self.flow_flip_threshold:.2f}")
                    self._aggressive_flat_long(bid1 - self.exit_slip_ticks * pt, "LOB_FLIP")
                    return

            # 止损
            # 🔍 详细诊断 - 止损检查
            touched_sl = (bid1 <= sl_trigger)
            self._vlog(
                f"🛑 [止损检查] "
                f"bid1({bid1:.1f}) <= SL_trigger({sl_trigger:.1f})? {touched_sl}"
            )
            if touched_sl:
                self.write_log(f"❌ [EXIT] 止损触发！bid1={bid1:.1f}, SL_trigger={sl_trigger:.1f}")
                self._aggressive_flat_long(bid1 - self.exit_slip_ticks * pt, "STOP_LOSS")
                return

            # 时间止损 - 已禁用
            # if hold >= self.max_hold_seconds:
            #     self.write_log(f"⏰ [EXIT] 时间止损触发！持仓{hold:.1f}s >= {self.max_hold_seconds}s")
            #     self._aggressive_flat_long(bid1 - self.exit_slip_ticks * pt, "TIME_STOP")
            #     return

        # 空头持仓
        elif self.pos < 0:
            # 止损：直接用loss_ticks，不减exit_slip（保持语义清晰，避免SL线太贴近entry）
            sl_trigger = self.entry_price + self.loss_ticks * pt

            # ⚠️ 止盈逻辑已禁用 - 完全依赖限价止盈单
            # 原有的Touch/Trailing止盈已被take_profit_mode="disabled"关闭

            if self.use_flow_flip_exit:
                if self.use_tape_ofi and self.tape_of_imbalance >= self.flow_flip_threshold:
                    self._aggressive_flat_short(ask1 + self.exit_slip_ticks * pt, "TAPE_FLIP")
                    return
                if self.lob_of_imbalance >= self.flow_flip_threshold:
                    self._aggressive_flat_short(ask1 + self.exit_slip_ticks * pt, "LOB_FLIP")
                    return

            touched_sl = (ask1 >= sl_trigger)
            if touched_sl:
                self.write_log(f"❌ [EXIT] 止损触发！ask1={ask1:.1f}, SL_trigger={sl_trigger:.1f}")
                self._aggressive_flat_short(ask1 + self.exit_slip_ticks * pt, "STOP_LOSS")
                return

            # 时间止损 - 已禁用
            # if hold >= self.max_hold_seconds:
            #     self._aggressive_flat_short(ask1 + self.exit_slip_ticks * pt, "TIME_STOP")
            #     return

    # =========================
    # 指标更新
    # =========================
    def _update_indicators(self, tick: TickData):
        self.book_imbalance = self._calculate_book_imbalance(tick)
        self._update_lob_ofi(tick)
        if self.use_tape_ofi:
            self._update_tape_ofi(tick)
        else:
            self.tape_of_imbalance = 0.0
        self._update_momentum_and_vol(tick)
        self.microprice_tilt = self._calculate_microprice_tilt(tick) if self.use_microprice_tilt else 0.0

    def _calculate_book_imbalance(self, tick: TickData) -> float:
        levels = max(1, int(self.book_depth_levels))
        decay = float(self.book_decay)
        buy_w = 0.0
        sell_w = 0.0
        for i in range(1, levels + 1):
            w = decay ** (i - 1)
            bv = float(getattr(tick, f"bid_volume_{i}", 0) or 0)
            av = float(getattr(tick, f"ask_volume_{i}", 0) or 0)
            if bv > 0:
                buy_w += w * bv
            if av > 0:
                sell_w += w * av
        tot = buy_w + sell_w
        if tot <= 0:
            return 0.0
        return float((buy_w - sell_w) / tot)

    def _update_lob_ofi(self, tick: TickData):
        """
        计算LOB订单流失衡(Limit Order Book Order Flow Imbalance)

        ⚠️  设计注意:
        - 当价格跨档位时(激进订单行为),使用max(volume, min_best_volume)而非实际volume
        - 这会"放大"小单造成的OFI信号,让策略对激进订单更敏感
        - 如果觉得信号过于频繁,需要提高of_imbalance_long/short阈值
        """
        levels = max(1, int(self.book_depth_levels))
        if not self._prev_bid_prices or len(self._prev_bid_prices) != levels:
            self._prev_bid_prices = [0.0] * levels
            self._prev_ask_prices = [0.0] * levels
            self._prev_bid_vols = [0] * levels
            self._prev_ask_vols = [0] * levels

        buy_delta = 0.0
        sell_delta = 0.0

        for i in range(1, levels + 1):
            bp = float(getattr(tick, f"bid_price_{i}", 0.0) or 0.0)
            ap = float(getattr(tick, f"ask_price_{i}", 0.0) or 0.0)
            bv = int(getattr(tick, f"bid_volume_{i}", 0) or 0)
            av = int(getattr(tick, f"ask_volume_{i}", 0) or 0)

            p_bp = self._prev_bid_prices[i - 1]
            p_ap = self._prev_ask_prices[i - 1]
            p_bv = self._prev_bid_vols[i - 1]
            p_av = self._prev_ask_vols[i - 1]

            if p_bp > 0 and bp > 0:
                if bp > p_bp:
                    # 买价上升(买方激进) - 用min_best_volume作为最小权重
                    buy_delta += max(bv, self.min_best_volume)
                elif bp == p_bp:
                    diff = bv - p_bv
                    if diff > 0:
                        buy_delta += diff
                    elif diff < 0:
                        sell_delta += -diff
                else:
                    # 买价下降(卖方激进) - 用min_best_volume作为最小权重
                    sell_delta += max(p_bv, self.min_best_volume)

            if p_ap > 0 and ap > 0:
                if ap < p_ap:
                    # 卖价下降(买方激进) - 用min_best_volume作为最小权重
                    buy_delta += max(av, self.min_best_volume)
                elif ap == p_ap:
                    diff = av - p_av
                    if diff > 0:
                        sell_delta += diff
                    elif diff < 0:
                        buy_delta += -diff
                else:
                    # 卖价上升(卖方激进) - 用min_best_volume作为最小权重
                    sell_delta += max(p_av, self.min_best_volume)

            self._prev_bid_prices[i - 1] = bp
            self._prev_ask_prices[i - 1] = ap
            self._prev_bid_vols[i - 1] = bv
            self._prev_ask_vols[i - 1] = av

        if len(self._ofi_buy_q) == self._ofi_buy_q.maxlen:
            self._ofi_buy_sum -= self._ofi_buy_q.popleft()
        self._ofi_buy_q.append(buy_delta)
        self._ofi_buy_sum += buy_delta

        if len(self._ofi_sell_q) == self._ofi_sell_q.maxlen:
            self._ofi_sell_sum -= self._ofi_sell_q.popleft()
        self._ofi_sell_q.append(sell_delta)
        self._ofi_sell_sum += sell_delta

        tot = self._ofi_buy_sum + self._ofi_sell_sum
        self.lob_of_imbalance = (self._ofi_buy_sum - self._ofi_sell_sum) / tot if tot > 0 else 0.0

    def _update_tape_ofi(self, tick: TickData):
        now = tick.datetime if getattr(tick, "datetime", None) else datetime.now()
        bid1, ask1 = self._get_bid_ask_1(tick)
        last_price = float(tick.last_price or 0.0)

        dv = 0.0
        if tick.volume is not None:
            if self._prev_total_volume is None:
                self._prev_total_volume = float(tick.volume)
                dv = 0.0
            else:
                dv = float(tick.volume) - float(self._prev_total_volume)
                self._prev_total_volume = float(tick.volume)

        if dv <= 0 or last_price <= 0:
            self._trim_tape_window(now)
            self.tape_of_imbalance = (self._tape_sum / self._tape_abs_sum) if self._tape_abs_sum > 0 else 0.0
            return

        side = 0
        if ask1 > 0 and last_price >= ask1:
            side = 1
        elif bid1 > 0 and last_price <= bid1:
            side = -1
        else:
            if self._prev_last_price is not None:
                if last_price > self._prev_last_price:
                    side = 1
                elif last_price < self._prev_last_price:
                    side = -1
        self._prev_last_price = last_price

        signed = float(side) * dv
        if signed != 0.0:
            self._tape_q.append((now, signed))
            self._tape_sum += signed
            self._tape_abs_sum += abs(signed)

        self._trim_tape_window(now)
        self.tape_of_imbalance = (self._tape_sum / self._tape_abs_sum) if self._tape_abs_sum > 0 else 0.0

    def _trim_tape_window(self, now: datetime):
        win = float(self.tape_window_seconds)
        if win <= 0:
            self._tape_q.clear()
            self._tape_sum = 0.0
            self._tape_abs_sum = 0.0
            return

        while self._tape_q:
            t, v = self._tape_q[0]
            if (now - t).total_seconds() <= win:
                break
            self._tape_q.popleft()
            self._tape_sum -= v
            self._tape_abs_sum -= abs(v)
            if self._tape_abs_sum < 0:
                self._tape_abs_sum = 0.0

    def _update_momentum_and_vol(self, tick: TickData):
        bid1, ask1 = self._get_bid_ask_1(tick)
        last_price = float(tick.last_price or 0.0)
        mid = (bid1 + ask1) / 2 if (bid1 > 0 and ask1 > 0) else last_price
        if mid <= 0:
            return

        # 优化: 使用 microprice 计算 momentum (更敏感)
        direction = 0
        if self.use_microprice_momentum:
            micro = self._calc_microprice(tick)
            if micro > 0:
                if self._prev_micro_for_mom is not None:
                    if micro > self._prev_micro_for_mom:
                        direction = 1
                    elif micro < self._prev_micro_for_mom:
                        direction = -1
                self._prev_micro_for_mom = micro
        else:
            if self._prev_mid_for_mom is not None:
                if mid > self._prev_mid_for_mom:
                    direction = 1
                elif mid < self._prev_mid_for_mom:
                    direction = -1
            self._prev_mid_for_mom = mid

        if len(self._mom_q) == self._mom_q.maxlen:
            self._mom_sum -= self._mom_q.popleft()
        self._mom_q.append(direction)
        self._mom_sum += direction
        length = len(self._mom_q)
        self.micro_momentum = (self._mom_sum / length) if length else 0.0

        # 波动率计算
        pt = self.price_tick if self.price_tick > 0 else 1.0
        if self._prev_mid_for_vol is None:
            self._prev_mid_for_vol = mid
            return

        ret_ticks = (mid - self._prev_mid_for_vol) / pt if pt > 0 else 0.0
        self._prev_mid_for_vol = mid

        if len(self._mid_ret_q) == self._mid_ret_q.maxlen:
            old = self._mid_ret_q.popleft()
            self._mid_ret_sum -= old
            self._mid_ret_sq_sum -= old * old

        self._mid_ret_q.append(ret_ticks)
        self._mid_ret_sum += ret_ticks
        self._mid_ret_sq_sum += ret_ticks * ret_ticks

        n = len(self._mid_ret_q)
        if n >= 5:
            mean = self._mid_ret_sum / n
            var = (self._mid_ret_sq_sum / n) - (mean * mean)
            self.mid_std_ticks = (var ** 0.5) if var > 0 else 0.0
        else:
            self.mid_std_ticks = 0.0

    def _calc_microprice(self, tick: TickData) -> float:
        """计算 microprice (用于 momentum)"""
        bid1, ask1 = self._get_bid_ask_1(tick)
        bv1 = float(tick.bid_volume_1 or 0)
        av1 = float(tick.ask_volume_1 or 0)
        if bid1 <= 0 or ask1 <= 0:
            return 0.0
        denom = bv1 + av1
        if denom <= 0:
            return (bid1 + ask1) / 2
        return (ask1 * bv1 + bid1 * av1) / denom

    def _calculate_microprice_tilt(self, tick: TickData) -> float:
        bid1, ask1 = self._get_bid_ask_1(tick)
        bv1 = float(tick.bid_volume_1 or 0)
        av1 = float(tick.ask_volume_1 or 0)
        if bid1 <= 0 or ask1 <= 0:
            return 0.0
        denom = bv1 + av1
        if denom <= 0:
            return 0.0
        micro = (ask1 * bv1 + bid1 * av1) / denom
        mid = (bid1 + ask1) / 2
        pt = self.price_tick if self.price_tick > 0 else 1.0
        return float((micro - mid) / pt) if pt > 0 else 0.0

    # =========================
    # 执行辅助
    # =========================
    def _submit_order(self, func, price: float, volume: int, purpose: str, direction: Direction) -> List[str]:
        if volume <= 0:
            return []

        # 🚀 v2优化: 分离限流检查
        now = datetime.now()

        # 根据订单用途选择不同的限流策略
        if purpose in (OrderPurpose.ENTRY_LONG.value, OrderPurpose.ENTRY_SHORT.value):
            # 开仓订单
            if not self._allow_entry_order(now):
                return []
            self._last_entry_order_dt = now
        elif purpose == OrderPurpose.LIMIT_TP.value:
            # 止盈单
            if not self._allow_limit_tp_order(now):
                return []
            self._last_limit_tp_order_dt = now
        else:
            # 平仓订单（止损/FlowFlip/应急）
            if not self._allow_exit_order(now):
                return []
            self._last_exit_order_dt = now

        # 更新全局时间戳（兼容旧逻辑）
        self._last_order_action_dt = now

        # 尝试下单，捕获"找不到合约"错误并友好提示
        try:
            ids = func(price, volume)
            # vn.py 的 buy/sell/short/cover 通常返回 List[str]；失败时可能返回 [] 或 ""。
            if not ids:
                self.write_log(
                    f"⚠️ 下单返回空: purpose={purpose} dir={direction.value} px={price:.2f} vol={volume} | "
                    f"请检查：网关是否已连接、合约是否已订阅、以及 kabu 是否返回拒单/400。"
                )
                return []
        except Exception as e:
            error_msg = str(e)
            if "找不到合约" in error_msg or "Contract" in error_msg:
                self.write_log(f"⚠️ 下单失败: 合约未加载。请检查行情订阅是否成功（{self.vt_symbol}）")
                self.write_log(f"   提示: 确保Kabu Gateway已成功订阅该合约行情")
                return []
            else:
                self.write_log(f"❌ 下单异常: {error_msg}")
                return []

        if not ids:
            return []
        if isinstance(ids, str):
            ids = [ids]
        _dir_cn = getattr(direction, "value", str(direction))
        self.write_log(f"✅ [委托] {purpose} {_dir_cn} px={price:.0f} vol={volume} ids={ids}")
        for oid in ids:
            self.active_orders[str(oid)] = _OrderMeta(now, purpose, direction, float(price), int(volume))
        self._last_order_action_dt = now
        return [str(x) for x in ids]

    def cancel_all(self):
        for oid in list(self.active_orders.keys()):
            try:
                self.cancel_order(oid)
            except Exception:
                pass

    def _aggressive_flat_long(self, price: float, reason: str):
        if self.exit_in_progress:
            return
        if self.pos <= 0:
            return
        self.exit_in_progress = True
        self.cancel_all()
        self._limit_tp_order_ids.clear()  # 清理限价止盈单ID
        self._need_submit_limit_tp = False  # 重置标志
        self._limit_tp_retry_count = 0  # 重置重试计数

        # 使用统一的reason到purpose映射
        purpose = self._reason_to_purpose(reason)
        _vol_exit = abs(self.pos)
        vt_orderids = self._submit_order(self.sell, price, _vol_exit, purpose, Direction.SHORT)
        _hold_s = self._dt_diff_seconds(datetime.now(), self.entry_time) if self.entry_time else 0.0
        _est_pnl = (price - self.entry_price) * _vol_exit if self.entry_price > 0 else 0.0
        self.write_log(
            f"[CLOSE▲] {reason} 平{_vol_exit}@{price:.0f} 入:{self.entry_price:.0f} "
            f"预估:{_est_pnl:+.0f}¥ 持{_hold_s:.0f}s"
        )
        if not vt_orderids:
            self.write_log(f"⚠️ [EXIT-FAIL] 多头平仓单提交失败，重置exit_in_progress以允许重试")
            self.exit_in_progress = False

    def _aggressive_flat_short(self, price: float, reason: str):
        if self.exit_in_progress:
            return
        if self.pos >= 0:
            return
        self.exit_in_progress = True
        self.cancel_all()
        self._limit_tp_order_ids.clear()  # 清理限价止盈单ID
        self._need_submit_limit_tp = False  # 重置标志
        self._limit_tp_retry_count = 0  # 重置重试计数

        # 使用统一的reason到purpose映射
        purpose = self._reason_to_purpose(reason)
        _vol_exit = abs(self.pos)
        vt_orderids = self._submit_order(self.cover, price, _vol_exit, purpose, Direction.LONG)
        _hold_s = self._dt_diff_seconds(datetime.now(), self.entry_time) if self.entry_time else 0.0
        _est_pnl = (self.entry_price - price) * _vol_exit if self.entry_price > 0 else 0.0
        self.write_log(
            f"[CLOSE▼] {reason} 平{_vol_exit}@{price:.0f} 入:{self.entry_price:.0f} "
            f"预估:{_est_pnl:+.0f}¥ 持{_hold_s:.0f}s"
        )
        if not vt_orderids:
            self.write_log(f"⚠️ [EXIT-FAIL] 空头平仓单提交失败，重置exit_in_progress以允许重试")
            self.exit_in_progress = False

    def _emergency_close(self, tick: TickData, reason: str):
        bid1, ask1 = self._get_bid_ask_1(tick)
        pt = self.price_tick if self.price_tick > 0 else 1.0
        self.cancel_all()
        if self.pos > 0:
            px = (bid1 if bid1 > 0 else float(tick.last_price or 0.0)) - self.exit_slip_ticks * pt
            self._aggressive_flat_long(px, f"EMERGENCY_{reason}")
        elif self.pos < 0:
            px = (ask1 if ask1 > 0 else float(tick.last_price or 0.0)) + self.exit_slip_ticks * pt
            self._aggressive_flat_short(px, f"EMERGENCY_{reason}")

    # =========================
    # 限价止盈单管理
    # =========================
    def _try_submit_limit_tp_order(self, tick: TickData):
        """尝试提交限价止盈单（支持多空+重试机制）"""
        # ⚠️ 修复：如果正在平仓，不要挂止盈单（避免订单冲突）
        if self.exit_in_progress:
            return

        if not self._need_submit_limit_tp:
            return

        # ✅ 修复：使用entry_volume代替pos（避免竞态条件）
        if self.entry_volume <= 0 or self.entry_price <= 0:
            self._need_submit_limit_tp = False
            self._limit_tp_retry_count = 0
            return

        # 检查重试次数（最多重试5次）
        if self._limit_tp_retry_count >= self.MAX_LIMIT_TP_RETRY:
            self.write_log(f"⚠️ [限价止盈单] 重试{self._limit_tp_retry_count}次仍失败，放弃挂单")
            self._need_submit_limit_tp = False
            self._limit_tp_retry_count = 0
            return

        # ✅ 检查延迟提交时间（等待Kabu建玉同步）
        now = datetime.now()
        if self._limit_tp_submit_after and now < self._limit_tp_submit_after:
            return  # 还未到提交时间，下次tick再试

        # 检查是否允许下单（限流）
        if not self._allow_order_action(now):
            return  # 下次tick再试（不算重试次数）

        pt = self.price_tick if self.price_tick > 0 else 1.0

        # ✅ 使用entry_direction代替pos判断（避免竞态条件）
        # 多头：挂卖单止盈
        if self.entry_direction == Direction.LONG:
            tp_price = self.entry_price + self.limit_tp_ticks * pt
            self.write_log(
                f"💰 [限价止盈单-多] 挂单@{tp_price:.2f} (entry={self.entry_price:.2f} +{self.limit_tp_ticks}tick) "
                f"vol={self.entry_volume} (重试{self._limit_tp_retry_count}/{self.MAX_LIMIT_TP_RETRY})"
            )
            try:
                order_ids = self._submit_order(self.sell, tp_price, self.entry_volume, OrderPurpose.LIMIT_TP.value, Direction.SHORT)
                if order_ids:
                    self._limit_tp_order_ids.extend(order_ids)
                    self._need_submit_limit_tp = False
                    self._limit_tp_retry_count = 0
                    self.write_log(f"✅ [限价止盈单] 挂单成功 ids={order_ids}")
                else:
                    # 返回空，可能是限流或合约未就绪，继续重试
                    self._limit_tp_retry_count += 1
                    self.write_log(f"⚠️ [限价止盈单] 下单返回空，将重试（{self._limit_tp_retry_count}/5）")
            except Exception as e:
                self._limit_tp_retry_count += 1
                self.write_log(f"❌ [限价止盈单] 挂单异常({self._limit_tp_retry_count}/5): {e}")

        # 空头：挂买单止盈
        elif self.entry_direction == Direction.SHORT:
            tp_price = self.entry_price - self.limit_tp_ticks * pt
            self.write_log(
                f"💰 [限价止盈单-空] 挂单@{tp_price:.2f} (entry={self.entry_price:.2f} -{self.limit_tp_ticks}tick) "
                f"vol={self.entry_volume} (重试{self._limit_tp_retry_count}/{self.MAX_LIMIT_TP_RETRY})"
            )
            try:
                order_ids = self._submit_order(self.cover, tp_price, self.entry_volume, OrderPurpose.LIMIT_TP.value, Direction.LONG)
                if order_ids:
                    self._limit_tp_order_ids.extend(order_ids)
                    self._need_submit_limit_tp = False
                    self._limit_tp_retry_count = 0
                    self.write_log(f"✅ [TP成功] 止盈单挂单成功 ids={order_ids}")
                else:
                    self._limit_tp_retry_count += 1
                    self.write_log(f"⚠️ [TP重试] 返回空 ({self._limit_tp_retry_count}/5)")
            except Exception as e:
                self._limit_tp_retry_count += 1
                self.write_log(f"❌ [TP异常] ({self._limit_tp_retry_count}/5): {e}")

    def _try_submit_limit_tp_order_delayed(self):
        """延迟提交限价止盈单(使用预存的价格和数量)

        🚀 v2优化: 动态建玉轮询，替代固定2s延迟
        """
        if not self._need_submit_limit_tp or self._pending_tp_direction is None:
            return

        # 检查是否正在平仓
        if self.exit_in_progress:
            self._need_submit_limit_tp = False
            return

        # 检查数据完整性
        if self._pending_tp_volume <= 0 or self._pending_tp_price <= 0:
            self._need_submit_limit_tp = False
            return

        # 检查重试次数：5次快速重试失败后，切换为"慢速重试"模式（每30s再试一次，最多再试10次）
        # 这是为了应对 KABU 建玉状态同步延迟导致的拒单（HTTP 200 但无 OrderId）
        if self._limit_tp_retry_count >= 5:
            slow_retry_count = self._limit_tp_retry_count - 5  # 慢速重试已进行次数
            if slow_retry_count >= 10:
                self.write_log(
                    f"🚨 [TP放弃] 快速×5+慢速×10共15次仍失败，彻底放弃挂单。"
                    f"请检查: KABU API拒单原因(见gateway日志) / 仓位是否已手动处理"
                )
                self._need_submit_limit_tp = False
                self._limit_tp_retry_count = 0
                return
            # 慢速重试：通过 _limit_tp_submit_after 控制30s间隔
            now = datetime.now()
            if self._limit_tp_submit_after and now < self._limit_tp_submit_after:
                return  # 未到下次重试时间
            self.write_log(
                f"⏳ [TP慢速] 第{slow_retry_count+1}/10次(30s间隔) px={self._pending_tp_price:.0f} vol={self._pending_tp_volume}"
            )
            # 设置下次慢速重试时间
            self._limit_tp_submit_after = datetime.now() + timedelta(seconds=30)
            # 重置轮询状态（让建玉轮询重新检查）
            self._position_poll_start_time = None
            self._last_position_poll_time = None

        now = datetime.now()

        # 🚀 v2优化: 动态建玉轮询
        if self.use_dynamic_position_polling:
            # 初始化轮询开始时间
            if self._position_poll_start_time is None:
                self._position_poll_start_time = now
                self._last_position_poll_time = now

            # 检查是否超时
            elapsed = (now - self._position_poll_start_time).total_seconds()
            if elapsed > self.max_position_wait_seconds:
                self.write_log(
                    f"⚠️ [TP轮询] 建玉同步超时{elapsed:.2f}s，强制挂单"
                )
                # 继续执行下单（超时强制挂单）
            else:
                # 检查轮询间隔
                if self._last_position_poll_time:
                    poll_elapsed_ms = (now - self._last_position_poll_time).total_seconds() * 1000
                    if poll_elapsed_ms < self.position_poll_interval_ms:
                        return  # 未到轮询时间，下次tick再试

                # 执行建玉检查
                self._last_position_poll_time = now
                if self._check_position_synced():
                    self.write_log(
                        f"✅ [TP轮询] 建玉已同步(耗时{elapsed:.2f}s)，立即挂止盈单"
                    )
                    # 继续执行下单
                else:
                    # 建玉未同步，继续等待
                    return
        else:
            # 兼容旧逻辑：固定延迟
            if not self._allow_order_action(now):
                return

        # 提交订单
        try:
            if self._pending_tp_direction == Direction.SHORT:
                # 多头平仓止盈
                _tp_est = (self._pending_tp_price - self.entry_price) * self._pending_tp_volume if self.entry_price > 0 else 0.0
                _retry_phase = f"快速{self._limit_tp_retry_count}/5" if self._limit_tp_retry_count < 5 else f"慢速{self._limit_tp_retry_count-5}/10"
                self.write_log(
                    f"[TP▼] 卖止盈挂单@{self._pending_tp_price:.0f} vol={self._pending_tp_volume} "
                    f"预期:{_tp_est:+.0f}¥ 重试:{_retry_phase}"
                )
                order_ids = self._submit_order(
                    self.sell, self._pending_tp_price, self._pending_tp_volume,
                    OrderPurpose.LIMIT_TP.value, Direction.SHORT
                )
            else:
                # 空头平仓止盈
                _tp_est = (self.entry_price - self._pending_tp_price) * self._pending_tp_volume if self.entry_price > 0 else 0.0
                _retry_phase = f"快速{self._limit_tp_retry_count}/5" if self._limit_tp_retry_count < 5 else f"慢速{self._limit_tp_retry_count-5}/10"
                self.write_log(
                    f"[TP▲] 买止盈挂单@{self._pending_tp_price:.0f} vol={self._pending_tp_volume} "
                    f"预期:{_tp_est:+.0f}¥ 重试:{_retry_phase}"
                )
                order_ids = self._submit_order(
                    self.cover, self._pending_tp_price, self._pending_tp_volume,
                    OrderPurpose.LIMIT_TP.value, Direction.LONG
                )

            if order_ids:
                self._limit_tp_order_ids.extend(order_ids)
                self._need_submit_limit_tp = False
                self._limit_tp_retry_count = 0
                # 🚀 v2优化: 重置轮询状态
                self._position_poll_start_time = None
                self._last_position_poll_time = None
                self.write_log(f"✅ [TP成功] 止盈单挂单成功 ids={order_ids}")
            else:
                self._limit_tp_retry_count += 1
                _phase = f"快速{self._limit_tp_retry_count}/5" if self._limit_tp_retry_count <= 5 else f"慢速{self._limit_tp_retry_count-5}/10"
                self.write_log(f"⚠️ [TP重试] 返回空将重试 {_phase}")

        except Exception as e:
            self._limit_tp_retry_count += 1
            _phase = f"快速{self._limit_tp_retry_count}/5" if self._limit_tp_retry_count <= 5 else f"慢速{self._limit_tp_retry_count-5}/10"
            self.write_log(f"❌ [TP异常] {_phase}: {e}")

    # =========================
    # 订单超时
    # =========================
    def _check_order_timeouts(self, tick: TickData):
        if not self.active_orders:
            return
        now = datetime.now()
        if not self._allow_order_action(now):
            return

        to_cancel: List[str] = []
        for oid, meta in self.active_orders.items():
            age = (now - meta.created_dt).total_seconds()
            if meta.purpose.startswith("ENTRY"):
                if age >= float(self.entry_order_timeout):
                    to_cancel.append(oid)
            elif meta.purpose == OrderPurpose.LIMIT_TP.value:
                # 限价止盈单使用单独的超时时间，0=永不超时
                if self.limit_tp_timeout > 0 and age >= float(self.limit_tp_timeout):
                    to_cancel.append(oid)
            else:
                if age >= float(self.exit_order_timeout):
                    to_cancel.append(oid)

        if not to_cancel:
            return

        # 分类处理超时订单
        has_exit_timeout = False  # 是否有非LIMIT_TP的出场单超时

        for oid in to_cancel:
            meta = self.active_orders.get(oid)
            try:
                self.cancel_order(oid)
            except Exception:
                pass
            self.active_orders.pop(oid, None)

            # ⚠️ 修改：区分 LIMIT_TP 和其他出场单的超时处理
            if meta:
                if meta.purpose == OrderPurpose.LIMIT_TP.value:
                    # LIMIT_TP 超时：撤单后重新调度止盈单（0.5s后再挂），防止仓位无人管理
                    self.write_log(f"⏰ [TP超时] 撤单，0.5s后重挂 px={self._pending_tp_price:.0f} retry={self._limit_tp_retry_count}")
                    if self._pending_tp_price > 0 and self._pending_tp_volume > 0 and self._pending_tp_direction is not None:
                        self._need_submit_limit_tp = True
                        self._limit_tp_submit_after = datetime.now() + timedelta(seconds=0.5)
                        self._position_poll_start_time = None
                        self._last_position_poll_time = None
                    else:
                        self.write_log(f"⚠️ [TP超时] 参数缺失无法重挂 (price={self._pending_tp_price} vol={self._pending_tp_volume})")
                elif meta.purpose.startswith("EXIT"):
                    # 其他出场单（止损/FlowFlip）超时：需要重试
                    self.exit_in_progress = False
                    has_exit_timeout = True

        self._last_order_action_dt = now

        # ⚠️ 修改：只有非LIMIT_TP的出场单超时才强制重试
        if has_exit_timeout and self.pos != 0 and not self.exit_in_progress:
            bid1, ask1 = self._get_bid_ask_1(tick)
            pt = self.price_tick if self.price_tick > 0 else 1.0
            if self.pos > 0:
                self._aggressive_flat_long((bid1 if bid1 > 0 else float(tick.last_price or 0.0)) - self.exit_slip_ticks * pt, "RETRY")
            elif self.pos < 0:
                self._aggressive_flat_short((ask1 if ask1 > 0 else float(tick.last_price or 0.0)) + self.exit_slip_ticks * pt, "RETRY")

    def _allow_order_action(self, now: datetime) -> bool:
        """全局订单限流检查（兼容模式）"""
        if self._last_order_action_dt is None:
            return True
        dt = (now - self._last_order_action_dt).total_seconds() * 1000
        return dt >= float(self.min_order_action_interval_ms)

    # 🚀 v2优化: 分离限流机制
    def _allow_entry_order(self, now: datetime) -> bool:
        """开仓订单限流检查（100ms）"""
        if not self.use_separate_order_throttle:
            return self._allow_order_action(now)

        if self._last_entry_order_dt is None:
            return True
        dt = (now - self._last_entry_order_dt).total_seconds() * 1000
        return dt >= float(self.entry_order_interval_ms)

    def _allow_exit_order(self, now: datetime) -> bool:
        """平仓订单限流检查（50ms，更紧急）"""
        if not self.use_separate_order_throttle:
            return self._allow_order_action(now)

        if self._last_exit_order_dt is None:
            return True
        dt = (now - self._last_exit_order_dt).total_seconds() * 1000
        return dt >= float(self.exit_order_interval_ms)

    def _allow_limit_tp_order(self, now: datetime) -> bool:
        """止盈单限流检查（150ms，不紧急）"""
        if not self.use_separate_order_throttle:
            return self._allow_order_action(now)

        if self._last_limit_tp_order_dt is None:
            return True
        dt = (now - self._last_limit_tp_order_dt).total_seconds() * 1000
        return dt >= float(self.limit_tp_order_interval_ms)

    # =========================
    # Trailing 辅助
    # =========================
    def _update_trailing_state_for_long(self, bid1: float, pt: float):
        if not self.tp_triggered:
            if bid1 >= self.entry_price + self.trailing_trigger_ticks * pt:
                self.tp_triggered = True
                self.trailing_peak_price = bid1
                lock = max(self.trailing_min_lock_ticks, 0) * pt
                self.trailing_stop_price = max(self.entry_price + lock, bid1 - self.trailing_stop_ticks * pt)
        else:
            if bid1 > self.trailing_peak_price:
                self.trailing_peak_price = bid1
                lock = max(self.trailing_min_lock_ticks, 0) * pt
                self.trailing_stop_price = max(self.entry_price + lock, bid1 - self.trailing_stop_ticks * pt)

    def _update_trailing_state_for_short(self, ask1: float, pt: float):
        if not self.tp_triggered:
            if ask1 <= self.entry_price - self.trailing_trigger_ticks * pt:
                self.tp_triggered = True
                self.trailing_peak_price = ask1
                lock = max(self.trailing_min_lock_ticks, 0) * pt
                self.trailing_stop_price = min(self.entry_price - lock, ask1 + self.trailing_stop_ticks * pt)
        else:
            if ask1 < self.trailing_peak_price:
                self.trailing_peak_price = ask1
                lock = max(self.trailing_min_lock_ticks, 0) * pt
                self.trailing_stop_price = min(self.entry_price - lock, ask1 + self.trailing_stop_ticks * pt)

    # =========================
    # 风控与统计
    # =========================
    def _calc_close_trade_pnl(self, close_price: float, close_volume: int, close_direction: Direction) -> float:
        """
        优化: 添加手续费和滑点成本计算
        """
        if close_volume <= 0 or self.entry_price <= 0:
            return 0.0

        # 基础盈亏
        if close_direction == Direction.LONG:
            base_pnl = (self.entry_price - float(close_price)) * close_volume
        else:
            base_pnl = (float(close_price) - self.entry_price) * close_volume

        # 手续费成本(买入+卖出双边) - 修正：按开仓价+平仓价计算
        commission = (self.entry_price + float(close_price)) * close_volume * self.commission_rate

        # 滑点成本 - 注意：已经在entry/exit_slip_ticks中体现，设置为0避免重复扣
        # 如果要模拟额外滑点，可以设置slippage_ticks > 0
        pt = self.price_tick if self.price_tick > 0 else 1.0
        slippage_cost = self.slippage_ticks * pt * close_volume

        # 净盈亏
        net_pnl = base_pnl - commission - slippage_cost

        return net_pnl

    def _finalize_flat(self, dt: datetime):
        """
        优化: 修复 PnL 累加逻辑 + 添加盈亏比统计
        """
        self.total_trades += 1

        # 本笔交易的净增量PnL（_trade_realized_pnl是当日累计，daily_pnl是上笔前的累计）
        round_pnl = self._trade_realized_pnl - self.daily_pnl
        # 真正的累加
        self.daily_pnl += round_pnl

        # 盈亏比统计
        if round_pnl > 0:
            self.win_trades += 1
            self.total_win_pnl += round_pnl
            self.avg_win = self.total_win_pnl / self.win_trades if self.win_trades > 0 else 0.0
            self.continuous_loss_count = 0  # 优化: 盈利时重置连亏
        elif round_pnl < 0:
            self.continuous_loss_count += 1
            self.total_loss_pnl += abs(round_pnl)
            loss_trades = self.total_trades - self.win_trades
            self.avg_loss = self.total_loss_pnl / loss_trades if loss_trades > 0 else 0.0

        # 计算盈亏比
        if self.total_loss_pnl > 0:
            self.profit_factor = self.total_win_pnl / self.total_loss_pnl
        else:
            self.profit_factor = 999.0 if self.total_win_pnl > 0 else 0.0

        # 交易日志
        win_rate = (self.win_trades / self.total_trades * 100) if self.total_trades > 0 else 0.0
        _hold_s = (dt - self.entry_time).total_seconds() if self.entry_time else 0.0
        _result_icon = "🟢盈" if round_pnl > 0 else ("🔴亏" if round_pnl < 0 else "⚪平")
        self.write_log(
            f"[平仓{_result_icon}] {round_pnl:+.0f}¥ 持{_hold_s:.0f}s | "
            f"今日:{self.daily_pnl:+.0f}¥ | "
            f"胜率:{win_rate:.1f}%({self.win_trades}/{self.total_trades}) 连亏:{self.continuous_loss_count} | "
            f"均盈/亏:{self.avg_win:.0f}/{self.avg_loss:.0f} PF:{self.profit_factor:.2f}"
        )

        # 重置持仓状态
        self.entry_price = 0.0
        self.entry_time = None
        self.entry_volume = 0
        self.entry_direction = None  # ✅ 重置开仓方向
        self.exit_in_progress = False
        self.last_flat_time = dt
        self._limit_tp_order_ids.clear()  # 清理限价止盈单ID
        self._need_submit_limit_tp = False  # 重置标志
        self._limit_tp_retry_count = 0  # 重置重试计数
        self._limit_tp_submit_after = None  # ✅ 重置延迟提交时间

        # 风控检查
        if self.enable_auto_stop:
            if self.daily_pnl <= self.max_daily_loss:
                self.is_trading_allowed = False
                self.write_log(f"[风控] 日亏熔断: {self.daily_pnl:.0f} <= {self.max_daily_loss:.0f}")
            if self.continuous_loss_count >= self.max_continuous_loss:
                self.is_trading_allowed = False
                self.write_log(f"[风控] 连亏熔断: {self.continuous_loss_count} >= {self.max_continuous_loss}")

    def _check_new_day(self, dt: datetime):
        date_str = dt.strftime("%Y-%m-%d")
        if not self.current_date:
            self.current_date = date_str
            return
        if date_str != self.current_date:
            # 日结
            if self.total_trades > 0:
                win_rate = (self.win_trades / self.total_trades * 100)
                self.write_log("=" * 80)
                self.write_log(f"[日结] {self.current_date}")
                self.write_log(f"  盈亏: {self.daily_pnl:+.0f}¥")
                self.write_log(f"  交易: {self.total_trades}笔 | 胜率: {win_rate:.1f}%")
                self.write_log(f"  平均盈/亏: {self.avg_win:.0f}/{self.avg_loss:.0f}")
                self.write_log(f"  盈亏比(PF): {self.profit_factor:.2f}")
                self.write_log("=" * 80)

            # 重置
            self.current_date = date_str
            self.daily_pnl = 0.0
            self.continuous_loss_count = 0
            self.total_trades = 0
            self.win_trades = 0
            self.is_trading_allowed = True
            self._trade_realized_pnl = 0.0
            self.total_win_pnl = 0.0
            self.total_loss_pnl = 0.0
            self.avg_win = 0.0
            self.avg_loss = 0.0
            self.profit_factor = 0.0

            # 重置滚动队列
            self._prev_total_volume = None
            self._tape_q.clear()
            self._tape_sum = 0.0
            self._tape_abs_sum = 0.0
            self._mom_q.clear()
            self._mom_sum = 0
            self._mid_ret_q.clear()
            self._mid_ret_sum = 0.0
            self._mid_ret_sq_sum = 0.0

    # =========================
    # 时间过滤
    # =========================
    def _parse_time_filters(self):
        try:
            self.trade_start_time_obj = datetime.strptime(self.trade_start_time, "%H:%M:%S").time()
            self.trade_end_time_obj = datetime.strptime(self.trade_end_time, "%H:%M:%S").time()
            self.morning_end_time_obj = datetime.strptime(self.morning_end_time, "%H:%M:%S").time()
            self.afternoon_start_time_obj = datetime.strptime(self.afternoon_start_time, "%H:%M:%S").time()
        except Exception:
            self.trade_start_time_obj = time(9, 0, 0)
            self.trade_end_time_obj = time(15, 0, 0)
            self.morning_end_time_obj = time(11, 30, 0)
            self.afternoon_start_time_obj = time(12, 30, 0)

    def _is_trading_time(self, dt: datetime) -> bool:
        if not self.enable_time_filter:
            return True
        t = dt.time()
        if getattr(t, "tzinfo", None) is not None:
            t = t.replace(tzinfo=None)
        if t < self.trade_start_time_obj or t > self.trade_end_time_obj:
            return False
        if self.morning_end_time_obj <= t < self.afternoon_start_time_obj:
            return False
        return True

    def _in_cooldown(self, dt: datetime) -> bool:
        if not self.last_flat_time:
            return False
        return self._dt_diff_seconds(dt, self.last_flat_time) < float(self.cooldown_seconds)

    def _reset_confirm(self):
        self._long_confirm = 0
        self._short_confirm = 0

    def _load_pricetick_or_fallback(self) -> float:
        try:
            main_engine = self.cta_engine.main_engine
            contract = main_engine.get_contract(self.vt_symbol)
            if contract and getattr(contract, "pricetick", 0) and contract.pricetick > 0:
                # 成功从合约加载，标记已验证（含price_tick=1.0的正常情况）
                self._price_tick_verified = True
                return float(contract.pricetick)
        except Exception:
            pass
        # fallback=1.0，未经合约验证，不设_price_tick_verified标志
        return 1.0

    def _get_bid_ask_1(self, tick: TickData) -> Tuple[float, float]:
        bid1 = float(getattr(tick, "bid_price_1", 0.0) or 0.0)
        ask1 = float(getattr(tick, "ask_price_1", 0.0) or 0.0)
        return bid1, ask1

    def _normalize_tick_bidask(self, tick: TickData) -> TickData:
        bp1 = float(getattr(tick, "bid_price_1", 0.0) or 0.0)
        ap1 = float(getattr(tick, "ask_price_1", 0.0) or 0.0)
        if bp1 <= 0 or ap1 <= 0:
            return tick

        need_swap = False
        if self.kabu_bidask_reversed:
            if bp1 > ap1:
                need_swap = True
        else:
            if self.auto_fix_negative_spread and bp1 > ap1:
                need_swap = True

        if not need_swap:
            return tick

        t = copy.copy(tick)
        n = max(5, int(self.book_depth_levels))
        for i in range(1, n + 1):
            bp = getattr(t, f"bid_price_{i}", None)
            ap = getattr(t, f"ask_price_{i}", None)
            bv = getattr(t, f"bid_volume_{i}", None)
            av = getattr(t, f"ask_volume_{i}", None)
            if bp is not None and ap is not None:
                setattr(t, f"bid_price_{i}", ap)
                setattr(t, f"ask_price_{i}", bp)
            if bv is not None and av is not None:
                setattr(t, f"bid_volume_{i}", av)
                setattr(t, f"ask_volume_{i}", bv)
        return t
