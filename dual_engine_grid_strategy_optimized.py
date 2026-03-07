from __future__ import annotations

import os
import math
import logging
import traceback
from typing import Any, Dict, List, Optional

from vnpy_ctastrategy import (
    CtaTemplate,
    StopOrder,
)
from vnpy.trader.object import TickData, BarData, OrderData, TradeData
from vnpy.trader.utility import BarGenerator, ArrayManager


class DualEngineGridStrategyOptimized(CtaTemplate):
    """
    双引擎微网格策略（优化版 - 只做多，只盈利平仓）

    核心改进：
    ✅ 1. 成本价跟踪：记录平均持仓成本
    ✅ 2. 只盈利平仓：卖单必须高于成本价 + 手续费 + 最小利润
    ✅ 3. 亏损持有：趋势失效时只撤单，不强制平仓
    ✅ 4. 智能止盈：可配置止盈比例

    一、趋势核心仓（CoreTrend）
    -------------------------
    - 通过 EMA20 / EMA60 + 收盘价判断"震荡上行"
    - 趋势成立时：目标持仓 = core_pos
    - 趋势失效时：只撤单，不平仓（等待反弹）

    二、微网格（MicroGrid）
    -------------------------
    - 趋势成立时维护微网格
    - 下方挂买单（不突破 max_pos）
    - ✅ 上方挂卖单：必须 >= 成本价 + 手续费 + 最小利润
    - 价格偏离中心超过 2 格 → 重建网格

    三、风险控制
    -------------------------
    - 趋势失效：撤单（但保留仓位）
    - 只盈利平仓，亏损持有
    """

    author = "Claude Optimized"

    # ---- 参数（可在 VeighNa UI 里设置）----
    ema_fast_window: int = 20
    ema_slow_window: int = 60
    core_pos: int = 1000                 # 核心仓目标（股数）
    grid_levels: int = 3                 # 网格层数（上下各几层）
    grid_step_pct: float = 0.3           # 网格每层间距（百分比）
    grid_volume: int = 100               # 每格下单数（股）
    max_pos: int = 2000                  # 最大多头仓位
    pricetick: float = 0.01              # 最小价格变动

    # —— 手续费/盈利过滤参数 ——
    fee_per_side: float = 80.0           # 预估单边手续费(JPY/100股)，例如80表示每100股单边手续费80日元
    min_profit_multiple: float = 2.0     # 期望净利润 >= 往返手续费 * 倍数（保守设为2倍以上）
    auto_adjust_step: bool = True        # 若网格步长不足覆盖手续费成本，是否自动放大步长

    # —— 新增：止盈参数 ——
    profit_take_pct: float = 0.5        # 止盈比例（%），价格 >= 成本价*(1+1%) 时卖出

    # —— 动态网格参数 ——
    active_grid_levels: int = 0         # 实际激活的网格层数（0=使用grid_levels，>0=限制层数）

    # —— 日志控制参数 ——
    debug_log: bool = True              # 是否输出详细DEBUG日志（生产环境建议关闭）

    parameters = [
        "ema_fast_window",
        "ema_slow_window",
        "core_pos",
        "grid_levels",
        "grid_step_pct",
        "grid_volume",
        "max_pos",
        "pricetick",
        "fee_per_side",
        "min_profit_multiple",
        "auto_adjust_step",
        "profit_take_pct",
        "active_grid_levels",
        "debug_log",
    ]

    variables = [
        "ema_fast",
        "ema_slow",
        "trend_up",
        "grid_center",
        "last_price",
        "avg_cost_price",
        "total_buy_amount",
        "total_buy_volume",
    ]

    def __init__(
        self,
        cta_engine: Any,
        strategy_name: str,
        vt_symbol: str,
        setting: Dict[str, Any],
    ) -> None:
        """"""
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)

        # ====== 关键：先定义 variables 中的字段 ======
        self.ema_fast: float = 0.0
        self.ema_slow: float = 0.0
        self.trend_up: bool = False
        self.grid_center: float = 0.0
        self.last_price: float = 0.0

        # ✅ 成本价跟踪系统
        self.avg_cost_price: float = 0.0        # 平均成本价
        self.total_buy_amount: float = 0.0      # 累计买入金额
        self.total_buy_volume: int = 0          # 累计买入数量

        # Tick → Bar 的合成器（1 分钟）
        self.bg: BarGenerator = BarGenerator(self.on_bar)

        # K 线序列管理（用于 EMA 等指标）
        self.am: ArrayManager = ArrayManager(100)

        # 网格挂单追踪
        self.buy_orderids: List[str] = []     # MicroGrid buy orders
        self.sell_orderids: List[str] = []    # MicroGrid sell orders

        # 核心仓补仓单追踪
        self.core_orderids: List[str] = []    # CoreTrend 补仓单

        # 文件日志
        self._pylogger: logging.Logger = self._setup_pylogger()

        # 记录趋势状态切换
        self._last_trend_up: Optional[bool] = None

        self.write_log(
            f"[__init__] 创建策略实例 strategy_name={strategy_name}, "
            f"vt_symbol={vt_symbol}, setting={setting}"
        )

    # ----------------------------------------------------------------------
    # 日志系统
    # ----------------------------------------------------------------------
    def _setup_pylogger(self) -> logging.Logger:
        """设置 Python 文件日志。"""
        try:
            base_dir = os.getcwd()
            log_dir = os.path.join(base_dir, "log", "cta_strategies")
            os.makedirs(log_dir, exist_ok=True)

            logger = logging.getLogger(f"{self.strategy_name}_{self.vt_symbol}")
            logger.setLevel(logging.INFO)

            if not logger.handlers:
                file_path = os.path.join(
                    log_dir, f"{self.strategy_name}_{self.vt_symbol}.log"
                )
                fh = logging.FileHandler(file_path, encoding="utf-8")
                fh.setLevel(logging.INFO)
                fmt = logging.Formatter(
                    "%(asctime)s | %(levelname)s | %(message)s"
                )
                fh.setFormatter(fmt)
                logger.addHandler(fh)

            logger.info("[logger] 文件日志初始化成功")
            return logger

        except Exception as e:
            self.write_log(f"[logger][ERROR] 初始化文件日志失败：{e!r}")
            self.write_log(traceback.format_exc())
            return logging.getLogger()

    def _log_file(self, msg: str) -> None:
        """写入文件日志"""
        try:
            if self._pylogger:
                self._pylogger.info(msg)
        except Exception:
            pass

    def write_log(self, msg: str) -> None:
        """覆盖 CtaTemplate.write_log，同步写到文件。"""
        super().write_log(msg)
        self._log_file(msg)

    def dbg(self, msg: str) -> None:
        """✅ 调试日志：只在 debug_log=True 时输出"""
        if self.debug_log:
            self.write_log(msg)

    # ----------------------------------------------------------------------
    # 常用工具函数
    # ----------------------------------------------------------------------
    def _get_tick(self) -> float:
        """获取 pricetick"""
        return float(self.pricetick) if self.pricetick and self.pricetick > 0 else 0.01

    def _round_price(self, price: float, direction: str = "buy") -> float:
        """按最小跳动修正价格"""
        tick = self._get_tick()
        if tick <= 0:
            return price
        if direction.lower() == "buy":
            return max(math.floor(price / tick) * tick, tick)
        if direction.lower() == "sell":
            return max(math.ceil(price / tick) * tick, tick)
        return max(round(price / tick) * tick, tick)

    def _fmt_price(self, price: float) -> str:
        """按 tick 决定展示小数位"""
        tick = self._get_tick()
        if tick >= 1:
            return f"{price:.0f}"
        decimals = max(0, int(round(-math.log10(tick))))
        return f"{price:.{decimals}f}"

    # ----------------------------------------------------------------------
    # 手续费/步长过滤工具
    # ----------------------------------------------------------------------
    def _calculate_fee(self, volume: int, side: str = "one") -> float:
        """
        统一的手续费计算函数（按每100股计算）

        Args:
            volume: 交易股数
            side: "one" = 单边（买或卖），"roundtrip" = 往返（买+卖）

        Returns:
            手续费金额（JPY）

        示例：
            - volume=100, side="one": 返回 80.0 JPY（单边）
            - volume=100, side="roundtrip": 返回 160.0 JPY（往返）
            - volume=300, side="one": 返回 240.0 JPY（300/100 * 80）
        """
        try:
            if volume <= 0:
                return 0.0
            # 按每100股计算手续费
            units = volume / 100.0
            if side == "roundtrip":
                # 往返 = 买入手续费 + 卖出手续费
                return units * float(self.fee_per_side) * 2.0
            else:
                # 单边手续费
                return units * float(self.fee_per_side)
        except Exception:
            return 0.0

    def _roundtrip_fee(self, volume: int = 100) -> float:
        """
        预估一轮买+卖的往返手续费（JPY）

        Args:
            volume: 交易股数，默认100股（一格的标准手数）

        Returns:
            往返总手续费（买入 + 卖出）
        """
        return self._calculate_fee(volume, side="roundtrip")

    def _min_step_pct_required(self, center: float) -> float:
        """
        根据中心价和每格股数，计算覆盖手续费所需的最小步长百分比

        Args:
            center: 网格中心价格

        Returns:
            所需最小步长百分比（%）

        逻辑：
            1. 计算每格往返手续费：fee = (grid_volume / 100) * fee_per_side * 2
            2. 计算期望净利润：target_profit = fee * min_profit_multiple
            3. 计算价格波动：price_diff = target_profit / grid_volume
            4. 转换为百分比：step_pct = (price_diff / center) * 100

        示例：
            - center=1000, grid_volume=100, fee_per_side=80, min_profit_multiple=2
            - 往返手续费 = 160 JPY
            - 期望利润 = 320 JPY
            - 每股利润 = 3.2 JPY
            - 最小步长 = 0.32%
        """
        if center <= 0 or self.grid_volume <= 0:
            return 0.0

        # 计算往返手续费
        roundtrip_fee = self._roundtrip_fee(self.grid_volume)

        # 期望净利润
        target_profit = roundtrip_fee * float(self.min_profit_multiple)

        # 转换为步长百分比
        required = (target_profit / (center * self.grid_volume)) * 100.0

        return max(required, 0.0)

    def _ensure_long_only(self) -> None:
        """发现净空头时的紧急处理：撤卖单+补回到0。"""
        if self.pos >= 0:
            return
        self.write_log(f"[Risk][LongOnly] 检测到空头持仓 pos={self.pos}，执行紧急回补")
        self._cancel_grid_orders()
        tick = self._get_tick()
        price = self.last_price or 0
        if price <= 0:
            price = getattr(self, "grid_center", 0) or 0
        cover_price = self._round_price(max(price + tick, tick), direction="buy")
        self.buy(cover_price, abs(self.pos))

    # ----------------------------------------------------------------------
    # ✅ 成本价跟踪系统
    # ----------------------------------------------------------------------
    def _update_cost_on_buy(self, trade_price: float, trade_volume: int) -> None:
        """
        买入成交时更新成本价（包含手续费）

        逻辑：
            1. 计算成交金额 = 价格 * 数量
            2. 计算买入手续费（按每100股）
            3. 总成本 = 成交金额 + 手续费
            4. 更新平均成本价 = 累计总成本 / 累计持仓

        Args:
            trade_price: 成交价格
            trade_volume: 成交数量

        示例：
            - 买入100股 @ 1000元/股，手续费80元
            - 总成本 = 100000 + 80 = 100080元
            - 平均成本价 = 100080 / 100 = 1000.8元/股
        """
        # 计算成交金额
        trade_amount = trade_price * trade_volume

        # ✅ 计算买入手续费（按每100股）
        fee = self._calculate_fee(trade_volume, side="one")

        # 更新累计成本和持仓
        self.total_buy_amount += trade_amount + fee
        self.total_buy_volume += trade_volume

        # 重新计算平均成本价
        if self.total_buy_volume > 0:
            self.avg_cost_price = self.total_buy_amount / self.total_buy_volume
        else:
            self.avg_cost_price = 0.0

        self.write_log(
            f"[Cost] 买入成交更新成本：price={self._fmt_price(trade_price)}, "
            f"vol={trade_volume}, 手续费={fee:.2f}JPY, "
            f"新成本价={self._fmt_price(self.avg_cost_price)}, "
            f"总持仓={self.total_buy_volume}"
        )

    def _update_cost_on_sell(self, trade_price: float, trade_volume: int) -> None:
        """
        卖出成交时更新成本价

        逻辑：
            1. 保存旧成本价（用于计算盈亏）
            2. 防止卖出数量超过持仓（边界保护）
            3. 按成本价减少累计金额
            4. 减少持仓数量
            5. 重新计算平均成本价
            6. 计算本次盈亏（卖价 - 成本价）* 数量

        注意：
            - 卖出手续费不计入成本价（因为已经平仓）
            - 盈亏计算使用旧成本价（卖出前的成本）
            - 如果全部卖出，成本价归零

        Args:
            trade_price: 成交价格
            trade_volume: 成交数量

        示例：
            - 持仓100股，平均成本1000.8元/股
            - 卖出50股 @ 1010元/股
            - 盈亏 = (1010 - 1000.8) * 50 = 460元
            - 剩余持仓50股，成本价仍为1000.8元/股
        """
        # 异常检查：成本价或持仓为0
        if self.total_buy_volume <= 0 or self.avg_cost_price <= 0:
            self.write_log(
                f"[Cost][Warning] 卖出时成本异常：avg_cost={self.avg_cost_price}, "
                f"total_vol={self.total_buy_volume}"
            )
            return

        # ✅ 保存旧成本价用于计算盈亏
        old_avg_cost = self.avg_cost_price

        # ✅ 防止卖出数量超过持仓（边界保护）
        real_volume = min(trade_volume, self.total_buy_volume)
        if real_volume < trade_volume:
            self.write_log(
                f"[Cost][Warning] 卖出数量({trade_volume})超过持仓({self.total_buy_volume})，"
                f"按实际持仓{real_volume}计算"
            )

        # 按成本价减少累计金额
        self.total_buy_amount -= old_avg_cost * real_volume
        self.total_buy_volume -= real_volume

        # 防止负数（边界保护）
        self.total_buy_amount = max(self.total_buy_amount, 0.0)
        self.total_buy_volume = max(self.total_buy_volume, 0)

        # 重新计算平均成本价
        if self.total_buy_volume > 0:
            self.avg_cost_price = self.total_buy_amount / self.total_buy_volume
        else:
            # 全部卖出，成本价归零
            self.avg_cost_price = 0.0
            self.total_buy_amount = 0.0

        # ✅ 使用旧成本价计算盈亏（注意：这是毛利润，未扣除卖出手续费）
        profit = (trade_price - old_avg_cost) * real_volume

        self.write_log(
            f"[Cost] 卖出成交更新成本：price={self._fmt_price(trade_price)}, "
            f"vol={real_volume}, 旧成本价={self._fmt_price(old_avg_cost)}, "
            f"新成本价={self._fmt_price(self.avg_cost_price)}, "
            f"剩余持仓={self.total_buy_volume}, 本次盈亏={profit:.2f}JPY（毛利润，未扣卖出手续费）"
        )

    def _calculate_min_sell_price(self) -> float:
        """
        计算最低卖出价 = max(基于手续费的最低价, 基于止盈比例的最低价)

        目的：确保卖出时能够覆盖成本+手续费+最小利润

        公式1（基于手续费覆盖）：
            1. 计算往返手续费：roundtrip_fee = (grid_volume / 100) * fee_per_side * 2
            2. 计算期望利润：target_profit = roundtrip_fee * min_profit_multiple
            3. 分摊到每股：profit_per_share = target_profit / grid_volume
            4. 最低卖价：min_price_fee = avg_cost_price + profit_per_share

        公式2（基于止盈比例）：
            min_price_take = avg_cost_price * (1 + profit_take_pct / 100)

        最终结果：
            取两者中的较大值，确保同时满足两个条件

        Args:
            无（使用实例变量）

        Returns:
            最低卖出价格（JPY/股）

        示例：
            假设：
            - avg_cost_price = 1000.8（包含买入手续费的成本价）
            - grid_volume = 100
            - fee_per_side = 80
            - min_profit_multiple = 2.0
            - profit_take_pct = 0.5

            计算过程：
            1. 往返手续费 = (100/100) * 80 * 2 = 160 JPY
            2. 期望利润 = 160 * 2 = 320 JPY
            3. 每股利润 = 320 / 100 = 3.2 JPY
            4. min_price_fee = 1000.8 + 3.2 = 1004.0
            5. min_price_take = 1000.8 * 1.005 = 1005.8
            6. 最终最低价 = max(1004.0, 1005.8) = 1005.8
        """
        if self.avg_cost_price <= 0 or self.grid_volume <= 0:
            return 0.0

        # 公式1：基于手续费覆盖的最低价
        roundtrip_fee = self._roundtrip_fee(self.grid_volume)
        target_profit = roundtrip_fee * self.min_profit_multiple
        profit_per_share = target_profit / self.grid_volume
        min_price_fee = self.avg_cost_price + profit_per_share

        # 公式2：基于止盈比例的最低价
        if self.profit_take_pct > 0:
            min_price_take = self.avg_cost_price * (1 + self.profit_take_pct / 100.0)
            # 取两者中的较大值
            return max(min_price_fee, min_price_take)

        return min_price_fee

    # ----------------------------------------------------------------------
    # 生命周期回调
    # ----------------------------------------------------------------------
    def on_init(self) -> None:
        """策略初始化：加载历史 K 线，预热 EMA。"""
        self.write_log(
            "[on_init] DualEngineGridStrategyOptimized 初始化："
            f"ema_fast_window={self.ema_fast_window}, "
            f"ema_slow_window={self.ema_slow_window}, "
            f"core_pos={self.core_pos}, grid_levels={self.grid_levels}, "
            f"active_grid_levels={self.active_grid_levels}, "
            f"grid_step_pct={self.grid_step_pct}, grid_volume={self.grid_volume}, "
            f"max_pos={self.max_pos}, pricetick={self.pricetick}, "
            f"fee_per_side={self.fee_per_side}, min_profit_multiple={self.min_profit_multiple}, "
            f"auto_adjust_step={self.auto_adjust_step}, profit_take_pct={self.profit_take_pct}"
        )

        # 加载 10 天历史 K 线
        self.load_bar(10)

        self.write_log("[on_init] DualEngineGridStrategyOptimized 初始化完成")
        self.put_event()

    def on_start(self) -> None:
        """策略启动"""
        self.write_log("[on_start] DualEngineGridStrategyOptimized 启动")
        self.write_log(f"[on_start] vt_symbol={self.vt_symbol}, pos={self.pos}, inited={self.inited}")

        # ✅ 如果启动时已有持仓，但成本价为0，则初始化成本价
        if self.pos > 0 and self.total_buy_volume == 0:
            # 使用当前价格作为近似成本价
            approx_cost = self.last_price if self.last_price > 0 else 0
            if approx_cost > 0:
                self.total_buy_volume = self.pos
                self.total_buy_amount = approx_cost * self.pos
                self.avg_cost_price = approx_cost
                self.write_log(
                    f"[on_start][CostInit] 检测到已有多头持仓 pos={self.pos}，"
                    f"但成本价系统未初始化。假设成本价≈{self._fmt_price(approx_cost)}，"
                    f"初始化成本系统（注意：这只是近似值，建议空仓启动策略）"
                )
            else:
                self.write_log(
                    f"[on_start][Warning] 检测到已有持仓 pos={self.pos}，但无法获取当前价格初始化成本价。"
                    f"建议空仓启动策略或手动平仓！"
                )

        self.write_log("[on_start] 等待接收行情数据...")
        self.put_event()

    def on_stop(self) -> None:
        """策略停止"""
        self.write_log("[on_stop] DualEngineGridStrategyOptimized 停止")
        self.put_event()

    # ----------------------------------------------------------------------
    # 行情回调
    # ----------------------------------------------------------------------
    def on_tick(self, tick: TickData) -> None:
        """Tick 回调：推入 BarGenerator。"""
        try:
            # 初始化计数器
            if not hasattr(self, '_tick_count'):
                self._tick_count = 0
                self.write_log("[on_tick] 首次接收到Tick数据！")

            self._tick_count += 1

            if tick.last_price:
                self.last_price = tick.last_price

            # 前5个tick全部输出，之后每10个输出一次
            if self._tick_count <= 5 or self._tick_count % 10 == 0:
                self.write_log(
                    f"[on_tick] #{self._tick_count} last_price={self._fmt_price(tick.last_price)}, "
                    f"bid={tick.bid_price_1}, ask={tick.ask_price_1}"
                )

            self.bg.update_tick(tick)
        except Exception as e:
            self.write_log(f"[EXCEPTION][on_tick] {e!r}")
            self.write_log(traceback.format_exc())
            raise

    def on_bar(self, bar: BarData) -> None:
        """1 分钟 K 线回调：驱动趋势引擎 + 网格引擎。"""
        try:
            self.last_price = bar.close_price

            # 长期只做多保护
            self._ensure_long_only()

            # 更新 K 线序列
            self.am.update_bar(bar)
            if not self.am.inited:
                self.write_log(
                    f"[on_bar][INIT] 预热中 time={bar.datetime}, "
                    f"close={bar.close_price}, am.count={self.am.count}, "
                    f"需要至少 {max(self.ema_fast_window, self.ema_slow_window)} 根K线"
                )
                return

            # === 1. 计算 EMA，判断趋势 ===
            self.ema_fast = float(self.am.ema(self.ema_fast_window, array=False))
            self.ema_slow = float(self.am.ema(self.ema_slow_window, array=False))

            self.trend_up = (
                self.ema_fast > self.ema_slow
                and bar.close_price > self.ema_fast
            )

            # 趋势切换日志
            if self._last_trend_up != self.trend_up:
                self.write_log(
                    f"[Trend] 趋势切换：last={self._last_trend_up} → now={self.trend_up} "
                    f"time={bar.datetime}, close={self._fmt_price(bar.close_price)}, "
                    f"ema_fast={self._fmt_price(self.ema_fast)}, ema_slow={self._fmt_price(self.ema_slow)}"
                )
                self._last_trend_up = self.trend_up

            self.write_log(
                f"[on_bar] time={bar.datetime}, close={self._fmt_price(bar.close_price)}, "
                f"ema_fast={self._fmt_price(self.ema_fast)}, ema_slow={self._fmt_price(self.ema_slow)}, "
                f"trend_up={self.trend_up}, pos={self.pos}, inited={self.inited}, "
                f"avg_cost={self._fmt_price(self.avg_cost_price)}, "
                f"grid_center={self._fmt_price(self.grid_center)}"
            )

            # 初始化阶段（load_bar 时）不交易
            if not self.inited:
                self.write_log(
                    f"[on_bar][WARNING] 策略未初始化 self.inited={self.inited}，跳过交易逻辑"
                )
                self.put_event()
                return

            # ✅ 持仓一致性检查
            if self.pos != self.total_buy_volume and self.total_buy_volume > 0:
                self.write_log(
                    f"[Warning] 持仓不一致：pos={self.pos}, total_buy_volume={self.total_buy_volume}，"
                    f"可能存在手动平仓或策略重启。建议检查！"
                )

            # === 2. 根据趋势驱动双引擎 ===
            self.write_log(
                f"[on_bar][DEBUG] 准备执行交易逻辑：trend_up={self.trend_up}"
            )

            if self.trend_up:
                self.write_log("[on_bar][DEBUG] 趋势向上，调用核心仓和网格逻辑")
                # 先调整核心仓，再维护网格
                self._sync_core_position(bar)
                self._update_grid(bar)
            else:
                self.write_log("[on_bar][DEBUG] 趋势失效，撤销所有挂单")
                # ✅ 趋势失效：只撤单，不平仓（等待反弹）
                self._cancel_all_orders_on_trend_fail(bar)

            self.put_event()

        except Exception as e:
            self.write_log(
                f"[EXCEPTION][on_bar] {e!r} | "
                f"time={bar.datetime}, close={bar.close_price}, pos={self.pos}, "
                f"ema_fast={getattr(self,'ema_fast',None)}, ema_slow={getattr(self,'ema_slow',None)}"
            )
            self.write_log(traceback.format_exc())
            raise

    def on_order(self, order: OrderData) -> None:
        """委托状态更新"""
        self.write_log(
            f"[on_order] vt_orderid={order.vt_orderid}, status={order.status}, "
            f"direction={order.direction}, offset={getattr(order, 'offset', '')}, "
            f"price={order.price}, volume={order.volume}, traded={order.traded}, "
            f"active={order.is_active()}, pos={self.pos}"
        )

        # 如果是核心补仓单，且已不再活动，则从 core_orderids 中移除
        if order.vt_orderid in self.core_orderids and not order.is_active():
            self.core_orderids.remove(order.vt_orderid)
            self.write_log(
                f"[on_order] 核心补仓单完成或被撤销，移除 vt_orderid={order.vt_orderid}, "
                f"剩余未完成核心单={self.core_orderids}"
            )

        # 网格订单已结束（成交/撤销）时，从记录里移除
        if not order.is_active():
            if order.vt_orderid in self.buy_orderids:
                self.buy_orderids.remove(order.vt_orderid)
            if order.vt_orderid in self.sell_orderids:
                self.sell_orderids.remove(order.vt_orderid)

        self.put_event()

    def on_trade(self, trade: TradeData) -> None:
        """成交回调"""
        self.write_log(
            f"[on_trade] time={getattr(trade, 'datetime', '')}, "
            f"vt_tradeid={trade.vt_tradeid}, order={trade.vt_orderid}, "
            f"direction={trade.direction}, offset={getattr(trade, 'offset', '')}, "
            f"price={trade.price}, volume={trade.volume}, pos={self.pos}"
        )

        # ✅ 更新成本价
        # 判断买卖方向：通过direction字符串判断
        direction_str = str(trade.direction).upper()
        if "LONG" in direction_str or "BUY" in direction_str:
            # 买入
            self._update_cost_on_buy(trade.price, trade.volume)
        elif "SHORT" in direction_str or "SELL" in direction_str:
            # 卖出
            self._update_cost_on_sell(trade.price, trade.volume)

        # 长期只做多保护
        self._ensure_long_only()

        self.put_event()

    def on_stop_order(self, stop_order: StopOrder) -> None:
        """停止单（本策略未使用）"""
        self.write_log(
            f"[on_stop_order] stop_orderid={stop_order.stop_orderid}, "
            f"vt_symbol={stop_order.vt_symbol}, "
            f"direction={stop_order.direction}, volume={stop_order.volume}"
        )

    # ----------------------------------------------------------------------
    # 引擎 A：趋势核心仓
    # ----------------------------------------------------------------------
    def _sync_core_position(self, bar: BarData) -> None:
        """
        趋势成立时，保持核心仓位 core_pos。
        - pos < core_pos → 买入补到 core_pos（不超过 max_pos）
        """
        self.dbg(
            f"[CoreTrend][DEBUG] 进入核心仓调整逻辑"
        )

        current_pos = self.pos
        target_core = min(self.core_pos, self.max_pos)

        self.dbg(
            f"[CoreTrend][DEBUG] 当前持仓={current_pos}, 目标核心仓={target_core}, "
            f"core_pos={self.core_pos}, max_pos={self.max_pos}"
        )

        if current_pos >= target_core:
            self.write_log(
                f"[CoreTrend] 核心仓已满足：pos={current_pos} >= target_core={target_core}"
            )
            return

        buy_volume = target_core - current_pos
        if buy_volume <= 0:
            self.write_log(
                f"[CoreTrend][DEBUG] buy_volume={buy_volume} <= 0，无需补仓"
            )
            return

        # 如果还有未完成核心补仓单，则不再重复下单
        if self.core_orderids:
            self.write_log(
                f"[CoreTrend] 已有核心补仓挂单未完成，core_orderids={self.core_orderids}，"
                f"当前pos={current_pos}，暂不重复下单"
            )
            return

        self.write_log(
            f"[CoreTrend] 调整核心仓检查：pos={current_pos}, "
            f"target_core={target_core}, max_pos={self.max_pos}, "
            f"bar_time={bar.datetime}, bar_close={bar.close_price}"
        )

        tick = self._get_tick()
        price = self._round_price(bar.close_price + tick, direction="buy")

        self.write_log(
            f"[CoreTrend][ORDER] 准备下单补仓 buy_volume={buy_volume} 股 → "
            f"目标={target_core}, 当前pos={current_pos}, price={self._fmt_price(price)}, tick={tick}"
        )

        vt_orderids = self.buy(price, buy_volume)

        self.write_log(
            f"[CoreTrend][ORDER] buy() 返回结果：vt_orderids={vt_orderids}"
        )

        if vt_orderids:
            self.core_orderids.extend(vt_orderids)
            self.write_log(
                f"[CoreTrend][SUCCESS] 核心补仓单已提交 core_orderids={self.core_orderids}"
            )
        else:
            self.write_log(
                f"[CoreTrend][ERROR] buy() 返回空，订单提交失败！"
            )

    # ----------------------------------------------------------------------
    # 引擎 B：微网格
    # ----------------------------------------------------------------------
    def _update_grid(self, bar: BarData) -> None:
        """
        在趋势成立时维护微网格：
        - 若尚未有 grid_center → 以当前收盘价为中心生成网格
        - 若价格偏离中心超过 2 格 → 重建网格
        """
        self.write_log(
            f"[MicroGrid][DEBUG] 进入网格更新逻辑，grid_levels={self.grid_levels}, "
            f"grid_volume={self.grid_volume}"
        )

        if self.grid_levels <= 0 or self.grid_volume <= 0:
            self.write_log(
                f"[MicroGrid][WARNING] grid_levels={self.grid_levels} 或 grid_volume={self.grid_volume} <=0，网格逻辑关闭"
            )
            return

        step = self.grid_step_pct / 100.0
        price = bar.close_price
        center = self.grid_center

        self.write_log(
            f"[MicroGrid][DEBUG] step={step:.6f}, price={self._fmt_price(price)}, "
            f"grid_center={self._fmt_price(center)}"
        )

        if center <= 0:
            # 初次建立网格
            self.grid_center = price
            self.write_log(
                f"[MicroGrid] 初始化网格，center={self._fmt_price(self.grid_center)}, "
                f"step={self.grid_step_pct}%，pos={self.pos}"
            )
            self._rebuild_grid_orders()
            return

        # 偏离中心超过 2 格 → 爬坡重建
        deviation = abs(price - center) / center if center > 0 else 0.0
        self.write_log(
            f"[MicroGrid][DEBUG] deviation={deviation:.6f} ({deviation*100:.2f}%), "
            f"threshold={2*step:.6f} ({2*step*100:.2f}%)"
        )

        if deviation >= 2 * step:
            old_center = self.grid_center
            self.write_log(
                f"[MicroGrid] 价格偏离中心超过 2 格，准备爬坡重建："
                f"old_center={self._fmt_price(old_center)}, new_center={self._fmt_price(price)}, "
                f"deviation={deviation * 100:.2f}%"
            )
            # ✅ 移除重复调用，_rebuild_grid_orders() 内部会先撤单
            self.grid_center = price
            self._rebuild_grid_orders()
        else:
            self.write_log(
                f"[MicroGrid][DEBUG] 价格偏离未超过阈值，不重建网格"
            )

    def _rebuild_grid_orders(self) -> None:
        """
        按当前 grid_center 重建网格挂单：
        - 下方挂买单（不突破 max_pos）
        - ✅ 上方挂卖单（必须 >= 成本价 + 手续费 + 最小利润）
        - 🆕 动态网格：根据 active_grid_levels 限制实际挂单层数
        """
        self.write_log(
            f"[MicroGrid][DEBUG] 进入重建网格挂单逻辑"
        )

        center = self.grid_center

        # === 手续费过滤/自动抬升步长 ===
        required_step_pct = self._min_step_pct_required(center)
        effective_step_pct = self.grid_step_pct

        self.write_log(
            f"[MicroGrid][DEBUG] required_step_pct={required_step_pct:.4f}%, "
            f"grid_step_pct={self.grid_step_pct:.4f}%, "
            f"auto_adjust_step={self.auto_adjust_step}"
        )

        if required_step_pct > effective_step_pct:
            if self.auto_adjust_step:
                effective_step_pct = required_step_pct
                self.write_log(
                    f"[MicroGrid][FeeFilter] grid_step_pct={self.grid_step_pct:.4f}% "
                    f"< required={required_step_pct:.4f}%，已自动放大为 {effective_step_pct:.4f}% "
                    f"以覆盖手续费 (roundtrip_fee≈{self._roundtrip_fee():.2f}JPY, vol={self.grid_volume})"
                )
            else:
                self.write_log(
                    f"[MicroGrid][FeeFilter] grid_step_pct={self.grid_step_pct:.4f}% "
                    f"< required={required_step_pct:.4f}%，为避免利润小于手续费，本次不重建网格。"
                )
                return

        step = effective_step_pct / 100.0

        current_pos = self.pos
        max_pos = self.max_pos

        self.write_log(
            f"[MicroGrid][DEBUG] step={step:.6f}, current_pos={current_pos}, max_pos={max_pos}"
        )

        # 保险：先撤掉记录中的旧网格单（若有）
        self._cancel_grid_orders()

        self.buy_orderids.clear()
        self.sell_orderids.clear()

        # 可用买入空间
        available_buy_space = max(max_pos - max(current_pos, 0), 0)

        if self.grid_volume > 0:
            max_buy_levels = int(available_buy_space // self.grid_volume)
            max_sell_levels = int(max(current_pos, 0) // self.grid_volume)
        else:
            max_buy_levels = 0
            max_sell_levels = 0

        self.write_log(
            f"[MicroGrid][DEBUG] available_buy_space={available_buy_space}, "
            f"max_buy_levels={max_buy_levels}, max_sell_levels={max_sell_levels}"
        )

        # ✅ 计算最低卖价（成本价 + 手续费 + 利润）
        min_sell_price = self._calculate_min_sell_price()

        self.write_log(
            f"[MicroGrid][DEBUG] min_sell_price={self._fmt_price(min_sell_price)}, "
            f"avg_cost_price={self._fmt_price(self.avg_cost_price)}"
        )

        # 每层最小毛利润阈值（JPY）
        # 注意：这里使用 grid_volume 计算该网格层的往返手续费
        min_profit_jpy = self._roundtrip_fee(self.grid_volume) * float(self.min_profit_multiple)

        self.write_log(
            f"[MicroGrid][DEBUG] min_profit_jpy={min_profit_jpy:.2f}, "
            f"roundtrip_fee={self._roundtrip_fee(self.grid_volume):.2f}"
        )

        tick = self._get_tick()

        # 🆕 动态网格：确定实际激活的层数
        if self.active_grid_levels > 0:
            # 使用 active_grid_levels 限制实际挂单层数
            effective_levels = min(self.active_grid_levels, self.grid_levels)
            self.write_log(
                f"[MicroGrid][Dynamic] 动态网格模式：grid_levels={self.grid_levels}, "
                f"active_grid_levels={self.active_grid_levels}, "
                f"实际激活层数={effective_levels}"
            )
        else:
            # active_grid_levels=0 时使用全部 grid_levels
            effective_levels = self.grid_levels
            self.write_log(
                f"[MicroGrid][DEBUG] 标准模式：使用全部层数 grid_levels={self.grid_levels}"
            )

        self.write_log(
            f"[MicroGrid][DEBUG] 开始循环建立网格挂单，effective_levels={effective_levels}"
        )

        for i in range(1, effective_levels + 1):
            self.write_log(
                f"[MicroGrid][DEBUG] 处理第 {i} 层网格"
            )

            # 下方买单
            if i <= max_buy_levels and self.grid_volume > 0:
                buy_price = center * (1.0 - step * i)
                buy_price = self._round_price(max(buy_price, tick), direction="buy")

                gross_profit = abs(center - buy_price) * self.grid_volume

                self.write_log(
                    f"[MicroGrid][DEBUG] 买单 L{i}: price={self._fmt_price(buy_price)}, "
                    f"gross_profit={gross_profit:.2f}, min_profit={min_profit_jpy:.2f}"
                )

                if gross_profit < min_profit_jpy:
                    self.write_log(
                        f"[MicroGrid][FeeFilter] 跳过买单 L{i} diff={abs(center-buy_price):.2f} "
                        f"gross={gross_profit:.2f} < min_profit={min_profit_jpy:.2f}"
                    )
                else:
                    self.write_log(
                        f"[MicroGrid][ORDER] 准备下买单 L{i} price={self._fmt_price(buy_price)} vol={self.grid_volume}"
                    )
                    vt_orderids = self.buy(buy_price, self.grid_volume)
                    self.write_log(
                        f"[MicroGrid][ORDER] buy() 返回：{vt_orderids}"
                    )
                    if vt_orderids:
                        self.buy_orderids.extend(vt_orderids)
                        self.write_log(
                            f"[MicroGrid][SUCCESS] 买单 L{i} 已提交"
                        )
                    else:
                        self.write_log(
                            f"[MicroGrid][ERROR] 买单 L{i} 提交失败！buy() 返回空"
                        )
            else:
                self.write_log(
                    f"[MicroGrid][DEBUG] 跳过买单 L{i}：i({i}) > max_buy_levels({max_buy_levels}) "
                    f"或 grid_volume({self.grid_volume}) <= 0"
                )

            # ✅ 上方卖单：必须 >= 最低卖价
            if i <= max_sell_levels and self.grid_volume > 0:
                sell_price = center * (1.0 + step * i)
                sell_price = self._round_price(max(sell_price, tick), direction="sell")

                self.write_log(
                    f"[MicroGrid][DEBUG] 卖单 L{i}: price={self._fmt_price(sell_price)}, "
                    f"min_sell_price={self._fmt_price(min_sell_price)}"
                )

                # ✅ 关键过滤：卖价必须 >= 最低卖价
                if min_sell_price > 0 and sell_price < min_sell_price:
                    self.write_log(
                        f"[MicroGrid][CostFilter] 跳过卖单 L{i} price={self._fmt_price(sell_price)} "
                        f"< min_sell_price={self._fmt_price(min_sell_price)} (成本价={self._fmt_price(self.avg_cost_price)})"
                    )
                    continue

                gross_profit = abs(sell_price - center) * self.grid_volume

                self.write_log(
                    f"[MicroGrid][DEBUG] 卖单 L{i}: gross_profit={gross_profit:.2f}, "
                    f"min_profit={min_profit_jpy:.2f}"
                )

                if gross_profit < min_profit_jpy:
                    self.write_log(
                        f"[MicroGrid][FeeFilter] 跳过卖单 L{i} diff={abs(sell_price-center):.2f} "
                        f"gross={gross_profit:.2f} < min_profit={min_profit_jpy:.2f}"
                    )
                else:
                    self.write_log(
                        f"[MicroGrid][ORDER] 准备下卖单 L{i} price={self._fmt_price(sell_price)} vol={self.grid_volume}"
                    )
                    vt_orderids = self.sell(sell_price, self.grid_volume)
                    self.write_log(
                        f"[MicroGrid][ORDER] sell() 返回：{vt_orderids}"
                    )
                    if vt_orderids:
                        self.sell_orderids.extend(vt_orderids)
                        self.write_log(
                            f"[MicroGrid][SUCCESS] 卖单 L{i} 已提交 (成本价={self._fmt_price(self.avg_cost_price)})"
                        )
                    else:
                        self.write_log(
                            f"[MicroGrid][ERROR] 卖单 L{i} 提交失败！sell() 返回空"
                        )
            else:
                self.write_log(
                    f"[MicroGrid][DEBUG] 跳过卖单 L{i}：i({i}) > max_sell_levels({max_sell_levels}) "
                    f"或 grid_volume({self.grid_volume}) <= 0"
                )

        # 计算实际激活层数
        if self.active_grid_levels > 0:
            effective_levels = min(self.active_grid_levels, self.grid_levels)
        else:
            effective_levels = self.grid_levels

        self.write_log(
            f"[MicroGrid] 重建网格完成：center={self._fmt_price(center)}, "
            f"min_sell_price={self._fmt_price(min_sell_price)}, "
            f"配置层数={self.grid_levels}, 激活层数={effective_levels}, "
            f"买层数={min(effective_levels, max_buy_levels)}, "
            f"卖层数={min(effective_levels, max_sell_levels)}, "
            f"pos={current_pos}, max_pos={max_pos}, "
            f"buy_order_count={len(self.buy_orderids)}, "
            f"sell_order_count={len(self.sell_orderids)}"
        )

    def _cancel_grid_orders(self) -> None:
        """取消所有已有的网格挂单。"""
        total_buy = len(self.buy_orderids)
        total_sell = len(self.sell_orderids)

        if not (total_buy or total_sell):
            self.write_log("[MicroGrid] 没有记录到网格挂单，无需取消")
            return

        self.write_log(
            f"[MicroGrid] 取消网格挂单：buy={total_buy}，sell={total_sell}"
        )

        # 遍历一个副本，避免遍历过程中修改原列表
        for vt_orderid in list(self.buy_orderids) + list(self.sell_orderids):
            self.cancel_order(vt_orderid)

        self.buy_orderids.clear()
        self.sell_orderids.clear()
        self.write_log("[MicroGrid] 网格挂单取消完成，记录已清空")

    # ----------------------------------------------------------------------
    # ✅ 风险控制：趋势失效 → 只撤单，不平仓
    # ----------------------------------------------------------------------
    def _cancel_all_orders_on_trend_fail(self, bar: BarData) -> None:
        """
        趋势失效时：
        - 撤掉网格单 + 核心单
        - ✅ 不平仓（等待反弹）
        - 重置 grid_center
        """
        self.write_log(
            f"[Risk] 趋势失效，撤销所有挂单（保留仓位等待反弹）：pos={self.pos}, "
            f"avg_cost={self._fmt_price(self.avg_cost_price)}, "
            f"time={bar.datetime}, close={self._fmt_price(bar.close_price)}"
        )

        # 撤单（网格）
        self._cancel_grid_orders()

        # 撤单（核心补仓）
        for vt_orderid in list(self.core_orderids):
            self.cancel_order(vt_orderid)
        self.core_orderids.clear()

        # ✅ 不平仓，只重置网格中心
        self.grid_center = 0.0
        self.write_log(
            "[Risk] 已撤销所有挂单，保留仓位等待反弹。"
            f"当前持仓={self.pos}, 成本价={self._fmt_price(self.avg_cost_price)}"
        )
