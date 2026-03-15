---
name: kabu-hft-microstructure
description: >
  Expert knowledge base for Japanese equity (TSE) high-frequency microstructure scalping
  using kabu Station API, VeighNa/vnpy CtaTemplate, and IBKR. Use this skill whenever
  the user asks about: order book imbalance (OBI/OFI), LOB-OFI, Tape-OFI, microprice,
  microprice tilt, flow flip detection, tick-by-tick scalping, kabu Station API integration,
  bid/ask field mapping (BidPrice/AskPrice naming convention), HFT execution optimization
  (rate limiting, latency), VeighNa CTA strategy template on_tick/on_bar, position
  management for intraday Japanese equities, or any L1/L2 data signals for alpha generation
  on TSE. Also trigger for Kyle's lambda, VPIN, order flow toxicity, microstructure-based
  entry/exit logic, kabu_signal_stack, or any kabu/vnpy code review and debugging.
  ALWAYS consult this skill before writing or reviewing kabu HFT strategy code.
---

# Kabu HFT 微观结构剥头皮策略 — 专家知识库

## 快速参考索引

| 主题 | 文件 | 关键内容 |
|------|------|----------|
| 字段映射 & API坑点 | `references/kabu-api.md` | **BidPrice=卖一/AskPrice=买一** 完整字段表 |
| 五大信号数学定义 | `references/signals.md` | OBI/LOB-OFI/Tape-OFI/Microprice/FlowFlip |
| VeighNa CTA模板 | `references/vnpy-template.md` | CtaTemplate结构、on_tick、下单API |
| 执行层优化 | `references/execution.md` | 限流/滑点/状态机/动态止盈 |
| 风险管理 | `references/risk.md` | 止损/连亏/仓位/PnL/日内风控 |

> 读取顺序：先本文件确认方向 → 按需读取 references/ 子文件

---

## ⚠️ 最高优先级：字段命名约定（必须先读）

kabu Station API 的字段命名与"直觉"和"旧版文档"均不同，**以官方最新 push.html 为准**：

| kabu API 字段 | 官方日文含义 | 标准金融含义 |
|--------------|-------------|-------------|
| `BidPrice` | 最良**売**気配値段 | **Ask（卖一价）** |
| `BidQty` | 最良**売**気配数量 | **Ask Size（卖一量）** |
| `AskPrice` | 最良**買**気配値段 | **Bid（买一价）** |
| `AskQty` | 最良**買**気配数量 | **Bid Size（买一量）** |
| `Sell1~10` | 売気配（各档） | **asks 盘口（从低到高）** |
| `Buy1~10` | 買気配（各档） | **bids 盘口（从高到低）** |

**适配器代码（所有策略必须先经此层）：**
```python
class KabuBoardAdapter:
    """官方字段适配：BidPrice=卖一(ask), AskPrice=买一(bid)"""
    
    @staticmethod
    def parse(raw: dict) -> dict:
        bid = raw.get('AskPrice', 0.0)   # 最良買気配値段 = 买一价
        ask = raw.get('BidPrice', 0.0)   # 最良売気配値段 = 卖一价
        bid_size = raw.get('AskQty', 0)
        ask_size = raw.get('BidQty', 0)
        
        # sanity check
        assert bid < ask or bid == 0, f"Invalid spread: bid={bid} ask={ask}"
        
        bids = [(b['Price'], b['Qty']) for b in
                [raw.get(f'Buy{i}', {}) for i in range(1, 11)] if b.get('Price', 0) > 0]
        asks = [(s['Price'], s['Qty']) for s in
                [raw.get(f'Sell{i}', {}) for i in range(1, 11)] if s.get('Price', 0) > 0]
        
        bids.sort(key=lambda x: -x[0])  # 降序
        asks.sort(key=lambda x: x[0])   # 升序
        
        return {
            'symbol': raw.get('Symbol'),
            'ts': raw.get('CurrentPriceTime'),
            'bid': bid, 'ask': ask,
            'bid_size': bid_size, 'ask_size': ask_size,
            'last': raw.get('CurrentPrice', 0.0),
            'volume': raw.get('TradingVolume', 0),
            'bids': bids,  # L2买档
            'asks': asks,  # L2卖档
        }
```

详细字段表见 `references/kabu-api.md`

---

## VeighNa CTA 策略结构（必须遵守）

```python
from vnpy_ctastrategy import (
    CtaTemplate, StopOrder,
    TickData, BarData, TradeData, OrderData,
    BarGenerator, ArrayManager,
)
from vnpy.trader.constant import Direction, Offset, Exchange
from vnpy.trader.object import TickData

class KabuHFTStrategy(CtaTemplate):
    """kabu Station HFT 微观结构策略 VeighNa 模板"""
    author = "HFT"
    
    # 策略参数（可在UI调整）
    entry_threshold = 0.4
    strong_threshold = 0.7
    take_profit_ticks = 3
    stop_loss_ticks = 2
    max_hold_seconds = 30
    
    parameters = ["entry_threshold", "strong_threshold",
                  "take_profit_ticks", "stop_loss_ticks", "max_hold_seconds"]
    variables = ["composite_score", "position_direction", "entry_price"]
    
    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        # 信号模块（详见 references/signals.md）
        self.signal_stack = KabuSignalStack()
        # 风控模块（详见 references/risk.md）
        self.risk_guard = RiskGuard()
        # 执行控制（详见 references/execution.md）
        self.exec_ctrl = ExecutionController()
        # 状态
        self.composite_score = 0.0
        self.position_direction = 0
        self.entry_price = 0.0
        self.entry_time = None
    
    def on_init(self):
        self.write_log("Strategy initializing")
        self.load_bar(1)  # HFT: 最少历史加载
    
    def on_start(self):
        self.write_log("Strategy started")
        self.put_event()
    
    def on_stop(self):
        self.write_log("Strategy stopped")
    
    def on_tick(self, tick: TickData):
        """核心：每个 tick 驱动信号 + 执行"""
        # 1. 字段适配（通过 gateway 层已完成，此处二次校验）
        if tick.bid_price_1 <= 0 or tick.ask_price_1 <= 0:
            return
        if tick.bid_price_1 >= tick.ask_price_1:
            self.write_log(f"[WARN] Invalid spread at {tick.datetime}")
            return
        
        # 2. 交易时段保护
        if not self._is_safe_to_trade(tick.datetime):
            return
        
        # 3. 信号计算
        self.composite_score = self.signal_stack.update(tick)
        
        # 4. 持仓管理（止盈/止损/超时）
        if self.position_direction != 0:
            self._manage_position(tick)
            return
        
        # 5. 开仓判断
        if self.risk_guard.can_open() and self.exec_ctrl.can_open():
            direction = self._check_entry(self.composite_score)
            if direction != 0:
                self._open_position(tick, direction)
    
    def on_bar(self, bar: BarData):
        """可选：用于长周期过滤信号（趋势过滤）"""
        pass
    
    def on_order(self, order: OrderData):
        self.put_event()
    
    def on_trade(self, trade: TradeData):
        """成交回报：更新持仓状态"""
        if trade.offset == Offset.OPEN:
            self.position_direction = 1 if trade.direction == Direction.LONG else -1
            self.entry_price = trade.price
            self.entry_time = trade.datetime
        else:
            pnl = self.risk_guard.record_close(
                self.entry_price, trade.price,
                self.position_direction, trade.volume
            )
            self.position_direction = 0
            self.entry_price = 0.0
        self.put_event()
```

完整 VeighNa 接口文档见 `references/vnpy-template.md`

---

## 信号体系速览

五大信号（权重参考：LOB-OFI > Tape-OFI ≈ OBI > momentum ≈ tilt）：

```
composite_score = (
    0.30 * z(lob_ofi) +
    0.25 * z(obi) +
    0.25 * z(tape_ofi) +
    0.10 * z(micro_momentum) +
    0.10 * z(microprice_tilt)
)
```

- **OBI**：加权盘口不平衡，识别供需失衡
- **LOB-OFI**：增量订单流不平衡，最强短期预测因子
- **Tape-OFI**：逐笔主动买卖方向（Quote Rule + Tick Rule）
- **Micro-momentum**：Microprice 滚动均值偏差
- **Microprice Tilt**：(microprice - mid) / half_spread，标准化压力

详细数学定义与代码见 `references/signals.md`

---

## TSE 关键规则（随时可能更新，使用前验证）

```
交易时段: 09:00-11:30 / 12:30-15:30（前场/后场）
订单受理: 08:00起 / 12:05起
开仓保护: 尾盘前5分钟停止（11:25 / 15:25）
最大注册: 50銘柄（PUSH配信）

Tick Size（呼値単位）参考值：
  < ¥1,000      → ¥1
  ¥1,000~3,000  → ¥5
  ¥3,000~10,000 → ¥10  ← 注意：旧版文档有误
  ¥10,000~30,000→ ¥10
  ≥ ¥50,000     → ¥50
  ※ TOPIX500/ETF/ETN 另有独立表，务必核查官方最新公告
```

---

## 策略开发 Checklist

审查或开发任何策略代码时，逐项确认：

- [ ] **字段适配**：AskPrice→bid, BidPrice→ask（官方定义）
- [ ] **数据适配层与策略逻辑分离**（KabuBoardAdapter 独立）
- [ ] **spread sanity check**：bid < ask，否则跳过
- [ ] **OFI 增量计算**（非全量重算）
- [ ] **microprice 公式**：`(ask_size * bid + bid_size * ask) / (bid_size + ask_size)`
- [ ] **开仓/平仓限流分离**（平仓优先）
- [ ] **状态机保护**：IDLE→PENDING_OPEN→OPEN→PENDING_CLOSE→CLOSED
- [ ] **PnL 含手续费和方向乘数**（非价差直接相减）
- [ ] **连亏冷却后需重新确认信号**（非直接重置）
- [ ] **午休/尾盘/异常 spread 保护**
- [ ] **信号→下单→成交延迟有日志记录**

---

## 禁止事项

- ❌ 不得在策略层直接用原始 `BidPrice` 作为 bid（必须经过适配层）
- ❌ 不得硬编码手续费率（券商规则随时变更）
- ❌ 不得把旧版 tick size 表当永久有效（需核对官方公告）
- ❌ 不得忽略部分成交场景
- ❌ 不得把"回测可行"等同于"实盘可行"
- ❌ 不得把所有问题归因于信号（执行层问题往往更多）
