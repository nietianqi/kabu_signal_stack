# VeighNa CTA 策略模板参考

## 包结构

```python
from vnpy_ctastrategy import (
    CtaTemplate,        # 策略基类（必须继承）
    StopOrder,          # 本地停止单
    TickData,           # Tick 数据容器
    BarData,            # K 线数据容器
    TradeData,          # 成交回报
    OrderData,          # 委托回报
    BarGenerator,       # Tick → K线生成器
    ArrayManager,       # K线时间序列（含技术指标）
)
from vnpy.trader.constant import Direction, Offset, Exchange, OrderType, Status
from vnpy.trader.object import TickData  # 也可从此导入
```

## CtaTemplate 生命周期

```
策略加载
  → on_init()    : 初始化，调用 load_bar(n) 加载历史
  → on_start()   : 启动，调用 put_event() 更新UI
  → on_tick()    : 每个 tick 触发（主要驱动）
  → on_bar()     : 每根 K 线触发（可选）
  → on_order()   : 委托状态更新
  → on_trade()   : 成交回报
  → on_stop()    : 策略停止
```

## TickData 字段

```python
tick.symbol          # 代码（不含交易所）
tick.exchange        # Exchange.SSE 等
tick.datetime        # datetime 对象
tick.gateway_name    # 网关名

# 价格
tick.last_price      # 最新价
tick.open_price      # 今日开盘
tick.high_price      # 今日最高
tick.low_price       # 今日最低

# 成交量
tick.volume          # 累计成交量
tick.turnover        # 累计成交额
tick.open_interest   # 持仓量（期货）

# L1 盘口
tick.bid_price_1     # 买一价（已适配：来自 AskPrice）
tick.ask_price_1     # 卖一价（已适配：来自 BidPrice）
tick.bid_volume_1    # 买一量
tick.ask_volume_1    # 卖一量

# L2 盘口（2~5档）
tick.bid_price_2     # 买二价
tick.bid_volume_2
# ... 最多 5 档
```

## 下单 API

```python
# 开仓
vt_orderids = self.buy(price, volume)    # 买入开仓
vt_orderids = self.short(price, volume)  # 卖出开仓（需信用账户）

# 平仓
vt_orderids = self.sell(price, volume)   # 卖出平仓
vt_orderids = self.cover(price, volume)  # 买入平仓

# 通用接口
vt_orderids = self.send_order(
    direction=Direction.LONG,
    offset=Offset.OPEN,
    price=price,
    volume=volume,
    stop=False,       # True = 本地停止单
    lock=False,       # True = 锁仓模式
)

# 撤单
self.cancel_order(vt_orderid)
self.cancel_all()

# 日志
self.write_log("message")

# UI 刷新（变量变更后调用）
self.put_event()
```

## 持仓查询

```python
# 查询当前持仓
pos = self.get_pos()  # 返回净持仓数量（多为正，空为负）

# 在 on_trade 中手动维护更可靠
def on_trade(self, trade: TradeData):
    if trade.direction == Direction.LONG:
        if trade.offset == Offset.OPEN:
            self.long_pos += trade.volume
        else:
            self.short_pos -= trade.volume
    else:
        if trade.offset == Offset.OPEN:
            self.short_pos += trade.volume
        else:
            self.long_pos -= trade.volume
    self.put_event()
```

## kabu Gateway 集成要点

### vt_symbol 格式
```python
# kabu: 6位股票代码 "7203"
# VeighNa: "7203.TSE" 或 "7203.TSE2"
vt_symbol = "7203.TSE"

def kabu_to_vt(symbol: str) -> str:
    return f"{symbol}.TSE"

def vt_to_kabu(vt_symbol: str) -> str:
    return vt_symbol.split('.')[0]
```

### Gateway 适配层（在 gateway 文件中处理字段反转）
```python
from vnpy.trader.object import TickData
from vnpy.trader.constant import Exchange

class KabuGateway:
    """在 gateway 层处理字段反转，策略层无需关心"""
    
    def _on_board(self, raw: dict):
        snap = KabuBoardAdapter.parse(raw)
        if snap is None:
            return
        
        tick = TickData(
            symbol=snap.symbol,
            exchange=Exchange.TSE,
            datetime=self._parse_ts(snap.ts),
            gateway_name=self.gateway_name,
            last_price=snap.last,
            volume=snap.volume,
            # ⚠️ 已适配：bid←AskPrice, ask←BidPrice
            bid_price_1=snap.bid,
            bid_volume_1=snap.bid_size,
            ask_price_1=snap.ask,
            ask_volume_1=snap.ask_size,
        )
        # 填充 L2
        for i, (p, q) in enumerate(snap.bids[1:5], 2):
            setattr(tick, f'bid_price_{i}', p)
            setattr(tick, f'bid_volume_{i}', q)
        for i, (p, q) in enumerate(snap.asks[1:5], 2):
            setattr(tick, f'ask_price_{i}', p)
            setattr(tick, f'ask_volume_{i}', q)
        
        self.on_tick(tick)
```

## 策略参数与变量声明规范

```python
class MyStrategy(CtaTemplate):
    author = "HFT"
    
    # parameters: 可在UI中修改的参数（必须是类属性）
    entry_threshold = 0.4
    stop_loss_ticks = 2
    
    # variables: 在UI中显示的运行时变量
    composite_score = 0.0
    position_direction = 0
    
    parameters = ["entry_threshold", "stop_loss_ticks"]
    variables = ["composite_score", "position_direction"]
    
    # 注意：parameters 和 variables 必须是字符串列表
    # 对应上方声明的类属性名
```

## 交易时段保护

```python
import datetime

def _is_safe_to_trade(self, dt: datetime.datetime) -> bool:
    t = dt.time()
    morning = datetime.time(9, 0) <= t <= datetime.time(11, 25)
    afternoon = datetime.time(12, 30) <= t <= datetime.time(15, 25)
    return morning or afternoon

def _is_morning_session(self, dt: datetime.datetime) -> bool:
    t = dt.time()
    return datetime.time(9, 0) <= t <= datetime.time(11, 30)
```

## Tick Size 计算

```python
def get_tick_size(price: float) -> float:
    """TSE 呼値単位（参考值，以官方最新公告为准）"""
    if price < 1000:     return 1.0
    elif price < 3000:   return 5.0
    elif price < 10000:  return 10.0
    elif price < 30000:  return 10.0
    elif price < 50000:  return 50.0
    elif price < 100000: return 100.0
    elif price < 300000: return 100.0
    elif price < 500000: return 500.0
    elif price < 1000000:return 1000.0
    elif price < 3000000:return 1000.0
    elif price < 5000000:return 5000.0
    else:                return 10000.0
    # ※ TOPIX500 / ETF / ETN 另有独立表
```
