# 执行层优化文档

## 1. 限流架构（开平仓分离）

```python
import time
from dataclasses import dataclass, field

@dataclass
class RateLimiter:
    interval_ms: float
    _last_time: float = field(default=0.0, init=False)
    
    def can_send(self) -> bool:
        now = time.time() * 1000
        if now - self._last_time >= self.interval_ms:
            self._last_time = now
            return True
        return False
    
    def remaining_ms(self) -> float:
        return max(0.0, self.interval_ms - (time.time() * 1000 - self._last_time))


class ExecutionController:
    def __init__(self):
        self.open_limiter  = RateLimiter(100)   # 开仓: 100ms
        self.close_limiter = RateLimiter(50)    # 平仓: 50ms（优先）
    
    def can_open(self) -> bool:
        return self.open_limiter.can_send()
    
    def can_close(self) -> bool:
        return self.close_limiter.can_send()
    
    def force_close_ready(self):
        """紧急平仓：重置平仓限流"""
        self.close_limiter._last_time = 0.0
```

**为何分离**：止损信号不应被开仓限流阻塞。平仓限流独立，止损延迟从 250ms 降至 10ms 以内。

---

## 2. 订单状态机

```python
from enum import Enum

class OrderState(Enum):
    IDLE          = "idle"
    PENDING_OPEN  = "pending_open"
    OPEN          = "open"
    PENDING_CLOSE = "pending_close"
    CLOSED        = "closed"
    ERROR         = "error"


class StrategyStateMachine:
    def __init__(self):
        self.state        = OrderState.IDLE
        self.entry_price  = 0.0
        self.entry_time   = None
        self.position_qty = 0
        self.direction    = 0
    
    def on_order_sent(self, direction: int, qty: int):
        if self.state == OrderState.IDLE:
            self.state = OrderState.PENDING_OPEN
            self.direction = direction
    
    def on_order_filled(self, fill_price: float, qty: int, is_open: bool):
        if is_open and self.state == OrderState.PENDING_OPEN:
            self.state        = OrderState.OPEN
            self.entry_price  = fill_price
            self.entry_time   = time.time()
            self.position_qty = qty
        elif not is_open and self.state == OrderState.PENDING_CLOSE:
            self.state        = OrderState.CLOSED
            self.position_qty = 0
    
    def on_close_signal(self):
        if self.state == OrderState.OPEN:
            self.state = OrderState.PENDING_CLOSE
    
    def reset(self):
        self.state        = OrderState.IDLE
        self.entry_price  = 0.0
        self.entry_time   = None
        self.position_qty = 0
        self.direction    = 0
    
    @property
    def can_open_new(self) -> bool:
        return self.state == OrderState.IDLE
    
    @property
    def hold_seconds(self) -> float:
        if self.entry_time is None:
            return 0.0
        return time.time() - self.entry_time
```

---

## 3. 动态止盈（基于 on_tick，非 sleep）

在 VeighNa 框架中，**不能使用 asyncio.sleep**，改为在 `on_tick` 中轮询：

```python
def _manage_position(self, tick):
    """在 on_tick 中调用，非阻塞持仓管理"""
    if self.position_direction == 0:
        return
    
    current_mp = (
        (tick.ask_volume_1 * tick.bid_price_1 + tick.bid_volume_1 * tick.ask_price_1)
        / (tick.bid_volume_1 + tick.ask_volume_1)
        if (tick.bid_volume_1 + tick.ask_volume_1) > 0
        else tick.last_price
    )
    
    tick_size = get_tick_size(self.entry_price)
    tp_price  = self.entry_price + self.position_direction * self.take_profit_ticks * tick_size
    sl_price  = self.entry_price - self.position_direction * self.stop_loss_ticks  * tick_size
    
    # 止盈
    if self.position_direction > 0 and current_mp >= tp_price:
        self._close_position(tick, "TAKE_PROFIT")
        return
    if self.position_direction < 0 and current_mp <= tp_price:
        self._close_position(tick, "TAKE_PROFIT")
        return
    
    # 止损
    if self.position_direction > 0 and current_mp <= sl_price:
        self._close_position(tick, "STOP_LOSS")
        return
    if self.position_direction < 0 and current_mp >= sl_price:
        self._close_position(tick, "STOP_LOSS")
        return
    
    # 超时平仓
    if self.sm.hold_seconds > self.max_hold_seconds:
        self._close_position(tick, "TIMEOUT")
        return
    
    # 反向强信号平仓
    if abs(self.composite_score) > self.strong_threshold:
        if (self.composite_score > 0) != (self.position_direction > 0):
            self._close_position(tick, "SIGNAL_FLIP")
```

---

## 4. 信号强度自适应确认

```python
class SignalConfirmGate:
    def __init__(self, entry_thr=0.4, strong_thr=0.7):
        self.entry_thr  = entry_thr
        self.strong_thr = strong_thr
        self._pending   = 0
        self._count     = 0
    
    def check(self, score: float) -> tuple:
        """返回 (should_trade, direction)"""
        if abs(score) < self.entry_thr:
            self._reset()
            return False, 0
        
        direction = 1 if score > 0 else -1
        
        # 强信号：直接开仓
        if abs(score) >= self.strong_thr:
            self._reset()
            return True, direction
        
        # 弱信号：需2次确认
        if self._pending == direction:
            self._count += 1
            if self._count >= 2:
                self._reset()
                return True, direction
        else:
            self._pending = direction
            self._count   = 1
        return False, 0
    
    def _reset(self):
        self._pending = 0
        self._count   = 0
```

---

## 5. 手续费模型

```python
@staticmethod
def calc_commission(price: float, qty: int, broker: str = "kabu") -> float:
    """
    往返手续费估算（以官方最新费率为准，此为参考值）
    kabu/SBI: 約定代金に応じた定額プラン or 一律
    IBKR: 約¥0.5/株, 最低¥80/注文
    实盘前务必核查券商当前费率页面
    """
    trade_value = price * qty
    if broker in ("kabu", "sbi"):
        # 参考值：定額プランでは実質0~最大0.385%
        # 以下为保守估算，实际可能更低
        rate = 0.00385
        min_fee = 55
        one_way = max(trade_value * rate, min_fee)
    elif broker == "ibkr":
        one_way = max(qty * 0.5, 80)
        one_way = min(one_way, trade_value * 0.005)
    else:
        one_way = 0
    return one_way * 2  # 往返

def calc_breakeven_ticks(price: float, commission: float) -> float:
    tick_size = get_tick_size(price)
    return commission / tick_size
```

---

## 6. 性能监控

```python
import statistics
from collections import deque

class PerfMonitor:
    def __init__(self, window=200):
        self.signal_to_order = deque(maxlen=window)  # ms
        self.fill_latency    = deque(maxlen=window)  # ms
        self.slippages       = deque(maxlen=window)
    
    def record_signal(self, ms: float): self.signal_to_order.append(ms)
    def record_fill(self, ms: float, slip: float):
        self.fill_latency.append(ms)
        self.slippages.append(slip)
    
    def summary(self) -> str:
        def p(d):
            if not d: return "N/A"
            lst = sorted(d)
            return f"p50={lst[len(lst)//2]:.1f} p95={lst[int(len(lst)*0.95)]:.1f}"
        return (f"SigToOrder: {p(self.signal_to_order)}ms | "
                f"Fill: {p(self.fill_latency)}ms | "
                f"Slip: mean={statistics.mean(self.slippages) if self.slippages else 0:.2f}")
```
