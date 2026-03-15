# 风险管理文档

## 1. PnL 计算（含方向、手续费、部分成交）

```python
def calc_trade_pnl(
    entry_price: float, exit_price: float,
    qty: int, direction: int,  # +1=多, -1=空
    commission: float = 0.0,
) -> float:
    """净盈亏 = 方向 × (出价-入价) × 数量 - 手续费"""
    gross = direction * (exit_price - entry_price) * qty
    return gross - commission
```

**⚠️ 常见 Bug**：`pnl += price_diff`（忘记方向乘数和手续费）

---

## 2. PnL 追踪器

```python
from dataclasses import dataclass, field
from typing import List
import statistics

@dataclass
class TradeRecord:
    entry_price: float
    exit_price: float
    qty: int
    direction: int
    commission: float
    entry_time: float
    exit_time: float
    exit_reason: str  # "TP"/"SL"/"TIMEOUT"/"SIGNAL_FLIP"
    
    @property
    def pnl(self): return calc_trade_pnl(
        self.entry_price, self.exit_price,
        self.qty, self.direction, self.commission)
    @property
    def hold_sec(self): return self.exit_time - self.entry_time
    @property
    def is_win(self): return self.pnl > 0


class PnLTracker:
    def __init__(self):
        self.trades: List[TradeRecord] = []
        self.cumulative = 0.0
        self.peak = 0.0
        self.max_drawdown = 0.0
    
    def record(self, trade: TradeRecord):
        self.trades.append(trade)
        self.cumulative += trade.pnl
        self.peak = max(self.peak, self.cumulative)
        self.max_drawdown = max(self.max_drawdown, self.peak - self.cumulative)
    
    def stats(self) -> dict:
        if not self.trades: return {}
        wins   = [t.pnl for t in self.trades if t.is_win]
        losses = [t.pnl for t in self.trades if not t.is_win]
        n = len(self.trades)
        avg_w  = sum(wins) / len(wins) if wins else 0
        avg_l  = sum(losses) / len(losses) if losses else 0
        all_pnl = [t.pnl for t in self.trades]
        mean_pnl = sum(all_pnl) / n
        std_pnl  = statistics.stdev(all_pnl) if n > 1 else 1
        return {
            'n': n,
            'win_rate': len(wins) / n,
            'avg_win': avg_w,
            'avg_loss': avg_l,
            'profit_factor': abs(avg_w / avg_l) if avg_l else 999,
            'cum_pnl': self.cumulative,
            'max_dd': self.max_drawdown,
            'expectancy': mean_pnl,
            'sharpe_approx': mean_pnl / std_pnl * (252**0.5) if std_pnl else 0,
            'avg_hold_sec': sum(t.hold_sec for t in self.trades) / n,
            'by_reason': {r: sum(1 for t in self.trades if t.exit_reason == r)
                         for r in ['TP','SL','TIMEOUT','SIGNAL_FLIP']},
        }
```

---

## 3. 连亏冷却（冷却后需重新确认信号）

```python
class ConsecutiveLossGuard:
    def __init__(self, max_losses=3, cooldown_sec=120.0):
        self.max_losses    = max_losses
        self.cooldown_sec  = cooldown_sec
        self._consecutive  = 0
        self._in_cooldown  = False
        self._until        = 0.0
        self._need_confirm = False  # 冷却后需确认
    
    def record_result(self, pnl: float):
        if pnl < 0:
            self._consecutive += 1
            if self._consecutive >= self.max_losses:
                self._in_cooldown  = True
                self._until        = time.time() + self.cooldown_sec
                self._need_confirm = True
        else:
            self._consecutive = 0
    
    def can_open(self, signal_confirmed: bool = False) -> bool:
        if not self._in_cooldown:
            return True
        if time.time() >= self._until:
            self._in_cooldown = False
            self._consecutive = 0
            # 冷却结束但需要信号重新确认
            if not signal_confirmed:
                return False
            self._need_confirm = False
            return True
        return False
```

---

## 4. 仓位控制

```python
def calc_qty(
    price: float,
    stop_loss_ticks: int,
    account_equity: float,
    risk_per_trade: float = 0.01,
    max_position_value: float = 1_000_000,
    lot_size: int = 100,
) -> int:
    tick_size = get_tick_size(price)
    max_loss_per_share = stop_loss_ticks * tick_size
    max_loss_total     = account_equity * risk_per_trade
    
    risk_qty  = int(max_loss_total / max_loss_per_share) if max_loss_per_share > 0 else 0
    value_qty = int(max_position_value / price)
    
    qty = min(risk_qty, value_qty)
    qty = (qty // lot_size) * lot_size  # 圆整到最小手数
    return max(qty, 0)
```

---

## 5. 三级风控状态机

```python
class RiskLevel:
    NORMAL   = 0   # 正常运行
    WARNING  = 1   # 减半仓位
    HALT     = 2   # 停止开仓（只平仓）
    EMERGENCY= 3   # 全部平仓

class RiskStateMachine:
    THRESHOLDS = {
        1: {'loss_pct': 0.005, 'consec': 3},
        2: {'loss_pct': 0.010, 'consec': 5},
        3: {'loss_pct': 0.020, 'consec': 8},
    }
    
    def __init__(self, equity: float):
        self.equity     = equity
        self.level      = 0
        self.daily_loss = 0.0
        self.consec     = 0
    
    def update(self, pnl: float):
        if pnl < 0:
            self.daily_loss += pnl
            self.consec += 1
        else:
            self.consec = 0
        
        loss_pct = abs(self.daily_loss) / self.equity
        for lvl in [3, 2, 1]:
            t = self.THRESHOLDS[lvl]
            if loss_pct >= t['loss_pct'] or self.consec >= t['consec']:
                if lvl > self.level:
                    self.level = lvl
                break
    
    @property
    def can_open(self): return self.level < RiskLevel.HALT
    @property
    def must_flatten(self): return self.level >= RiskLevel.EMERGENCY
    @property
    def size_scale(self):
        return {0: 1.0, 1: 0.5, 2: 0.0, 3: 0.0}.get(self.level, 0.0)
```

---

## 6. 日内风控限制

```python
@dataclass
class DailyRiskLimits:
    max_daily_loss:   float = 50_000      # ¥5万
    max_daily_trades: int   = 50
    
    _daily_pnl:    float = 0.0
    _daily_trades: int   = 0
    halted:        bool  = False
    halt_reason:   str   = ""
    
    def check(self, pnl: float) -> bool:
        """返回 False = 触发风控"""
        self._daily_pnl    += pnl
        self._daily_trades += 1
        
        if self._daily_pnl < -self.max_daily_loss:
            self.halted, self.halt_reason = True, f"日亏损超限: ¥{self._daily_pnl:.0f}"
        elif self._daily_trades >= self.max_daily_trades:
            self.halted, self.halt_reason = True, f"日交易次数超限: {self._daily_trades}"
        
        return not self.halted
    
    def reset(self):
        self._daily_pnl = self._daily_trades = 0
        self.halted = False; self.halt_reason = ""
```

---

## 7. 综合风控 Guard

```python
class RiskGuard:
    """策略层调用的统一风控入口"""
    
    def __init__(self, equity=5_000_000):
        self.rsm     = RiskStateMachine(equity)
        self.daily   = DailyRiskLimits()
        self.loss_gd = ConsecutiveLossGuard()
        self.tracker = PnLTracker()
    
    def can_open(self, signal_confirmed=False) -> bool:
        return (
            self.rsm.can_open and
            not self.daily.halted and
            self.loss_gd.can_open(signal_confirmed)
        )
    
    def record_close(self, entry, exit_p, direction, qty,
                     exit_reason="", broker="kabu") -> float:
        commission = calc_commission(entry, qty, broker)
        pnl = calc_trade_pnl(entry, exit_p, qty, direction, commission)
        
        record = TradeRecord(
            entry_price=entry, exit_price=exit_p, qty=qty,
            direction=direction, commission=commission,
            entry_time=time.time()-5, exit_time=time.time(),
            exit_reason=exit_reason
        )
        self.tracker.record(record)
        self.rsm.update(pnl)
        self.daily.check(pnl)
        self.loss_gd.record_result(pnl)
        return pnl
```

---

## 8. 日志规范

```python
import logging

class HFTLogger:
    def __init__(self, symbol: str):
        self.log = logging.getLogger(f"HFT.{symbol}")
    
    def signal(self, score, direction, confirmed):
        self.log.info(f"[SIGNAL] score={score:.3f} dir={'+' if direction>0 else '-'} confirmed={confirmed}")
    
    def order(self, side, price, qty, oid):
        self.log.info(f"[ORDER] {side} {price:.1f}×{qty} id={oid}")
    
    def fill(self, price, qty, latency_ms):
        self.log.info(f"[FILL] {price:.1f}×{qty} lat={latency_ms:.1f}ms")
    
    def close(self, reason, pnl, hold_sec):
        lvl = logging.INFO if pnl >= 0 else logging.WARNING
        self.log.log(lvl, f"[CLOSE] reason={reason} pnl={pnl:+.0f} hold={hold_sec:.1f}s")
    
    def risk(self, event, detail):
        self.log.warning(f"[RISK] {event}: {detail}")
```
