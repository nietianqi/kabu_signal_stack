# 信号体系详细文档

## 目录
1. Microprice & Mid
2. 加权盘口不平衡 OBI
3. LOB-OFI（增量订单流不平衡）
4. Tape-OFI（逐笔主动方向）
5. Micro-momentum
6. Microprice Tilt
7. Flow Flip 检测
8. 信号归一化与融合
9. KabuSignalStack 完整实现

---

## 1. Microprice & Mid

```python
mid = (bid + ask) / 2

# 按对手盘深度加权 → 比mid更准确反映短期价格压力
microprice = (ask_size * bid + bid_size * ask) / (bid_size + ask_size)
```

**直觉**：bid_size >> ask_size → microprice > mid → 买压强，短期倾向上行

---

## 2. 加权盘口不平衡 OBI

```python
def calc_weighted_obi(bids: list, asks: list, levels: int = 5) -> float:
    """
    bids/asks: [(price, size), ...] 已排序
    返回: [-1, +1], 正值=买压, 负值=卖压
    """
    weights = [1 / (1 + i) for i in range(levels)]
    bid_vol = sum(w * s for w, (_, s) in zip(weights, bids[:levels]))
    ask_vol = sum(w * s for w, (_, s) in zip(weights, asks[:levels]))
    total = bid_vol + ask_vol
    return 0.0 if total == 0 else (bid_vol - ask_vol) / total
```

参数建议：levels=3~5（日股流动性集中前3档）

---

## 3. LOB-OFI（Cont, Kukanov, Stoikov 2014）

追踪最优档增量变化，对短期价格预测力最强。

```python
def calc_lob_ofi_incremental(
    prev_bid, prev_bid_size, prev_ask, prev_ask_size,
    curr_bid, curr_bid_size, curr_ask, curr_ask_size,
) -> float:
    # Bid side
    if curr_bid > prev_bid:
        bid_ofi = curr_bid_size
    elif curr_bid == prev_bid:
        bid_ofi = curr_bid_size - prev_bid_size
    else:
        bid_ofi = -prev_bid_size
    
    # Ask side
    if curr_ask < prev_ask:
        ask_ofi = -curr_ask_size
    elif curr_ask == prev_ask:
        ask_ofi = -(curr_ask_size - prev_ask_size)
    else:
        ask_ofi = prev_ask_size
    
    return bid_ofi + ask_ofi


from collections import deque

class LOBOFIAccumulator:
    def __init__(self, window: int = 20):
        self.buffer = deque(maxlen=window)
        self.prev = None
    
    def update(self, bid, bid_size, ask, ask_size) -> float:
        if self.prev:
            delta = calc_lob_ofi_incremental(*self.prev, bid, bid_size, ask, ask_size)
            self.buffer.append(delta)
        self.prev = (bid, bid_size, ask, ask_size)
        return sum(self.buffer)
```

---

## 4. Tape-OFI（逐笔主动方向）

```python
def classify_tape_aggressor(
    trade_price: float, bid: float, ask: float,
    prev_trade_price: float = None
) -> int:
    """
    Quote Rule（优先）+ Tick Rule（价差内fallback）
    返回: +1=主动买, -1=主动卖, 0=不确定
    """
    if trade_price >= ask:
        return +1
    if trade_price <= bid:
        return -1
    # 价格在价差内：用 tick rule
    if prev_trade_price is not None:
        if trade_price > prev_trade_price:
            return +1
        if trade_price < prev_trade_price:
            return -1
    return 0


def calc_tape_ofi(aggressors: list, window: int = 20) -> float:
    """
    aggressors: [(aggressor, volume), ...]
    返回: (买量 - 卖量) / 总量
    """
    buy_vol = sum(v for a, v in aggressors[-window:] if a == +1)
    sell_vol = sum(v for a, v in aggressors[-window:] if a == -1)
    total = buy_vol + sell_vol
    return 0.0 if total == 0 else (buy_vol - sell_vol) / total
```

**kabu注意**：`CurrentPriceChangeStatus` 可辅助判断方向（"0056"=上昇等），但需实盘验证code含义

---

## 5. Micro-momentum

```python
class MicroMomentum:
    def __init__(self, window: int = 10):
        self.prices = deque(maxlen=window)
    
    def update(self, microprice: float) -> float:
        self.prices.append(microprice)
        if len(self.prices) < 3:
            return 0.0
        mean = sum(self.prices) / len(self.prices)
        return (self.prices[-1] - mean) / mean if mean != 0 else 0.0
```

注意：成交稀疏时（午休前后）降权，窗口不超过20 ticks

---

## 6. Microprice Tilt

```python
def calc_microprice_tilt(bid, ask, bid_size, ask_size) -> float:
    """
    标准化盘口压力方向
    返回: [-1, +1]，与OBI高相关但对spread变化更敏感
    """
    mid = (bid + ask) / 2
    total = bid_size + ask_size
    if total == 0:
        return 0.0
    microprice = (ask_size * bid + bid_size * ask) / total
    half_spread = (ask - bid) / 2
    return 0.0 if half_spread == 0 else (microprice - mid) / half_spread
```

---

## 7. Flow Flip 检测

```python
class FlowFlipDetector:
    def __init__(self, flip_threshold: int = 3):
        self.threshold = flip_threshold
        self.consecutive = 0
        self.last_dir = 0
    
    def update(self, aggressor: int) -> bool:
        """返回 True = 检测到 flow flip（可能趋势反转）"""
        if aggressor == 0:
            return False
        if aggressor == self.last_dir:
            self.consecutive += 1
            return False
        else:
            is_flip = self.consecutive >= self.threshold
            self.consecutive = 1
            self.last_dir = aggressor
            return is_flip
```

---

## 8. 信号归一化（Z-score）

```python
class ZNormalizer:
    def __init__(self, lookback: int = 100):
        self.buffer = deque(maxlen=lookback)
    
    def normalize(self, value: float) -> float:
        self.buffer.append(value)
        if len(self.buffer) < 10:
            return 0.0
        mean = sum(self.buffer) / len(self.buffer)
        std = (sum((x - mean)**2 for x in self.buffer) / len(self.buffer)) ** 0.5
        return 0.0 if std < 1e-10 else (value - mean) / std
```

---

## 9. KabuSignalStack 完整实现

```python
class KabuSignalStack:
    """
    五信号融合栈
    输入: VeighNa TickData（已经过 KabuBoardAdapter）
    输出: composite_score [-∞, +∞]，正=多头偏向，负=空头偏向
    """
    
    WEIGHTS = {
        'lob_ofi':  0.30,
        'obi':      0.25,
        'tape_ofi': 0.25,
        'momentum': 0.10,
        'tilt':     0.10,
    }
    
    def __init__(self, ofi_window=20, tape_window=20, momentum_window=10):
        self.lob_acc = LOBOFIAccumulator(ofi_window)
        self.momentum = MicroMomentum(momentum_window)
        self.flip = FlowFlipDetector()
        self.normalizers = {k: ZNormalizer() for k in self.WEIGHTS}
        self.tape_buffer = deque(maxlen=tape_window)
        self.prev_last = None
    
    def update(self, tick) -> float:
        bid = tick.bid_price_1
        ask = tick.ask_price_1
        bid_size = tick.bid_volume_1
        ask_size = tick.ask_volume_1
        last = tick.last_price
        
        # 1. LOB-OFI
        lob = self.lob_acc.update(bid, bid_size, ask, ask_size)
        
        # 2. OBI（需要 L2，fallback 到 L1）
        bids_l2 = [(getattr(tick, f'bid_price_{i}', 0),
                    getattr(tick, f'bid_volume_{i}', 0)) for i in range(1, 6)]
        asks_l2 = [(getattr(tick, f'ask_price_{i}', 0),
                    getattr(tick, f'ask_volume_{i}', 0)) for i in range(1, 6)]
        obi = calc_weighted_obi(
            [(p, s) for p, s in bids_l2 if p > 0],
            [(p, s) for p, s in asks_l2 if p > 0],
        )
        
        # 3. Tape-OFI
        mp = (ask_size * bid + bid_size * ask) / (bid_size + ask_size) if (bid_size + ask_size) > 0 else (bid + ask) / 2
        aggressor = classify_tape_aggressor(last, bid, ask, self.prev_last)
        self.tape_buffer.append((aggressor, 1))
        tape = calc_tape_ofi(list(self.tape_buffer))
        self.prev_last = last
        
        # 4. Micro-momentum
        mom = self.momentum.update(mp)
        
        # 5. Microprice Tilt
        tilt = calc_microprice_tilt(bid, ask, bid_size, ask_size)
        
        # 归一化 + 加权融合
        scores = {
            'lob_ofi': self.normalizers['lob_ofi'].normalize(lob),
            'obi': self.normalizers['obi'].normalize(obi),
            'tape_ofi': self.normalizers['tape_ofi'].normalize(tape),
            'momentum': self.normalizers['momentum'].normalize(mom),
            'tilt': self.normalizers['tilt'].normalize(tilt),
        }
        
        composite = sum(self.WEIGHTS[k] * v for k, v in scores.items())
        return composite
```
