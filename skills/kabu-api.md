# kabu Station API 字段参考 & 适配指南

## ⚠️ 核心字段命名真相（基于官方 push.html）

kabu API 的字段命名规则：**Bid = 売（卖方），Ask = 買（买方）**  
这与 Bloomberg/国际惯例（Bid=买方）完全相反。

| kabu 字段 | 官方日文 | 实际含义 | VeighNa映射 |
|-----------|---------|---------|------------|
| `BidPrice` | 最良**売**気配値段 | 卖一价（Ask） | `ask_price_1` |
| `BidQty` | 最良**売**気配数量 | 卖一量（Ask Size） | `ask_volume_1` |
| `AskPrice` | 最良**買**気配値段 | 买一价（Bid） | `bid_price_1` |
| `AskQty` | 最良**買**気配数量 | 买一量（Bid Size） | `bid_volume_1` |
| `Sell1~10` | 売気配 各档 | asks 盘口 | `ask_price_2~10` |
| `Buy1~10` | 買気配 各档 | bids 盘口 | `bid_price_2~10` |

## WebSocket PUSH 完整字段表

```
端点: ws://localhost:18080/kabusapi/websocket（本番）
      ws://localhost:18081/kabusapi/websocket（検証）
最大注册: 50銘柄
推送时机: 值更新时触发，午休/引け後不推送
```

| 字段 | 类型 | 内容 | 策略用途 |
|------|------|------|---------|
| Symbol | str | 銘柄コード（6桁） | 必须 |
| CurrentPrice | float | 現値（最終成交） | last_price |
| CurrentPriceTime | str | 現値時刻 | 时间戳 |
| CurrentPriceChangeStatus | str | 前值比较（"0056"等） | aggressor 参考 |
| TradingVolume | int | 売買高（出来高） | 成交量 |
| VWAP | float | 売買高加重平均価格 | 参考 |
| **BidPrice** | float | 最良売気配値段 | → **ask（卖一价）** |
| **BidQty** | int | 最良売気配数量 | → **ask_size** |
| BidTime | str | 最良売気配時刻 | 时间参考 |
| BidSign | str | 最良売気配フラグ | 气配状态 |
| MarketOrderSellQty | int | 売成行数量 | 市价卖单参考 |
| **AskPrice** | float | 最良買気配値段 | → **bid（买一价）** |
| **AskQty** | int | 最良買気配数量 | → **bid_size** |
| AskSign | str | 最良買気配フラグ | 气配状态 |
| MarketOrderBuyQty | int | 買成行数量 | 市价买单参考 |
| Sell1~10 | list | 売気配 Price/Qty | → asks L2 |
| Buy1~10 | list | 買気配 Price/Qty | → bids L2 |

## 完整适配器实现

```python
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import time

@dataclass
class BoardSnapshot:
    symbol: str
    ts: str
    bid: float          # 买一价（来自AskPrice）
    ask: float          # 卖一价（来自BidPrice）
    bid_size: int       # 买一量（来自AskQty）
    ask_size: int       # 卖一量（来自BidQty）
    last: float
    volume: int
    bids: List[Tuple[float, int]]  # [(price, qty), ...] 降序
    asks: List[Tuple[float, int]]  # [(price, qty), ...] 升序
    bid_market_qty: int = 0   # 買成行数量
    ask_market_qty: int = 0   # 売成行数量
    
    @property
    def spread(self) -> float:
        return self.ask - self.bid
    
    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2
    
    @property
    def microprice(self) -> float:
        total = self.bid_size + self.ask_size
        if total == 0:
            return self.mid
        return (self.ask_size * self.bid + self.bid_size * self.ask) / total
    
    def is_valid(self) -> bool:
        return (
            self.bid > 0 and self.ask > 0 and
            self.bid < self.ask and
            self.spread < self.bid * 0.05  # 价差不超过5%
        )


class KabuBoardAdapter:
    """
    kabu Station PUSH 数据适配器
    职责：原始字段 → 标准 BoardSnapshot
    规则：bid=AskPrice, ask=BidPrice（官方定义）
    """
    
    @staticmethod
    def parse(raw: dict) -> Optional[BoardSnapshot]:
        try:
            bid = float(raw.get('AskPrice') or 0)    # 最良買気配 = bid
            ask = float(raw.get('BidPrice') or 0)    # 最良売気配 = ask
            bid_size = int(raw.get('AskQty') or 0)
            ask_size = int(raw.get('BidQty') or 0)
            
            # 解析 L2 bids（Buy1~10 是买盘）
            bids = []
            for i in range(1, 11):
                entry = raw.get(f'Buy{i}', {})
                p = entry.get('Price', 0) if entry else 0
                q = entry.get('Qty', 0) if entry else 0
                if p > 0:
                    bids.append((float(p), int(q)))
            bids.sort(key=lambda x: -x[0])
            
            # 解析 L2 asks（Sell1~10 是卖盘）
            asks = []
            for i in range(1, 11):
                entry = raw.get(f'Sell{i}', {})
                p = entry.get('Price', 0) if entry else 0
                q = entry.get('Qty', 0) if entry else 0
                if p > 0:
                    asks.append((float(p), int(q)))
            asks.sort(key=lambda x: x[0])
            
            snap = BoardSnapshot(
                symbol=raw.get('Symbol', ''),
                ts=raw.get('CurrentPriceTime', ''),
                bid=bid, ask=ask,
                bid_size=bid_size, ask_size=ask_size,
                last=float(raw.get('CurrentPrice') or 0),
                volume=int(raw.get('TradingVolume') or 0),
                bids=bids, asks=asks,
                bid_market_qty=int(raw.get('MarketOrderBuyQty') or 0),
                ask_market_qty=int(raw.get('MarketOrderSellQty') or 0),
            )
            
            return snap if snap.is_valid() else None
            
        except (TypeError, ValueError) as e:
            return None
    
    @staticmethod
    def to_vnpy_tick(snap: BoardSnapshot, exchange, gateway_name: str):
        """转换为 VeighNa TickData"""
        from vnpy.trader.object import TickData
        from datetime import datetime
        
        tick = TickData(
            symbol=snap.symbol,
            exchange=exchange,
            datetime=datetime.now(),  # 建议解析 snap.ts
            gateway_name=gateway_name,
            last_price=snap.last,
            volume=snap.volume,
            bid_price_1=snap.bid,
            bid_volume_1=snap.bid_size,
            ask_price_1=snap.ask,
            ask_volume_1=snap.ask_size,
        )
        # 填充 L2（最多5档）
        for i, (p, q) in enumerate(snap.bids[:4], 2):
            setattr(tick, f'bid_price_{i}', p)
            setattr(tick, f'bid_volume_{i}', q)
        for i, (p, q) in enumerate(snap.asks[:4], 2):
            setattr(tick, f'ask_price_{i}', p)
            setattr(tick, f'ask_volume_{i}', q)
        return tick
```

## REST API 下单

```python
import requests

KABU_BASE = "http://localhost:18080/kabusapi"

class KabuOrderClient:
    def __init__(self, api_token: str):
        self.headers = {
            'Content-Type': 'application/json',
            'X-API-KEY': api_token,
        }
    
    def send_order(self, symbol: str, side: str, qty: int,
                   price: float = 0, order_type: str = "2") -> dict:
        """
        side: "2"=買（买入）, "1"=売（卖出）
        order_type: "2"=指値（限价）, "5"=成行（市价）
        fund_type: "11"=現物買, "13"=現物売
        """
        fund_type = "11" if side == "2" else "13"
        body = {
            "Password": "",
            "Symbol": symbol,
            "Exchange": 1,          # 1=東証
            "SecurityType": 1,      # 1=株式
            "Side": side,
            "CashMargin": 1,        # 1=現物
            "MarginTradeType": 0,
            "DelivType": 2,
            "FundType": fund_type,
            "AccountType": 4,       # 4=特定口座
            "Qty": qty,
            "Price": price,
            "ExpireDay": 0,         # 0=当日限り
            "OrderType": order_type,
        }
        resp = requests.post(
            f"{KABU_BASE}/sendorder",
            headers=self.headers,
            json=body,
            timeout=1.0,
        )
        return resp.json()
    
    def cancel_order(self, order_id: str) -> dict:
        resp = requests.put(
            f"{KABU_BASE}/cancelorder",
            headers=self.headers,
            json={"OrderId": order_id, "Password": ""},
            timeout=0.5,
        )
        return resp.json()

def get_token(password: str) -> str:
    resp = requests.post(
        f"{KABU_BASE}/token",
        json={"APIPassword": password}
    )
    return resp.json()['Token']
```

## 常见坑点

1. **WebSocket 每日重连**：kabu Station 每天重启，需自动重连逻辑
2. **Token 每次重获**：策略启动时必须重新获取 token
3. **午休不推送**：12:00~12:30 无 PUSH 数据
4. **成交确认延迟**：需轮询 `/positions` 确认，不能只看 sendorder 返回
5. **L2 数据权限**：多档气配需要 SBI/kabu 特定契約等级
6. **BidSign/AskSign**：气配フラグ "0101"=通常, "0107"=特別気配，注意排除特殊气配
