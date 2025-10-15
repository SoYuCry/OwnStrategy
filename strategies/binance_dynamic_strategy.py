from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from api.bn_client import BinanceClient
from logger import setup_logger
from utils.helpers import round_to_precision


@dataclass
class StrategyParameters:
    initial_allocation_pct: float = 0.05
    take_profit_activation_pct: float = 0.03
    trailing_stop_gap_pct: float = 0.005
    dip_threshold_pct: float = 0.01
    max_additions: int = 3
    poll_interval: float = 2.0
    max_loss_pct: float = 0.30


class BinanceDynamicStrategy:
    """动态仓位管理策略：首仓、追踪止盈、补仓。"""

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        symbol: str,
        strategy_config: Optional[Dict[str, Any]] = None,
        exchange_config: Optional[Dict[str, Any]] = None,
        direction: str = "LONG",
    ) -> None:
        if not api_key or not secret_key:
            raise ValueError("API key 和 Secret key 均需提供")

        config = strategy_config or {}
        params_dict = asdict(StrategyParameters())
        overrides = {k: v for k, v in config.items() if v is not None and k in params_dict}
        params_dict.update(overrides)
        self.params = StrategyParameters(**params_dict)

        exchange_cfg = dict(exchange_config or {})
        exchange_cfg.setdefault("api_key", api_key)
        exchange_cfg.setdefault("secret_key", secret_key)
        self.client = BinanceClient(exchange_cfg)

        self.symbol = symbol.upper()
        self.direction = direction.upper()
        if self.direction not in {"LONG", "SHORT"}:
            raise ValueError("direction 必须是 LONG 或 SHORT")

        self.logger = setup_logger(f"bn_strategy_{self.symbol.lower()}")

        self.market_limits = self.client.get_market_limits(self.symbol)
        if not self.market_limits or isinstance(self.market_limits, dict) and self.market_limits.get("error"):
            raise ValueError(f"无法获取 {self.symbol} 的市场限制: {self.market_limits}")

        self.base_precision = int(float(self.market_limits.get("base_precision", 4)))
        self.quote_precision = int(float(self.market_limits.get("quote_precision", 4)))
        self.tick_size = float(self.market_limits.get("tick_size", 0.0) or 0.0)
        self.min_order_size = float(self.market_limits.get("min_order_size", 0.0) or 0.0)
        if self.min_order_size <= 0:
            self.min_order_size = 10 ** (-self.base_precision)
        self.quote_asset = (self.market_limits.get("quote_asset") or "USDT").upper()

        self.position_qty: float = 0.0
        self.entry_price: Optional[float] = None
        self.trailing_extreme: Optional[float] = None
        self.trailing_active = False
        self.last_add_price: Optional[float] = None
        self.last_allocation_notional: float = 0.0
        self.additions_done = 0
        self._stop = False

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def _get_last_price(self) -> Optional[float]:
        ticker = self.client.get_ticker(self.symbol)
        if isinstance(ticker, dict) and ticker.get("error"):
            self.logger.error("获取行情失败: %s", ticker.get("error"))
            return None

        price_fields = ("lastPrice", "markPrice", "price", "close")
        for field in price_fields:
            value = ticker.get(field)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        self.logger.error("行情返回数据缺少价格字段: %s", ticker)
        return None

    def _fetch_account_equity(self) -> Optional[float]:
        balances = self.client.get_balance()
        if isinstance(balances, dict) and balances.get("error"):
            self.logger.error("获取余额失败: %s", balances.get("error"))
            return None

        total_equity = 0.0
        for data in balances.values():
            try:
                total_equity += float(data.get("total", 0.0))
            except (TypeError, ValueError):
                continue
        return total_equity

    def _fetch_available_quote(self) -> float:
        balances = self.client.get_balance()
        if isinstance(balances, dict) and balances.get("error"):
            self.logger.error("获取余额失败: %s", balances.get("error"))
            return 0.0

        asset_info = balances.get(self.quote_asset)
        if not asset_info:
            available = 0.0
            for data in balances.values():
                try:
                    available += float(data.get("available", 0.0))
                except (TypeError, ValueError):
                    continue
            return available

        try:
            return float(asset_info.get("available", 0.0))
        except (TypeError, ValueError):
            return 0.0

    def _refresh_position_state(self) -> None:
        positions = self.client.get_positions(self.symbol)
        if isinstance(positions, dict) and positions.get("error"):
            self.logger.error("获取仓位失败: %s", positions.get("error"))
            return

        net_qty = 0.0
        entry_price = None
        for pos in positions:
            try:
                qty = float(pos.get("netQuantity", "0"))
            except (TypeError, ValueError):
                continue

            if abs(qty) < 1e-12:
                continue

            if self.direction == "LONG":
                if qty <= 0:
                    continue
                net_qty += qty
            else:
                if qty >= 0:
                    continue
                net_qty += abs(qty)

            raw_entry = pos.get("entryPrice") or pos.get("markPrice")
            try:
                entry_price = float(raw_entry)
            except (TypeError, ValueError):
                entry_price = entry_price or None

        self.position_qty = net_qty
        self.entry_price = entry_price
        if self.position_qty <= 0:
            self.trailing_active = False
            self.trailing_extreme = None

    def _round_quantity(self, quantity: float) -> float:
        rounded = round_to_precision(quantity, self.base_precision)
        if rounded < self.min_order_size:
            rounded = round_to_precision(self.min_order_size, self.base_precision)
        return rounded

    def _current_pnl_ratio(self, price: float) -> Optional[float]:
        if not self.entry_price or self.entry_price <= 0:
            return None
        if self.direction == "LONG":
            return (price - self.entry_price) / self.entry_price
        return (self.entry_price - price) / self.entry_price

    # ------------------------------------------------------------------
    # Trade actions
    # ------------------------------------------------------------------

    def _open_initial_position(self, price: float) -> bool:
        total_equity = self._fetch_account_equity()
        if total_equity is None or total_equity <= 0:
            self.logger.error("账户权益为零，无法建立首仓")
            return False

        allocation = max(total_equity * self.params.initial_allocation_pct, 0.0)
        if allocation <= 0:
            self.logger.error("无有效首仓资金配置")
            return False

        qty = self._round_quantity(allocation / price)
        if qty <= 0:
            self.logger.error("计算得到的首仓数量无效: %s", qty)
            return False

        order = self.client.execute_order(
            {
                "symbol": self.symbol,
                "side": "BUY" if self.direction == "LONG" else "SELL",
                "type": "MARKET",
                "quantity": qty,
            }
        )
        if isinstance(order, dict) and order.get("error"):
            self.logger.error("首仓下单失败: %s", order.get("error"))
            return False

        self.logger.info("首仓下单成功，数量 %s，预期建仓成本 %.4f", qty, price)
        self.last_allocation_notional = allocation
        time.sleep(1.0)
        self._refresh_position_state()
        self.last_add_price = self.entry_price or price
        return True

    def _close_position(self) -> None:
        if self.position_qty <= 0:
            return

        qty = self._round_quantity(self.position_qty)
        order = self.client.execute_order(
            {
                "symbol": self.symbol,
                "side": "SELL" if self.direction == "LONG" else "BUY",
                "type": "MARKET",
                "quantity": qty,
                "reduceOnly": True,
            }
        )
        if isinstance(order, dict) and order.get("error"):
            self.logger.error("平仓失败: %s", order.get("error"))
            return

        self.logger.info("仓位已平仓，数量 %s", qty)
        time.sleep(1.0)
        self._refresh_position_state()
        self.last_add_price = None
        self.last_allocation_notional = 0.0
        self.additions_done = 0

    def _maybe_activate_trailing(self, price: float) -> None:
        if self.trailing_active or not self.entry_price:
            return

        if self.direction == "LONG":
            activation_price = self.entry_price * (1 + self.params.take_profit_activation_pct)
            if price >= activation_price:
                self.trailing_active = True
                self.trailing_extreme = price
                self.logger.info(
                    "追踪止盈已激活，触发价格 %.4f (入场价 %.4f)",
                    activation_price,
                    self.entry_price,
                )
        else:
            activation_price = self.entry_price * (1 - self.params.take_profit_activation_pct)
            if price <= activation_price:
                self.trailing_active = True
                self.trailing_extreme = price
                self.logger.info(
                    "追踪止盈已激活，触发价格 %.4f (入场价 %.4f)",
                    activation_price,
                    self.entry_price,
                )

    def _maybe_trigger_trailing(self, price: float) -> None:
        if not self.trailing_active or self.trailing_extreme is None:
            return

        if self.direction == "LONG":
            if price > self.trailing_extreme:
                self.trailing_extreme = price
                return

            threshold_price = self.trailing_extreme * (1 - self.params.trailing_stop_gap_pct)
            if price <= threshold_price:
                self.logger.info(
                    "价格从最高 %.4f 回落至 %.4f (回撤 %.2f%%)，执行止盈",
                    self.trailing_extreme,
                    price,
                    self.params.trailing_stop_gap_pct * 100,
                )
                self._close_position()
                self._stop = True
        else:
            if price < self.trailing_extreme:
                self.trailing_extreme = price
                return

            threshold_price = self.trailing_extreme * (1 + self.params.trailing_stop_gap_pct)
            if price >= threshold_price:
                self.logger.info(
                    "价格从最低 %.4f 反弹至 %.4f (回撤 %.2f%%)，执行止盈",
                    self.trailing_extreme,
                    price,
                    self.params.trailing_stop_gap_pct * 100,
                )
                self._close_position()
                self._stop = True

    def _maybe_add_position(self, price: float) -> None:
        if self.additions_done >= self.params.max_additions:
            return
        if not self.last_add_price:
            return

        if self.direction == "LONG":
            trigger_price = self.last_add_price * (1 - self.params.dip_threshold_pct)
            should_add = price <= trigger_price
        else:
            trigger_price = self.last_add_price * (1 + self.params.dip_threshold_pct)
            should_add = price >= trigger_price

        if not should_add:
            return

        desired_notional = self.last_allocation_notional * 2 if self.last_allocation_notional else 0.0
        if desired_notional <= 0:
            return

        available_quote = self._fetch_available_quote()
        if available_quote <= 0:
            self.logger.warning("可用资金不足，无法补仓")
            return

        allocation = min(desired_notional, available_quote)
        qty = self._round_quantity(allocation / price)
        if qty <= 0:
            self.logger.warning("补仓数量不足最小要求，跳过")
            return

        order_side = "BUY" if self.direction == "LONG" else "SELL"
        order = self.client.execute_order(
            {
                "symbol": self.symbol,
                "side": order_side,
                "type": "MARKET",
                "quantity": qty,
            }
        )
        if isinstance(order, dict) and order.get("error"):
            self.logger.error("补仓失败: %s", order.get("error"))
            return

        self.logger.info(
            "补仓成功，方向 %s，数量 %s，触发价格 %.4f，补仓资金 %.4f",
            self.direction,
            qty,
            price,
            allocation,
        )
        self.additions_done += 1
        self.last_allocation_notional = allocation
        time.sleep(1.0)
        self._refresh_position_state()
        self.last_add_price = price

    def _check_max_loss(self, price: float) -> None:
        pnl_ratio = self._current_pnl_ratio(price)
        if pnl_ratio is None:
            return

        if pnl_ratio <= -self.params.max_loss_pct:
            self.logger.warning(
                "触发最大亏损风控 %.2f%% (当前收益 %.2f%%)，立即平仓",
                self.params.max_loss_pct * 100,
                pnl_ratio * 100,
            )
            self._close_position()
            self._stop = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        self.logger.info("启动 Binance 动态策略: %s (%s)", self.symbol, self.direction)
        initial_price = self._get_last_price()
        if initial_price is None:
            self.logger.error("无法获取首仓参考价格，策略停止")
            return

        if not self._open_initial_position(initial_price):
            self.logger.error("首仓建立失败，策略停止")
            return

        while not self._stop:
            time.sleep(self.params.poll_interval)
            price = self._get_last_price()
            if price is None:
                continue

            self._refresh_position_state()
            if self.position_qty <= 0:
                self.logger.info("当前无持仓，策略完成")
                break

            self._maybe_activate_trailing(price)
            self._maybe_trigger_trailing(price)
            if self._stop:
                break
            self._check_max_loss(price)
            if self._stop:
                break
            self._maybe_add_position(price)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Binance 动态仓位策略脚本")
    parser.add_argument("--symbol", required=True, help="交易标的 (例如: BTCUSDT)")
    parser.add_argument("--direction", choices=["long", "short"], default="long", help="开仓方向 (默认 long)")
    parser.add_argument("--initial-allocation", type=float, default=0.05, help="首仓资金占比 (默认 0.05)")
    parser.add_argument("--activation-profit", type=float, default=0.03, help="追踪止盈启动利润阈值 (默认 0.03)")
    parser.add_argument("--trailing-gap", type=float, default=0.005, help="追踪止盈回撤幅度 (默认 0.005)")
    parser.add_argument("--dip-threshold", type=float, default=0.01, help="补仓触发跌幅阈值 (默认 0.01)")
    parser.add_argument("--max-additions", type=int, default=3, help="最大补仓次数 (默认 3)")
    parser.add_argument("--poll-interval", type=float, default=2.0, help="行情轮询间隔秒 (默认 2.0)")
    parser.add_argument("--max-loss", type=float, default=0.30, help="最大亏损比例 (默认 0.30)")
    parser.add_argument("--api-key", type=str, help="可选，覆盖 .env 中的 API Key")
    parser.add_argument("--secret-key", type=str, help="可选，覆盖 .env 中的 Secret Key")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    strategy_config = {
        "initial_allocation_pct": args.initial_allocation,
        "take_profit_activation_pct": args.activation_profit,
        "trailing_stop_gap_pct": args.trailing_gap,
        "dip_threshold_pct": args.dip_threshold,
        "max_additions": args.max_additions,
        "poll_interval": args.poll_interval,
        "max_loss_pct": args.max_loss,
    }

    api_key = args.api_key or os.getenv("BINANCE_API_KEY") or os.getenv("API_KEY")
    secret_key = args.secret_key or os.getenv("BINANCE_SECRET_KEY") or os.getenv("SECRET_KEY")

    if not api_key or not secret_key:
        print("缺少 API Key 或 Secret Key，请在 .env 中设置 BINANCE_API_KEY/BINANCE_SECRET_KEY 或 API_KEY/SECRET_KEY", file=sys.stderr)
        sys.exit(1)

    exchange_config: Dict[str, Any] = {}

    try:
        strategy = BinanceDynamicStrategy(
            api_key=api_key,
            secret_key=secret_key,
            symbol=args.symbol,
            strategy_config=strategy_config,
            exchange_config=exchange_config,
            direction=args.direction.upper(),
        )
    except Exception as exc:
        print(f"初始化策略失败: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        strategy.run()
    except KeyboardInterrupt:
        print("接收到中断信号，正在退出...")
        strategy.stop()
    except Exception as exc:
        print(f"策略运行异常: {exc}", file=sys.stderr)
        strategy.stop()
        raise


if __name__ == "__main__":
    main()
