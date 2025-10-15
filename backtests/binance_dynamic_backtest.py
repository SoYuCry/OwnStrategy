from __future__ import annotations

import argparse
import glob
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class BacktestConfig:
    symbol: str
    direction: str = "LONG"
    initial_equity: float = 10_000.0
    initial_allocation_pct: float = 0.05
    take_profit_activation_pct: float = 0.03
    trailing_stop_gap_pct: float = 0.005
    dip_threshold_pct: float = 0.01
    max_additions: int = 3
    max_loss_pct: float = 0.30
    poll_interval_minutes: int = 1


@dataclass
class TradeEvent:
    timestamp: pd.Timestamp
    event: str
    price: float
    quantity: float
    equity: float
    note: str = ""


@dataclass
class BacktestResult:
    config: BacktestConfig
    start: pd.Timestamp
    end: pd.Timestamp
    final_equity: float
    return_pct: float
    max_drawdown_pct: float
    max_drawdown_start: Optional[pd.Timestamp] = None
    max_drawdown_end: Optional[pd.Timestamp] = None
    trades: List[TradeEvent] = field(default_factory=list)
    equity_curve: pd.Series | None = None


class BinanceDynamicBacktester:
    """Portfolio simulator for `BinanceDynamicStrategy` using historical klines."""

    def __init__(self, klines: pd.DataFrame, config: BacktestConfig) -> None:
        if klines.empty:
            raise ValueError("Kline DataFrame is empty")

        self.df = klines.sort_index()
        self.cfg = config
        self.direction = config.direction.upper()
        if self.direction not in {"LONG", "SHORT"}:
            raise ValueError("direction 必须是 LONG 或 SHORT")

        self.sign = 1 if self.direction == "LONG" else -1

        # State
        self.cash = config.initial_equity
        self.position_qty = 0.0
        self.avg_entry_price: Optional[float] = None
        self.position_margin = 0.0
        self.trailing_active = False
        self.trailing_extreme: Optional[float] = None
        self.last_add_price: Optional[float] = None
        self.last_allocation_notional = 0.0
        self.additions_done = 0
        self.reentry_pending = False

        self.trades: List[TradeEvent] = []
        self.equity_points: Dict[pd.Timestamp, float] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _alloc_to_qty(self, allocation: float, price: float) -> float:
        if price <= 0:
            return 0.0
        qty = allocation / price
        # Align with exchange precision assumptions (round to 6 decimals)
        return math.floor(qty * 1e6) / 1e6

    def _record(self, ts: pd.Timestamp, event: str, price: float, qty: float, note: str = "") -> None:
        equity = self._equity(price)
        self.trades.append(TradeEvent(ts, event, price, qty, equity, note))

    def _equity(self, price: float) -> float:
        unrealized = self._unrealized_pnl(price)
        return self.cash + self.position_margin + unrealized

    def _unrealized_pnl(self, price: float) -> float:
        if not self.position_qty or self.avg_entry_price is None:
            return 0.0
        return self.sign * self.position_qty * (price - self.avg_entry_price)

    def _pnl_ratio(self, price: float) -> Optional[float]:
        if not self.position_qty or not self.avg_entry_price:
            return None
        return self.sign * (price - self.avg_entry_price) / self.avg_entry_price

    def _update_trailing(self, price: float, ts: pd.Timestamp) -> bool:
        """Returns True if trailing exit occurs."""
        if not self.trailing_active or self.trailing_extreme is None:
            return False

        if self.direction == "LONG":
            if price > self.trailing_extreme:
                self.trailing_extreme = price
                return False
            threshold = self.trailing_extreme * (1 - self.cfg.trailing_stop_gap_pct)
            if price <= threshold:
                self._record(ts, "TRAIL_STOP", price, -self.position_qty, "Trailing stop hit")
                self._close_position(price, ts, allow_reentry=True)
                return True
            return False

        # SHORT
        if price < self.trailing_extreme:
            self.trailing_extreme = price
            return False
        threshold = self.trailing_extreme * (1 + self.cfg.trailing_stop_gap_pct)
        if price >= threshold:
            self._record(ts, "TRAIL_STOP", price, -self.position_qty, "Trailing stop hit")
            self._close_position(price, ts, allow_reentry=True)
            return True
        return False

    # ------------------------------------------------------------------
    # Trade actions
    # ------------------------------------------------------------------

    def _open_initial(self, price: float, ts: pd.Timestamp) -> None:
        total_equity = self._equity(price)
        allocation = total_equity * self.cfg.initial_allocation_pct
        allocation = min(allocation, self.cash)
        if allocation <= 0:
            raise RuntimeError("资金不足，无法开仓")
        qty = self._alloc_to_qty(allocation, price)
        if qty <= 0:
            raise RuntimeError("首仓数量过小，无法执行")

        self.cash -= allocation
        self.position_margin += allocation
        self.position_qty = qty
        self.avg_entry_price = price
        self.last_allocation_notional = allocation
        self.last_add_price = price
        self.additions_done = 0
        self.trailing_active = False
        self.trailing_extreme = None
        self.reentry_pending = False

        side = "BUY" if self.direction == "LONG" else "SELL"
        self._record(ts, f"{side}_OPEN", price, qty, "Initial position")

    def _close_position(self, price: float, ts: pd.Timestamp, allow_reentry: bool = True) -> None:
        if not self.position_qty or self.avg_entry_price is None:
            return

        realized = self.sign * self.position_qty * (price - self.avg_entry_price)
        payout = self.position_margin + realized
        self.cash += payout

        qty = self.position_qty

        # Reset state before logging so equity reflects post-close balance
        self.position_qty = 0.0
        self.avg_entry_price = None
        self.position_margin = 0.0
        self.trailing_active = False
        self.trailing_extreme = None
        self.last_add_price = None
        self.last_allocation_notional = 0.0
        self.additions_done = 0
        self.reentry_pending = allow_reentry

        self._record(ts, "CLOSE", price, -qty, f"Realized PnL: {realized:.2f}")

    def _maybe_activate_trailing(self, price: float, ts: pd.Timestamp) -> None:
        if self.trailing_active or not self.avg_entry_price:
            return

        trigger = (
            self.avg_entry_price * (1 + self.cfg.take_profit_activation_pct)
            if self.direction == "LONG"
            else self.avg_entry_price * (1 - self.cfg.take_profit_activation_pct)
        )
        condition = price >= trigger if self.direction == "LONG" else price <= trigger
        if condition:
            self.trailing_active = True
            self.trailing_extreme = price
            self._record(ts, "TRAIL_ARMED", price, 0.0, f"Activation price {trigger:.4f}")

    def _maybe_max_loss(self, price: float, ts: pd.Timestamp) -> bool:
        ratio = self._pnl_ratio(price)
        if ratio is None:
            return False
        if ratio <= -self.cfg.max_loss_pct:
            self._record(ts, "MAX_LOSS", price, -self.position_qty, f"Unrealized {ratio:.2%}")
            self._close_position(price, ts, allow_reentry=True)
            return True
        return False

    def _maybe_add(self, price: float, ts: pd.Timestamp) -> None:
        if self.additions_done >= self.cfg.max_additions:
            return
        if self.last_add_price is None or self.last_allocation_notional <= 0:
            return

        trigger_price = (
            self.last_add_price * (1 - self.cfg.dip_threshold_pct)
            if self.direction == "LONG"
            else self.last_add_price * (1 + self.cfg.dip_threshold_pct)
        )
        condition = price <= trigger_price if self.direction == "LONG" else price >= trigger_price
        if not condition:
            return

        desired = self.last_allocation_notional * 2
        allocation = min(desired, self.cash)
        if allocation <= 0:
            return

        qty = self._alloc_to_qty(allocation, price)
        if qty <= 0:
            return

        # Update cash/margin
        self.cash -= allocation
        self.position_margin += allocation
        new_qty = self.position_qty + qty
        if new_qty <= 0:
            return

        # Weighted average entry
        prev_cost = self.avg_entry_price * self.position_qty if self.avg_entry_price else 0.0
        self.avg_entry_price = (prev_cost + price * qty) / new_qty
        self.position_qty = new_qty
        self.last_add_price = price
        self.last_allocation_notional = allocation
        self.additions_done += 1

        side = "BUY" if self.direction == "LONG" else "SELL"
        self._record(ts, f"{side}_ADD", price, qty, "Averaging down")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> BacktestResult:
        first_idx = self.df.index[0]
        price = float(self.df.loc[first_idx, "close"])
        self._open_initial(price, first_idx)
        self.equity_points[first_idx] = self._equity(price)

        for ts, row in self.df.iloc[1:].iterrows():
            price = float(row["close"])

            if self.position_qty <= 0 and self.reentry_pending:
                try:
                    self._open_initial(price, ts)
                except RuntimeError as exc:
                    self._record(ts, "REENTRY_FAIL", price, 0.0, str(exc))
                    self.reentry_pending = False

            self.equity_points[ts] = self._equity(price)

            if not self.position_qty:
                continue

            self._maybe_activate_trailing(price, ts)
            if self._update_trailing(price, ts):
                continue

            if self._maybe_max_loss(price, ts):
                continue

            self._maybe_add(price, ts)

        # Close at final bar if still open
        final_ts = self.df.index[-1]
        final_price = float(self.df.iloc[-1]["close"])
        if self.position_qty:
            self._record(final_ts, "FORCE_CLOSE", final_price, -self.position_qty, "End of dataset")
            self._close_position(final_price, final_ts, allow_reentry=False)
            self.equity_points[final_ts] = self._equity(final_price)

        equity_series = pd.Series(self.equity_points).sort_index()
        equity_series.index = pd.to_datetime(equity_series.index)

        final_equity = equity_series.iloc[-1]
        return_pct = (final_equity / self.cfg.initial_equity) - 1
        dd, dd_start, dd_end = self._max_drawdown(equity_series)

        return BacktestResult(
            config=self.cfg,
            start=equity_series.index[0],
            end=equity_series.index[-1],
            final_equity=final_equity,
            return_pct=return_pct,
            max_drawdown_pct=dd,
            max_drawdown_start=dd_start,
            max_drawdown_end=dd_end,
            trades=self.trades,
            equity_curve=equity_series,
        )

    @staticmethod
    def _max_drawdown(equity: pd.Series) -> tuple[float, Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        if equity.empty:
            return 0.0, None, None

        rolling_max = equity.cummax()
        drawdown = equity / rolling_max - 1
        min_drawdown = float(drawdown.min())
        if math.isclose(min_drawdown, 0.0, abs_tol=1e-12):
            return min_drawdown, None, None

        end_idx = drawdown.idxmin()
        peak_slice = equity.loc[:end_idx]
        peak_value = peak_slice.max()
        peak_candidates = peak_slice[peak_slice == peak_value]
        start_idx = peak_candidates.index[-1] if not peak_candidates.empty else None
        return min_drawdown, start_idx, end_idx


# ----------------------------------------------------------------------
# CLI helpers
# ----------------------------------------------------------------------

def load_klines(data_dir: str, symbol: str, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    pattern = os.path.join(data_dir, "**", f"{symbol}-1m-*.csv")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(f"未在 {data_dir} 找到 {symbol} 的 1m 数据")

    frames = []
    for path in files:
        df = pd.read_csv(
            path,
            header=None,
            names=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_volume",
                "trades",
                "taker_buy_base",
                "taker_buy_quote",
                "ignore",
            ],
        )
        if df.iloc[0, 0] == "open_time":
            df = df.iloc[1:]
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    data["open_time"] = pd.to_numeric(data["open_time"], errors="coerce")
    data["open_time"] = pd.to_datetime(data["open_time"], unit="ms")
    data.set_index("open_time", inplace=True)
    data = data.sort_index()
    if start:
        start_ts = pd.to_datetime(start)
        data = data.loc[data.index >= start_ts]
    if end:
        end_ts = pd.to_datetime(end) + pd.Timedelta(days=1)
        data = data.loc[data.index < end_ts]
    data = data.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="回测 Binance 动态策略 (1m 数据)")
    parser.add_argument("--data-dir", required=True, help="build downloader 保存数据的根目录")
    parser.add_argument("--symbol", required=True, help="交易对 (例如 BTCUSDT)")
    parser.add_argument("--direction", choices=["long", "short"], default="long", help="策略方向")
    parser.add_argument("--start", help="开始时间 (YYYY-MM-DD)")
    parser.add_argument("--end", help="结束时间 (YYYY-MM-DD)")
    parser.add_argument("--initial-equity", type=float, default=10_000.0, help="初始资金")
    parser.add_argument("--initial-allocation", type=float, default=0.05, help="首仓占比")
    parser.add_argument("--activation-profit", type=float, default=0.03, help="追踪止盈启动阈值")
    parser.add_argument("--trailing-gap", type=float, default=0.005, help="追踪止盈回撤幅度")
    parser.add_argument("--dip-threshold", type=float, default=0.01, help="补仓触发跌幅")
    parser.add_argument("--max-additions", type=int, default=3, help="最大补仓次数")
    parser.add_argument("--max-loss", type=float, default=0.30, help="最大亏损比例")
    parser.add_argument(
        "--show-events",
        choices=["none", "last", "all"],
        default="last",
        help="事件输出方式: none 不打印, last 仅展示最后5条, all 打印全部事件",
    )
    parser.add_argument(
        "--summary-path",
        help="将全部事件保存为 CSV/Parquet 等格式 (依据扩展名)；如未提供则不写入文件",
    )
    return parser.parse_args()


def summarize(result: BacktestResult, show_events: str, summary_path: Optional[str]) -> None:
    print(f"Backtest {result.config.symbol} {result.config.direction}")
    print(f"Period: {result.start} -> {result.end}")
    print(f"Final equity: {result.final_equity:.2f}")
    print(f"Return: {result.return_pct:.2%}")
    print(f"Max drawdown: {result.max_drawdown_pct:.2%}")
    if result.max_drawdown_start and result.max_drawdown_end:
        print(f"Max drawdown window: {result.max_drawdown_start} -> {result.max_drawdown_end}")
    print(f"Trades executed: {len(result.trades)}")
    if result.trades:
        if show_events == "all":
            print("Event log:")
            events_to_show = result.trades
        elif show_events == "last":
            print("Last 5 events:")
            events_to_show = result.trades[-5:]
        else:
            events_to_show = []

        for event in events_to_show:
            print(
                f" {event.timestamp} | {event.event:<12} | price={event.price:.4f} | qty={event.quantity:.6f} | equity={event.equity:.2f} | {event.note}"
            )

    if summary_path:
        df = pd.DataFrame(
            [
                {
                    "timestamp": event.timestamp,
                    "event": event.event,
                    "price": event.price,
                    "quantity": event.quantity,
                    "equity": event.equity,
                    "note": event.note,
                }
                for event in result.trades
            ]
        )
        if summary_path.lower().endswith(".parquet"):
            df.to_parquet(summary_path, index=False)
        else:
            df.to_csv(summary_path, index=False)
        print(f"Event summary saved to: {summary_path}")


def main() -> None:
    args = parse_args()
    klines = load_klines(args.data_dir, args.symbol, args.start, args.end)
    cfg = BacktestConfig(
        symbol=args.symbol.upper(),
        direction=args.direction.upper(),
        initial_equity=args.initial_equity,
        initial_allocation_pct=args.initial_allocation,
        take_profit_activation_pct=args.activation_profit,
        trailing_stop_gap_pct=args.trailing_gap,
        dip_threshold_pct=args.dip_threshold,
        max_additions=args.max_additions,
        max_loss_pct=args.max_loss,
    )
    engine = BinanceDynamicBacktester(klines, cfg)
    result = engine.run()
    summarize(result, show_events=args.show_events, summary_path=args.summary_path)


if __name__ == "__main__":
    main()
