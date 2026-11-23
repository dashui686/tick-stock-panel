"""DYNAINFO snapshot from the latest visible trading session."""

from __future__ import annotations

import math
from itertools import pairwise

import polars as pl

MORNING_OPEN = 9 * 60 + 30
MORNING_CLOSE = 11 * 60 + 30
AFTERNOON_OPEN = 13 * 60
AFTERNOON_CLOSE = 15 * 60


def _minute_of_day(time: str) -> int:
    return int(time[11:13]) * 60 + int(time[14:16])


def _price(value: float | None) -> float:
    return value if value is not None and math.isfinite(value) and value > 0 else math.nan


def _trade_value(value: float | None) -> float:
    return value if value is not None and math.isfinite(value) and value >= 0 else math.nan


def _minute_summary(rows: list[dict]) -> tuple[float, float, float, float, float]:
    minutes = [_minute_of_day(row["time"]) for row in rows]
    starts_at_open = minutes[0] in (MORNING_OPEN, MORNING_OPEN + 1)
    opening = _price(rows[0].get("open")) if starts_at_open else math.nan
    # Minute feeds label bars by either their start or end. The lunch break
    # is a session boundary, while an intraday gap makes totals incomplete.
    contiguous = starts_at_open and all(
        later == earlier + 1
        or (
            MORNING_CLOSE - 1 <= earlier <= MORNING_CLOSE
            and AFTERNOON_OPEN <= later <= AFTERNOON_OPEN + 1
        )
        for earlier, later in pairwise(minutes)
    )
    if not contiguous:
        return opening, math.nan, math.nan, math.nan, math.nan

    highs = [_price(row.get("high")) for row in rows]
    lows = [_price(row.get("low")) for row in rows]
    high = max(highs) if all(map(math.isfinite, highs)) else math.nan
    low = min(lows) if all(map(math.isfinite, lows)) else math.nan
    volume = sum(_trade_value(row.get("volume")) for row in rows)
    amount = sum(_trade_value(row.get("amount")) for row in rows)
    return opening, high, low, volume, amount


def quote_snapshot(frame: pl.DataFrame, timeframe: str, asset_type: str) -> dict[int, float]:
    if frame.is_empty():
        return {}
    frame = frame.sort("time")
    last = frame.row(-1, named=True)
    if timeframe == "minute":
        session = frame.filter(pl.col("time").str.starts_with(last["time"][:10])).to_dicts()
        session = [
            row
            for row in session
            if (
                MORNING_OPEN <= _minute_of_day(row["time"]) <= MORNING_CLOSE
                or AFTERNOON_OPEN <= _minute_of_day(row["time"]) <= AFTERNOON_CLOSE
            )
        ]
        if not session:
            return {}
        last = session[-1]
        opening, high, low, volume, amount = _minute_summary(session)
    else:
        opening, high, low = (_price(last.get(field)) for field in ("open", "high", "low"))
        volume, amount = (_trade_value(last.get(field)) for field in ("volume", "amount"))

    close = _price(last.get("close"))
    prev_close = _price(last.get("prev_close"))
    average = (
        amount / volume / 100 if asset_type != "index" and volume > 0 and amount > 0 else math.nan
    )
    raw_close = _price(last.get("raw_close"))
    if timeframe == "daily" and math.isfinite(raw_close):
        # Daily amount is unadjusted. The minute projection already adjusts amount with prices.
        average *= close / raw_close
    change = close - prev_close
    return {
        3: prev_close,
        4: opening,
        5: high,
        6: low,
        7: close,
        8: volume,
        10: amount,
        11: average,
        12: change,
        13: (high - low) / prev_close,
        14: change / prev_close,
    }
