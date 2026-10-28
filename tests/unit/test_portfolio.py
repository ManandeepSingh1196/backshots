"""Portfolio accounting against hand-computed numbers (see T015)."""

from datetime import UTC, datetime

import pytest

from bt.core.errors import PortfolioError, TimezoneError
from bt.core.types import Fill, Instrument, InstrumentKind, PortfolioView, Side
from bt.portfolio.portfolio import Portfolio

T1 = datetime(2024, 1, 2, tzinfo=UTC)
T2 = datetime(2024, 1, 3, tzinfo=UTC)
T3 = datetime(2024, 1, 4, tzinfo=UTC)
T4 = datetime(2024, 1, 5, tzinfo=UTC)


def inst(symbol: str, step: float, kind: InstrumentKind = InstrumentKind.SPOT) -> Instrument:
    lev = 1.0 if kind == InstrumentKind.SPOT else 5.0
    return Instrument(
        symbol, symbol, kind, symbol[:3], "USDT", 0.0, step, 0.0, 0.01, 0.001, 0.001, lev
    )


AAA = inst("AAAUSDT", 0.001)
BBB = inst("BBBUSDT", 0.1)


def fill(
    sym: str, side: Side, qty: float, price: float, fee: float, ts: datetime = T1, slip: float = 0.0
) -> Fill:
    return Fill("o", sym, ts, side, qty, price, fee, slip, False)


def mk() -> Portfolio:
    return Portfolio(1000.0, [AAA, BBB])


def test_full_round_trip_hand_computed() -> None:
    p = mk()
    p.apply_fill(fill("AAAUSDT", Side.BUY, 5.0, 102.102, 0.51051, T1, 0.51))
    assert p.cash == pytest.approx(488.97949, abs=1e-9)
    p.mark(T1, {"AAAUSDT": 108.0})
    assert p.equity == pytest.approx(1028.97949, abs=1e-9)
    pv = p.position("AAAUSDT")
    assert pv.qty == pytest.approx(5.0, abs=1e-9)
    assert pv.avg_entry == pytest.approx(102.102, abs=1e-9)
    assert pv.market_value == pytest.approx(540.0, abs=1e-9)
    assert pv.unrealized_pnl == pytest.approx(29.49, abs=1e-9)

    p.apply_fill(fill("AAAUSDT", Side.SELL, 2.0, 107.892, 0.215784, T2))
    assert p.cash == pytest.approx(704.547706, abs=1e-9)
    p.mark(T2, {"AAAUSDT": 110.0})
    assert p.snapshot().realized_pnl == pytest.approx(11.58, abs=1e-9)
    assert p.position("AAAUSDT").qty == pytest.approx(3.0, abs=1e-9)
    assert p.position("AAAUSDT").avg_entry == pytest.approx(102.102, abs=1e-9)
    assert p.equity == pytest.approx(1034.547706, abs=1e-9)

    p.apply_fill(fill("AAAUSDT", Side.BUY, 1.0, 112.5, 0.1125, T3))
    assert p.cash == pytest.approx(591.935206, abs=1e-9)
    assert p.position("AAAUSDT").avg_entry == pytest.approx(104.7015, abs=1e-9)
    p.mark(T3, {"AAAUSDT": 114.0})
    assert p.equity == pytest.approx(1047.935206, abs=1e-9)

    p.apply_fill(fill("AAAUSDT", Side.SELL, 4.0, 113.886, 0.455544, T4))
    assert p.cash == pytest.approx(1047.023662, abs=1e-9)
    assert len(p.positions()) == 0
    assert p.position("AAAUSDT").qty == 0.0
    assert p.position("AAAUSDT").avg_entry == 0.0
    assert p.equity == pytest.approx(p.cash, abs=1e-9)
    p.mark(T4, {"AAAUSDT": 115.0})
    s = p.snapshot()
    assert s.realized_pnl == pytest.approx(48.318, abs=1e-9)
    assert s.fees_paid == pytest.approx(1.294338, abs=1e-9)
    assert s.slippage_paid == pytest.approx(0.51, abs=1e-9)
    assert s.gross_exposure == 0.0
    assert s.net_exposure == 0.0
    assert s.funding_paid == 0.0
    assert s.tax_paid == 0.0


def test_dust_rule() -> None:
    p = mk()
    p.apply_fill(fill("BBBUSDT", Side.BUY, 0.3, 10.0, 0.0))
    for _ in range(3):
        p.apply_fill(fill("BBBUSDT", Side.SELL, 0.1, 10.0, 0.0))
    assert p.position("BBBUSDT").qty == 0.0
    assert len(p.positions()) == 0


def test_oversell_and_no_position_rejected_atomically() -> None:
    p = mk()
    p.apply_fill(fill("AAAUSDT", Side.BUY, 0.001, 100.0, 0.01))
    p.mark(T1, {"AAAUSDT": 100.0})
    before = (p.cash, p.position("AAAUSDT"), p.snapshot())
    with pytest.raises(PortfolioError):
        p.apply_fill(fill("AAAUSDT", Side.SELL, 0.002, 100.0, 0.5))
    assert (p.cash, p.position("AAAUSDT"), p.snapshot()) == before
    with pytest.raises(PortfolioError):
        p.apply_fill(fill("BBBUSDT", Side.SELL, 0.1, 10.0, 0.0))
    assert p.cash == before[0]


def test_overspend_tolerance() -> None:
    p = mk()
    with pytest.raises(PortfolioError):
        p.apply_fill(fill("AAAUSDT", Side.BUY, 10.0, 100.0, 0.000002))
    assert p.cash == 1000.0
    assert len(p.positions()) == 0
    p.apply_fill(fill("AAAUSDT", Side.BUY, 10.0, 100.0, 0.0000005))
    assert p.cash == pytest.approx(-5e-7, abs=1e-9)


def test_missing_bar_keeps_old_mark() -> None:
    p = mk()
    p.apply_fill(fill("AAAUSDT", Side.BUY, 1.0, 100.0, 0.0))
    p.apply_fill(fill("BBBUSDT", Side.BUY, 2.0, 50.0, 0.0))
    p.mark(T1, {"AAAUSDT": 100.0, "BBBUSDT": 60.0})
    p.mark(T2, {"AAAUSDT": 110.0})
    assert p.equity == pytest.approx(800.0 + 110.0 + 120.0, abs=1e-9)


def test_bad_marks_rejected_and_unchanged() -> None:
    p = mk()
    p.apply_fill(fill("AAAUSDT", Side.BUY, 1.0, 100.0, 0.0))
    p.mark(T2, {"AAAUSDT": 120.0})
    for bad in (
        {"ZZZ": 1.0},
        {"AAAUSDT": 0.0},
        {"AAAUSDT": float("nan")},
        {"AAAUSDT": 130.0, "BBBUSDT": -1.0},
    ):
        with pytest.raises(PortfolioError):
            p.mark(T3, bad)
        assert p.equity == pytest.approx(1020.0, abs=1e-9)
        assert p.ts == T2
    with pytest.raises(PortfolioError):
        p.mark(T1, {"AAAUSDT": 130.0})
    with pytest.raises(TimezoneError):
        p.mark(datetime(2024, 1, 9), {"AAAUSDT": 130.0})  # noqa: DTZ001
    p.mark(T2, {"AAAUSDT": 125.0})  # equal ts allowed
    assert p.equity == pytest.approx(1025.0, abs=1e-9)
    with pytest.raises(PortfolioError):
        p.apply_fill(fill("AAAUSDT", Side.BUY, 1.0, 100.0, 0.0, T1))
    with pytest.raises(PortfolioError):
        p.apply_fill(fill("ZZZ", Side.BUY, 1.0, 100.0, 0.0, T3))
    with pytest.raises(PortfolioError):
        p.position("ZZZ")


def test_fill_before_any_mark_uses_fill_price() -> None:
    p = mk()
    p.apply_fill(fill("AAAUSDT", Side.BUY, 1.0, 100.0, 0.1))
    assert p.equity == pytest.approx(999.9, abs=1e-9)
    assert p.ts is None
    assert p.available_cash == p.cash


def test_snapshot_before_mark_and_ctor_errors() -> None:
    with pytest.raises(PortfolioError):
        mk().snapshot()
    with pytest.raises(PortfolioError, match="M6"):
        Portfolio(1000.0, [inst("PPPUSDT", 0.001, InstrumentKind.PERP)])
    for cash in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(PortfolioError):
            Portfolio(cash, [AAA])
    with pytest.raises(PortfolioError):
        Portfolio(1000.0, [AAA, AAA])
    with pytest.raises(PortfolioError):
        Portfolio(1000.0, [])


def test_view_is_live_and_read_only() -> None:
    p = mk()
    v: PortfolioView = p.view()
    assert v.cash == 1000.0
    assert v.ts is None
    p.apply_fill(fill("BBBUSDT", Side.BUY, 1.0, 10.0, 0.0))
    p.mark(T1, {"BBBUSDT": 12.0})
    assert v.cash == pytest.approx(990.0, abs=1e-9)
    assert v.equity == pytest.approx(1002.0, abs=1e-9)
    assert v.ts == T1
    assert list(v.positions()) == ["BBBUSDT"]
    assert v.position("BBBUSDT").qty == 1.0
    assert not hasattr(v, "apply_fill")
    assert not hasattr(v, "mark")
    assert not hasattr(v, "__dict__")


def test_instances_independent_and_positions_read_only() -> None:
    a, b = mk(), mk()
    a.apply_fill(fill("AAAUSDT", Side.BUY, 1.0, 10.0, 0.0))
    assert b.cash == 1000.0
    with pytest.raises(TypeError):
        a.positions()["X"] = a.position("AAAUSDT")  # type: ignore[index]
