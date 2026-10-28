"""Hypothesis accounting invariants for the spot Portfolio (P1-P6 of T015)."""

from datetime import UTC, datetime, timedelta

import pytest
from hypothesis import HealthCheck, event, given, settings
from hypothesis import strategies as st

from bt.core.errors import PortfolioError
from bt.core.types import Fill, Instrument, InstrumentKind, Side
from bt.portfolio.portfolio import Portfolio

CASH_TOL = 1e-6
START = 1000.0
T0 = datetime(2024, 1, 1, tzinfo=UTC)
SYMS = ("AAAUSDT", "BBBUSDT")
STEPS = {"AAAUSDT": 0.001, "BBBUSDT": 0.1}

SETTINGS = settings(
    derandomize=True,
    max_examples=300,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


def _inst(sym: str) -> Instrument:
    return Instrument(
        sym, sym, InstrumentKind.SPOT, sym[:3], "USDT", 0.0, STEPS[sym], 0.0, 0.01, 0.001, 0.001
    )


def _new() -> Portfolio:
    return Portfolio(START, [_inst(s) for s in SYMS])


price_st = st.one_of(st.floats(0.5, 100.0), st.floats(0.5, 1e5))
op_st = st.tuples(
    st.sampled_from(["buy", "sell", "mark", "oversell", "overspend"]),
    st.sampled_from(SYMS),
    st.integers(1, 300),  # qty in steps
    price_st,
    st.floats(0.0, 0.01),  # fee fraction of notional
    st.floats(0.0, 1.0),  # sell fraction of holdings
    st.integers(0, 3),  # ts advance in days
    price_st,  # second mark price
)


def _tol(x: float, rel: float) -> float:
    return rel * max(1.0, abs(x))


def _check_invariants(p: Portfolio) -> None:
    eq = p.equity
    mv = sum(p.position(s).market_value for s in SYMS)
    assert eq == pytest.approx(p.cash + mv, abs=_tol(eq, 1e-9))  # P1
    for s in SYMS:
        assert p.position(s).qty >= 0.0  # P3
    assert p.cash >= -CASH_TOL


def _state(p: Portfolio) -> tuple[float, dict[str, tuple[float, float, float, float]], object]:
    pos = {
        s: (v.qty, v.avg_entry, v.market_value, v.unrealized_pnl) for s, v in p.positions().items()
    }
    return p.cash, pos, p.snapshot() if p.ts is not None else None


@SETTINGS
@given(st.lists(op_st, max_size=40))
def test_accounting_invariants(
    ops: list[tuple[str, str, int, float, float, float, int, float]],
) -> None:
    p = _new()
    held: dict[str, int] = dict.fromkeys(SYMS, 0)
    ts = T0
    expected_cash = START
    notional_sum = 0.0
    n = 0
    # Early mark: exercises the fill/mark ordering guard (later fills must not predate it).
    p.mark(ts, {SYMS[0]: 1.0})
    for kind, sym, nsteps, price, feef, frac, days, price2 in ops:
        step = STEPS[sym]
        n += 1
        if kind == "mark":
            ts = ts + timedelta(days=days)
            p.mark(
                ts,
                {sym: price, SYMS[1 - SYMS.index(sym)]: price2} if feef > 0.005 else {sym: price},
            )
        elif kind in ("oversell", "overspend"):
            if kind == "oversell":
                f = Fill(
                    f"o{n}", sym, ts, Side.SELL, (held[sym] + 1) * step, price, 0.0, 0.0, False
                )
            else:
                qty = (p.cash * 1.5 + 10.0) / price
                f = Fill(f"o{n}", sym, ts, Side.BUY, qty, price, 0.0, 0.0, False)
            before = _state(p)
            with pytest.raises(PortfolioError):
                p.apply_fill(f)
            assert _state(p) == before  # P6
        elif kind == "buy":
            qty = nsteps * step
            fee = feef * qty * price
            cost = qty * price + fee
            f = Fill(f"o{n}", sym, ts, Side.BUY, qty, price, fee, 0.0, False)
            if p.cash - cost < -CASH_TOL:
                before = _state(p)
                with pytest.raises(PortfolioError):
                    p.apply_fill(f)
                assert _state(p) == before  # P6
            else:
                p.apply_fill(f)
                held[sym] += nsteps
                expected_cash -= cost
                notional_sum += qty * price
        else:  # sell
            k = min(held[sym], max(1, int(frac * held[sym]))) if held[sym] else 0
            if k == 0:
                continue
            qty = k * step
            fee = feef * qty * price
            p.apply_fill(Fill(f"o{n}", sym, ts, Side.SELL, qty, price, fee, 0.0, False))
            held[sym] -= k
            expected_cash += qty * price - fee
            notional_sum += qty * price
        _check_invariants(p)
        if p.ts is not None:
            s = p.snapshot()
            unreal = sum(v.unrealized_pnl for v in p.positions().values())
            assert p.equity - START == pytest.approx(  # P2
                s.realized_pnl + unreal - s.fees_paid, abs=_tol(p.equity, 1e-7)
            )
        assert p.cash == pytest.approx(expected_cash, abs=_tol(notional_sum, 1e-7))  # P4
        for s_ in SYMS:
            assert p.position(s_).qty == pytest.approx(held[s_] * STEPS[s_], abs=1e-9)
    event("flat before close-out" if not p.positions() else "open before close-out")
    for sym in SYMS:  # final close-out of everything still held
        if held[sym]:
            qty = held[sym] * STEPS[sym]
            p.apply_fill(Fill(f"c{sym}", sym, ts, Side.SELL, qty, 10.0, 0.0, 0.0, False))
            held[sym] = 0
    _check_invariants(p)
    assert len(p.positions()) == 0


@settings(derandomize=True, max_examples=2000, deadline=None)
@given(
    st.lists(st.integers(1, 5_000_000), min_size=10, max_size=10),
    st.floats(0.5, 1e5),
)
def test_sell_of_summed_grid_buys_closes_exactly(nsteps: list[int], price: float) -> None:
    p = Portfolio(1e18, [_inst("AAAUSDT")])
    total = 0.0
    for i, n in enumerate(nsteps):
        qty = n * 0.001
        p.apply_fill(Fill(f"b{i}", "AAAUSDT", T0, Side.BUY, qty, price, 0.0, 0.0, False))
        total += qty
    p.apply_fill(Fill("s", "AAAUSDT", T0, Side.SELL, total, price, 0.0, 0.0, False))
    assert p.position("AAAUSDT").qty == 0.0
    assert len(p.positions()) == 0


@SETTINGS
@given(
    st.floats(0.5, 1e5),
    st.lists(st.tuples(st.sampled_from(SYMS), st.integers(1, 100), st.booleans()), max_size=30),
)
def test_zero_fee_same_price_round_trip_is_flat_cash(
    price: float, ops: list[tuple[str, int, bool]]
) -> None:
    p = _new()
    held: dict[str, int] = dict.fromkeys(SYMS, 0)
    notional = 0.0
    for i, (sym, nsteps, is_buy) in enumerate(ops):
        step = STEPS[sym]
        if is_buy:
            if p.cash - nsteps * step * price < 0:
                continue
            p.apply_fill(Fill(f"b{i}", sym, T0, Side.BUY, nsteps * step, price, 0.0, 0.0, False))
            held[sym] += nsteps
            notional += nsteps * step * price
        else:
            k = min(nsteps, held[sym])
            if k == 0:
                continue
            p.apply_fill(Fill(f"s{i}", sym, T0, Side.SELL, k * step, price, 0.0, 0.0, False))
            held[sym] -= k
            notional += k * step * price
    for sym in SYMS:  # close out
        if held[sym]:
            qty = held[sym] * STEPS[sym]
            p.apply_fill(Fill(f"c{sym}", sym, T0, Side.SELL, qty, price, 0.0, 0.0, False))
            notional += qty * price
    assert len(p.positions()) == 0
    assert p.cash == pytest.approx(START, abs=_tol(notional, 1e-9))  # P5
