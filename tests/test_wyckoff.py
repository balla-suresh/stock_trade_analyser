"""Causal event and long-only execution checks for the supplied Wyckoff script."""

from dataclasses import FrozenInstanceError, fields
import numpy as np
import pandas as pd
import pytest

from stock_trade_analyser.backtest.engine import BacktestConfig, LongOnlyBacktester
from stock_trade_analyser.strategy.wyckoff import WyckoffStrategy


SETTINGS = {
    "lookback": 100, "volume_multiplier": 1.5, "climax_volume_addition": 0.5,
    "range_period": 20, "narrow_range_multiplier": 0.6,
    "preliminary_period": 50, "climax_period": 30, "sweep_period": 20,
    "rally_change_pct": 0.01, "secondary_test_tolerance_pct": 0.005,
    "spring_recovery_pct": 0.01, "test_range_tolerance": 0.02,
    "test_volume_multiplier": 0.8, "sos_volume_multiplier": 1.3,
    "buy_events": ["SPRING", "SOS", "LPS"],
    "exit_events": ["BC", "UTAD", "LPSY"],
}
EVENTS = (
    "ps", "sc", "ar", "st", "spring", "test", "sos", "lps",
    "psy", "bc", "ar_dist", "st_dist", "utad", "lpsy",
)


def strategy(**overrides):
    return WyckoffStrategy(**{**SETTINGS, **overrides})


def spring_and_upthrust():
    """A 100-bar range followed by an independently calculated round trip."""
    rows = [[100, 102, 98, 100, 100]] * 100 + [
        [100, 103, 95, 102, 150],  # lower sweep and recovery => Spring.
        [103, 105, 101, 104, 100],  # fill buy at 103, close equity 100001.
        [105, 110, 100, 101, 150],  # upper sweep and rejection => UTAD.
        [99, 101, 97, 100, 100],   # fill closing sell at 99.
    ]
    return pd.DataFrame(
        rows, columns=["open", "high", "low", "close", "volume"],
        index=pd.date_range("2024-01-01", periods=len(rows), name="date"), dtype=float,
    )


def scalar_events(frame, settings):
    """Evaluate source predicates chronologically, without pandas rolling/valuewhen."""
    rows = frame.to_dict("records")
    sc_low = spring_or_st_low = bc_high = utad_or_st_high = float("nan")
    phase, pattern = "None", "None"
    output = []
    for i, row in enumerate(rows):
        o, h, l, c, v = (row[key] for key in ("open", "high", "low", "close", "volume"))
        def window(key, period, end=i):
            return [bar[key] for bar in rows[max(0, end - period + 1):end + 1]]
        def average(values, period):
            return sum(values) / period if len(values) == period else float("nan")
        av = average(window("volume", settings["lookback"]), settings["lookback"])
        ranges = [bar["high"] - bar["low"] for bar in rows[max(0, i-settings["range_period"]+1):i+1]]
        arange = average(ranges, settings["range_period"])
        highest = max(window("high", settings["lookback"]))
        lowest = min(window("low", settings["lookback"]))
        middle = (highest + lowest) / 2
        previous = rows[i-1] if i else None
        rising = c > o
        falling = c < o
        high_volume = v > av * settings["volume_multiplier"]
        climax = v > av * (settings["volume_multiplier"] + settings["climax_volume_addition"])
        narrow = h-l < arange * settings["narrow_range_multiplier"]
        e = {}
        e["ps"] = l == min(window("low", settings["preliminary_period"])) and high_volume and rising
        e["sc"] = climax and falling and l <= min(window("low", settings["climax_period"])) and bool(previous and c < previous["close"])
        if e["sc"]:
            sc_low = l  # ta.valuewhen includes the current SC, not only earlier bars.
        e["ar"] = rising and v > av and bool(previous and previous["low"] == min(window("low", settings["climax_period"], i-1)) and c > previous["close"] * (1+settings["rally_change_pct"]))
        tolerance = settings["secondary_test_tolerance_pct"]
        e["st"] = l < sc_low*(1+tolerance) and l > sc_low*(1-tolerance) and v < av and rising
        e["spring"] = bool(previous and l < min(window("low", settings["sweep_period"], i-1))) and rising and v > av and c > l*(1+settings["spring_recovery_pct"])
        e["test"] = abs(l-sc_low) < (highest-lowest)*settings["test_range_tolerance"] and v < av*settings["test_volume_multiplier"] and narrow
        e["sos"] = bool(previous and c > previous["close"]) and v > av*settings["sos_volume_multiplier"] and c-o > arange and c > middle
        if e["spring"] or e["st"]:
            spring_or_st_low = l
        e["lps"] = l > spring_or_st_low and rising and v < av and c > middle
        e["psy"] = h == max(window("high", settings["preliminary_period"])) and high_volume and falling
        e["bc"] = climax and rising and h >= max(window("high", settings["climax_period"]))
        if e["bc"]:
            bc_high = h
        e["ar_dist"] = falling and v > av and bool(previous and previous["high"] == max(window("high", settings["climax_period"], i-1)) and c < previous["close"]*(1-settings["rally_change_pct"]))
        e["st_dist"] = h > bc_high*(1-tolerance) and h < bc_high*(1+tolerance) and v < av
        e["utad"] = bool(previous and h > max(window("high", settings["sweep_period"], i-1))) and falling and v > av and c < h*(1-settings["spring_recovery_pct"])
        if e["utad"] or e["st_dist"]:
            utad_or_st_high = h
        e["lpsy"] = h < utad_or_st_high and falling and v < av and c < middle
        # Source has two ordered if/else blocks; distribution can override accumulation.
        if e["ps"] or e["sc"]:
            phase, pattern = "Phase A", "Accumulation"
        elif e["ar"] and phase == "Phase A":
            phase, pattern = "Phase B", "Accumulation"
        elif e["test"] and phase == "Phase B":
            phase, pattern = "Phase C", "Accumulation"
        elif e["spring"] and phase == "Phase C":
            pattern = "Spring"
        elif e["sos"] and phase in {"Phase B", "Phase C"}:
            phase, pattern = "Phase D", "SOS"
        elif e["lps"] and phase == "Phase D":
            phase, pattern = "Phase E", "LPS"
        if e["psy"] or e["bc"]:
            phase, pattern = "Phase A (Dist)", "Distribution"
        elif e["ar_dist"] and phase == "Phase A (Dist)":
            phase, pattern = "Phase B (Dist)", "Distribution"
        elif e["utad"] and phase == "Phase B (Dist)":
            phase, pattern = "Phase C (Dist)", "UTAD"
        elif e["lpsy"] and phase == "Phase C (Dist)":
            phase, pattern = "Phase D (Dist)", "LPSY"
        output.append({**e, "phase": phase, "pattern": pattern})
    return output


def test_spring_then_upthrust_fills_following_opens_and_never_shorts():
    result = strategy().generate_signals(spring_and_upthrust())
    assert result.iloc[100]["spring"]
    assert result.iloc[100]["signal"] == "BUY"
    assert result.iloc[102]["utad"]
    assert result.iloc[102]["signal"] == "EXIT"
    # The higher low after the Spring also meets LPS; an open long cannot pyramid.
    assert result["buy_signal"].sum() == 2
    assert result["exit_signal"].sum() == 1
    assert result["entry_stop"].isna().all()
    # Spring is actionable even without a full schematic or a visible label.
    assert result.iloc[100]["phase"] == "None"
    report = LongOnlyBacktester(BacktestConfig()).run(result)
    assert report.metrics["initial_capital"] == 100_000
    assert report.metrics["final_equity"] == 99_996
    assert report.metrics["total_buys"] == report.metrics["total_sells"] == report.metrics["total_trades"] == 1
    assert report.metrics["net_profit"] == -4
    assert report.metrics["win_rate_pct"] == 0
    assert report.metrics["max_drawdown_pct"] == pytest.approx(5/100_001*100)
    trade = report.trades.iloc[0]
    assert trade["entry_price"] == 103
    assert trade["exit_price"] == 99
    assert trade["entry_time"] == result.index[101]
    assert trade["exit_time"] == result.index[103]
    assert trade["exit_reason"] == "EXIT_SIGNAL"
    assert report.equity["quantity"].between(0, 1).all()


def test_sweeps_exclude_the_current_candle_and_allow_partial_extrema_history():
    frame = spring_and_upthrust().iloc[:101]
    result = strategy(lookback=20).generate_signals(frame)
    assert result.iloc[-1]["low"] == result.iloc[-1]["support"] == 95
    assert result.iloc[-1]["spring"]
    assert result.iloc[0]["support"] == 98
    assert result.iloc[0]["resistance"] == 102
    assert not result.iloc[0]["spring"]
    assert not result.iloc[0]["utad"]


def test_current_secondary_test_updates_valuewhen_before_lps_evaluation():
    frame = spring_and_upthrust().iloc[:100].copy()
    frame.loc[frame.index[-2], ["open", "high", "low", "close", "volume"]] = [99, 100, 95, 96, 1_000]
    frame.loc[frame.index[-1], ["open", "high", "low", "close", "volume"]] = [97, 103, 95.2, 102, 20]
    # The SC needs a completed 100-bar volume average, so use lookback 20 here.
    result = strategy(lookback=20).generate_signals(frame)
    assert result.iloc[-2]["sc"]
    assert result.iloc[-1]["st"]
    assert not result.iloc[-1]["lps"]
    # On the ST bar, low > valuewhen(ST, low, 0) is low > itself: false.


def test_current_distribution_test_updates_valuewhen_before_lpsy_evaluation():
    frame = spring_and_upthrust().iloc[:100].copy()
    frame.iloc[-3] = [100, 110, 99, 109, 1_000]  # BC reference high = 110.
    frame.iloc[-2] = [109, 111, 99, 100, 200]   # UTAD reference high = 111.
    frame.iloc[-1] = [100, 110.2, 95, 97, 20]   # ST_DIST updates reference to 110.2.
    result = strategy(lookback=20).generate_signals(frame)
    assert result.iloc[-3]["bc"]
    assert result.iloc[-2]["utad"]
    assert result.iloc[-1]["st_dist"]
    assert not result.iloc[-1]["lpsy"]


def test_simultaneous_climax_and_strength_keeps_raw_events_and_prioritizes_exit():
    frame = spring_and_upthrust().iloc[:101].copy()
    frame.loc[frame.index[-1], ["open", "high", "low", "close", "volume"]] = [100, 110, 99, 109, 1_000]
    row = strategy().generate_signals(frame).iloc[-1]
    assert row["bc"] and row["sos"]
    assert {"BC", "SOS"} <= set(row["events"].split(","))
    assert row["exit_signal"] and not row["buy_signal"]
    assert row["signal"] == "EXIT"
    assert row["phase"] == "Phase A (Dist)"
    assert row["pattern"] == "Distribution"


def test_configured_event_mapping_does_not_change_detector():
    frame = spring_and_upthrust()
    default = strategy().generate_signals(frame)
    mapped = strategy(buy_events=["UTAD"], exit_events=["SPRING"]).generate_signals(frame)
    pd.testing.assert_frame_equal(default[list(EVENTS)], mapped[list(EVENTS)])
    assert mapped.iloc[100]["signal"] == "EXIT"
    assert mapped.iloc[102]["signal"] == "BUY"


@pytest.mark.parametrize("seed", [173, 721])
def test_all_fourteen_events_and_phase_order_match_independent_scalar_history(seed):
    rng = np.random.default_rng(seed)
    count = 1_000
    opens = 100 + np.cumsum(rng.normal(0, 0.4, count))
    closes = opens + rng.normal(0, 2.0, count)
    frame = pd.DataFrame({
        "open": opens, "close": closes,
        "high": np.maximum(opens, closes) + rng.uniform(0.05, 2.0, count),
        "low": np.minimum(opens, closes) - rng.uniform(0.05, 2.0, count),
        "volume": rng.choice([10, 40, 100, 180, 1_000], count),
    }, index=pd.date_range("2021-01-01", periods=count))
    parameters = {
        **SETTINGS, "lookback": 20, "preliminary_period": 12,
        "climax_period": 10, "sweep_period": 8,
        "secondary_test_tolerance_pct": 0.02, "test_range_tolerance": 0.2,
    }
    expected = pd.DataFrame(scalar_events(frame, parameters), index=frame.index)
    actual = WyckoffStrategy(**parameters).generate_signals(frame)
    for event in EVENTS:
        assert expected[event].any(), f"fixture must exercise {event}"
        assert actual[event].tolist() == expected[event].tolist(), event
    assert actual["phase"].tolist() == expected["phase"].tolist()
    assert actual["pattern"].tolist() == expected["pattern"].tolist()
    for _, row in actual.iterrows():
        expected_events = {event.upper() for event in EVENTS if row[event]}
        assert set(row["events"].split(",")) - {""} == expected_events


def test_no_future_lookahead_input_mutation_or_display_spacing_filters():
    frame = spring_and_upthrust()
    original = frame.copy(deep=True)
    full = strategy().generate_signals(frame)
    for length in (1, 20, 99, 100, 101, 103):
        partial = strategy().generate_signals(frame.iloc[:length])
        pd.testing.assert_frame_equal(partial, full.iloc[:length])
    pd.testing.assert_frame_equal(frame, original)
    assert full.iloc[100]["spring"] and full.iloc[102]["utad"]
    # Both raw alerts survive despite the source's 20-bar *label* distance.
    assert full.iloc[100]["signal"] == "BUY"
    assert full.iloc[102]["signal"] == "EXIT"


def test_missing_volume_is_rejected_instead_of_creating_zero_volume():
    with pytest.raises(ValueError, match="[Vv]olume"):
        strategy().generate_signals(spring_and_upthrust().drop(columns="volume"))


@pytest.mark.parametrize("bad_volume", [-1, np.nan, np.inf, "bad"])
def test_invalid_volume_is_rejected(bad_volume):
    frame = spring_and_upthrust().astype({"volume": object})
    frame.iloc[0, frame.columns.get_loc("volume")] = bad_volume
    with pytest.raises(ValueError, match="[Vv]olume"):
        strategy().generate_signals(frame)


def test_zero_volume_is_valid_but_cannot_trigger_any_event():
    frame = spring_and_upthrust()
    frame["volume"] = 0
    result = strategy().generate_signals(frame)
    assert not result[list(EVENTS)].any().any()
    assert result["signal"].eq("HOLD").all()


@pytest.mark.parametrize("name,value", [
    ("lookback", 19), ("lookback", 301), ("lookback", True),
    ("range_period", 0), ("range_period", 1.5),
    ("preliminary_period", 0), ("climax_period", -1), ("sweep_period", True),
    ("volume_multiplier", 0), ("volume_multiplier", 0.5), ("volume_multiplier", np.inf),
    ("climax_volume_addition", -1), ("narrow_range_multiplier", 0),
    ("rally_change_pct", -0.1), ("secondary_test_tolerance_pct", np.nan),
    ("spring_recovery_pct", -0.1), ("test_range_tolerance", -0.1),
    ("test_volume_multiplier", 0), ("sos_volume_multiplier", 0),
    ("buy_events", ["UNKNOWN"]), ("buy_events", ["SPRING", "SPRING"]),
    ("exit_events", "BC"),
])
def test_invalid_config_identifies_parameter(name, value):
    with pytest.raises(ValueError, match=name):
        strategy(**{name: value})


def test_all_parameters_are_required_and_configuration_is_immutable():
    with pytest.raises(TypeError):
        WyckoffStrategy()
    assert len(fields(WyckoffStrategy)) == len(SETTINGS)
    with pytest.raises(FrozenInstanceError):
        strategy().lookback = 20


def test_empty_history_and_uppercase_input_are_supported():
    frame = spring_and_upthrust()
    uppercase = frame.rename(columns=str.upper).iloc[::-1]
    result = strategy().generate_signals(uppercase)
    assert result.index.is_monotonic_increasing
    assert result["spring"].sum() == 1
    empty = strategy().generate_signals(frame.iloc[:0])
    assert empty.empty
    assert empty["buy_signal"].dtype == bool
    assert empty["exit_signal"].dtype == bool
