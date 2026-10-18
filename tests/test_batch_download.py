"""Batch Yahoo downloads remain shared while each ticker gets isolated bars."""

import logging

import numpy as np
import pandas as pd
import pytest

from stock_trade_analyser.strategy import cli


@pytest.fixture(autouse=True)
def prevent_external_cache_initialization(monkeypatch):
    monkeypatch.setattr(cli, "_warm_download_cache", lambda: None)


def prices(start=100.0, *, dates=None):
    index = dates if dates is not None else pd.date_range("2025-01-01", periods=3)
    opening = np.arange(len(index), dtype=float) + start
    return pd.DataFrame(
        {"Open": opening, "High": opening + 2, "Low": opening - 1,
         "Close": opening + 1, "Volume": np.full(len(index), 1000.0)},
        index=index,
    )


def batch(*, orientation="ticker", names=True):
    frame = pd.concat({"ALPHA": prices(), "BETA": prices(500.0)}, axis=1)
    frame.columns.names = ["Ticker", "Price"] if names else [None, None]
    if orientation == "price":
        frame = frame.swaplevel(axis=1)
    return frame


@pytest.mark.parametrize("already_disabled", [False, True])
def test_download_fetches_all_tickers_once_and_returns_the_same_object(monkeypatch, capsys, caplog, already_disabled):
    import yfinance as yf

    downloaded = batch()
    tickers = ["ALPHA", "BETA"]
    calls = []
    logger = logging.getLogger("yfinance")
    monkeypatch.setattr(logger, "disabled", already_disabled)

    def download(*args, **kwargs):
        calls.append((args, kwargs))
        logger.error("Provider diagnostic should not appear on screen")
        return downloaded

    monkeypatch.setattr(yf, "download", download)

    assert cli.download_prices(tickers, "7y", "1d") is downloaded
    assert calls == [((), {"tickers": tickers, "period": "7y", "interval": "1d",
                           "auto_adjust": False, "progress": False,
                           "threads": True, "group_by": "ticker"})]
    assert logger.disabled is already_disabled
    assert not caplog.records
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


def test_download_failure_is_quiet_and_restores_provider_logging(monkeypatch, capsys, caplog):
    import yfinance as yf

    logger = logging.getLogger("yfinance")
    monkeypatch.setattr(logger, "disabled", False)

    def fail_download(*args, **kwargs):
        logger.error("Provider diagnostic should not appear on screen")
        raise OSError("Download unavailable")

    monkeypatch.setattr(yf, "download", fail_download)
    with pytest.raises(OSError, match="Download unavailable"):
        cli.download_prices(["ALPHA", "BETA"], "7y", "1d")
    assert logger.disabled is False
    assert not caplog.records
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


@pytest.mark.parametrize("downloaded", [None, pd.DataFrame()])
def test_empty_batch_is_an_error_without_per_ticker_retries(monkeypatch, downloaded):
    import yfinance as yf

    calls = []

    def download(*args, **kwargs):
        calls.append(kwargs)
        return downloaded

    monkeypatch.setattr(yf, "download", download)
    with pytest.raises(ValueError):
        cli.download_prices(["ALPHA", "BETA"], "7y", "1d")
    assert len(calls) == 1


@pytest.mark.parametrize("orientation", ["ticker", "price"])
@pytest.mark.parametrize("names", [True, False])
def test_extract_supports_both_yahoo_column_orders_without_mutating_batch(orientation, names):
    downloaded = batch(orientation=orientation, names=names)
    original = downloaded.copy(deep=True)

    alpha = cli.extract_downloaded_prices(downloaded, "ALPHA", ["ALPHA", "BETA"])
    beta = cli.extract_downloaded_prices(downloaded, "BETA", ["ALPHA", "BETA"])

    assert list(alpha.columns) == ["open", "high", "low", "close", "volume"]
    assert alpha["close"].tolist() == [101.0, 102.0, 103.0]
    assert beta["close"].tolist() == [501.0, 502.0, 503.0]
    alpha.loc[alpha.index[0], "close"] = 9999
    assert beta["close"].tolist() == [501.0, 502.0, 503.0]
    pd.testing.assert_frame_equal(downloaded, original)


@pytest.mark.parametrize("orientation", ["ticker", "price"])
@pytest.mark.parametrize("ticker_level", ["Ticker", "Symbol"])
def test_named_ticker_level_resolves_symbol_that_is_also_a_price_field(orientation, ticker_level):
    downloaded = pd.concat({"Open": prices(200.0), "TEST": prices(900.0)}, axis=1)
    downloaded.columns.names = [ticker_level, "Price"]
    if orientation == "price":
        downloaded = downloaded.swaplevel(axis=1)

    extracted = cli.extract_downloaded_prices(downloaded, "Open", ["Open", "TEST"])
    assert extracted["open"].tolist() == [200.0, 201.0, 202.0]
    assert extracted["close"].tolist() == [201.0, 202.0, 203.0]


def test_single_level_ohlcv_is_accepted_only_for_one_requested_ticker():
    downloaded = prices()
    actual = cli.extract_downloaded_prices(downloaded, "ALPHA", ["ALPHA"])
    assert actual["close"].tolist() == [101.0, 102.0, 103.0]

    with pytest.raises(ValueError):
        cli.extract_downloaded_prices(downloaded, "ALPHA", ["ALPHA", "BETA"])


def test_extract_matches_lowercase_request_to_uppercase_yahoo_symbol():
    extracted = cli.extract_downloaded_prices(batch(), "alpha", ["alpha", "beta"])
    assert extracted["close"].tolist() == [101.0, 102.0, 103.0]


def test_single_level_ohlcv_cannot_be_assigned_to_an_unrequested_ticker():
    with pytest.raises(ValueError):
        cli.extract_downloaded_prices(prices(), "BETA", ["ALPHA"])


def test_extract_omits_only_that_tickers_outer_join_padding():
    alpha_dates = pd.to_datetime(["2025-01-01", "2025-01-03"])
    beta_dates = pd.to_datetime(["2025-01-02", "2025-01-03"])
    downloaded = pd.concat(
        {"ALPHA": prices(dates=alpha_dates), "BETA": prices(500.0, dates=beta_dates)},
        axis=1, sort=True,
    )
    original = downloaded.copy(deep=True)

    alpha = cli.extract_downloaded_prices(downloaded, "ALPHA", ["ALPHA", "BETA"])
    beta = cli.extract_downloaded_prices(downloaded, "BETA", ["ALPHA", "BETA"])

    assert alpha.index.tolist() == alpha_dates.tolist()
    assert beta.index.tolist() == beta_dates.tolist()
    assert alpha["close"].tolist() == [101.0, 102.0]
    assert beta["close"].tolist() == [501.0, 502.0]
    pd.testing.assert_frame_equal(downloaded, original)


def test_extract_converts_timezone_to_utc_naive_and_sorts_dates():
    dates = pd.to_datetime(["2025-01-02 09:15", "2025-01-01 09:15"]).tz_localize("Asia/Kolkata")
    downloaded = pd.concat({"ALPHA": prices(dates=dates)}, axis=1)
    actual = cli.extract_downloaded_prices(downloaded, "ALPHA", ["ALPHA"])

    assert actual.index.tz is None
    assert actual.index.tolist() == [pd.Timestamp("2025-01-01 03:45"), pd.Timestamp("2025-01-02 03:45")]
    assert actual["close"].tolist() == [102.0, 101.0]


def test_absent_or_all_missing_ticker_is_not_filled_from_another_symbol():
    downloaded = batch()
    downloaded.loc[:, "ALPHA"] = np.nan

    with pytest.raises(ValueError):
        cli.extract_downloaded_prices(downloaded, "GAMMA", ["ALPHA", "BETA", "GAMMA"])
    with pytest.raises(ValueError):
        cli.extract_downloaded_prices(downloaded, "ALPHA", ["ALPHA", "BETA"])
    beta = cli.extract_downloaded_prices(downloaded, "BETA", ["ALPHA", "BETA"])
    assert beta["close"].tolist() == [501.0, 502.0, 503.0]


def test_partial_missing_bar_is_rejected_instead_of_silently_dropped():
    downloaded = batch()
    downloaded.loc[downloaded.index[1], ("ALPHA", "Close")] = np.nan

    with pytest.raises(ValueError):
        cli.extract_downloaded_prices(downloaded, "ALPHA", ["ALPHA", "BETA"])


@pytest.mark.parametrize("column,value", [("Close", np.inf), ("Open", -1), ("High", 1), ("Low", 900)])
def test_nonfinite_nonpositive_and_invalid_ohlc_geometry_are_rejected(column, value):
    downloaded = batch()
    downloaded.loc[downloaded.index[1], ("ALPHA", column)] = value

    with pytest.raises(ValueError):
        cli.extract_downloaded_prices(downloaded, "ALPHA", ["ALPHA", "BETA"])


@pytest.mark.parametrize("index", [
    pd.Index([1, 2, 3]),
    pd.Index(["2025-01-01", "bad-date", "2025-01-03"]),
    pd.DatetimeIndex(["2025-01-01", None, "2025-01-03"]),
    pd.DatetimeIndex(["2025-01-01", "2025-01-01", "2025-01-03"]),
])
def test_missing_duplicate_or_invalid_dates_are_rejected(index):
    downloaded = batch()
    downloaded.index = index

    with pytest.raises(ValueError):
        cli.extract_downloaded_prices(downloaded, "ALPHA", ["ALPHA", "BETA"])


def test_duplicate_normalized_price_columns_are_rejected():
    downloaded = prices()
    downloaded[" close "] = downloaded["Close"]

    with pytest.raises(ValueError):
        cli.extract_downloaded_prices(downloaded, "ALPHA", ["ALPHA"])


@pytest.mark.parametrize("volumes", [[1000.0, np.nan, 1000.0], [np.nan, np.nan, np.nan]])
def test_optional_missing_volume_does_not_invalidate_prices_or_change_batch(volumes):
    downloaded = batch()
    downloaded.loc[:, ("ALPHA", "Volume")] = volumes
    original = downloaded.copy(deep=True)

    extracted = cli.extract_downloaded_prices(downloaded, "ALPHA", ["ALPHA", "BETA"])
    assert extracted["close"].tolist() == [101.0, 102.0, 103.0]
    if "volume" in extracted:
        assert extracted["volume"].notna().all()
        assert extracted["volume"].iloc[1] == 0
    pd.testing.assert_frame_equal(downloaded, original)


def test_missing_required_price_column_is_an_error():
    downloaded = prices().drop(columns="Close")

    with pytest.raises(ValueError):
        cli.extract_downloaded_prices(downloaded, "ALPHA", ["ALPHA"])


def test_volume_is_optional_when_download_contains_only_ohlc_prices():
    downloaded = prices().drop(columns="Volume")

    extracted = cli.extract_downloaded_prices(downloaded, "ALPHA", ["ALPHA"])
    assert list(extracted.columns) == ["open", "high", "low", "close"]
    assert extracted["close"].tolist() == [101.0, 102.0, 103.0]
