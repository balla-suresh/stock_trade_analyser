"""Shared tool boundaries preserve ticker, report and optional training behavior."""

import json
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

from stock_trade_analyser.tools import data_utils, downloader, file_utils


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_isolated(source, *, without_site=False):
    command = [sys.executable]
    if without_site:
        command.append("-S")
    command.extend(["-c", textwrap.dedent(source)])
    result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert result.stderr == ""


@pytest.mark.parametrize("module", ["downloader", "file_utils", "data_utils"])
def test_shared_modules_import_without_optional_dependencies(module):
    run_isolated(f"""
        import importlib
        import logging
        import sys
        logging.basicConfig(level=logging.INFO)
        importlib.import_module('src.stock_trade_analyser.tools.{module}')
        optional = {{'numpy', 'pandas', 'torch', 'yfinance', 'tvDatafeed'}}
        assert optional.isdisjoint(sys.modules), optional.intersection(sys.modules)
    """, without_site=True)


@pytest.mark.parametrize("path_kind", ["bare", "relative", "absolute"])
def test_legacy_ticker_paths_use_the_shared_root_resolution(tmp_path, path_kind):
    (tmp_path / "data").mkdir()
    ticker_path = tmp_path / "data" / "universe.csv"
    ticker_path.write_text("symbol\nALPHA\n", encoding="utf-8")
    supplied = {
        "bare": "universe.csv",
        "relative": "data/universe.csv",
        "absolute": str(ticker_path),
    }[path_kind]
    legacy = file_utils.FileUtils()
    legacy.ROOT_DIR = tmp_path

    assert file_utils.resolve_ticker_file(supplied, root_dir=tmp_path) == ticker_path
    assert legacy._resolve_ticker_file(supplied) == str(ticker_path)


def test_legacy_ticker_reader_handles_headers_blanks_bom_and_global_duplicates(tmp_path):
    ticker_path = tmp_path / "universe.csv"
    ticker_path.write_text(
        "\ufeffsymbol,company\nALPHA,Alpha Ltd\n\n ,blank\nBETA,Beta Ltd\nALPHA,duplicate\n",
        encoding="utf-8",
    )
    legacy = file_utils.FileUtils(ticker_file=str(ticker_path))

    assert legacy.read_ticker() == ["ALPHA", "BETA"]


def test_legacy_ticker_reader_uses_the_shared_reader(tmp_path, monkeypatch):
    ticker_path = tmp_path / "universe.csv"
    ticker_path.touch()
    legacy = file_utils.FileUtils(ticker_file=str(ticker_path))
    calls = []

    def read(path):
        calls.append(path)
        return ["ALPHA"]

    monkeypatch.setattr(file_utils, "read_tickers", read)
    assert legacy.read_ticker() == ["ALPHA"]
    assert calls == [ticker_path]


def test_legacy_default_ticker_path_and_missing_explicit_file_are_preserved(tmp_path):
    legacy = file_utils.FileUtils()
    legacy.ROOT_DIR = tmp_path

    assert legacy._resolve_ticker_file(None) == str(tmp_path / "data" / "tickers.csv")
    with pytest.raises(FileNotFoundError, match="Ticker file not found"):
        legacy._resolve_ticker_file("missing.csv")


def test_report_json_round_trips_numerical_scalars_dates_and_missing_values(tmp_path):
    path = tmp_path / "metrics.json"
    file_utils.write_json(path, {
        "trades": np.int64(3), "net_profit": np.float64(4.5),
        "missing": [float("nan"), np.float64(float("inf"))],
        "date": pd.Timestamp("2025-01-02 09:15:00"),
    })

    assert file_utils.read_json(path) == {
        "trades": 3, "net_profit": 4.5, "missing": [None, None],
        "date": "2025-01-02T09:15:00",
    }
    assert path.read_text(encoding="utf-8").endswith("\n")
    assert json.loads(path.read_text(encoding="utf-8"))["missing"] == [None, None]


def test_malformed_json_error_identifies_the_path_and_line(tmp_path):
    path = tmp_path / "day.json"
    path.write_text('{\n  "download":\n}', encoding="utf-8")

    with pytest.raises(ValueError) as caught:
        file_utils.read_json(path)
    assert str(path) in str(caught.value)
    assert "Invalid JSON in config" in str(caught.value)
    assert "(line 3)" in str(caught.value)


def test_legacy_yahoo_cache_warmup_uses_the_shared_initializer(monkeypatch):
    calls = []
    monkeypatch.setattr(downloader, "_warm_download_cache", lambda: calls.append("warm"))

    downloader.Downloader._warm_yf_cache()

    assert calls == ["warm"]


def test_downloader_import_does_not_read_a_ticker_file():
    run_isolated("""
        from src.stock_trade_analyser.tools import file_utils
        def unexpected_read(self):
            raise AssertionError('ticker file read during module import')
        file_utils.FileUtils.read_ticker = unexpected_read
        file_utils.FileUtils.read_tv_ticker = unexpected_read
        from src.stock_trade_analyser.tools import downloader
    """, without_site=True)


def test_yahoo_constructor_uses_supplied_file_helpers_without_tv_dependencies():
    run_isolated("""
        import os
        import sys
        from src.stock_trade_analyser.tools.downloader import Downloader
        os.environ.pop('TV_USERNAME', None)
        class Files:
            def __init__(self):
                self.reads = 0
            def read_ticker(self):
                self.reads += 1
                return ['ALPHA', 'BETA']
            def read_tv_ticker(self):
                raise AssertionError('Yahoo constructor requested TV tickers')
        files = Files()
        instance = Downloader(downloader='yahoo', file_utils=files)
        assert instance.file_utils is files
        assert instance.ticker_list == ['ALPHA', 'BETA']
        assert files.reads == 1
        assert {'tvDatafeed', 'yfinance', 'pandas'}.isdisjoint(sys.modules)
    """, without_site=True)


def test_legacy_normalizer_values_and_inverse_are_preserved():
    values = np.array([2.0, 4.0, 6.0, 8.0])
    normalizer = data_utils.Normalizer()

    actual = normalizer.fit_transform(values)

    np.testing.assert_allclose(actual, np.array([-3.0, -1.0, 1.0, 3.0]) / np.sqrt(5))
    np.testing.assert_allclose(normalizer.inverse_transform(actual), values)


def test_legacy_windowing_keeps_next_day_labels_and_unseen_window():
    values = np.arange(6, dtype=float)
    windows, unseen = data_utils.prepare_data_x(values, window_size=3)

    np.testing.assert_array_equal(windows, [[0, 1, 2], [1, 2, 3], [2, 3, 4]])
    np.testing.assert_array_equal(unseen, [3, 4, 5])
    np.testing.assert_array_equal(data_utils.prepare_data_y(values, window_size=3), [3, 4, 5])
    split, train_x, train_y, val_x, val_y, last = data_utils.prepare_data(
        values, {"data": {"window_size": 3, "train_split_size": 0.5}},
    )
    assert split == 1
    np.testing.assert_array_equal(train_x, [[0, 1, 2]])
    np.testing.assert_array_equal(train_y, [3])
    np.testing.assert_array_equal(val_x, [[1, 2, 3], [2, 3, 4]])
    np.testing.assert_array_equal(val_y, [4, 5])
    np.testing.assert_array_equal(last, unseen)


def test_torch_is_loaded_only_when_the_legacy_dataset_is_requested():
    run_isolated("""
        import sys
        import types
        import numpy as np
        from src.stock_trade_analyser.tools import data_utils
        normalizer = data_utils.Normalizer()
        normalizer.fit_transform(np.array([1., 2., 3.]))
        data_utils.prepare_data_x(np.arange(6.), 3)
        assert 'torch' not in sys.modules

        # Supply only the optional Dataset interface so no torch installation
        # is needed to exercise the legacy dataset's numpy conversion.
        torch = types.ModuleType('torch')
        utils = types.ModuleType('torch.utils')
        data = types.ModuleType('torch.utils.data')
        class Dataset:
            pass
        data.Dataset = Dataset
        sys.modules.update({'torch': torch, 'torch.utils': utils, 'torch.utils.data': data})
        dataset_type = data_utils.TimeSeriesDataset
        dataset = dataset_type(np.array([[1., 2.], [3., 4.]]), np.array([5., 6.]))
        assert isinstance(dataset, Dataset)
        assert data_utils.TimeSeriesDataset is dataset_type
        assert len(dataset) == 2
        assert dataset.x.dtype == np.float32
        assert dataset.y.dtype == np.float32
        assert dataset.x.shape == (2, 2, 1)
        np.testing.assert_array_equal(dataset[0][0], [[1.], [2.]])
        assert dataset[0][1] == 5.
    """)
