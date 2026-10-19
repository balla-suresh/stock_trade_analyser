"""Shared price-data utilities and optional legacy training helpers."""

from __future__ import annotations

import csv
import logging
import math
from pathlib import Path

__all__ = [
    "load_prices", "extract_downloaded_prices", "normalize_ohlcv",
    "prepare_data_x", "prepare_data_y", "prepare_data", "run_epoch",
    "Normalizer", "TimeSeriesDataset",
]


def normalize_ohlcv(data: pd.DataFrame) -> pd.DataFrame:
    """Copy, validate and chronologically order a single-symbol OHLCV frame.

    Open, high, low and close are required; volume-dependent strategies check
    its presence separately. Numeric strings are accepted, while
    missing prices and duplicate timestamps are rejected rather than filled.
    """
    import numpy as np
    import pandas as pd

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Price data must be a pandas DataFrame.")
    if isinstance(data.columns, pd.MultiIndex):
        raise ValueError("Price data must have single-level OHLCV columns.")
    columns = [str(column).strip().lower() for column in data.columns]
    if len(columns) != len(set(columns)):
        raise ValueError("Price data has duplicate column names.")
    frame = data.copy()
    frame.columns = columns
    required = ("open", "high", "low", "close")
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Missing required OHLC columns: {', '.join(missing)}")
    if isinstance(frame.index, pd.MultiIndex):
        raise ValueError("Price data must have a single timestamp index.")
    if frame.index.hasnans:
        raise ValueError("Price data has missing timestamps.")
    if frame.index.has_duplicates:
        raise ValueError("Price data has duplicate timestamps.")
    try:
        frame = frame.sort_index()
    except TypeError as exc:
        raise ValueError("Price timestamps must be chronologically sortable.") from exc

    numeric_columns = list(required) + (["volume"] if "volume" in frame else [])
    for column in numeric_columns:
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Column {column!r} must contain numeric values.") from exc
        values = frame[column].to_numpy()
        if not np.isfinite(values).all():
            raise ValueError(f"Column {column!r} must contain finite values.")
        if column == "volume":
            if (values < 0).any():
                raise ValueError("Volume cannot be negative.")
        elif (values <= 0).any():
            raise ValueError(f"Column {column!r} must contain positive prices.")

    invalid_high = frame["high"] < frame[["open", "low", "close"]].max(axis=1)
    invalid_low = frame["low"] > frame[["open", "high", "close"]].min(axis=1)
    if invalid_high.any() or invalid_low.any():
        raise ValueError("OHLC prices must satisfy low <= open/close <= high.")
    return frame


def load_prices(path: Path):
    """Load ordinary OHLC CSVs with a date column or exported date index."""
    import pandas as pd

    # pandas silently renames repeated headers (Close, Close -> Close, Close.1),
    # so inspect the source header before loading or choosing any OHLC column.
    with path.open(newline="", encoding="utf-8-sig") as handle:
        header = next((row for row in csv.reader(handle) if row), None)
    if header is None:
        raise ValueError("CSV contains no header or price rows")
    normalized_header = [column.strip().lower().replace(" ", "_") for column in header]
    if len(set(normalized_header)) != len(normalized_header):
        raise ValueError("CSV contains duplicate column names")
    # Keep timestamps as text: an integer YYYYMMDD must never become nanoseconds
    # since the epoch before date-format validation runs.
    frame = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    if frame.empty:
        raise ValueError("CSV contains no price rows")
    frame.columns = [str(column).strip().lower().replace(" ", "_") for column in frame.columns]
    if frame.columns.duplicated().any():
        raise ValueError("CSV contains duplicate column names")
    date_column = next((name for name in ("date", "datetime", "timestamp", "time") if name in frame.columns), None)
    if date_column is None:
        first_column = frame.columns[0]
        if first_column.startswith("unnamed:") or first_column not in {"open", "high", "low", "close", "volume", "adj_close"}:
            date_column = first_column
    if date_column is None:
        raise ValueError("CSV needs a date/datetime/timestamp column or a saved date index")
    date_text = frame.pop(date_column).astype("string").str.strip()
    compact_dates = date_text.str.fullmatch(r"\d{8}", na=False)
    numeric_dates = date_text.str.fullmatch(r"[+-]?\d+(?:\.\d+)?", na=False)
    if (numeric_dates & ~compact_dates).any():
        raise ValueError("CSV contains ambiguous numeric dates; use ISO dates or YYYYMMDD")
    dates = pd.to_datetime(date_text, format="mixed", errors="coerce", utc=True)
    if compact_dates.any():
        dates.loc[compact_dates] = pd.to_datetime(
            date_text.loc[compact_dates], format="%Y%m%d", errors="coerce", utc=True,
        )
    if dates.isna().any():
        raise ValueError("CSV contains invalid dates")
    frame.index = pd.DatetimeIndex(dates).tz_convert(None)
    frame.index.name = "date"
    if frame.index.has_duplicates:
        raise ValueError("CSV contains duplicate dates")
    required = {"open", "high", "low", "close"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {', '.join(sorted(missing))}")
    for column in required | ({"volume"} if "volume" in frame.columns else set()):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not frame[list(required)].map(lambda value: math.isfinite(value)).all().all():
        raise ValueError("CSV contains missing or non-finite OHLC values")
    return frame.sort_index()


def extract_downloaded_prices(downloaded, ticker: str, tickers: list[str]):
    """Take one symbol from a batch without mutating or rereading the batch."""
    import pandas as pd

    if not isinstance(downloaded, pd.DataFrame) or downloaded.empty:
        raise ValueError("Yahoo Finance returned no price data")
    required = {"open", "high", "low", "close"}
    if isinstance(downloaded.columns, pd.MultiIndex):
        if downloaded.columns.nlevels != 2:
            raise ValueError("Downloaded columns must contain ticker and price levels")

        def symbol_label(level):
            return next((label for label in downloaded.columns.get_level_values(level).unique()
                         if str(label).upper() == ticker.upper()), None)

        named_levels = [index for index, name in enumerate(downloaded.columns.names)
                        if str(name).lower() in {"ticker", "symbol"}]
        if len(named_levels) == 1:
            level = named_levels[0]
            label = symbol_label(level)
            if label is None:
                raise ValueError(f"Downloaded data does not contain ticker {ticker}")
        else:
            candidates = []
            for index in range(2):
                label = symbol_label(index)
                if label is not None:
                    fields = downloaded.xs(label, axis=1, level=index).columns
                    if required.issubset({str(field).strip().lower() for field in fields}):
                        candidates.append((index, label))
            if len(candidates) != 1:
                raise ValueError(f"Cannot identify downloaded OHLC columns for ticker {ticker}")
            level, label = candidates[0]
        frame = downloaded.xs(label, axis=1, level=level).copy()
    elif len(tickers) == 1 and ticker.upper() == tickers[0].upper():
        frame = downloaded.copy()
    else:
        raise ValueError("Multiple tickers require downloaded columns grouped by ticker")

    frame.columns = [str(column).strip().lower().replace(" ", "_") for column in frame.columns]
    if frame.columns.has_duplicates:
        raise ValueError("Downloaded data has duplicate column names")
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Downloaded data is missing OHLC columns: {', '.join(sorted(missing))}")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("Downloaded prices must have a datetime index")
    if frame.index.tz is not None:
        frame.index = frame.index.tz_convert(None)
    frame.index.name = "date"
    # Different listing dates and trading calendars create empty rows in a
    # batch's outer join. A partially missing OHLC row still fails validation.
    frame = frame.dropna(subset=sorted(required), how="all")
    if frame.empty:
        raise ValueError(f"Yahoo Finance returned no price data for ticker {ticker}")
    if "volume" in frame:
        frame["volume"] = frame["volume"].fillna(0)
    return normalize_ohlcv(frame)


logger = logging.getLogger(__name__)

def prepare_data_x(x, window_size):
    import numpy as np

    # perform windowing
    n_row = x.shape[0] - window_size + 1
    output = np.lib.stride_tricks.as_strided(x, shape=(n_row,window_size), strides=(x.strides[0],x.strides[0]))
    return output[:-1], output[-1]

def prepare_data_y(x, window_size):
    # # perform simple moving average
    # output = np.convolve(x, np.ones(window_size), 'valid') / window_size

    # use the next day as label
    output = x[window_size:]
    return output

def prepare_data(normalized_data_close_price, config):
    window_size = config["data"]["window_size"]
    if normalized_data_close_price.shape[0] <= window_size:
        return None
        
    data_x, data_x_unseen = prepare_data_x(normalized_data_close_price, window_size=window_size)
    data_y = prepare_data_y(normalized_data_close_price, window_size=window_size)

    # split dataset

    split_index = int(data_y.shape[0]*config["data"]["train_split_size"])
    data_x_train = data_x[:split_index]
    data_x_val = data_x[split_index:]
    data_y_train = data_y[:split_index]
    data_y_val = data_y[split_index:]

    return split_index, data_x_train, data_y_train, data_x_val, data_y_val, data_x_unseen

def run_epoch(model, optimizer, criterion, scheduler, config, dataloader, is_training=False):
    epoch_loss = 0

    if is_training:
        model.train()
    else:
        model.eval()

    for idx, (x, y) in enumerate(dataloader):
        if is_training:
            optimizer.zero_grad()

        batchsize = x.shape[0]

        x = x.to(config["training"]["device"])
        y = y.to(config["training"]["device"])

        out = model(x)
        loss = criterion(out.contiguous(), y.contiguous())

        if is_training:
            loss.backward()
            optimizer.step()

        epoch_loss += (loss.detach().item() / batchsize)

    lr = scheduler.get_last_lr()[0]

    return epoch_loss, lr

class Normalizer():
    def __init__(self):
        self.mu = None
        self.sd = None

    def fit_transform(self, x):
        import numpy as np

        self.mu = np.mean(x, axis=(0), keepdims=True)
        self.sd = np.std(x, axis=(0), keepdims=True)
        normalized_x = (x - self.mu)/self.sd
        return normalized_x

    def inverse_transform(self, x):
        return (x*self.sd) + self.mu


def __getattr__(name):
    """Load the optional torch dataset only when its public class is requested."""
    if name != "TimeSeriesDataset":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    import numpy as np
    from torch.utils.data import Dataset

    class TimeSeriesDataset(Dataset):
        def __init__(self, x, y):
            x = np.expand_dims(x, 2) # in our case, we have only 1 feature, so we need to convert `x` into [batch, sequence, features] for LSTM
            self.x = x.astype(np.float32)
            self.y = y.astype(np.float32)

        def __len__(self):
            return len(self.x)

        def __getitem__(self, idx):
            return (self.x[idx], self.y[idx])

    # Keep the original public class identity for imports and pickling.
    TimeSeriesDataset.__qualname__ = "TimeSeriesDataset"
    globals()[name] = TimeSeriesDataset
    return TimeSeriesDataset
