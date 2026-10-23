import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features


# =========================================================
# Base Dataset（统一 + TQNet + 你现在）
# =========================================================
class BaseDataset(Dataset):
    def __init__(
        self,
        root_path,
        data_path,
        flag,
        size,
        features,
        target,
        scale,
        timeenc,
        freq,
        cycle=24,
    ):
        if size is None:
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 96
        else:
            self.seq_len, self.label_len, self.pred_len = size

        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.cycle = cycle

        self.root_path = root_path
        self.data_path = data_path

        self.__read_data__()

    # =====================================================
    # 时间编码
    # =====================================================
    def _time_features(self, df_stamp):
        df_stamp = df_stamp.copy()

        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.dt.month
            df_stamp['day'] = df_stamp.date.dt.day
            df_stamp['weekday'] = df_stamp.date.dt.weekday
            df_stamp['hour'] = df_stamp.date.dt.hour
            return df_stamp.drop(['date'], axis=1).values
        else:
            # Models may use 't' as their minute-level embedding key, while
            # pandas 3 requires 'min' when parsing a frequency string.
            pandas_freq = 'min' if str(self.freq).lower() == 't' else self.freq
            return time_features(df_stamp, timeenc=1, freq=pandas_freq)

    # =====================================================
    # window
    # =====================================================
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len

        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        x = self.data_x[s_begin:s_end]
        y = self.data_y[r_begin:r_end]
        xm = self.data_stamp[s_begin:s_end]
        ym = self.data_stamp[r_begin:r_end]

        cycle = torch.tensor(self.cycle_index[s_end])

        return (
            torch.from_numpy(x).float(),
            torch.from_numpy(y).float(),
            torch.from_numpy(xm).float(),
            torch.from_numpy(ym).float(),
            cycle
        )

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def get_data_dim(self):
        return self.data_x.shape[1]

    def get_scaler(self):
        return self.scaler


# =========================================================
# ETTh
# =========================================================
class Dataset_ETT_hour(BaseDataset):
    def __read_data__(self):
        self.scaler = StandardScaler()
        df = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]

        border1, border2 = border1s[self.set_type], border2s[self.set_type]

        df_stamp = df[['date']].copy()
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])

        if self.features == 'S':
            df_data = df[[self.target]]
        else:
            df_data = df.iloc[:, 1:]

        data = df_data.values

        if self.scale:
            self.scaler.fit(data[border1s[0]:border2s[0]])
            data = self.scaler.transform(data)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = self._time_features(df_stamp.iloc[border1:border2].copy())

        self.cycle_index = (np.arange(len(data)) % self.cycle)[border1:border2]


# =========================================================
# ETTm
# =========================================================
class Dataset_ETT_minute(Dataset_ETT_hour):
    def __read_data__(self):
        self.scaler = StandardScaler()
        df = pd.read_csv(os.path.join(self.root_path, self.data_path))

        scale = 4
        border1s = [
            0,
            12 * 30 * 24 * scale - self.seq_len,
            12 * 30 * 24 * scale + 4 * 30 * 24 * scale - self.seq_len
        ]
        border2s = [
            12 * 30 * 24 * scale,
            12 * 30 * 24 * scale + 4 * 30 * 24 * scale,
            12 * 30 * 24 * scale + 8 * 30 * 24 * scale
        ]

        border1, border2 = border1s[self.set_type], border2s[self.set_type]

        df_stamp = df[['date']].copy()
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])

        if self.features == 'S':
            df_data = df[[self.target]]
        else:
            df_data = df.iloc[:, 1:]

        data = df_data.values

        if self.scale:
            self.scaler.fit(data[border1s[0]:border2s[0]])
            data = self.scaler.transform(data)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = self._time_features(df_stamp.iloc[border1:border2].copy())

        self.cycle_index = (np.arange(len(data)) % self.cycle)[border1:border2]


# =========================================================
# Custom / ECL / Traffic / Exchange（统一）
# =========================================================
class Dataset_Custom(BaseDataset):
    def __read_data__(self):
        self.scaler = StandardScaler()
        df = pd.read_csv(os.path.join(self.root_path, self.data_path))

        num_train = int(len(df) * 0.7)
        num_val = int(len(df) * 0.1)
        num_test = len(df) - num_train - num_val

        border1s = [0, num_train - self.seq_len, len(df) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_val, len(df)]

        border1, border2 = border1s[self.set_type], border2s[self.set_type]

        df_stamp = df[['date']].copy()
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])

        if self.features == 'S':
            df_data = df[[self.target]]
        else:
            df_data = df.drop(columns=['date'])

        data = df_data.values

        if self.scale:
            self.scaler.fit(data[border1s[0]:border2s[0]])
            data = self.scaler.transform(data)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = self._time_features(df_stamp.iloc[border1:border2].copy())

        self.cycle_index = (np.arange(len(data)) % self.cycle)[border1:border2]


# =========================================================
# AirQualityUCI（支持 xlsx / csv）
# =========================================================
class Dataset_Air(BaseDataset):
    def __read_data__(self):
        self.scaler = StandardScaler()
        file_path = os.path.join(self.root_path, self.data_path)

        if self.data_path.lower().endswith('.xlsx') or self.data_path.lower().endswith('.xls'):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)

        # -------------------------------------------------
        # 去掉全空列 / 常见无效列
        # -------------------------------------------------
        df = df.dropna(axis=1, how='all')

        unnamed_cols = [c for c in df.columns if str(c).lower().startswith('unnamed')]
        if len(unnamed_cols) > 0:
            df = df.drop(columns=unnamed_cols)

        # -------------------------------------------------
        # 构造 date 列
        # 兼容：
        # 1) 已有 date 列
        # 2) 有 Date 和 Time 两列
        # -------------------------------------------------
        cols_lower = {str(c).lower(): c for c in df.columns}

        if 'date' in cols_lower:
            date_col = cols_lower['date']
            df['date'] = pd.to_datetime(df[date_col], errors='coerce')
        elif 'datetime' in cols_lower:
            dt_col = cols_lower['datetime']
            df['date'] = pd.to_datetime(df[dt_col], errors='coerce')
        elif 'date' in df.columns and 'time' in df.columns:
            df['date'] = pd.to_datetime(
                df['Date'].astype(str) + ' ' + df['Time'].astype(str),
                errors='coerce'
            )
        elif 'Date' in df.columns and 'Time' in df.columns:
            df['date'] = pd.to_datetime(
                df['Date'].astype(str) + ' ' + df['Time'].astype(str),
                errors='coerce'
            )
        else:
            # 如果完全没有时间列，就人工构造
            df['date'] = pd.date_range(start='2000-01-01', periods=len(df), freq=self.freq)

        # -------------------------------------------------
        # 删除原始时间字符串列，避免进入特征
        # -------------------------------------------------
        drop_cols = []
        for c in df.columns:
            if c == 'date':
                continue
            if str(c).lower() in ['date', 'time', 'datetime']:
                drop_cols.append(c)
        if len(drop_cols) > 0:
            df = df.drop(columns=drop_cols)

        # -------------------------------------------------
        # 缺失值处理：AirQualityUCI 里常见 -200 表示缺失
        # -------------------------------------------------
        df = df.replace(-200, np.nan)
        df = df.replace(-200.0, np.nan)

        # 仅保留数值列 + date
        feature_cols = []
        for c in df.columns:
            if c == 'date':
                continue
            df[c] = pd.to_numeric(df[c], errors='coerce')
            feature_cols.append(c)

        df = df[['date'] + feature_cols]

        # 按时间排序
        df = df.sort_values('date').reset_index(drop=True)

        # 去掉无法解析时间的行
        df = df.dropna(subset=['date']).reset_index(drop=True)

        # 先做一个简单插值，再前后填充
        for c in feature_cols:
            df[c] = df[c].interpolate(method='linear', limit_direction='both')
            df[c] = df[c].ffill().bfill()

        # 如果还有全空列，删掉
        valid_feature_cols = [c for c in feature_cols if not df[c].isna().all()]
        df = df[['date'] + valid_feature_cols]

        if len(valid_feature_cols) == 0:
            raise ValueError("Air dataset has no valid numeric feature columns after preprocessing.")

        num_train = int(len(df) * 0.7)
        num_val = int(len(df) * 0.1)
        num_test = len(df) - num_train - num_val

        border1s = [0, num_train - self.seq_len, len(df) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_val, len(df)]

        border1, border2 = border1s[self.set_type], border2s[self.set_type]

        df_stamp = df[['date']].copy()
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])

        if self.features == 'S':
            if self.target not in df.columns:
                raise ValueError(f"Target column '{self.target}' not found in air dataset.")
            df_data = df[[self.target]]
        else:
            df_data = df.drop(columns=['date'])

        data = df_data.values.astype(np.float32)

        if self.scale:
            self.scaler.fit(data[border1s[0]:border2s[0]])
            data = self.scaler.transform(data)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = self._time_features(df_stamp.iloc[border1:border2].copy())

        self.cycle_index = (np.arange(len(data)) % self.cycle)[border1:border2]


# =========================================================
# Abilene / GEANT / CERNET: origin-destination traffic CSVs
# =========================================================
class Dataset_InternetTraffic(BaseDataset):
    """No calendar features; 7:1:2 split, train-only scaling, gap-safe windows.

    CSV values are preserved: missing/non-finite values raise an error rather
    than being imputed. All OD columns (including zero-valued flows) are kept.
    Validation/test histories may use previous split rows; future targets stay
    entirely within their own split. No window crosses a timestamp gap.
    Timestamps are used only to validate ordering and window continuity.
    x_mark/y_mark have four all-zero columns for the existing training API.
    """

    def __init__(self, *args, expected_interval=None, network_cycle=None, **kwargs):
        self.expected_interval = expected_interval
        self.network_cycle = network_cycle
        super().__init__(*args, **kwargs)

    def __read_data__(self):
        if self.seq_len <= 0 or self.pred_len <= 0:
            raise ValueError('seq_len and pred_len must be positive.')
        if not 0 <= self.label_len <= self.seq_len:
            raise ValueError('label_len must lie between 0 and seq_len.')
        if self.features not in ('M', 'S', 'MS'):
            raise ValueError("features must be 'M', 'S', or 'MS'.")

        file_path = os.path.join(self.root_path, self.data_path)
        df = pd.read_csv(file_path)
        time_cols = [c for c in df.columns
                     if str(c).lower() in ('time', 'date', 'datetime', 'timestamp')]
        if len(time_cols) != 1:
            raise ValueError(f'{file_path}: expected exactly one timestamp column.')
        time_col = time_cols[0]
        text_dates = df[time_col].astype(str)
        if text_dates.str.fullmatch(r'\d{4}-\d{2}-\d{2}-\d{2}-\d{2}').all():
            dates = pd.to_datetime(text_dates, format='%Y-%m-%d-%H-%M', errors='raise')
        else:
            dates = pd.to_datetime(text_dates, errors='raise')
        if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
            raise ValueError(f'{file_path}: timestamps must be valid, unique, and increasing.')
        if len(df) < 2:
            raise ValueError(f'{file_path}: not enough rows.')

        # Do not silently drop columns or fill missing observations.
        feature_cols = [c for c in df.columns if c != time_col]
        if not feature_cols:
            raise ValueError(f'{file_path}: no traffic columns.')
        values = df[feature_cols].apply(pd.to_numeric, errors='raise').to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f'{file_path}: traffic contains missing or non-finite values.')
        if self.features in ('S', 'MS'):
            if self.target not in feature_cols:
                raise ValueError(f"Target '{self.target}' not found. Use an OD column name; "
                                 f"for example '{feature_cols[0]}'.")
            # Common forecasting engines select the final channel for MS.
            feature_cols = ([self.target] if self.features == 'S' else
                            [c for c in feature_cols if c != self.target] + [self.target])
            values = df[feature_cols].to_numpy(dtype=np.float64)
        self.feature_names = feature_cols

        deltas = dates.diff().iloc[1:]
        interval = (pd.Timedelta(self.expected_interval) if self.expected_interval is not None
                    else deltas.mode().iloc[0])
        if interval <= pd.Timedelta(0):
            raise ValueError('The sampling interval must be positive.')
        self.sampling_interval = interval

        total = len(df)
        num_train, num_val = int(total * 0.7), int(total * 0.1)
        split_starts = [0, num_train, num_train + num_val]
        split_ends = [num_train, num_train + num_val, total]
        split_start, split_end = split_starts[self.set_type], split_ends[self.set_type]
        if num_train < self.seq_len + self.pred_len:
            raise ValueError(f'{file_path}: training split is too short for the requested window.')
        border1 = max(0, split_start - self.seq_len) if self.set_type else 0
        border2 = split_end
        self.split_range = (split_start, split_end)
        self.data_offset = border1
        self.scaler = StandardScaler()
        if self.scale:
            self.scaler.fit(values[:num_train])
            values = self.scaler.transform(values)

        self.data_x = values[border1:border2]
        self.data_y = self.data_x
        # No month/day/weekday/hour/minute values are supplied to the model.
        # Four zero placeholders match infer_time_dim('h', ...) in train.py.
        self.data_stamp = np.zeros((len(self.data_x), 4), dtype=np.float32)

        # Preserve BaseDataset's existing cycle by default for model compatibility.
        # A custom network_cycle must also match the consuming model's setting.
        if self.network_cycle is not None:
            self.cycle = int(self.network_cycle)
            if self.cycle <= 0:
                raise ValueError('network_cycle must be positive.')
        # Preserve the row-position cycle used by the original TXT/PEMS loaders;
        # this is a sequence index, not a calendar-time feature.
        self.cycle_index = (np.arange(total) % self.cycle)[border1:border2]

        local_dates = dates.iloc[border1:border2]
        gaps = np.zeros(len(local_dates), dtype=np.int64)
        gaps[1:] = (local_dates.diff().iloc[1:] != interval).to_numpy(dtype=np.int64)
        cumulative_gaps = np.cumsum(gaps)
        window_len = self.seq_len + self.pred_len
        candidate_count = max(0, len(self.data_x) - window_len + 1)
        starts = np.arange(candidate_count, dtype=np.int64)
        ends = starts + window_len - 1
        # Exclude gaps strictly inside each complete input + forecast window.
        valid = (cumulative_gaps[ends] - cumulative_gaps[starts]) == 0
        # Explicitly ensure that forecast targets remain in the owning split.
        valid &= starts + border1 + self.seq_len >= split_start
        self.window_starts = starts[valid]
        self.gap_count = int((deltas != interval).sum())
        self.skipped_windows = int(candidate_count - len(self.window_starts))
        if not len(self.window_starts):
            raise ValueError(f'{file_path}: no continuous forecasting windows in split '
                             f'{self.set_type}; reduce seq_len/pred_len or inspect timestamp gaps.')
        flag = ('train', 'val', 'test')[self.set_type]
        print(f'[INFO] InternetTraffic={self.data_path} split={flag} '
              f'interval={interval} gaps={self.gap_count} '
              f'windows={len(self.window_starts)} skipped={self.skipped_windows} '
              f'cycle={self.cycle} time_features=disabled')

    def __getitem__(self, index):
        # Keep the existing (x, y, x_mark, y_mark, cycle) return interface.
        return super().__getitem__(int(self.window_starts[index]))

    def __len__(self):
        return len(self.window_starts)




class Dataset_Service(BaseDataset):
    """BizITObs-Service KPI forecasting from the official single Service.csv.

    The full file has 72 KPIs and 35 IT-event alerts. Only KPI columns are
    inputs and targets here; alerts are excluded from both. This is a
    KPI-only forecasting protocol, not the papers' exogenous-alert setting.
    Split rows chronologically 6:2:2 and fit scaling on training rows only.
    No filling, sorting, resampling, or calendar features are applied.
    """

    KPI_SUFFIXES = (':calls.sum', ':erroneousCalls.sum',
                    ':latency.mean', ':latency.sum')

    def __init__(self, *args, network_cycle=None, **kwargs):
        self.network_cycle = network_cycle
        super().__init__(*args, **kwargs)

    def __read_data__(self):
        if self.seq_len <= 0 or self.pred_len <= 0:
            raise ValueError('seq_len and pred_len must be positive.')
        if not 0 <= self.label_len <= self.seq_len:
            raise ValueError('label_len must lie between 0 and seq_len.')
        if self.features not in ('M', 'S', 'MS'):
            raise ValueError("features must be 'M', 'S', or 'MS'.")

        file_path = os.path.join(self.root_path, self.data_path)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(
                f'Service CSV not found: {file_path}. Download the official '
                'BizITObs Complete/Time Series/RobotShop/Service.csv and '
                'place it at data/Service.csv; no download runs during training.')
        df = pd.read_csv(file_path)
        time_cols = [c for c in df.columns if str(c).lower() in
                     ('time', 'date', 'datetime', 'timestamp')]
        # Also accept an export whose datetime index is the first unnamed column.
        if not time_cols and len(df.columns) and str(df.columns[0]) == 'Unnamed: 0':
            time_cols = [df.columns[0]]
        if len(time_cols) != 1:
            raise ValueError(f'{file_path}: expected one timestamp column or '
                             'a leading Unnamed: 0 datetime index.')
        dates = pd.to_datetime(df[time_cols[0]], errors='raise')
        if (len(df) < 2 or dates.isna().any() or dates.duplicated().any()
                or not dates.is_monotonic_increasing):
            raise ValueError(f'{file_path}: timestamps must be valid, unique, '
                             'and increasing in source order.')
        self.sampling_interval = pd.Timedelta(seconds=10)
        if not (dates.diff().iloc[1:] == self.sampling_interval).all():
            raise ValueError(f'{file_path}: expected continuous 10-second '
                             'sampling; no timestamp gaps are filled.')

        feature_cols = [c for c in df.columns
                        if str(c).endswith(self.KPI_SUFFIXES)]
        if not feature_cols:
            raise ValueError(f'{file_path}: no Service KPI columns found. '
                             'Use the official CSV with its original column names.')
        if self.features in ('S', 'MS'):
            if self.target not in feature_cols:
                raise ValueError(f"Service target '{self.target}' is not a KPI. "
                                 f"Use features='M' to predict all KPIs, or a "
                                 f"KPI target such as '{feature_cols[0]}'.")
            feature_cols = ([self.target] if self.features == 'S' else
                            [c for c in feature_cols if c != self.target] + [self.target])
        self.feature_names = feature_cols
        values = (df[feature_cols].apply(pd.to_numeric, errors='raise')
                  .to_numpy(dtype=np.float64))
        if not np.isfinite(values).all():
            raise ValueError(f'{file_path}: KPI values contain missing or '
                             'non-finite observations; no filling is applied.')

        total = len(df)
        num_train, num_val = int(total * 0.6), int(total * 0.2)
        num_test = total - num_train - num_val
        if num_train < self.seq_len + self.pred_len:
            raise ValueError(f'Service training split has {num_train} rows: '
                             'seq_len + pred_len exceeds its length.')
        if self.pred_len > min(num_val, num_test):
            raise ValueError(f'Service validation/test splits have {num_val}/'
                             f'{num_test} rows: reduce pred_len.')
        split_starts = [0, num_train, num_train + num_val]
        split_ends = [num_train, num_train + num_val, total]
        split_start = split_starts[self.set_type]
        split_end = split_ends[self.set_type]
        border1 = split_start - self.seq_len if self.set_type else 0
        self.split_range = (split_start, split_end)
        self.data_offset = border1

        self.scaler = StandardScaler()
        if self.scale:
            self.scaler.fit(values[:num_train])
            values = self.scaler.transform(values)
        # Validation/test inputs may use earlier rows. All forecast targets
        # remain within their own split, with no future rows in the inputs.
        self.data_x = values[border1:split_end]
        self.data_y = self.data_x
        self.data_stamp = np.zeros((len(self.data_x), 4), dtype=np.float32)

        if self.network_cycle is not None:
            self.cycle = int(self.network_cycle)
        if self.cycle <= 0:
            raise ValueError('cycle must be positive.')
        # Preserve the existing row-index cycle for model compatibility.
        self.cycle_index = np.arange(border1, split_end) % self.cycle
        flag = ('train', 'val', 'test')[self.set_type]
        print(f'[INFO] Service={self.data_path} split={flag} '
              f'interval={self.sampling_interval} input_dim={len(feature_cols)} '
              f'target_rows=[{split_start},{split_end}) windows={len(self)} '
              f'cycle={self.cycle} time_features=disabled')


class Dataset_TXT(BaseDataset):
    def __read_data__(self):
        self.scaler = StandardScaler()
        data = np.loadtxt(os.path.join(self.root_path, self.data_path), delimiter=',')

        T = len(data)
        num_test = int(T * 0.2)
        num_val = int(T * 0.1)
        num_train = T - num_val - num_test

        border1s = [0, num_train - self.seq_len, T - num_test - self.seq_len]
        border2s = [num_train, num_train + num_val, T]
        border1, border2 = border1s[self.set_type], border2s[self.set_type]

        if self.scale:
            self.scaler.fit(data[border1s[0]:border2s[0]])
            data = self.scaler.transform(data)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = np.zeros((border2 - border1, 1), dtype=np.float32)

        self.cycle_index = (np.arange(len(data)) % self.cycle)[border1:border2]


# =========================================================
# Solar / TXT
# =========================================================
class Dataset_Solar(BaseDataset):
    def __read_data__(self):
        self.scaler = StandardScaler()

        data = np.loadtxt(os.path.join(self.root_path, self.data_path), delimiter=',')

        num_train = int(len(data) * 0.7)
        num_val = int(len(data) * 0.1)
        num_test = len(data) - num_train - num_val

        border1s = [0, num_train - self.seq_len, len(data) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_val, len(data)]

        border1, border2 = border1s[self.set_type], border2s[self.set_type]

        if self.scale:
            self.scaler.fit(data[border1s[0]:border2s[0]])
            data = self.scaler.transform(data)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        self.data_stamp = np.zeros((len(self.data_x), 1), dtype=np.float32)

        self.cycle_index = (np.arange(len(data)) % self.cycle)[border1:border2]


# =========================================================
# PEMS
# =========================================================
class Dataset_PEMS(BaseDataset):
    def __read_data__(self):
        self.scaler = StandardScaler()

        data = np.load(os.path.join(self.root_path, self.data_path))['data']

        if data.ndim == 3:
            data = data[:, :, 0]

        num_train = int(len(data) * 0.6)
        num_val = int(len(data) * 0.2)
        num_test = len(data) - num_train - num_val

        border1s = [0, num_train - self.seq_len, len(data) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_val, len(data)]

        border1, border2 = border1s[self.set_type], border2s[self.set_type]

        if self.scale:
            self.scaler.fit(data[border1s[0]:border2s[0]])
            data = self.scaler.transform(data)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        self.data_stamp = np.zeros((len(self.data_x), 1), dtype=np.float32)

        self.cycle_index = (np.arange(len(data)) % self.cycle)[border1:border2]
