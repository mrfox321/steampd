"""
Op represents the interface for batching + streaming transformations against

1. dataframe
2. stream of imputs
"""
from collections import deque
from typing import Protocol, List, Any, Callable, Optional, Dict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class Op(Protocol):
    def apply(self, series: pd.Series) -> pd.Series:
        ...

    def push(self, value: float) -> float:
        ...

    @property
    def value(self) -> float:
        ...


class ListOp(Protocol):
    def apply(self, series: pd.Series) -> List[pd.Series]:
        ...

    def push(self, value: float) -> List[float]:
        ...

    @property
    def value(self) -> List[float]:
        ...


@dataclass
class Id:
    # aggregator
    val: float = float('nan')

    def apply(self, series: pd.Series) -> pd.Series:
        return series

    def push(self, value: float) -> float:
        self.val = value
        return value

    @property
    def value(self) -> float:
        return self.val


@dataclass
class Lag:
    windows: List[int]
    # aggregator
    values: deque[float] = field(init=False)

    def __post_init__(self) -> None:
        assert len(self.windows) > 0
        max_lag = max(self.windows)
        self.values = deque([float('nan')] * max_lag, maxlen=max_lag)

    def apply(self, series: pd.Series) -> List[pd.Series]:
        return [series.shift(window) for window in self.windows]

    def push(self, value: float) -> List[float]:
        self.values.append(value)
        return self.value

    @property
    def value(self) -> List[float]:
        return [self.values[-window] for window in self.windows]


@dataclass
class Delta:
    # aggregator
    values: List[Any] = field(init=False, default_factory=lambda: [float('nan')] * 2)
    parity: bool = False

    def apply(self, series: pd.Series) -> pd.Series:
        return series.diff()

    def push(self, value: float) -> float:
        next_parity = not self.parity
        cur, prev = int(self.parity), int(next_parity)
        self.values[cur] = value
        self.parity = next_parity
        return value - self.values[prev]

    @property
    def value(self) -> float:
        cur, prev = int(self.parity), int(not self.parity)
        return self.values[cur] - self.values[prev]


@dataclass
class Mapper:
    fn: Callable
    # aggregator
    val: float = float('nan')

    def apply(self, series: pd.Series) -> pd.Series:
        return self.fn(series)

    def push(self, value: float) -> float:
        self.val = self.fn(value)
        return self.value

    @property
    def value(self) -> float:
        return self.val


@dataclass
class Ewm:
    alpha: float
    # aggregator
    ewm: float = 0.0
    empty: bool = True

    def apply(self, series: pd.Series) -> pd.Series:
        return series.ewm(alpha=self.alpha, adjust=False).mean()

    def push(self, value: float) -> float:
        if self.empty:
            self.ewm = value
            self.empty = False
        else:
            self.ewm = (1.0 - self.alpha) * self.ewm + self.alpha * value
        return self.ewm

    @property
    def value(self) -> float:
        return self.ewm


@dataclass
class RollingVar:
    window: int  # width of rolling calculation
    # accumulators
    squares: np.ndarray = field(init=False)  # buffer of squared values
    means: np.ndarray = field(init=False)  # buffer of values
    acc_squares: float = 0.0  # rolling sum of squares
    acc_sum: float = 0.0  # rolling sum
    cnt: int = field(init=False, default=0)  # incremement count

    def __post_init__(self):
        self.squares = np.zeros(self.window)
        self.means = np.zeros(self.window)

    def apply(self, series: pd.Series) -> pd.Series:
        return series.rolling(self.window).var()

    def push(self, X: float) -> float:
        if self.cnt >= self.window:
            self.acc_squares -= self.squares[self.ptr]
            self.acc_sum -= self.means[self.ptr]
        # nan's will permanently corrupt accumulators
        if np.isnan(X):
            X = 0.0

        X_pow = X ** 2

        self.acc_squares += X_pow
        self.acc_sum += X

        self.squares[self.ptr] = X_pow
        self.means[self.ptr] = X

        self.cnt += 1

        return self.var

    @property
    def value(self) -> float:
        return self.var

    @property
    def num(self) -> int:
        return min(self.cnt, self.window)

    @property
    def mean(self) -> float:
        return self.acc_sum / self.num

    @property
    def var(self) -> float:
        """
        Unbiased estimator of variance
        """
        ddof = 1
        N = self.num
        if N < self.window:
            return float('nan')
        return (self.acc_squares - self.acc_sum ** 2 / N) / (N - ddof)

    @property
    def std(self) -> float:
        """
        Unbiased estimator of standard deviation
        """
        return np.sqrt(self.var)

    @property
    def ptr(self) -> int:
        return self.cnt % self.window


@dataclass
class Node:
    name: str
    op: Op = field(default_factory=lambda: Id())
    parent: Optional['Node'] = field(default=None)
    children: List['Node'] = field(init=False, default_factory=list)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.apply_node(df)
        for child in self.children:
            df = child.apply(df)
        return df

    def apply_node(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.parent:
            df[self.name] = self.op.apply(df[self.parent.name])
        return df

    def push(self, value: float, values: Dict[str, float]) -> Dict[str, float]:
        value = self.push_node(value, values)
        for child in self.children:
            values = child.push(value, values)
        return values

    def push_node(self, value: float, values: Dict[str, float]) -> float:
        value =  self.op.push(value)
        values[self.name] = value
        return value

    def lag(self, windows: List[int]) -> 'Node':
        name = f'{self.name}_lag'
        return self._create_list(name, Lag(windows))

    def delta(self) -> 'Node':
        name = f'{self.name}_delta'
        return self._create(name, Delta())

    def mapper(self, fn: Callable, name: str) -> 'Node':
        name = f'{self.name}_{name}'
        return self._create(name, Mapper(fn))

    def rolling_var(self, window: int) -> 'Node':
        name = f'{self.name}_rolling_var_{window}'
        return self._create(name, RollingVar(window))

    def ewm(self, alpha: float) -> 'Node':
        name = f'{self.name}_ewa_{alpha}'
        return self._create(name, Ewm(alpha))

    def _create(self, name, op: Op) -> 'Node':
        node = Node(name, op, self)
        self.children.append(node)
        return node

    def _create_list(self, name, op: ListOp) -> 'ListNode':
        node = ListNode(name, op, self)
        self.children.append(node)
        return node


"""
Groups of nodes (Lag features) share a transformation

e.g.

1. Shift operations (multiple shifts)
"""


@dataclass
class ListNode(Node):
    op: ListOp

    def apply_node(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.parent:
            series_list = self.op.apply(df[self.parent.name])
            for i, series in enumerate(series_list):
                df[f'{self.name}_{i}'] = series
        return df

    def push_node(self, value: float, values: Dict[str, float]) -> Dict[str, float]:
        values_list = self.op.push(value)
        for i, val in enumerate(values_list):
            values[f'{self.name}_{i}'] = val
        return values


"""
Simple declarative set of transformations against a set of signals.

Derived features are there transformed like:

for node in derived_transforms(...):
    df = node.apply(df)
"""


def derived_transforms(
        signals: List[str],
        lags: List[int],
        alphas: List[float],
        alphas_lags: List[int],
        rolling_vars: List[int],
        rolling_vars_lags: List[int],
        delta_rolling_vars: List[int],
        delta_rolling_vars_lags: List[int]
) -> List[Node]:
    """
    Lazily compute:

    1. ewma log(price) for various alphas
    2. historical time series of log(p_i) - log(p_{i-1})
    3. rolling variance of log(price) for various window sizes
    """
    nodes = [Node(signal) for signal in signals]
    for node in nodes:
        node_log = node.mapper(np.log, 'log')
        node_log_delta = node_log.delta()
        # last observed ewm log(price)
        for alpha in alphas:
            node_log.ewm(alpha).lag(alphas_lags)
        # historical movements
        node_log_delta.lag(lags)

        # last observed rolling variance
        # values
        for var_window in rolling_vars:
            node_log.rolling_var(var_window).lag(rolling_vars_lags)
        # deltas
        for var_window in delta_rolling_vars:
            node_log_delta.rolling_var(var_window).lag(delta_rolling_vars_lags)
    return nodes
