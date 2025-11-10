import pandas as pd
import numpy as np
import warnings
from pandas.api import types as ptypes
from config.logging_conf import setup_logger
logger = setup_logger(__name__)

def _check_required_cols(df, cols, raise_on_error=True):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        msg = f'Missing columns for indicators: {missing}'
        if raise_on_error:
            raise logger.error(msg)
        warnings.warn(msg)
        return False
    return True


def _ensure_numeric(df, cols, raise_on_error=True):
    for c in cols:
        if not ptypes.is_numeric_dtype(df[c]):
            msg = f"Column {c} is not numeric"
            if raise_on_error:
                raise logger.error(msg)
            warnings.warn(msg)
            return False
    return True


# ----------------------------
# 이동평균 (EMA)
# ----------------------------
def add_ema(df, target_column='Close', windows=[7,20,60]):
    if not _check_required_cols(df, [target_column], raise_on_error=True): return df
    if not _ensure_numeric(df, [target_column], raise_on_error=True): return df
    for w in windows:
        df[f"EMA{w}"] = df[target_column].ewm(span=w, adjust=False).mean()
    return df

# ----------------------------
# RSI
# ----------------------------
def add_rsi(df, target_column='Close', period=14, raise_on_error=True):
    if not _check_required_cols(df, [target_column], raise_on_error=True): return df
    if len(df) < period + 1:
        msg = f"Not enough rows for RSI (need >={period+1}), got {len(df)}"
        if raise_on_error: raise logger.error(msg)
        warnings.warn(msg)
        return df
    delta = df[target_column].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(period).mean()
    roll_down = down.rolling(period).mean()
    RS = roll_up / roll_down
    df['RSI'] = 100 - (100 / (1 + RS))
    return df

# ----------------------------
# MACD (EMA차이로 골든/데드 크로스 판단)
# ----------------------------
def add_macd(df, target_column='Close', fast=12, slow=26, signal=9, raise_on_error=True):
    if not _check_required_cols(df, [target_column], raise_on_error): return df
    if len(df) < max(fast, slow):
        msg = f"Not enough rows for MACD (need >={max(fast, slow)}), got {len(df)}"
        if raise_on_error: raise logger.error(msg)
        warnings.warn(msg);
        return df
    ema_fast = df[target_column].ewm(span=fast, adjust=False).mean()
    ema_slow = df[target_column].ewm(span=slow, adjust=False).mean()
    df['MACD'] = ema_fast - ema_slow
    df['MACD_signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    return df

# ----------------------------
# Bollinger Bands (가격 변동 범위를 시각화, 밴드 밖으로 나가면 과매수/과매도 신호)
# ----------------------------
# 중간선: 20일 단순이동평균(MA20)
# 상단 밴드: MA20 + 2σ, 하단 밴드: MA20 - 2σ
def add_bollinger_bands(df, target_column='Close', k_period=20, ):
    df[f'MA{k_period}'] = df[target_column].rolling(k_period).mean()
    df['BB_upper'] = df[f'MA{k_period}'] + 2*df[target_column].rolling(k_period).std()
    df['BB_lower'] = df[f'MA{k_period}'] - 2*df[target_column].rolling(k_period).std()
    df['BB_width'] = df['BB_upper'] - df['BB_lower']
    return df


# ----------------------------
# ATR (Average True Range) ==> 변동성 지표, 값이 클수록 가격이 급격히 움직임
# ----------------------------
# True Range(TR) = 하루 변동폭 계산으로 리스크 판단
# high_low : 오늘 고가-저가, high_close : 오늘 고가-어제 종가, low_close : 오늘 저가-어제 종가
# 셋 중 절댓값 최대값 사용
def add_atr(df, target_column='Close', k_period=14, raise_on_error=True ):
    if not _check_required_cols(df, ['High', 'Low', 'Close'], raise_on_error): return df
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df[target_column].shift())
    low_close = np.abs(df['Low'] - df[target_column].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR14'] = tr.rolling(k_period).mean()
    return df


# ----------------------------
# Stochastic (%K, %D)
# ----------------------------
# 최근 14일 범위에서 종가 위치를 비율로 계산 → %K
# %D = %K의 3일 이동평균
# 의미: 과매수/과매도 판단
# %K > 0.8 → 과매수
# %K < 0.2 → 과매도
def add_stoch(df, target_column='Close', k_period=14, d_period=3, raise_on_error=True):
    if not _check_required_cols(df, ['High', 'Low', 'Close'], raise_on_error): return df
    if len(df) < k_period:
        msg = f"Not enough rows for Stochastic (need >={k_period}), got {len(df)}"
        if raise_on_error: raise logger.error(msg)
        warnings.warn(msg);
        return df
    low_min = df['Low'].rolling(k_period).min()
    high_max = df['High'].rolling(k_period).max()
    df['Stoch_K'] = (df[target_column] - low_min) / (high_max - low_min) * 100
    df['Stoch_D'] = df['Stoch_K'].rolling(d_period).mean()
    return df



def add_all_indicators(df):
    df = add_ema(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_atr(df)
    df = add_stoch(df)
    return df