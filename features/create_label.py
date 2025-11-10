import pandas as pd
import numpy as np
from config.logging_conf import setup_logger
logger = setup_logger(__name__)

def create_label(df, target_column='Close', short_ma=7, long_ma=20, window=3 ):
    """
    mv_spread 기반 directional label 생성
    """
    if target_column not in df.columns:
        raise logger.error(f'DataFrame does not contain {target_column} column')

    if len(df) < 30:
        raise logger.error(f'DataFrame only has {len(df)} rows')

    # 이동평균
    df['MA_short'] = df[target_column].rolling(short_ma, min_periods=1).mean()
    df['MA_long'] = df[target_column].rolling(long_ma, min_periods=1).mean()

    # mv_spread와 변화량
    df['mv_spread'] = df['MA_short'] - df['MA_long']
    df['mv_diff'] = df['mv_spread'].diff()

    #window = 3  # 최근 3일 평균 변화
    df['mv_diff_mean'] = df['mv_spread'].diff().rolling(window).mean()
    df['trend'] = np.where(df['mv_diff_mean'] > 0, '상승', '하락')

    return df

