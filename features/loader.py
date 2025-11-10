import yfinance as yf
import pandas as pd
from config.logging_conf import setup_logger
logger = setup_logger(__name__)

def load_price(ticker, period="5y", interval="1d"):
    """
    yfinance 기반 가격데이터 로드
    index Datetime
    """
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval)
        if df.empty:
            raise logger.error(f'[ERROR] Data is empty. Ticker may be invalid: {ticker}')

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

        return df
    except Exception as e:
        raise logger.error(f"[ERROR] Failed to load ticker {ticker}: {e}") from e

