import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from features.feature_list import FEATURE_COLS, LABEL_COL
from config.logging_conf import setup_logger
logger = setup_logger(__name__)

def scale_features(df):
    """
    cols 컬럼들 정규화 → 스케일링한 df 반환
    """
    if not isinstance(df, pd.DataFrame):
        raise logger.error("df must be pandas DataFrame")

    for c in FEATURE_COLS:
        if c not in df.columns:
            raise logger.error(f"{c} not in df columns")

    numerical_cols = df[FEATURE_COLS].select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df[FEATURE_COLS].select_dtypes(include=['object', 'category']).columns.tolist()

    logger.info("Numerical: {}".format( ','.join(numerical_cols)))
    logger.info("Categorical: {}".format(','.join(categorical_cols)))

    scaler = StandardScaler()
    num_scaled = scaler.fit_transform(df[numerical_cols])
    num_scaled_df = pd.DataFrame(num_scaled, columns=numerical_cols, index=df.index)

    ohe = OneHotEncoder(sparse_output=False, drop='first')
    cat_encoded = ohe.fit_transform(df[categorical_cols])
    cat_encoded_df = pd.DataFrame(cat_encoded, columns=ohe.get_feature_names_out(categorical_cols), index=df.index)
    df_processed = pd.concat([num_scaled_df, cat_encoded_df], axis=1)

    other_cols = [c for c in df.columns if c not in FEATURE_COLS]
    df_processed = pd.concat([df_processed, df[other_cols]], axis=1)
    return df_processed



