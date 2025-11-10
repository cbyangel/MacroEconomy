import pandas as pd
import numpy as np

# ==== modules ====
from features.data.tickers import dict_tickers
from features.loader import load_price
from features.indicators import add_all_indicators
from features.create_label import create_label
from features.feature_list import FEATURE_COLS, LABEL_COL
from preprocessing.scaler import scale_features
from training.train import run_train
from training.inference import run_inference
from utils.slack_notifier import send_slack
from config.logging_conf import setup_logger
logger = setup_logger(__name__)

# main.py 에서 하는 일
#	1.	loader 로 가격 로드
#	2.	indicators 계산
#	3.	label 생성
#	4.	normalization 적용된 df 로 만들기
#	5.	train 실행 → model.pth 저장
#	6.	inference 실행 → 오늘자 예측 출력

# ==== config ====
#ticker = "360750.KS"     # 예측할 ETF


def main():
    dict_result = {'ticker_nm':[], 'accuracy':[], 'f1':[], 'pred':[]}
    for ticker in dict_tickers.keys():
        # 1) load
        df = load_price(ticker)
        logger.info(f"1. 데이터 로드 완료: {len(df)} rows")

        # 2) indicators
        df = add_all_indicators(df)
        logger.info("2. 지표 계산 완료")
        logger.info(df.columns.tolist())

        # 3) label
        df = create_label(df)
        logger.info("3. 지표 계산 완료")

        # 4) drop NaN (MA; MACD 초반 구간 제거)
        df = df.dropna().copy()

        # 5) scaling
        df = scale_features(df)
        logger.info("4. 정규화 완료")

        # 6) Train
        logger.info("\n========= 5. TRAINING START =========")
        model, metrics = run_train(df, FEATURE_COLS, LABEL_COL, dict_tickers[ticker])
        logger.info("========= 5. TRAIN END =========")


        # 7) TODAY inference
        logger.info("\n========= 6. TODAY PREDICT =========")
        pred = run_inference(df, FEATURE_COLS, dict_tickers[ticker])
        logger.info(f"오늘 예측 결과 → {pred}\n")

        dict_result['ticker_nm'].append(dict_tickers[ticker])
        dict_result['accuracy'].append(metrics['accuracy'])
        dict_result['f1'].append(metrics['f1_macro'])
        dict_result['pred'].append(pred[0])

    df_result = pd.DataFrame(dict_result)
    text = df_result[['ticker_nm', 'accuracy', 'pred']].to_string(index=False)
    send_slack(text)

if __name__ == "__main__":
    main()



