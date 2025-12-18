#%%
import FinanceDataReader as fdr
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
import pickle, time, os
import argparse
from pathlib import Path
from tqdm import tqdm


# 중간 저장용 함수
def save_chunk(data, chunk_idx, where):
    with open(f'pkl_{where}/{where}_chunk_{chunk_idx}.pkl', 'wb') as f:
        pickle.dump(data, f)


def get_data_from_yf(list_tickers, where):
    chunk_size = 100
    for chunk_idx in range(0, len(list_tickers), chunk_size):
        chunk = list_tickers[chunk_idx:chunk_idx+chunk_size]
        chunk_data = {}

        for ticker in tqdm(chunk, desc=f'Processing chunk {chunk_idx//chunk_size + 1}'):
            retries = 2
            while retries > 0:
                try:
                    tkr = yf.Ticker(ticker)

                    # 기본 재무제표, 배당 등
                    balance_sheet = tkr.quarterly_balance_sheet.T
                    financials = tkr.quarterly_financials.T
                    dividends = tkr.dividends
                    info = tkr.info
                    cashflow = tkr.quarterly_cashflow.T
                    incomestmt = tkr.quarterly_income_stmt.T
                    history = tkr.history(period="1y", interval="1d")

                    chunk_data[ticker] = {
                        'balance_sheet': balance_sheet,
                        'financials': financials,
                        'dividends': dividends,
                        'cashflow' : cashflow,
                        'incomestmt' : incomestmt,
                        'info': info,
                        'history': history
                    }
                    break
                except Exception as e:
                    print(f"{ticker} failed with error: {e}, retries left: {retries-1}")
                    retries -= 1
                    time.sleep(1)
            time.sleep(0.1)

        # chunk 저장
        save_chunk(chunk_data, chunk_idx//chunk_size + 1, where)


def calc_indicator(dict_us_stocks, list_tickers, str_type, return_df=False):

    def safe_div(a, b, alt=0.0):
        try:
            if b is None or pd.isna(b) or b == 0:
                return alt
            return a / b
        except Exception:
            return alt

    def get_col(df, col, fill = 0.0):
        if col in df.columns:
            return df[col].fillna(0)
        else:
            return pd.Series([fill] * len(df), index=df.index)

    def recent_and_past(s):
        if s is None or len(s) == 0:
            return 0.0, 0.0
        try:
            return float(s.iloc[0]), float(s.iloc[-1])
        except Exception:
            try:
                r = s.iloc[0]
                p = s.iloc[-1]
                r = float(r) if not pd.isna(r) else 0.0
                p = float(p) if not pd.isna(p) else 0.0
                return r, p
            except Exception:
                return 0.0, 0.0

    dict_tickers_summarize = {
        'Symbol':[], 'sector':[], 'industry':[], 'market_cap':[], 'price':[], 'eps':[], 'pbs':[], 'per':[], 'pbr':[], 'psr':[], 'roe':[],
        'net_income':[],  #'ev/ebitda':[], 'freecashflow':[],
        'is_share_reduced':[], 'is_share_same':[], 'is_eps_inc':[], 'is_revenue_inc':[], 'is_operating_income_inc':[], 'is_net_income_inc':[],
        'is_debt_inc':[], 'is_debt_long_inc':[], 'is_debt_short_inc':[], 'is_debt_payables_inc':[], 'is_debt_deferredTax':[],
        'total_revenue':[], 'current(%)':[], 'quick(%)':[], 'op(%)':[], 'net(%)':[],
        'dividend_rate':[], 'dividend_yield':[], 'payout(%)':[],
        'debt(%)':[], 'debt_long(%)':[], 'debt_short(%)':[], 'debt_payables(%)':[], 'debt_tax(%)':[], 'is_ok_payables':[], 'is_ok_tax':[]
    }

    for ticker in tqdm(list_tickers):
        bs = dict_us_stocks[ticker]['balance_sheet']
        financials = dict_us_stocks[ticker]['financials']
        dividends = dict_us_stocks[ticker]['dividends']
        info = dict_us_stocks[ticker]['info']

        if bs.empty or len(bs) == 0: continue
        if financials.empty or len(financials) == 0: continue
        if dividends.empty: continue

        eps = info.get('trailingEps',0)
        pbs = info.get('bookValue', 0)
        per = info.get('trailingPE', 0)
        pbr = info.get('priceToBook', 0)
        psr = info.get('priceToSalesTrailing12Months', 0)
        roe = info.get('returnOnEquity', 0)
        market_cap = info.get('marketCap', 0)
        dividend_rate = info.get('dividendRate', 0)     ## 연간배당금 총액
        dividend_yield = info.get('dividendYield', 0)   ## 배당수익률 (배당률)
        payoutRatio = info.get('payoutRatio', 0)        ## 배당성향
        currentPrice = info.get('currentPrice', 0)
        sector = info.get('sector', '')
        industry = info.get('industry', '')

        df_base = bs.join(financials)
        df_base.dropna(inplace=True, how='all')
        df_base = df_base.iloc[:5]

        if len(df_base) < 4: continue                       ## 최근 1년치가 있는지.. 분기로 4분기
        if 'Diluted EPS' not in df_base.columns: continue   ## 이익을 보고 있는가?

        share_issued_s = get_col(df_base, 'Share Issued')
        share_recent, share_past = recent_and_past(share_issued_s)

        diluted_eps_s = get_col(df_base, 'Diluted EPS')
        eps_recent, eps_past = recent_and_past(diluted_eps_s)

        total_revenue_s = get_col(df_base, 'Total Revenue')
        rev_recent, rev_past = recent_and_past(total_revenue_s)

        operating_income_s = get_col(df_base, 'Operating Income')
        op_recent, op_past = recent_and_past(operating_income_s)

        net_income_s = get_col(df_base, 'Net Income')
        net_recent, net_past = recent_and_past(net_income_s)

        ## 발행주식 수가 동일하면서 YoY 대비 eps가 증가하고 있는지?
        ## 부가적으로 YoY 대비 매출액과 영업이익, 순이익이 증가하고 있는지?
        is_share_reduced = share_recent < share_past
        is_share_same = share_recent == share_past
        is_eps_inc = eps_recent > eps_past
        is_revenue_inc = rev_recent > rev_past
        is_net_income_inc = net_recent > net_past
        is_operating_income_inc = False
        if str_type in ('bs_fin', 'no_bs_fin'):
            is_operating_income_inc = op_recent > op_past

        ## QoQ 기반에서 성장률 계산할 수 없음. 오직 YoY
        peg = 0
        # diluted_eps = [v for v in df_base['Diluted EPS'].values if not pd.isna(v) ]
        # eps_arr = np.array(diluted_eps, dtype=float)  # 확실히 float
        # if np.any(eps_arr <= 0) or np.isnan(eps_arr).any():
        #     eps_growth = np.nan  # 계산 불가
        # else:
        #     eps_growth = (np.power(eps_arr[0] / eps_arr[-1], 1/(len(eps_arr)-1)) - 1) * 100
        #     peg = per / eps_growth if eps_growth > 0 else 0
        is_debt_inc = False
        debt_ratio_val = 0
        if str_type in ('bs_fin', 'no_bs_fin'):
            noncurrent_liab = get_col(df_base, 'Total Non Current Liabilities Net Minority Interest')
            current_liab = get_col(df_base, 'Current Liabilities')
            total_liab_s = noncurrent_liab + current_liab
            df_base = df_base.assign(**{'Total Liabilities':total_liab_s})

            stockholders_equity_s = get_col(df_base, 'Stockholders Equity')

            total_liab_recent = recent_and_past(df_base['Total Liabilities'])[0]
            stockholders_equity_recent = recent_and_past(stockholders_equity_s)[0]
            debt_ratio_val = safe_div(total_liab_recent, stockholders_equity_recent, alt=0.0) * 100

            debt_ratio_s = (df_base['Total Liabilities'] / stockholders_equity_s.replace(0, np.nan)).fillna(0) * 100
            try:
                is_debt_inc = bool(debt_ratio_s.iloc[0] > debt_ratio_s.iloc[-1])
            except Exception:
                is_debt_inc = False


        current_ratio_val = 0
        quick_ratio_val = 0
        if str_type == 'bs_fin':
            curr_assets_s = get_col(df_base, 'Current Assets', fill=0.0)
            inventory_s = get_col(df_base, 'Inventory', fill=0.0)
            curr_liab_s = get_col(df_base, 'Current Liabilities', fill=0.0)

            current_ratio_val = safe_div(recent_and_past(curr_assets_s)[0], recent_and_past(curr_liab_s)[0], alt=0.0) * 100
            quick_ratio_val = safe_div(recent_and_past(curr_assets_s)[0] - recent_and_past(inventory_s)[0],
                                       recent_and_past(curr_liab_s)[0], alt=0.0) * 100

        ## 단기부채, 장기부채, 지급채무/매입채무(외상으로 물건/서비스를 받고 아직 돈을 안 준 금액), 세금성 부채
        is_debt_long_inc = False
        is_debt_short_inc = False
        is_debt_payables_inc = False
        is_debt_deferredTax = False

        is_ok_payables = False
        is_ok_deferredTax = False
        if str_type in ('bs_fin', 'no_bs_fin'):
            debt_short_s = (get_col(df_base, 'Current Debt')
                            + get_col(df_base, 'Other Current Borrowing')
                            + get_col(df_base, 'Commercial Paper')
                            + get_col(df_base, 'Line Of Credit')).fillna(0)
            debt_long_s = np.maximum(get_col(df_base, 'Long Term Debt'), get_col(df_base, 'Long Term Debt And Capital Lease Obligation')).fillna(0)
            debt_payables_s = get_col(df_base, 'Accounts Payable').fillna(0)

            debt_short_recent = recent_and_past(debt_short_s)[0]
            debt_long_recent = recent_and_past(pd.Series(debt_long_s))[0] if isinstance(debt_long_s, (pd.Series, np.ndarray)) else recent_and_past(debt_long_s)[0]
            debt_payables_recent = recent_and_past(debt_payables_s)[0]

            # deferred tax : 'Deferred Tax Liabilities' 우선, 없으면 'DeferredTax' 컬럼 사용
            if 'Deferred Tax Liabilities'in df_base.columns:
                deferred_tax_s = get_col(df_base, 'Deferred Tax Liabilities')
            else:
                deferred_tax_s = get_col(df_base, 'DeferredTax', fill=0.0)
            deferred_tax_recent = recent_and_past(deferred_tax_s)[0]

            # 현금 선택: 가능한 한 상세한 컬럼 우선 사용 (coalesce)
            cash_candidates = []
            if 'Cash And Cash Equivalents' in df_base.columns:
                cash_candidates.append(get_col(df_base, 'Cash And Cash Equivalents'))
            if 'Cash Cash Equivalents And Short Term Investments' in df_base.columns:
                cash_candidates.append(get_col(df_base, 'Cash Cash Equivalents And Short Term Investments'))
            # coalesce: 첫번째 non-zero를 recent 값으로 사용
            cash_recent = 0.0
            for s in cash_candidates:
                val = recent_and_past(s)[0]
                if val != 0.0:
                    cash_recent = val
                    break

            # fallback: 0
            st_inv_s = (get_col(df_base, 'Other Short Term Investments')
                        + get_col(df_base,'Short Term Investments')).fillna(0)
            st_inv_recent = recent_and_past(st_inv_s)[0]

            ocf_recent = recent_and_past(get_col(df_base, 'Operating Cashflow'))[0]

            #증가여부
            is_debt_long_inc = False
            is_debt_short_inc = False
            is_debt_payables_inc = False
            is_debt_deferredTax = False
            try:
                is_debt_long_inc = bool(debt_long_s.iloc[0] > debt_long_s.iloc[-1])
                is_debt_short_inc = bool(debt_short_s.iloc[0] > debt_short_s.iloc[-1])
                is_debt_payables_inc = bool(debt_payables_s.iloc[0] > debt_payables_s.iloc[-1])
                is_debt_deferredTax = bool(deferred_tax_s.iloc[0] > deferred_tax_s.iloc[-1])
            except Exception:
                pass

            # 지급채무(Accounts Payable) 커버 여부 (보수적 판단)
            is_ok_payables = False
            if debt_payables_recent < (cash_recent + st_inv_recent):
                is_ok_payables = True
            elif str_type == 'bs_fin':
                if debt_payables_recent < recent_and_past(curr_assets_s)[0]:
                    is_ok_payables = True

            # DeferredTax 커버 여부 : (1) deferred/total_liabilities < 0.25 OR (2) deferred < OCF (OCF > 0)
            is_ok_deferredTax = False
            if total_liab_recent > 0:
                if safe_div(deferred_tax_recent, total_liab_recent, alt=1.0) < 0.25:
                    is_ok_deferredTax = True
            if not is_ok_deferredTax and ocf_recent > 0:
                if deferred_tax_recent < ocf_recent:
                    is_ok_deferredTax = True


        total_revenue = rev_recent
        op_inc_ratio = safe_div(op_recent, total_revenue, alt=np.nan) * 100.0
        net_inc_ratio = safe_div(net_recent, total_revenue, alt=np.nan) * 100.0

        debt_long_ratio = 0
        debt_short_ratio = 0
        debt_payables_ratio = 0
        debt_tax_ratio = 0
        if str_type in ('bs_fin', 'no_bs_fin'):
            total_debt = total_liab_recent
            debt_long_ratio = safe_div(debt_long_recent, total_debt, alt=0.0) * 100.0
            debt_short_ratio = safe_div(debt_short_recent, total_debt, alt=0.0) * 100.0
            debt_payables_ratio = safe_div(debt_payables_recent, total_debt, alt=0.0) * 100.0
            debt_tax_ratio = safe_div(deferred_tax_recent, total_debt, alt=0.0) * 100.0


        list_summary = [
            ticker, sector, industry, market_cap, currentPrice, eps, pbs, per, pbr, psr, roe, net_recent,  #enterpriseToEbitda, freecashflow,
            is_share_reduced, is_share_same, is_eps_inc, is_revenue_inc, is_operating_income_inc, is_net_income_inc,
            is_debt_inc, is_debt_long_inc, is_debt_short_inc, is_debt_payables_inc, is_debt_deferredTax,
            total_revenue, (round(current_ratio_val, 2) if not pd.isna(current_ratio_val) else np.nan),
            (round(quick_ratio_val, 2) if not pd.isna(quick_ratio_val) else np.nan),
            (round(op_inc_ratio, 2) if not pd.isna(op_inc_ratio) else np.nan), (round(net_inc_ratio, 2) if not pd.isna(net_inc_ratio) else np.nan),
            dividend_rate, dividend_yield, payoutRatio,
            round(debt_ratio_val, 2), round(debt_long_ratio, 2), round(debt_short_ratio, 2), round(debt_payables_ratio, 2),
            round(debt_tax_ratio, 2), bool(is_ok_payables), bool(is_ok_deferredTax)
        ]


        for k, v in zip(list(dict_tickers_summarize.keys()), list_summary):
            dict_tickers_summarize[k].append(v)

    if return_df:
        return pd.DataFrame(dict_tickers_summarize)
    return dict_tickers_summarize




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--is_collect_stocks', type=lambda x: x.lower() == 'true', default=False)
    args = parser.parse_args()

    #######################################################################
    #                            1.US데이터 수집
    #######################################################################
    nasdaq_stocks = fdr.StockListing('NASDAQ')
    nyse_stocks = fdr.StockListing('NYSE')
    all_stocks = pd.concat([nasdaq_stocks, nyse_stocks])

    if args.is_collect_stocks:
        list_nasdaq_stocks = nasdaq_stocks['Symbol'].tolist()
        get_data_from_yf(list_nasdaq_stocks, 'nasdaq')

        list_nyse_stocks = nyse_stocks['Symbol'].tolist()
        get_data_from_yf(list_nyse_stocks, 'nyse')

        all_data = {}
        cwd = Path.cwd()
        folder_nasdaq_path = "/".join([str(cwd), "pkl_nasdaq"])
        folder_nyse_path = "/".join([str(cwd), "pkl_nyse"])
        files_nasdaq = [f for f in os.listdir(folder_nasdaq_path) if os.path.isfile(os.path.join(folder_nasdaq_path, f))]
        files_nyse = [f for f in os.listdir(folder_nyse_path) if os.path.isfile(os.path.join(folder_nyse_path, f))]

        for file_nm in tqdm(files_nasdaq, desc=f'Processing {len(folder_nasdaq_path)} files'):
            dict_file_nasdaq = pd.read_pickle(os.path.join(folder_nasdaq_path, file_nm))
            all_data.update(dict_file_nasdaq)

        for file_nm in tqdm(files_nyse, desc=f'Processing {len(folder_nyse_path)} files'):
            dict_file_nyse = pd.read_pickle(os.path.join(folder_nyse_path, file_nm))
            all_data.update(dict_file_nyse)

        with open(f'yf_chunk_all.pkl', 'wb') as f:
            pickle.dump(all_data, f)

    #######################################################################
    #                            2.티커 분류
    #######################################################################
    dict_us_stocks = pd.read_pickle('yf_chunk_all.pkl')

    list_bs_fin_tickers = []
    list_bs_no_fin_tickers = []
    list_no_bs_fin_tickers = []
    list_no_bs_ca_fin_tickers = []
    list_no_bs_no_fin_tickers = []

    for ticker in dict_us_stocks.keys():
        bs = dict_us_stocks[ticker]['balance_sheet']
        financials = dict_us_stocks[ticker]['financials']

        if bs.empty or len(bs) == 0: continue
        if financials.empty or len(financials) == 0: continue

        set_core_bs_cols = {"Total Assets", "Total Non Current Assets", "Current Assets", "Inventory"}
        set_core_fin_cols = {"Total Revenue", "Cost Of Revenue", "Gross Profit", "Operating Income", "Net Income"}
        set_mtch_bs_cols = set(bs.columns) & set_core_bs_cols
        set_mtch_fin_cols = set(financials.columns) & set_core_fin_cols

        if len(set_mtch_bs_cols) == len(set_core_bs_cols): #유동자산과 재고가 있다면
            if len(set_mtch_fin_cols) == len(set_core_fin_cols): #매출관련 지표가 있다면
                list_bs_fin_tickers.append(ticker) ## 제약, 의료장비
            else:
                list_bs_no_fin_tickers.append(ticker) ## 방송, 연구, cost_of_revenue 없음
        else: # 유동자산이 없다면
            if 'Current Assets' in set_mtch_bs_cols:
                if len(set_mtch_fin_cols) == len(set_core_fin_cols):
                    list_no_bs_fin_tickers.append(ticker) ## 소프트웨어
                else:
                    list_no_bs_ca_fin_tickers.append(ticker) ## 매출원가, 총매출이익이 없음 (생명공학, 투자지주)
            else:
                list_no_bs_no_fin_tickers.append(ticker) ##매출원가, 총매출이익이 없음 (은행)

    # no_bs_no_fin은 PER/FCF 쓰면 안됨. ROE / NIM /건전성 지표만 사용
    print("전체 주식회사 : {}, bs_fin : {}, bs_no_fin : {}, no_bs_fin : {}, no_bs_ca_fin : {}, no_bs_no_fin : {}".format(
        len(dict_us_stocks.keys()),
        len(list_bs_fin_tickers),
        len(list_bs_no_fin_tickers),
        len(list_no_bs_fin_tickers),
        len(list_no_bs_ca_fin_tickers),
        len(list_no_bs_no_fin_tickers)   ))

    #######################################################################
    #                            3.가치 후보 추출
    #           현재로서, bs_fin만.. no_bs_no_fin은 나중에 하자 (은행권)
    #######################################################################

    dict_bs_fin_tickers_report = calc_indicator(dict_us_stocks, list_bs_fin_tickers, 'bs_fin', True)
    df_bs_fin_report = pd.DataFrame(dict_bs_fin_tickers_report)
    df_bs_fin_report['per'] = df_bs_fin_report['per'].astype(float)
    df_bs_fin_report.to_pickle('bs_fin.pickle')


