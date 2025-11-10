# inference.py
import numpy as np
import torch
from models.transformer import TimeSeriesTransformer
from utils.config import trend_map_inv
# trend_map_inv = {0:'하락', 1:'상승'}

def run_inference(df_processed, features, ticker_nm, window=7, model_path="model.pth"):
    """
    df_processed : 오늘까지 포함한 full df (indicator + label 은 없어도 됨)
    """
    model_path = f"./artifacts/model_{ticker_nm}.pth"

    # ------------ window cut -------------
    if len(df_processed) < window:
        raise ValueError(f"df length < window({window})")

    # 마지막 window seq (오늘 포함 직전 window일)
    seq = df_processed[features].values[-window:]
    X = np.array(seq).reshape(1, window, len(features))  # (1,7,feature_dim)

    device = torch.device('mps' if torch.backends.mps.is_available() else
                          ('cuda' if torch.cuda.is_available() else 'cpu'))

    model = TimeSeriesTransformer(input_dim=len(features)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        preds = model(X_tensor)             # shape (1,2)
        pred_idx = torch.argmax(preds, dim=1).item()
        pred_label = trend_map_inv[pred_idx]

    return pred_label, preds.cpu().numpy()