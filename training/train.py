# train.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from models.transformer import TimeSeriesDataset, TimeSeriesTransformer
from training.metrics import evaluate_model
from utils.config import trend_map
from config.logging_conf import setup_logger
logger = setup_logger(__name__)


def make_sequences_by_label(df, features, label, window=7):
    X_list, y_list, idx_list = [], [], []
    for label_name, label_val in trend_map.items():
        label_indices = df.index[df[label] == label_name]
        for idx in label_indices:
            i = df.index.get_loc(idx)
            if i >= window:
                seq = df[features].values[i-window:i]
                X_list.append(seq)
                y_list.append(label_val)
                idx_list.append(idx)
    X = np.array(X_list).reshape(-1, window, len(features))
    y = np.array(y_list)
    idx_arr = np.array(idx_list)
    return X, y, idx_arr


def run_train(df_processed, features, label, ticker_nm, window=7, epochs=50, batch_size=16):

    # ---------- Sequence ------------------
    X, y, idx_arr = make_sequences_by_label(df_processed, features, label, window)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, idx_arr, test_size=0.2, shuffle=True, stratify=y
    )

    # ---------- SMOTE ---------------------
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X_train_flat, y_train)
    X_res = X_res.reshape(-1, window, X_train.shape[2])

    train_loader = DataLoader(TimeSeriesDataset(X_res, y_res),
                              batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(TimeSeriesDataset(X_test, y_test),
                              batch_size=batch_size, shuffle=False)

    # ---------- Device / Model ------------
    device = torch.device('mps' if torch.backends.mps.is_available() else
                          ('cuda' if torch.cuda.is_available() else 'cpu'))

    model = TimeSeriesTransformer(input_dim=len(features)).to(device)

    class_counts = np.bincount(y_res, minlength=2)
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # ---------- train loop ----------------
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        logger.info(f"Epoch {epoch+1}/{epochs} - loss: {total_loss / len(train_loader):.4f}")


    # ---------- Eval ----------------------
    metrics = evaluate_model(model, test_loader, y_test, device)
    logger.info("ACC : {}".format(metrics["accuracy"]))
    logger.info("F1  : {}".format(metrics["f1_macro"]))
    #logger.info(metrics["report"])
    logger.info(metrics["confusion_matrix"])


    # ---------- Save ----------------------
    torch.save(model.state_dict(),f"./artifacts/model_{ticker_nm}.pth")
    logger.info(f"✅ 모델 저장 완료: model_{ticker_nm}.pth")


    return model, metrics