# models/metrics.py
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

def evaluate_model(model, test_loader, y_test, device):
    model.eval()
    preds_encoded = []

    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            out = model(X_batch)
            pred = torch.argmax(out, dim=1).cpu()
            preds_encoded.append(pred)

    preds_encoded = torch.cat(preds_encoded).numpy()

    acc = accuracy_score(y_test, preds_encoded)
    f1 = f1_score(y_test, preds_encoded, average='macro')
    report = classification_report(y_test, preds_encoded, target_names=['Down','Up'])
    cm = confusion_matrix(y_test, preds_encoded)

    return {
        "accuracy": acc,
        "f1_macro": f1,
        "report": report,
        "confusion_matrix": cm,
        "preds": preds_encoded
    }