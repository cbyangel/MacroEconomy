import torch
import torch.nn as nn
from torch.utils.data import Dataset


## 언더스코어 (__) 2개는 파이썬의 "특수 메소드" 또는 "매직 메소드"를 의미합니다.
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y): ## 객체를 생성할 때 자동으로 실행되는 초기화 메소. 클래스에 데이터를 넣거나, 변수를 준비하는데 사용
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): ## 데이터 개수 반환 len(dataset)
        return len(self.X)
    def __getitem__(self, idx): ## 한개 샘플 반환 dataset[10]
        return self.X[idx], self.y[idx]


class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim=1, num_heads=4, num_layers=3, hidden_dim=64, num_classes=2, mlp_dim=128):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        ## --- 여기 MLP HEAD 추가 ---
        # self.fc = nn.Linear(hidden_dim, num_classes)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes)

        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.mlp(x)