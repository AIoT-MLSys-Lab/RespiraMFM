import torch.nn as nn


class ContrastiveProjectionHead(nn.Module):
    def __init__(self, in_dim=768, out_dim=2048, hidden_dim=1024, dropout=0.1):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.projector(x)

class ContrastiveProjectionHead_new(nn.Module):
    def __init__(self, in_dim=768, out_dim=2048, hidden_dim=2048, dropout=0.1):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.projector(x)
