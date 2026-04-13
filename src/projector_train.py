import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import yaml
from model_utils import AudioTextDataset, Config



# ------------------ Contrastive Projection and Loss ------------------
class ContrastiveProjectionHead(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=512, out_dim=2048, dropout=0.3):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
            # nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        return self.projector(x)

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0]).to(device)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        return -mean_log_prob_pos.mean()


# ------------------ Training Loop ------------------
def train_contrastive(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for _, x, _, _, y in tqdm(dataloader, desc='Training'):
        x, y = x.to(device), y.to(device)
        projected = model(x)
        projected = F.normalize(projected, dim=1)
        loss = criterion(projected, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# ------------------ Main ------------------
def main(train_text=False):
    config_path = './src/config.yaml'
    with open(config_path, 'r') as file:
        configs = yaml.safe_load(file)
    configs = Config(configs)
    # Settings
    batch_size = 64
    num_epochs = 100
    learning_rate = 1e-2
    dataset_name = 'ukcovid19'
    if dataset_name == 'ukcovid19':
        dataset_root_dir = "/local/scratch1/siam/dataset/resp-dataset/uk-covid-19/"
        train_path = dataset_root_dir + f"custom_files/train_features.h5"
        test_path = dataset_root_dir + f"custom_files/test_features.h5"
    elif dataset_name == 'coswara':
        dataset_root_dir = "/local/scratch1/siam/dataset/cough-sound-data/coswara/download/Coswara-Data/"
        cough_type = "cough-shallow"
        train_path = dataset_root_dir + f"custom_files/alldata_{cough_type}_features_qc.h5"

    h5_path = train_path

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = AudioTextDataset(configs, h5_path, dataset_name='ukcovid19')
    # dataset = AudioEmbeddingDataset(h5_path, out_text=train_text)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ContrastiveProjectionHead().to(device)
    criterion = SupConLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    for epoch in range(num_epochs):
        loss = train_contrastive(model, dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")

    # Save model
    if not train_text:
        model_save_path = f'contrastive_projection_head_{dataset_name}.pth'
    else:
        model_save_path = f'contrastive_projection_head_text_{dataset_name}.pth'
    torch.save(model.state_dict(), model_save_path)

if __name__ == '__main__':
    main(train_text=False)