import torch
import torch.nn as nn
import torch.nn.functional as F
# from nemo.collections.common.data import ConcatDataset
from torch.utils.data import ConcatDataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import yaml
import h5py
from transformers import AutoTokenizer, LlamaModel, AutoModel
from model_utils import get_prompt, Config
from model_projector import ContrastiveProjectionHead, ContrastiveProjectionHead_new
from utils import get_dataset_dir, get_subset
import os

class AudioTextDataset(Dataset):
    def __init__(self, configs, dataset_path, dataset_name, d_llm, suffix= 'none', cache_path="text_embeddings_cache.npy"):
        with h5py.File(dataset_path, "r") as hf:
            self.audio_embeddings = hf["audio_embeddings"][:]
            self.contexts = [x.decode('utf-8') if isinstance(x, bytes) else str(x) for x in hf["metadata"][:]]
            self.labels = hf["labels"][:]

        self.prompt = get_prompt(configs, dataset=dataset_name, label="covid", modality="cough")

        self.cache_path = f'./cached_files/{dataset_name}_{suffix}_{d_llm}.npy'
        if os.path.exists(self.cache_path):
            print("Loading cached text embeddings...")
            self.text_embeddings = np.load(self.cache_path)
        else:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            print("Generating and caching text embeddings...")
            if d_llm==2048:
                model_id = "meta-llama/Llama-3.2-1B"
            elif d_llm == 3072:
                model_id = "meta-llama/Llama-3.2-3B"
            elif d_llm == 4096:
                model_id = "ContactDoctor/Bio-Medical-Llama-3-8B-CoT-012025"
            elif d_llm==2560:
                model_id = "microsoft/phi-2"
            elif d_llm==1536:
                model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
            elif d_llm==1024:
                model_id = "openai-community/gpt2-medium"
            elif d_llm==1280:
                model_id = "openai-community/gpt2-large"
            else:
                raise NotImplementedError('invalid d_llm value : d_llm = ', d_llm)

            tokenizer = AutoTokenizer.from_pretrained(model_id)
            tokenizer.pad_token = tokenizer.eos_token
            # model = LlamaModel.from_pretrained(model_id).eval().cuda()
            model = AutoModel.from_pretrained(model_id).eval().cuda()
            self.text_embeddings = []
            for context in tqdm(self.contexts, desc="Encoding text"):
                with torch.no_grad():
                    tokens = tokenizer(context, return_tensors="pt", padding=True, truncation=True, max_length=125).to("cuda")
                    output = model(**tokens)
                    text_embed = output.last_hidden_state.mean(dim=1).squeeze().detach().cpu().numpy()
                    self.text_embeddings.append(text_embed)
                    # print(text_embed.shape)
            self.text_embeddings = np.array(self.text_embeddings)
            np.save(self.cache_path, self.text_embeddings)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        audio_embed = torch.tensor(self.audio_embeddings[idx], dtype=torch.float32)
        text_embed = torch.tensor(self.text_embeddings[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return audio_embed, text_embed, label


def contrastive_loss(audio_proj, text_embed, temperature=0.07):
    audio_proj = F.normalize(audio_proj, dim=1)
    text_embed = F.normalize(text_embed, dim=1)
    symmetric_loss = False
    if symmetric_loss:
        logits_a2t = torch.matmul(audio_proj, text_embed.T) / temperature
        labels = torch.arange(audio_proj.size(0)).to(audio_proj.device)
        loss_a2t = F.cross_entropy(logits_a2t, labels)

        # Backward loss: text → audio
        logits_t2a = torch.matmul(text_embed, audio_proj.T) / temperature
        loss_t2a = F.cross_entropy(logits_t2a, labels)
        loss = (loss_a2t + loss_t2a) / 2
        return loss
    else:
        # print('shape check: ', audio_proj.shape, text_embed.shape)
        logits = torch.matmul(audio_proj, text_embed.T) / temperature
        labels = torch.arange(audio_proj.size(0)).to(audio_proj.device)
        return F.cross_entropy(logits, labels)

def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for audio_embed, text_embed, _ in dataloader:
        audio_embed, text_embed = audio_embed.to(device), text_embed.to(device)
        projected = model(audio_embed)
        loss = contrastive_loss(projected, text_embed)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def train_projector_all_dataset(train_datasets, test_datasets, diag_disease, num_epochs=100, learning_rate=0.001):
    config_path = './src/config.yaml'
    with open(config_path, 'r') as file:
        configs = yaml.safe_load(file)
    configs = Config(configs)

    if configs.llm_model in ["llama3-8b", "contactdoctor-8b", "openbiollm", "llama3-8b-instruct", "GPT-j"]:
        d_llm = 4096
    elif configs.llm_model in ["llama3-1b", "contactdoctor-1b"]:
        d_llm = 2048
    elif configs.llm_model in ["contactdoctor-3b"]:
        d_llm = 2048
    elif configs.llm_model in ["phi-2"]:
        d_llm = 2560
    elif configs.llm_model in ["deepseek-1.5b"]:
        d_llm = 1536
    elif configs.llm_model in ["GPT2-m"]:
        d_llm = 1024
    elif configs.llm_model == "GPT2-l":
        d_llm = 1280
    elif configs.llm_model == "llama3-3b":
        d_llm = 3072
    else:
        raise NotImplementedError(f'🚨 No d_llm defined for {configs.llm_model}.')

    all_datasets = []
    for dataset_name in train_datasets:
        dataset_root_dir, model_type, train_path, test_path = get_dataset_dir(dataset_name)
        train_dataset = AudioTextDataset(configs, train_path, dataset_name, d_llm, suffix='train')
        all_datasets.append(train_dataset)
    for dataset_name in test_datasets:
        dataset_root_dir, model_type, train_path, test_path = get_dataset_dir(dataset_name)
        test_dataset = AudioTextDataset(configs, test_path, dataset_name, d_llm, suffix='test')
        all_datasets.append(test_dataset)
    all_datasets_concat = ConcatDataset(all_datasets)
    data_loader = DataLoader(all_datasets_concat, batch_size=64, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ContrastiveProjectionHead(out_dim=d_llm).to(device)
    # model = ContrastiveProjectionHead_new(out_dim=d_llm).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    projector_saved_path = f'/local/scratch1/siam/saved_models/acl_2026/contrastive_audio_to_text_{diag_disease}_{d_llm}_final.pth'
    print(f'Projector will be saved at: {projector_saved_path}')
    for epoch in tqdm(range(num_epochs)):
        loss = train(model, data_loader, optimizer, device)
        if num_epochs >=10:
            if (epoch) % (num_epochs//10) == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")

    torch.save(model.state_dict(), projector_saved_path)
    print('✅️ Projector model saved successfully...')

def main():
    config_path = './src/config.yaml'
    with open(config_path, 'r') as file:
        configs = yaml.safe_load(file)
    configs = Config(configs)

    batch_size = 64
    num_epochs = 50
    learning_rate = 1e-3
    dataset_name = 'ukcovid19'
    dataset_root_dir = "/local/scratch1/siam/dataset/resp-dataset/uk-covid-19/"
    train_path = dataset_root_dir + f"custom_files/train_features.h5"
    test_path = dataset_root_dir + f"custom_files/test_features.h5"
    train_cache_path = dataset_root_dir + f"custom_files/text_embeddings_train_cache_{dataset_name}.npy"
    test_cache_path = dataset_root_dir + f"custom_files/text_embeddings_test_cache_{dataset_name}.npy"

    coswara_dataset_root_dir = "/local/scratch1/siam/dataset/cough-sound-data/coswara/download/Coswara-Data/"
    cough_type = "cough-shallow"
    test_path_coswara = coswara_dataset_root_dir + f"custom_files/alldata_{cough_type}_features_qc.h5"
    test_cache_path_coswara = coswara_dataset_root_dir + f"custom_files/text_embeddings_test_cache_{dataset_name}.npy"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = AudioTextDataset(configs, train_path, dataset_name, cache_path=train_cache_path)
    test_dataset = AudioTextDataset(configs, test_path, dataset_name, cache_path=test_cache_path)
    test_dataset_coswara = AudioTextDataset(configs, test_path_coswara, 'coswara', cache_path=test_cache_path_coswara)

    dataset = ConcatDataset([train_dataset, test_dataset, test_dataset_coswara])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ContrastiveProjectionHead().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-2)

    for epoch in range(num_epochs):
        loss = train(model, dataloader, optimizer, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")

    torch.save(model.state_dict(), f'/local/scratch1/siam/saved_models/acl_2026/contrastive_audio_to_text_{dataset_name}.pth')

if __name__ == '__main__':
    main()