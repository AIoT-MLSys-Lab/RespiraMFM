from torchmetrics.functional import f1_score
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset
import torch
import yaml
import os
import sys
import pandas as pd
from utils import get_dataset_dir, get_subset
from model_utils import Config, AudioTextDataset, AudioLLM
from projector_train_text_audio import train_projector_all_dataset
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import numpy as np
import random
import argparse
import json

def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)  # Python built-in hash seed
    random.seed(seed)                         # Python random module
    np.random.seed(seed)                      # NumPy
    torch.manual_seed(seed)                   # PyTorch CPU
    torch.cuda.manual_seed(seed)              # PyTorch single GPU
    torch.cuda.manual_seed_all(seed)          # PyTorch multi-GPU
    torch.backends.cudnn.deterministic = True # Enforce deterministic behavior
    torch.backends.cudnn.benchmark = False    # Disable benchmarking to avoid non-deterministic conv algorithms


def train(model, dataloader, optimizer, criterion, device):
    model.train()
    progress_bar = tqdm(dataloader, desc=f"Epoch", unit="batch")
    for audio_filenames, x_audio_encoder, x_prompt, x_context, labels in progress_bar:
        x_audio_encoder, labels = x_audio_encoder.to(device), labels.to(device)
        model_outputs = model(audio_filenames, x_audio_encoder, x_prompt, x_context)
        outputs = model_outputs.logits
        outputs = outputs.squeeze(-1)  
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

def evaluate(model, dataloader, dataset_name, device, configs):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for x_audio_filename, x_audio_encoder, x_prompt, x_context, labels in tqdm(dataloader, desc=f"Evaluating {dataset_name}", unit="batch"):
            x_audio_encoder, labels = x_audio_encoder.to(device), labels.to(device)
            outputs = model(x_audio_filename, x_audio_encoder, x_prompt, x_context).logits
            probabilities = torch.softmax(outputs, dim=-1)  
            _, predictions = torch.max(outputs, dim=-1)  

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predictions.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy()[:, 1])  

    # Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_probs)
    F1score = f1_score(all_labels, all_preds)

    print("LLM Model Selected: ", configs.llm_model)
    print(f"{dataset_name}: AUC: {roc_auc:.4f}    F1 score: {F1score:.4f}    Accuracy: {accuracy:.4f}")
    return accuracy, roc_auc, all_labels, all_probs


def main(train_datasets, test_datasets, diag_disease='all', retrain_projector=True):
    config_path = './src/config.yaml'
    with open(config_path, 'r') as file:
        configs = yaml.safe_load(file)
    configs = Config(configs)
    print("LLM Model Selected: ", configs.llm_model)

    batch_size = configs.batch_size

    # Define Training Dataset
    all_train_dataset = []
    for dataset_name in train_datasets:
        dataset_root_dir, model_type, train_path, test_path = get_dataset_dir(dataset_name)
        train_dataset = AudioTextDataset(configs, train_path, dataset_name)
        all_train_dataset.append(train_dataset)

    combined_training_dataset = ConcatDataset(all_train_dataset)
    if configs.dataset_subset_fraction < 1:
        combined_training_dataset = get_subset(combined_training_dataset, fraction=configs.dataset_subset_fraction, seed=42)
    train_loader = DataLoader(combined_training_dataset, batch_size=batch_size, shuffle=True)

    all_dataset_projector_train = all_train_dataset
    # Define Test Dataset
    all_test_loader = []
    for dataset_name in test_datasets:
        dataset_root_dir, model_type, train_path, test_path = get_dataset_dir(dataset_name)
        test_dataset = AudioTextDataset(configs, test_path, dataset_name)
        if configs.dataset_subset_fraction < 1:
            test_dataset = get_subset(test_dataset, fraction=configs.dataset_subset_fraction,
                                               seed=42)
        all_dataset_projector_train.append(test_dataset)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        all_test_loader.append(test_loader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if retrain_projector:
        if configs.multimodal_config != 3 and configs.aligner != "projection":
            train_projector_all_dataset(train_datasets, diag_disease, num_epochs=500)

    model = AudioLLM(configs, diag_disease=diag_disease).to(device)

    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-5,  
        betas=(0.9, 0.95),  
        eps=1e-8,  
        weight_decay=0.1  
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=10000,  
        eta_min=1e-6  
    )
    criterion = torch.nn.CrossEntropyLoss()

    print("🔥 Training Started....")
    num_epoch = 20
    best_avg = 0

    for epoch in range(num_epoch):
        print(f"Epoch {epoch + 1} of {num_epoch}")
        current_results_dict = {name: 0 for name in test_datasets}
        train(model, train_loader, optimizer, criterion, device)
        scheduler.step()

        # Evaluation
        save_flag = True
        for i, test_loader in enumerate(all_test_loader):
            dataset_name = test_datasets[i]
            acc, roc, all_labels, all_probs = evaluate(model, test_loader, dataset_name, device, configs)
            current_results_dict[dataset_name] = round(roc, 4)

        current_avg = np.mean(list(current_results_dict.values()))
        current_results_dict['epoch'] = epoch
        current_results_dict['Agerage'] = current_avg
        if current_avg > best_avg:
            save_flag = True
            best_avg = current_avg

        if save_flag:
            model_saved_path = f"/local/scratch1/siam/saved_models/acl_2026/respiramfm/audio_llm_model_{diag_disease}_{configs.llm_model}_{configs.aligner}_{configs.multimodal_config}_epoch{epoch}.pth"
            torch.save(model.state_dict(), model_saved_path)
            print(f"Full model with projector saved in directory: {model_saved_path}")

        print(current_results_dict)
        out_save_file_json = f'./results_log/final/RespiraMFM_{configs.llm_model}_{diag_disease}_{configs.aligner}_{configs.multimodal_config}.jsonl'
        with open(out_save_file_json, 'a') as f:
            json.dump(current_results_dict, f)
            f.write('\n')

if __name__ == "__main__":
    set_seed(seed=420)
    parser = argparse.ArgumentParser()
    parser.add_argument("--disease", default="all")
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()

    diag_disease = args.disease
    retrain_projector = args.retrain

    if diag_disease == "covid":
        train_datasets = ["ukcovid19", "coughvid"]
        test_datasets = ["ukcovid19", "coughvid", "coswara"]
    elif diag_disease == "TB":
        train_datasets = ["TBscreen"]
        test_datasets = ["codaTB", "TBscreen"] # TBscreen or codaTB
    elif diag_disease == "copd":
        train_datasets = ["icbhi"]
        test_datasets = ["icbhi", "kauh_copd", "kauh_asthma", "kauh_pneumonia"]
    elif diag_disease == "all":
        train_datasets = ["ukcovid19", "coughvid", "TBscreen", "icbhi"]
        test_datasets = ["ukcovid19", "coughvid", "TBscreen", "icbhi", "coswara", "codaTB",  "kauh_copd", "kauh_asthma", "kauh_pneumonia"]

    print(f'Model Training on {diag_disease} : {train_datasets} dataset')
    print(f'Model will be evaluated on {diag_disease} : {test_datasets} dataset')
    main(train_datasets, test_datasets, diag_disease = diag_disease, retrain_projector=retrain_projector)
