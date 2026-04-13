import numpy as np
import random
import argparse
import torch
import yaml
import os
import sys
import csv
from torchmetrics.functional import f1_score
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import get_dataset_dir
from model_utils import Config, AudioTextDataset, AudioLLM
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)  # Python built-in hash seed
    random.seed(seed)                         # Python random module
    np.random.seed(seed)                      # NumPy
    torch.manual_seed(seed)                   # PyTorch CPU
    torch.cuda.manual_seed(seed)              # PyTorch single GPU
    torch.cuda.manual_seed_all(seed)          # PyTorch multi-GPU
    torch.backends.cudnn.deterministic = True # Enforce deterministic behavior
    torch.backends.cudnn.benchmark = False    # Disable benchmarking to avoid non-deterministic conv algorithms


def evaluate(model, dataloader, dataset_name, device):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for x_audio_filename, x_audio_encoder, x_prompt, x_context, labels in tqdm(dataloader, desc=f"Evaluating {dataset_name}", unit="batch"):
            x_audio_encoder, labels = x_audio_encoder.to(device), labels.to(device)

            # Forward pass
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

    print(f"{dataset_name}: AUC: {roc_auc:.4f}    F1 score: {F1score:.4f}    Accuracy: {accuracy:.4f}")
    return accuracy, roc_auc, all_labels, all_probs


def main(test_datasets, projector_type, diag_disease='all'):
    config_path = './src/config.yaml'
    with open(config_path, 'r') as file:
        configs = yaml.safe_load(file)
    configs = Config(configs)
    print("LLM Model Selected: ", configs.llm_model)

    batch_size = configs.batch_size
    configs.aligner = projector_type

    all_test_loader = []
    for dataset_name in test_datasets:
        dataset_root_dir, model_type, train_path, test_path = get_dataset_dir(dataset_name)
        test_dataset = AudioTextDataset(configs, test_path, dataset_name)

        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        all_test_loader.append(test_loader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioLLM(configs, diag_disease=diag_disease).to(device)
    model_saved_path = f"/local/scratch1/siam/saved_models/acl_2026/final_models/audio_llm_model_{diag_disease}_{configs.llm_model}_{projector_type}_{configs.multimodal_config}_final.pth"
    print("Model will be loaded from: ", model_saved_path)
    model.load_state_dict(torch.load(model_saved_path))

    best_results_dict = {name: 0 for name in test_datasets}

    # Evaluation
    for i, test_loader in enumerate(all_test_loader):
        dataset_name = test_datasets[i]
        acc, roc, all_labels, all_probs = evaluate(model, test_loader, dataset_name, device)
        best_roc = roc
        best_results_dict[dataset_name] = round(best_roc,3)

    print(best_results_dict)
    with open("results.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Task_ID", "Dataset_name", "AUROC"])
        for i, (dataset, auroc) in enumerate(best_results_dict.items(), start=1):
            writer.writerow([f"T{i}", dataset, auroc])

if __name__ == "__main__":
    set_seed(seed=42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--disease", default="all")
    parser.add_argument("--projector", default="contrastive")
    args = parser.parse_args()

    diag_disease = args.disease
    projector_type = args.projector
    test_datasets = ["ukcovid19", "coughvid", "TBscreen", "icbhi",
                     "coswara", "codaTB",  "kauh_copd", "kauh_asthma", "kauh_pneumonia"]
    print(f'Model will be evaluated on {diag_disease} : {test_datasets} dataset')
    main(test_datasets, projector_type, diag_disease = diag_disease)
