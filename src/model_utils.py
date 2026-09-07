from torch.utils.data import DataLoader, Dataset, ConcatDataset
import h5py
import numpy as np
import torch
import torch.nn as nn
from transformers import (AutoTokenizer,  AutoModelForSequenceClassification, LlamaForSequenceClassification, TrainingArguments, Trainer)
from peft import LoraConfig, TaskType, get_peft_model, IA3Config
import os
import sys
from utils import get_prompt, get_dataset_dir, get_subset
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from RespFeatureExtractor.feature_extractor import get_encoder_path, initialize_pretrained_model, OperaExtractor
from model_projector import ContrastiveProjectionHead


class AudioLLM(nn.Module):
    def __init__(self, configs, diag_disease=None, return_attn=False):
        super(AudioLLM, self).__init__()
        self.loss = nn.CrossEntropyLoss()
        self.n_cls = configs.n_cls # 2 class for covid/non-covid
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.d_ff = configs.d_ff
        if hasattr(configs, "encode_from_audio"):
            self.encode_from_audio = configs.encode_from_audio
            self.noise_type = configs.noise_type
        else:
            self.encode_from_audio = False  # or some default

        if configs.llm_model in ["llama3-8b", "openbiollm", "contactdoctor-8b", "llama3-8b-instruct", "GPT-j"]:
            self.d_llm = 4096
        elif configs.llm_model in ["llama3-1b", "llama3-1b-custom", "contactdoctor-1b", "contactdoctor-3b"]:
            self.d_llm = 2048
        elif configs.llm_model in ["llama3-3b"]:
            self.d_llm = 3072
        elif configs.llm_model in ["phi-2"]:
            self.d_llm = 2560
        elif configs.llm_model in ['deepseek-1.5b']:
            self.d_llm = 1536
        elif configs.llm_model in ["GPT2-m"]:
            self.d_llm = 1024
        elif configs.llm_model in ["GPT2-l"]:
            self.d_llm = 1280
        else:
            print('Invalid Model Selection')


        self.audio_peft = configs.audio_peft
        self.d_audio = configs.enc_dim       # 768 if use OPERA-CT
        self.spectrogram_type = configs.spectrogram_type

        self.proj_hidden = configs.proj_hidden
        self.patch_nums = configs.patch_nums
        self.max_seq_length = configs.max_seq_length # custom added

        # self.head_nf = self.d_ff * self.patch_nums
        self.head_nf = self.max_seq_length * self.d_llm # modified, if error use prev line

        self.llm_peft = configs.llm_peft
        self.llm_lora_rank = configs.llm_lora_rank
        self.llm_lora_alpha = configs.llm_lora_alpha
        self.llm_lora_dropout = configs.llm_lora_dropout

        self.use_audio = configs.use_audio
        self.multimodal_config = configs.multimodal_config
        # self.aligner = nn.Linear(self.d_audio, self.d_llm)  # (768 -> 4096)

        self.output_attn = return_attn


        if configs.llm_model == "llama3-8b":
            model_id = "meta-llama/Meta-Llama-3-8B"
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.llm_model = LlamaForSequenceClassification.from_pretrained(model_id)
            # self.positional_encoding = PositionalEncoding(dim=4096)
            print(f'{configs.llm_model} model Loaded 📦📦📦...')

        if configs.llm_model == "llama3-8b-instruct":
            model_id = "meta-llama/Llama-3.1-8B-Instruct"
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.llm_model = LlamaForSequenceClassification.from_pretrained(model_id)
            # self.positional_encoding = PositionalEncoding(dim=4096)
            print(f'{configs.llm_model} model Loaded 📦📦📦...')

        elif configs.llm_model == "llama3-1b":
            model_id = "meta-llama/Llama-3.2-1B"
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.llm_model = LlamaForSequenceClassification.from_pretrained(model_id)
            print(f'{configs.llm_model} model Loaded 📦...')

        elif configs.llm_model == "llama3-3b":
            model_id = "meta-llama/Llama-3.2-3B"
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.llm_model = LlamaForSequenceClassification.from_pretrained(model_id)
            print(f'{configs.llm_model} model Loaded 📦...')

        elif configs.llm_model == "llama3-1b-custom":
            model_id = "/local/scratch1/siam/saved_models/emnlp/llama_3.1"
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.llm_model = LlamaForSequenceClassification.from_pretrained(model_id)
            print(f'{configs.llm_model} model Loaded 📦...')

        elif configs.llm_model == "openbiollm":
            model_id = "aaditya/OpenBioLLM-Llama3-8B"
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.llm_model = LlamaForSequenceClassification.from_pretrained(model_id)
            print(f'{configs.llm_model} model Loaded 📦📦📦💊...')

        elif configs.llm_model == "contactdoctor-1b":
            model_id = "ContactDoctor/Bio-Medical-Llama-3-2-1B-CoT-012025"
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.llm_model = LlamaForSequenceClassification.from_pretrained(model_id)
            print(f'{configs.llm_model} model Loaded 📦💊...')

        elif configs.llm_model == "contactdoctor-3b":
            model_id = "ContactDoctor/Bio-Medical-3B-CoT-012025"
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.llm_model = LlamaForSequenceClassification.from_pretrained(model_id)
            print(f'{configs.llm_model} model Loaded 📦📦💊...')

        elif configs.llm_model == "contactdoctor-8b":
            model_id = "ContactDoctor/Bio-Medical-Llama-3-8B-CoT-012025"
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.llm_model = LlamaForSequenceClassification.from_pretrained(model_id)
            print(f'{configs.llm_model} model Loaded 📦📦📦💊...')

        elif configs.llm_model == "phi-2":
            model_id = "microsoft/phi-2"
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.llm_model = AutoModelForSequenceClassification.from_pretrained(model_id)
            print(f'{configs.llm_model} model Loaded 📦📦...')

        elif configs.llm_model == "deepseek-1.5b":
            model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.llm_model = AutoModelForSequenceClassification.from_pretrained(model_id)
            print(f'{configs.llm_model} model Loaded 📦📦...')

        elif configs.llm_model == "GPT2-m":
            model_id = "openai-community/gpt2-medium"
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.llm_model = AutoModelForSequenceClassification.from_pretrained(model_id)
            configs.LLM_TARGET_MODULES_ALLPROJ = ["c_attn", "c_proj"]
            print(f'{configs.llm_model} model Loaded 📦...')

        elif configs.llm_model == "GPT2-l":
            model_id = "openai-community/gpt2-large"
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.llm_model = AutoModelForSequenceClassification.from_pretrained(model_id)
            configs.LLM_TARGET_MODULES_ALLPROJ = ["c_attn", "c_proj"]
            print(f'{configs.llm_model} model Loaded 📦...')

        elif configs.llm_model == "GPT-j":
            model_id = "EleutherAI/gpt-j-6b"
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.llm_model = AutoModelForSequenceClassification.from_pretrained(model_id)
            # configs.LLM_TARGET_MODULES_ALLPROJ = ["c_attn", "c_proj"]
            print(f'{configs.llm_model} model Loaded 📦...')

        else:
            raise NotImplementedError('LLM model is not defined')

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            # print('PAD Token : ', self.tokenizer.pad_token)
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        self.llm_model.config.pad_token_id = self.llm_model.config.eos_token_id

        if self.llm_peft == "lora":
            self.peft_config = LoraConfig(
                r=self.llm_lora_rank,
                lora_alpha=self.llm_lora_alpha,
                lora_dropout=self.llm_lora_dropout,
                # target_modules=LLM_TARGET_MODULES_ALLPROJ,
            )
            if configs.llm_lora_allproj:
                self.peft_config = LoraConfig(
                    r=self.llm_lora_rank,
                    lora_alpha=self.llm_lora_alpha,
                    lora_dropout=self.llm_lora_dropout,
                    target_modules=configs.LLM_TARGET_MODULES_ALLPROJ,
                )
            try:
                self.llm_model = get_peft_model(self.llm_model, self.peft_config)
            except ValueError:
                print(self.llm_model)
                if configs.llm_model == "phi":
                    self.peft_config = LoraConfig(
                        r=self.llm_lora_rank,
                        lora_alpha=self.llm_lora_alpha,
                        lora_dropout=self.llm_lora_dropout,
                        target_modules=["qkv_proj"]
                    )
                self.llm_model = get_peft_model(self.llm_model, self.peft_config)
            self.llm_model.print_trainable_parameters()
            # print('LoRA Training LLM')
        elif self.llm_peft == "frozen":
            for param in self.llm_model.parameters():
                param.requires_grad = False
        elif self.llm_peft == "full_ft":
            for param in self.llm_model.parameters():
                param.requires_grad = True
            print("LLM: Full fine-tuning (all parameters trainable)")
        else:
            return NotImplementedError("LLM fine-tuning mode undefined")

        if configs.audio_encoder == "operaCT":
            self.audio_encoder = initialize_pretrained_model(configs.audio_encoder).encoder
            if self.encode_from_audio == True:
                self.FeatureExtractor = OperaExtractor(pretrain="operaCT")

        if self.audio_peft == "frozen":
            for name, param in self.audio_encoder.named_parameters():
                param.requires_grad = False
            self.audio_encoder.eval()
            # print("Audio Encoder: Frozen")
        elif self.audio_peft == "full":
            for name, param in self.audio_encoder.named_parameters():
                param.requires_grad = True
            self.audio_encoder.train()
            print("Audio Encoder: Full model fine-tune")
            ckpt = torch.load("cks/model/encoder-operaCT.ckpt")
            self.audio_encoder.load_state_dict(ckpt["state_dict"], strict=False)
        else:
            # peft
            if self.audio_peft == "lora":
                peft_config = LoraConfig(
                    # task_type=TaskType.CAUSAL_LM, inference_mode=False,
                    r=configs.audio_lora_rank, lora_alpha=32, lora_dropout=0.1,
                    target_modules=target_module_dict[configs.audio_encoder]
                )
            elif self.audio_peft == "IA3":
                peft_config = IA3Config(
                    target_modules=target_module_dict[configs.audio_encoder],
                    feedforward_modules=['proj']
                )
            else:
                return NotImplementedError("audio fine-tuning mode undefined")
            self.audio_encoder = get_peft_model(self.audio_encoder, peft_config)
            self.audio_encoder.print_trainable_parameters()

        if configs.aligner == "projection":
            self.freeze_aligner = False
            self.aligner = nn.Linear(self.d_audio, self.d_llm)

        elif configs.aligner == "contrastive" or configs.aligner == "mixed":
            if configs.aligner == "mixed":
                self.freeze_aligner = False
            else:
                self.freeze_aligner = True
            self.aligner = ContrastiveProjectionHead(out_dim=self.d_llm)
            # self.aligner = ContrastiveProjectionHead_new(out_dim=self.d_llm)
            aligner_model_path = f'/local/scratch1/siam/saved_models/acl_2026/final_models/contrastive_audio_to_text_{diag_disease}_{self.d_llm}_final.pth'


            print(f'🔴 Alignment Module Loaded from:  {aligner_model_path}')

            self.aligner.load_state_dict(torch.load(aligner_model_path))
            self.aligner.eval()

        else:
            return NotImplementedError("aligner module undefined")

        self.head_dropout = configs.head_dropout
        self.output_projection = FlattenHead(self.head_nf, self.n_cls, head_dropout=self.head_dropout)

        self.print_trainable()

    def reinitialize_clf(self, n_cls):
        self.output_projection = FlattenHead(self.head_nf, n_cls, head_dropout=self.head_dropout)

    def print_trainable(self):
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("total trainable parameters (M) : ", round(trainable_params/1000000,2))

    def reset_trainable(self):
        if self.llm_peft == "lora":
            for name, param in self.audio_encoder.named_parameters():
                if "lora" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif self.llm_peft == "frozen":
            for param in self.llm_model.parameters():
                param.requires_grad = False
        elif self.llm_peft == "full_ft":
            for param in self.llm_model.parameters():
                param.requires_grad = True

        if not self.freeze_aligner:
            for param in self.aligner.parameters():
                param.requires_grad = True

        if self.audio_peft == "frozen":
            for param in self.audio_encoder.parameters():
                param.requires_grad = False
        elif self.audio_peft == "full":
            for param in self.audio_encoder.parameters():
                param.requires_grad = True

        for param in self.output_projection.parameters():
            param.requires_grad = True
        self.print_trainable()

    def forward(self, audio_files, audio_features, x_prompt, x_context, no_fc=False, noise_db=None):
        # if self.spectrogram_type == "frozen":
        enc_out = audio_features # shape= [N x 768]
        if self.encode_from_audio:
            # print('ℹ️ enc_out shape: ', enc_out.shape, enc_out.device)
            # print(audio_files)
            ext_features = []
            for input_file in audio_files:
                features = self.FeatureExtractor.extract_opera_features_add_noise(input_file, noise_type = self.noise_type, noise_db=noise_db)
                ext_features.append(features)
            ext_features = torch.tensor(ext_features).squeeze(1).to(enc_out.device)
            print('distance : ', torch.norm(audio_features - ext_features, p=2))
            # print('ℹ️ Features shape: ', ext_features.shape, ext_features.device)
            enc_out = ext_features

        x_enc = enc_out

        enc_out = self.aligner(enc_out)

        if len(enc_out.shape) == 2:
            enc_out = enc_out.unsqueeze(dim=1) # do not unsqeeze if multi-token
        # print("shape check 3: ", enc_out.shape)  #torch.Size([16, 1, 52, 2048])
        audio_attention_mask = torch.ones((enc_out.size(0), enc_out.size(1)), dtype=torch.long).to(audio_features.device)  #


        prompt_tokens = self.tokenizer(x_prompt, return_tensors="pt", padding='max_length', truncation=True, max_length=125)
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt_tokens['input_ids'].to(x_enc.device))  # (batch, prompt_token, dim)
        prompt_attention_mask = prompt_tokens['attention_mask'].to(x_enc.device)


        context_tokens = self.tokenizer(x_context, return_tensors="pt", padding='max_length', truncation=True, max_length=125)
        context_embeddings = self.llm_model.get_input_embeddings()(context_tokens['input_ids'].to(x_enc.device))  # (batch, prompt_token, dim)
        context_attention_mask = context_tokens['attention_mask'].to(x_enc.device)


        if self.multimodal_config == 1: # Audio + Metadata
            llama_enc_out = torch.cat([prompt_embeddings, context_embeddings, enc_out], dim=1)
            combined_attention_mask = torch.cat([audio_attention_mask, prompt_attention_mask, context_attention_mask], dim=1)


        elif self.multimodal_config == 2: # Audio only
            llama_enc_out = enc_out
            combined_attention_mask = audio_attention_mask

        elif self.multimodal_config == 3: # Metadata only
            llama_enc_out = torch.cat([prompt_embeddings, context_embeddings], dim=1)
            combined_attention_mask = torch.cat([prompt_attention_mask, context_attention_mask], dim=1)

        else:
            raise NotImplementedError("Multimodal Configuration Undefined")


        if self.output_attn :
            dec_out = self.llm_model(
                inputs_embeds=llama_enc_out,
                attention_mask=combined_attention_mask,
                output_attentions=True,
                return_dict=True
            )
        else:
            dec_out = self.llm_model(inputs_embeds=llama_enc_out, attention_mask=combined_attention_mask)
        return dec_out

    def get_audio_embedding(self, audio_features):
        enc_out = audio_features
        enc_out = self.aligner(enc_out)
        return enc_out


class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)


def collate_fn_fixed_size(batch, max_len=256):
    filename, embeddings, prompts, contexts, labels, seq_len = zip(*batch)

    padded_embeddings = []
    for emb in embeddings:
        if emb.size(0) > max_len:  # Truncate
            padded_embeddings.append(emb[:max_len, :])
        else:  # Pad
            padding = torch.zeros(max_len - emb.size(0), emb.size(1))
            padded_embeddings.append(torch.cat((emb, padding), dim=0))
    padded_embeddings = torch.stack(padded_embeddings)

    # Labels, prompts, and contexts remain unchanged
    labels = torch.stack(labels)
    return filename, padded_embeddings, prompts, contexts, labels, seq_len

class FlattenHead(nn.Module):
    def __init__(self, nf, out_dim, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, out_dim)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x, no_fc=False):
        x = self.flatten(x)
        if no_fc:
            return x
        x = self.linear(x)
        x = self.dropout(x)
        return x

class AudioTextDataset(Dataset):
    def __init__(self, configs, dataset_path, dataset_name, external_context=None):
        with h5py.File(dataset_path, "r") as hf:
            audio_filenames = [x.decode('utf-8') for x in hf["filenames"]]
            cough_labels = hf["labels"][:]
            cough_embedding = hf["audio_embeddings"][:]
            metadata_embedding = hf["text_embeddings"][:]
            metadata_context = hf["metadata"][:]

        self.audio_embedding = np.array(cough_embedding)   # Shape: N x 768
        self.audio_filenames = np.array(audio_filenames)
        self.labels = cough_labels  # Shape: (N,)
        self.contexts = [x.decode('utf-8') if isinstance(x, bytes) else str(x) for x in metadata_context]
        self.prompt = get_prompt(configs, dataset=dataset_name, label="covid", modality="cough")
        if external_context:
            self.prompt = self.prompt + external_context
        # print(self.prompt)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        embedding = torch.tensor(self.audio_embedding[idx], dtype=torch.float32)
        prompt = self.prompt
        audio_file_name = self.audio_filenames[idx]
        context = self.contexts[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return audio_file_name, embedding, prompt, context, label