import os
import numpy as np
import torch
import librosa
# from Notebooks.visualize_embedding import x_data
from RespFeatureExtractor.models_cola import Cola
# from RespFeatureExtractor.models_mae import mae_vit_small
from OPERA.src.model.models_mae import mae_vit_small
from huggingface_hub.file_download import hf_hub_download
from RespFeatureExtractor.util import get_split_signal_librosa, get_entire_signal_librosa, get_entire_signal_librosa_trimmed
from tqdm import tqdm


SR = 16000


ENCODER_PATH_OPERA_CE_EFFICIENTNET = "cks/model/encoder-operaCE.ckpt"
ENCODER_PATH_OPERA_CT_HT_SAT = "cks/model/encoder-operaCT.ckpt"
ENCODER_PATH_OPERA_GT_VIT = "cks/model/encoder-operaGT.ckpt"


def get_encoder_path(pretrain):
    encoder_paths = {
        "operaCT": ENCODER_PATH_OPERA_CT_HT_SAT,
        "operaCE": ENCODER_PATH_OPERA_CE_EFFICIENTNET,
        "operaGT": ENCODER_PATH_OPERA_GT_VIT
    }
    if not os.path.exists(encoder_paths[pretrain]):
        print("model ckpt not found, trying to download from huggingface")
        download_ckpt(pretrain)
    return encoder_paths[pretrain]


def download_ckpt(pretrain):
    model_repo = "evelyn0414/OPERA"
    model_name = "encoder-" + pretrain + ".ckpt"
    hf_hub_download(model_repo, model_name, local_dir="cks/model")


def initialize_pretrained_model(pretrain):
    if pretrain == "operaCT":
        model = Cola(encoder="htsat")
    elif pretrain == "operaCE":
        model = Cola(encoder="efficientnet")
    elif pretrain == "operaGT":
        model = mae_vit_small(norm_pix_loss=False,
                              in_chans=1, audio_exp=True,
                              img_size=(256, 64),
                              alpha=0.0, mode=0, use_custom_patch=False,
                              split_pos=False, pos_trainable=False, use_nce=False,
                              decoder_mode=0,  # decoder mode 0: global attn 1: swined local attn
                              mask_2d=False, mask_t_prob=0.7, mask_f_prob=0.3,
                              no_shift=False).float()
    else:
        raise NotImplementedError(f"Model not exist: {pretrain}, please check the parameter.")
    return model


def crop_window_around_peak(input_audio_path, window_duration=8.18):
    audio, sr = librosa.load(input_audio_path, sr=None)

    # Calculate the number of samples in the window
    window_samples = int(window_duration * sr)
    half_window = window_samples // 2

    # Detect the sample index of the maximum audio peak
    peak_index = np.argmax(np.abs(audio))

    # Define the start and end of the window
    start_index = max(0, peak_index - half_window)
    end_index = min(len(audio), peak_index + half_window)

    # Adjust window if it doesn't meet the required size
    if end_index - start_index < window_samples:
        if start_index == 0:
            end_index = min(len(audio), window_samples)
        elif end_index == len(audio):
            start_index = max(0, len(audio) - window_samples)

    # Crop the audio
    cropped_audio = audio[start_index:end_index]
    return cropped_audio


def get_spectrogram_patches(mel_spectrogram, patch_size=16):
    # Define patch size
    patch_height, patch_width = patch_size, patch_size

    # Split the mel-spectrogram into patches
    patches = []
    for i in range(0, mel_spectrogram.shape[0], patch_height):  # Up to down
        # for j in range(0, mel_spectrogram.shape[1], patch_width):  # Left to right
        # patch = mel_spectrogram[i:i + patch_height, j:j + patch_width]
        patch = mel_spectrogram[i:i + patch_height, :]
        patches.append(patch)
        # print(patch.shape)

    # Convert to numpy array for easier processing and verify dimensions

    patches = np.array(patches)
    return patches

class AuscultabaseExtractor:
    def __init__(self, pretrain="operaCT", from_spec=False, pad0=False):
        self.MAE = ("mae" in pretrain or "GT" in pretrain)
        self.from_spec = from_spec
        self.pad0 = pad0
        if pretrain == "operaCT":
            self.dim = 768
        else:
            self.dim = 1280
        encoder_path = "/local/scratch1/siam/pretrain_models/resp/auscultabase_model.ckpt"
        ckpt = torch.load(encoder_path)
        self.model = initialize_pretrained_model(pretrain)
        # self.model.eval()
        self.model.load_state_dict(ckpt["state_dict"], strict=False)
        print("Extracting feature from Auscultabase model...")

    def get_spectrogram(self, audio_file, input_sec=8):
        data = get_entire_signal_librosa_trimmed("", audio_file, spectrogram=True,
                                         input_sec=input_sec, pad=True)
        data = np.array(data)
        # data = np.expand_dims(data, axis=0)
        x = torch.tensor(data, dtype=torch.float)
        return x


    def extract_features(self, sound_dir_loc, input_sec=8):
        opera_features = []
        # for audio_file in sound_dir_loc:
        audio_file = sound_dir_loc
        if self.MAE:
            if self.from_spec:
                data = [audio_file[i: i + 256] for i in range(0, len(audio_file), 256)]
            else:
                data = get_split_signal_librosa("", audio_file, spectrogram=True,
                                                input_sec=input_sec)  ##8.18s --> T=256
            features = []
            for x in data:
                if x.shape[0] >= 16:  # Kernel size can't be greater than actual input size
                    x = np.expand_dims(x, axis=0)
                    x = torch.tensor(x, dtype=torch.float)
                    fea = self.model.forward_feature(x).detach().numpy()
                    features.append(fea)
            features_sta = np.mean(features, axis=0)
            # print('MAE ViT feature dim:', features_sta.shape)
            opera_features.append(features_sta.tolist())
        else:
            #  put entire audio into the model
            if self.from_spec:
                data = audio_file
            else:
                # input is filename of an audio
                if self.pad0:
                    data = get_entire_signal_librosa("", audio_file, spectrogram=True,
                                                     input_sec=input_sec, pad=True, types='zero')
                else:
                    data = get_entire_signal_librosa("", audio_file, spectrogram=True,
                                                     input_sec=input_sec, pad=True)

            data = np.array(data)

            # for entire audio, batchsize = 1
            data = np.expand_dims(data, axis=0)

            x = torch.tensor(data, dtype=torch.float)
            features = self.model.extract_feature(x, self.dim).detach().numpy()

            # for entire audio, batchsize = 1
            # opera_features.append(features.tolist()[0])

        # x_data = np.array(features.tolist()[0])
        x_data = np.array(features)
        if self.MAE: x_data = x_data.squeeze(1)
        # print(x_data.shape)
        return x_data

class OperaExtractor:
    def __init__(self, pretrain="operaCT", from_spec=False, pad0=False):
        self.MAE = ("mae" in pretrain or "GT" in pretrain)
        self.from_spec = from_spec
        self.pad0 = pad0
        if pretrain == "operaCT":
            self.dim = 768
        else:
            self.dim = 1280
        encoder_path = get_encoder_path(pretrain)
        ckpt = torch.load(encoder_path)
        self.model = initialize_pretrained_model(pretrain)
        # self.model.eval()
        self.model.load_state_dict(ckpt["state_dict"], strict=False)
        print("Extracting feature from {} model.".format(pretrain))

    def get_spectrogram(self, audio_file, input_sec=8):
        data = get_entire_signal_librosa_trimmed("", audio_file, spectrogram=True,
                                         input_sec=input_sec, pad=True)
        data = np.array(data)
        # data = np.expand_dims(data, axis=0)
        x = torch.tensor(data, dtype=torch.float)
        return x


    def extract_opera_features(self, sound_dir_loc, input_sec=8):
        opera_features = []
        # for audio_file in sound_dir_loc:
        audio_file = sound_dir_loc
        if self.MAE:
            if self.from_spec:
                data = [audio_file[i: i + 256] for i in range(0, len(audio_file), 256)]
            else:
                data = get_split_signal_librosa("", audio_file, spectrogram=True,
                                                input_sec=input_sec)  ##8.18s --> T=256
            features = []
            for x in data:
                if x.shape[0] >= 16:  # Kernel size can't be greater than actual input size
                    x = np.expand_dims(x, axis=0)
                    x = torch.tensor(x, dtype=torch.float)
                    fea = self.model.forward_feature(x).detach().numpy()
                    features.append(fea)
            features_sta = np.mean(features, axis=0)
            # print('MAE ViT feature dim:', features_sta.shape)
            opera_features.append(features_sta.tolist())
        else:
            #  put entire audio into the model
            if self.from_spec:
                data = audio_file
            else:
                # input is filename of an audio
                if self.pad0:
                    data = get_entire_signal_librosa("", audio_file, spectrogram=True,
                                                     input_sec=input_sec, pad=True, types='zero')
                else:
                    data = get_entire_signal_librosa("", audio_file, spectrogram=True,
                                                     input_sec=input_sec, pad=True)

            data = np.array(data)

            # for entire audio, batchsize = 1
            data = np.expand_dims(data, axis=0)

            x = torch.tensor(data, dtype=torch.float)
            features = self.model.extract_feature(x, self.dim).detach().numpy()

            # for entire audio, batchsize = 1
            # opera_features.append(features.tolist()[0])

        # x_data = np.array(features.tolist()[0])
        x_data = np.array(features)
        if self.MAE: x_data = x_data.squeeze(1)
        # print(x_data.shape)
        return x_data

    def extract_opera_features_add_noise(self, sound_dir_loc, noise_type = 'gaussian', noise_db=-20, input_sec=8):
        opera_features = []
        audio_file = sound_dir_loc
        data = get_entire_signal_librosa("", audio_file, spectrogram=True, input_sec=input_sec, pad=True,
                                         noise_type=noise_type, noise_db=noise_db)
        data = np.array(data)

        # if noise_db is not None:
        #     # print('Noise : ', noise_db)
        #     # Convert noise_db (dB) to a linear noise factor
        #     if noise_type=='gaussian':
        #         rms_signal = np.sqrt(np.mean(data ** 2))
        #         rms_noise = rms_signal / (10 ** (noise_db / 20))
        #         noise = np.random.normal(0, rms_noise, data.shape)
        #     elif noise_type=='babble':
        #         print('babble..')
        #         babble_sources_files = ['/home/siam.5/Resp/Resp-Disease-Identification/src/noise_input/1089-134691-0002.flac',
        #                           '/home/siam.5/Resp/Resp-Disease-Identification/src/noise_input/8224-274381-0016.flac']
        #
        #         babble_sources = []
        #         for babble_source_file in babble_sources_files:
        #             babble_sources.append(librosa.load(babble_source_file)[0])
        #
        #         babble = np.zeros_like(data)
        #         print('data shape = ', data.shape)
        #         print('babble shape = ', babble_sources[0].shape)
        #         for src in babble_sources:
        #             src = librosa.util.fix_length(src, size=len(data))  # match length
        #             babble += src
        #
        #         babble /= len(babble_sources)  # normalize
        #
        #         rms_signal = np.sqrt(np.mean(data ** 2))
        #         rms_babble = np.sqrt(np.mean(babble ** 2))
        #         desired_rms_babble = rms_signal / (10 ** (noise_db / 20))
        #         babble = babble * (desired_rms_babble / (rms_babble + 1e-8))
        #         noise = babble
        #
        #     elif noise_type=='reverberation':
        #         pass
        #
        #     data = data + noise

        data = np.expand_dims(data, axis=0)
        x = torch.tensor(data, dtype=torch.float32)
        features = self.model.extract_feature(x, self.dim).detach().numpy()
        x_data = np.array(features)
        return x_data

    def extract_opera_features_split_all(self, sound_dir_loc, input_sec=1):
        audio_file = sound_dir_loc
        data = get_split_signal_librosa("", audio_file, spectrogram=True, input_sec=input_sec)  ##8.18s --> T=256
        features = []
        for x in data:
            if x.shape[0] >= 16:  # Kernel size can't be greater than actual input size
                x = np.expand_dims(x, axis=0)
                x = torch.tensor(x, dtype=torch.float)
                fea = self.model.extract_feature(x, self.dim).detach().numpy()
                features.append(fea)

        return np.array(features).squeeze(1)

    def extract_opera_features_multi_token(self, sound_dir_loc, input_sec=8.18):
        audio_file = sound_dir_loc
        # audio file -> mel-spectrogram
        mel_spectrogram = get_entire_signal_librosa_trimmed("", audio_file, pad=True, spectrogram=True, input_sec=input_sec)  ##8.18s --> T=256
        # print("Mel spectrogram shape: ", mel_spectrogram.shape)
        # get 16x16 patches from entire mel-spec
        spec_patches = get_spectrogram_patches(mel_spectrogram, patch_size=4)

        features = []
        for x in spec_patches:
            if x.shape[0] >= 4:  # Kernel size can't be greater than actual input size
                x = np.expand_dims(x, axis=0)
                x = torch.tensor(x, dtype=torch.float)
                fea = self.model.extract_feature(x, self.dim).detach().numpy()
                features.append(fea)

        return np.array(features).squeeze(1)




# def extract_opera_feature(sound_dir_loc, pretrain="operaCT", input_sec=8, from_spec=False, dim=768, pad0=False):
#
#     print("extracting feature from {} model with input_sec {}".format(pretrain, input_sec))
#
#     MAE = ("mae" in pretrain or "GT" in pretrain)
#
#     encoder_path = get_encoder_path(pretrain)
#     ckpt = torch.load(encoder_path)
#     model = initialize_pretrained_model(pretrain)
#     model.eval()
#     model.load_state_dict(ckpt["state_dict"], strict=False)
#
#     opera_features = []
#
#     for audio_file in tqdm(sound_dir_loc):
#
#         if MAE:
#             if from_spec:
#                 data = [audio_file[i: i + 256] for i in range(0, len(audio_file), 256)]
#             else:
#                 data = get_split_signal_librosa("", audio_file[:-4], spectrogram=True,
#                                                 input_sec=input_sec)  ##8.18s --> T=256
#             features = []
#             for x in data:
#                 if x.shape[0] >= 16:  # Kernel size can't be greater than actual input size
#                     x = np.expand_dims(x, axis=0)
#                     x = torch.tensor(x, dtype=torch.float)
#                     fea = model.forward_feature(x).detach().numpy()
#                     features.append(fea)
#             features_sta = np.mean(features, axis=0)
#             # print('MAE ViT feature dim:', features_sta.shape)
#             opera_features.append(features_sta.tolist())
#         else:
#             #  put entire audio into the model
#             if from_spec:
#                 data = audio_file
#             else:
#                 # input is filename of an audio
#                 if pad0:
#                     data = get_entire_signal_librosa("", audio_file[:-4], spectrogram=True,
#                                                      input_sec=input_sec, pad=True, types='zero')
#                 else:
#                     data = get_entire_signal_librosa("", audio_file[:-4], spectrogram=True,
#                                                      input_sec=input_sec, pad=True)
#
#             data = np.array(data)
#
#             # for entire audio, batchsize = 1
#             data = np.expand_dims(data, axis=0)
#
#             x = torch.tensor(data, dtype=torch.float)
#             features = model.extract_feature(x, dim).detach().numpy()
#
#             # for entire audio, batchsize = 1
#             opera_features.append(features.tolist()[0])
#
#     x_data = np.array(opera_features)
#     if MAE: x_data = x_data.squeeze(1)
#     print(x_data.shape)
#     return x_data
#
#
#
#

