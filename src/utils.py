import numpy as np
import pywt
import torch
import torchaudio.transforms as transforms
from torch.utils.data import random_split


data_mapping_disease = {'icbhi': 'copd',
                        'kauh_copd': 'copd',
                        'kauh_asthma': 'asthma',
                        'kauh_pneumonia': 'pneumonia',
                        'ukcovid19': 'covid',
                        'coswara': 'covid',
                        'coughvid': 'covid',
                        'codaTB': 'TB',
                        'TBscreen': 'TB'}

def get_dataset_dir(dataset_name):
    if dataset_name == "coughvid":
        dataset_root_dir = "/local/scratch1/siam/dataset/cough-sound-data/COUGHVIDv3/"
        model_type = 'covid_'
        train_path = dataset_root_dir + f"custom_files/train_features.h5"
        test_path = dataset_root_dir + f"custom_files/test_features.h5"

    elif dataset_name == "ukcovid19":
        dataset_root_dir = "/local/scratch1/siam/dataset/resp-dataset/uk-covid-19/"
        model_type = 'covid_'
        train_path = dataset_root_dir + f"custom_files/train_features.h5"
        test_path = dataset_root_dir + f"custom_files/test_features.h5"

    elif dataset_name == "coswara":
        dataset_root_dir = "/local/scratch1/siam/dataset/cough-sound-data/coswara/download/Coswara-Data/"
        model_type = 'covid_'
        cough_type = "cough-shallow"
        train_path = dataset_root_dir + f"custom_files/alldata_{cough_type}_features_qc.h5"
        test_path = dataset_root_dir + f"custom_files/alldata_{cough_type}_features_qc.h5"
        # test_path = dataset_root_dir + f"custom_files/balanceddata2_cough-shallow_features_whole_operaCT.h5"

    elif dataset_name == "TBscreen":
        dataset_root_dir = "/local/scratch1/siam/dataset/cough-sound-data/TBscreen/data/TBscreen_Dataset/"
        model_type = 'TB_'
        train_path = dataset_root_dir + f"custom_files/train_features.h5"
        test_path = dataset_root_dir + f"custom_files/test_features.h5"

    elif dataset_name == "codaTB":
        dataset_root_dir = "/local/scratch1/siam/dataset/cough-sound-data/CODA_DREAM/Train/"
        model_type = 'TB_'
        train_path = dataset_root_dir + f"custom_files/train_features.h5"
        test_path = dataset_root_dir + f"custom_files/test_features.h5"

    elif dataset_name == "icbhi":
        dataset_root_dir = "/local/scratch1/siam/dataset/cough-sound-data/ICBHI/data/"
        model_type = 'copd_'
        train_path = dataset_root_dir + f"custom_files/train_features.h5"
        test_path = dataset_root_dir + f"custom_files/test_features.h5"

    elif dataset_name == "kauh_copd":
        dataset_root_dir = "/local/scratch1/siam/dataset/cough-sound-data/KAUH/"
        model_type = 'copd_'
        train_path = dataset_root_dir + f"custom_files/test_features_copd.h5"
        test_path = dataset_root_dir + f"custom_files/test_features_copd.h5"

    elif dataset_name == "kauh_asthma":
        dataset_root_dir = "/local/scratch1/siam/dataset/cough-sound-data/KAUH/"
        model_type = 'copd_'
        train_path = dataset_root_dir + f"custom_files/test_features_asthma.h5"
        test_path = dataset_root_dir + f"custom_files/test_features_asthma.h5"

    elif dataset_name == "kauh_pneumonia":
        dataset_root_dir = "/local/scratch1/siam/dataset/cough-sound-data/KAUH/"
        model_type = 'copd_'
        train_path = dataset_root_dir + f"custom_files/test_features_pneumonia.h5"
        test_path = dataset_root_dir + f"custom_files/test_features_pneumonia.h5"

    return dataset_root_dir, model_type, train_path, test_path


def get_prompt(configs, dataset="ukcovid19", label="covid", modality="cough"):
    label = data_mapping_disease.get(dataset)
    data_description = {
        "covid19sounds": "This data comes from the COVID-19 Sounds dataset.",
        "ukcovid19": "This data comes from the UK COVID-19 Vocal Audio Dataset.",
        "icbhi": "This data comes from the ICBHI Respiratory Sound Database Dataset.",
        "coswara": "This data comes from the Coswara Covid-19 dataset. ",
        "coughvid": "This data comes from the CoughVID dataset. ",
        "TBscreen": "This data comes from the TBscreen dataset. ",
        "codaTB": "This data comes from the CODA TB DREAM CHALLENGE dataset. ",
        "kauh_copd": "This data comes from the KAUH lung sound dataset, containing lung sounds recorded from the chest wall using an electronic stethoscope.",
        "kauh_asthma": "This data comes from the KAUH lung sound dataset, containing lung sounds recorded from the chest wall using an electronic stethoscope.",
        "kauh_pneumonia": "This data comes from the KAUH lung sound dataset, containing lung sounds recorded from the chest wall using an electronic stethoscope.",
    }

    task_description = {
       "covid": "whether the participant has COVID-19",
        "TB": "whether the participant has Tuberculosis (TB)",
        "copd": " whether the person has Chronic obstructive pulmonary disease (COPD)",
        "asthma": " whether the person has asthma",
        "pneumonia": " whether the person has pneumonia",
    }

    classes = {
        "covid": "non-COVID19, COVID19",
        "TB": "Non-TB, TB",
        "copd": "healthy, COPD",
        "asthma": "healthy, asthma",
        "pneumonia": "healthy, pneumonia",
    }

    # n_cls = len(classes[label].split(","))
    n_cls = 2

    if configs.multimodal_config == 1:
        prompt = (
                    f"<|start_prompt|>"
                    f"Dataset description: {data_description[dataset]} "
                    f"Task description: classify {task_description[label]} given the following information and audio of the person's {modality} sounds. "
                    f"The {n_cls} classes are: {classes[label]}. "
                    f"Please output 0 for {classes[label].split(',')[0]} and 1 for {classes[label].split(',')[1]}."
                    "<|<end_prompt>|>"
                )
    elif configs.multimodal_config == 2:
        prompt = (
            f"<|start_prompt|> Dataset description: {data_description[dataset]} "
            f"Task description: classify {task_description[label]} given the following information."
            f"The {n_cls} classes are: {classes[label]}. <|<end_prompt>|>"
        )

    elif configs.multimodal_config == 3:
        prompt = (
            f"<|start_prompt|> Dataset description: {data_description[dataset]} "
            f"Task description: classify {task_description[label]} given the audio of the person's {modality} sounds."
            f"The {n_cls} classes are: {classes[label]}. <|<end_prompt>|>"
        )

    return prompt


class wavelet:
    def __init__(self, f_n: int, f_min: float, f_max: float, dt: float):
        self.number_of_frequences: np.ndarray = np.array(int(f_n))
        self.frequency_range: np.ndarray = np.array((f_min, f_max))
        self.dt: np.ndarray = np.array(dt)
        self.s_spacing: np.ndarray = (1.0 / (self.number_of_frequences - 1)) * np.log2(
            self.frequency_range.max() / self.frequency_range.min())
        self.scale: np.ndarray = np.power(2, np.arange(0, self.number_of_frequences) * self.s_spacing)
        self.frequency_axis: np.ndarray = self.frequency_range.min() * np.flip(self.scale)
        self.wave_scales: np.ndarray = 1.0 / (self.frequency_axis * self.dt)
        self.frequency_axis = (pywt.scale2frequency("cmor1.5-1.0", self.wave_scales) / self.dt)
        self.mother = pywt.ContinuousWavelet("cmor1.5-1.0")
        self.cone_of_influence: np.ndarray = np.ceil(np.sqrt(2) * self.wave_scales).astype(np.int64)

    def __call__(self, audio):
        audio = audio.detach().cpu().numpy()
        complex_spectrum, _ = pywt.cwt(audio, self.wave_scales, self.mother, self.dt)
        fill_value: float = 0
        assert complex_spectrum.shape[0] == self.cone_of_influence.shape[0]
        for frequency_id in range(0, self.cone_of_influence.shape[0]):
            start_id: int = 0
            end_id: int = int(np.min((self.cone_of_influence[frequency_id], complex_spectrum.shape[1])))
            complex_spectrum[frequency_id, start_id:end_id] = fill_value
            start_id = np.max((complex_spectrum.shape[1] - self.cone_of_influence[frequency_id] - 1, 0,))
            end_id = complex_spectrum.shape[1]
            complex_spectrum[frequency_id, start_id:end_id] = fill_value
            complex_spec = torch.from_numpy(abs(complex_spectrum) ** 2)
            complex_spec = torch.unsqueeze(complex_spec, dim=0)
            complex_spec = transforms.AmplitudeToDB()(complex_spec)
        return complex_spec


def get_subset(dataset, fraction=0.2, seed=42):
    total_len = len(dataset)
    subset_len = int(total_len * fraction)
    rest_len = total_len - subset_len
    return random_split(dataset, [subset_len, rest_len], generator=torch.Generator().manual_seed(seed))[0]
