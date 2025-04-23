import csv
import math
import random
from pathlib import Path
import os
import fnmatch
import blobfile as bf
import warnings
from mpi4py import MPI
import numpy as np
from numpy import ndarray
import torch
from torch.utils.data import DataLoader, Dataset
from torchaudio import functional as af
import soundfile as sf
import pyloudnorm as pyln

from typing import Optional, Union

def normalize_waveform(w: torch.Tensor, sr: int) -> torch.Tensor:
    meter = pyln.Meter(sr)  # 建立測量器
    loudness = meter.integrated_loudness(w.numpy())
    # print(loudness, w.std())
    delta_loudness = 2.66 - loudness
    gain = np.power(10.0, delta_loudness/20.0)
    # print(gain)
    w = gain * w
    # loudness = meter.integrated_loudness(w.numpy())
    # print(loudness, w.std())
    return w

def load_data(*, dataset, batch_size, max_len, deterministic=False, num_workers=0, cond_drop_rate=0.5):
    """

    """

    def _collate(data: tuple[torch.Tensor, ...], is_cond=False):
        crop_len = min(max(w.shape[0] for w in data), max_len if not is_cond else 80000)
        cropped_data = torch.zeros((len(data), crop_len))
        mask = torch.ones_like(cropped_data, dtype=torch.bool)
        for i, w in enumerate(data):
            # w = normalize_waveform(w, dataset.sampling_rate)
            if crop_len <= w.shape[0]:
                s = random.randint(0, w.shape[0] - crop_len)
                cropped_data[i] = w[s:s+crop_len] - w[s:s+crop_len].mean() * random.uniform(0, 2)
                mask[i, :] = False
            else:
                if is_cond:
                    s = random.randint(0, crop_len - w.shape[0])
                    cropped_data[i, :w.shape[0]] = w - w.mean() * random.uniform(0, 2)
                    mask[i, :w.shape[0]] = False
                else:
                    s = random.randint(0,  crop_len - w.shape[0])
                    cropped_data[i, s:s+w.shape[0]] = w - w.mean() * random.uniform(0, 2)
                    mask[i, s:s+w.shape[0]] = False
            cropped_data[i] = cropped_data[i] / torch.max(torch.abs(cropped_data[i])).clamp_min(1)

        if is_cond:
            cond = torch.rand(mask.shape[0], 1) <= cond_drop_rate
            mask = torch.logical_or(cond, mask)
        return cropped_data, mask

    def collate_fn(data: list[tuple[torch.Tensor, torch.Tensor]]):
        datas = list(zip(*data))
        return (_collate(datas[0]), _collate(datas[1], True))

    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=not deterministic, num_workers=num_workers,
                        collate_fn=collate_fn, drop_last=True, pin_memory=True)
    while True:
        yield from loader


class VCTKDataset(Dataset):
    def __init__(self,
                 root: str,
                 mic_id: str = "mic2",
                 sampling_rate: int=16_000,
                 cache_dir: Optional[str] = None) -> None:
        super().__init__()
        self.root = root
        self.sampling_rate = sampling_rate
        self.cache_dir = cache_dir

        with open(f"{root}/speaker-info.txt", newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=' ', skipinitialspace=True)
            self.spks = [row["ID"] for row in reader]
        
        self.files = {}
        self.cache_files = {}
        self.mapping_table = []

        # for spk in self.spks:
        for spk in os.listdir(f"{root}/wav48_silence_trimmed/"):
            if not os.path.isdir(f"{root}/wav48_silence_trimmed/{spk}"):
                continue
            self.files[spk] = []
            self.cache_files[spk] = []
            for f in os.listdir(f"{root}/wav48_silence_trimmed/{spk}"):
                if not fnmatch.fnmatch(f, f"*{mic_id}.flac"):
                    continue
                self.files[spk].append(f"{root}/wav48_silence_trimmed/{spk}/{f}")
                if os.path.isfile(f"{self.cache_dir}/{spk}/{f}"):
                    self.cache_files[spk].append(f"{self.cache_dir}/{spk}/{f}")
                else:
                    self.cache_files[spk].append(f"{root}/wav48_silence_trimmed/{spk}/{f}")
                self.mapping_table.append((spk, len(self.files[spk])-1))
 
    def __len__(self) -> int:
        return len(self.mapping_table)
    
    def __getitem__(self, index):
        spk, i = self.mapping_table[index]
        w1 = self._get_data(spk, i)
        j = random.randint(0, len(self.files[spk])-1)
        while j == i:
            j = random.randint(0, len(self.files[spk])-1)
        w2 = self._get_data(spk, j)

        return w1, w2

    def _get_data(self, spk: str, i: int):

        f = self.cache_files[spk][i]
        try:
            w, r = sf.read(f)
        except:
            print(f)
            f = self.files[spk][i]
            w, r = sf.read(f)
        w = torch.from_numpy(w)

        if r != self.sampling_rate:
            if f != self.files[spk][i]:
                f = self.files[spk][i]
                w, r = sf.read(f)
                w = torch.from_numpy(w)
            if r != self.sampling_rate:
                w = af.resample(w, r, self.sampling_rate)

            if self.cache_dir is not None:
                name = os.path.basename(f)
                self.cache_files[spk][i] = f"{self.cache_dir}/{spk}/{name}"
                Path(f"{self.cache_dir}/{spk}").mkdir(parents=True, exist_ok=True)
                sf.write(self.cache_files[spk][i], w.numpy(), self.sampling_rate)

        return w

class LibriTTSDataset(Dataset):
    def __init__(self,
                 root: str,
                 subsets: Union[list[str], str] = "train-clean-100",
                 sampling_rate: int=16_000,
                 cache_dir: Optional[str] = None) -> None:
        super().__init__()
        self.root = root
        self.sampling_rate = sampling_rate
        self.cache_dir = cache_dir
        self.subsets = [subsets] if isinstance(subsets, str) else subsets
        self.files = {}
        self.cache_files = {}
        self.mapping_table = []

        for subset in self.subsets:
            for spk in os.listdir(f"{root}/{subset}/"):
                if not os.path.isdir(f"{root}/{subset}/{spk}"):
                    continue
                if spk not in self.files:
                    self.files[spk] = []
                    self.cache_files[spk] = []
                for chapter in os.listdir(f"{root}/{subset}/{spk}"):
                    if not os.path.isdir(f"{root}/{subset}/{spk}/{chapter}"):
                        continue
                    for f in os.listdir(f"{root}/{subset}/{spk}/{chapter}"):
                        if not fnmatch.fnmatch(f, f"*.wav"):
                            continue
                        self.files[spk].append(f"{root}/{subset}/{spk}/{chapter}/{f}")
                        if os.path.isfile(f"{self.cache_dir}/{spk}/{f}"):
                            self.cache_files[spk].append(f"{self.cache_dir}/{spk}/{f}")
                        else:
                            self.cache_files[spk].append(f"{root}/{subset}/{spk}/{chapter}/{f}")
                        self.mapping_table.append((spk, len(self.files[spk])-1))
 
    def __len__(self) -> int:
        return len(self.mapping_table)
    
    def __getitem__(self, index):
        spk, i = self.mapping_table[index]
        w1 = self._get_data(spk, i)
        j = random.randint(0, len(self.files[spk])-1)
        while j == i and len(self.files[spk]) > 1:
            j = random.randint(0, len(self.files[spk])-1)
        w2 = self._get_data(spk, j)

        return w1, w2

    def _get_data(self, spk: str, i: int):

        f = self.cache_files[spk][i]
        try:
            w, r = sf.read(f)
        except:
            print(f)
            f = self.files[spk][i]
            w, r = sf.read(f)
        w = torch.from_numpy(w)

        if r != self.sampling_rate:
            if f != self.files[spk][i]:
                f = self.files[spk][i]
                w, r = sf.read(f)
                w = torch.from_numpy(w)
            if r != self.sampling_rate:
                w = af.resample(w, r, self.sampling_rate)

            if self.cache_dir is not None:
                name = os.path.basename(f)
                self.cache_files[spk][i] = f"{self.cache_dir}/{spk}/{name}"
                Path(f"{self.cache_dir}/{spk}").mkdir(parents=True, exist_ok=True)
                sf.write(self.cache_files[spk][i], w.numpy(), self.sampling_rate)

        return w
# libritts = LibriTTSDataset("/media/8tsp/dataset/LibriTTS_R", cache_dir="./libritts")
# for (w, mask_w), (ref, mask_ref) in load_data(dataset=libritts, batch_size=64, max_len=32_000, num_workers=8):
#     print(w.shape, ref.shape)
#     break