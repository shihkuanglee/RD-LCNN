import os, torch, torchaudio
import torch.nn.functional as F
import numpy               as np
from torch.utils.data import Dataset

from librosa import util
from scipy.fft import dct
from scipy.signal import get_window

from ADFA.adfa import adfa_arb, mdfa_arb, cqa_arb

# return dict of cms with 1 (for bona fide) and 0 (for spoof)
# def dict_cm_protocols_ASVspoofing_2019(path_cm_protocols):

#     ndarray = np.genfromtxt(path_cm_protocols, dtype=str)
#     list_filename = list()
#     list_______cm = list()
#     for line in ndarray:
#         list_filename.append(str(line[1]))
#         list_______cm.append(str(line[4]))
#     dict_cm = dict(zip(list_filename, list_______cm))

#     list_set__cms = list(set(list_______cm))
#     list_set__cms.sort()
#     assert list_set__cms == ['bonafide', 'spoof']

#     list_filename = list()
#     list_______cm = list()
#     for key in list(dict_cm):
#         list_filename.append(key)
#         if dict_cm[key] == 'bonafide':
#             list_______cm.append(1)
#         else :
#             list_______cm.append(0)
#     dict_cm = dict(zip(list_filename, list_______cm))

#     return dict_cm

def dict_cm_protocols_ASVspoofing_2019(path_cm_protocols):
    ndarray = np.genfromtxt(path_cm_protocols, dtype=str)

    list_filename = [str(line[1]) for line in ndarray]
    list_______cm = [str(line[4]) for line in ndarray]
    dict_cm = {filename: cm for filename, cm in zip(list_filename, list_______cm)}

    set_cms = set(list_______cm)
    set_cms = sorted(set_cms)
    assert set_cms == ['bonafide', 'spoof']

    dict_cm = {key: 1 if value == 'bonafide' else 0 for key, value in dict_cm.items()}

    return dict_cm

# ===========================================
# return list of paths, dict of cms, asv data
# =============================================================================
def get_list_dict_task_online(path_data, task, feature_folder, file_extension):

    path_data__root = path_data / task
    path_cm_protocols__root =         path_data__root / f'ASVspoof2019_{task}_cm_protocols'
    path_cm_protocols_train = path_cm_protocols__root / f'ASVspoof2019.{task}.cm.train.trn.txt'
    path_cm_protocols___dev = path_cm_protocols__root / f'ASVspoof2019.{task}.cm.dev.trl.txt'
    path_cm_protocols__eval = path_cm_protocols__root / f'ASVspoof2019.{task}.cm.eval.trl.txt'
    dict_cm_train = dict_cm_protocols_ASVspoofing_2019(path_cm_protocols_train)
    dict_cm___dev = dict_cm_protocols_ASVspoofing_2019(path_cm_protocols___dev)
    dict_cm__eval = dict_cm_protocols_ASVspoofing_2019(path_cm_protocols__eval)

    path_asv_scores__root =       path_data__root / f'ASVspoof2019_{task}_asv_scores'
    path_asv_scores___dev = path_asv_scores__root / f'ASVspoof2019.{task}.asv.dev.gi.trl.scores.txt'
    path_asv_scores__eval = path_asv_scores__root / f'ASVspoof2019.{task}.asv.eval.gi.trl.scores.txt'
    asv_data__dev = np.genfromtxt(path_asv_scores___dev, dtype=str)
    asv_data_eval = np.genfromtxt(path_asv_scores__eval, dtype=str)

    path_data_train =  path_data__root / f'ASVspoof2019_{task}_train' / feature_folder
    path_data___dev =  path_data__root / f'ASVspoof2019_{task}_dev'   / feature_folder
    path_data__eval =  path_data__root / f'ASVspoof2019_{task}_eval'  / feature_folder
    list_path_train = [path_data_train / f'{name}.{file_extension}' for name in list(dict_cm_train)]
    list_path___dev = [path_data___dev / f'{name}.{file_extension}' for name in list(dict_cm___dev)]
    list_path__eval = [path_data__eval / f'{name}.{file_extension}' for name in list(dict_cm__eval)]

    return list_path_train, dict_cm_train, \
           list_path___dev, dict_cm___dev, asv_data__dev, \
           list_path__eval, dict_cm__eval, asv_data_eval

# return list of paths, dict of cms, asv data
def get_list_dict_train(path_data, task, feature_folder, file_extension):

    path_data__root = path_data / task
    path_cm_protocols__root =         path_data__root / f'ASVspoof2019_{task}_cm_protocols'
    path_cm_protocols_train = path_cm_protocols__root / f'ASVspoof2019.{task}.cm.train.trn.txt'
    dict_cm_train = dict_cm_protocols_ASVspoofing_2019(path_cm_protocols_train)

    path_data_train =  path_data__root / f'ASVspoof2019_{task}_train' / feature_folder
    list_path_train = [path_data_train / f'{name}.{file_extension}' for name in list(dict_cm_train)]

    return list_path_train, dict_cm_train

# ================================================================
# create empty array then fill it with feature according to dmode.
# ================================================================
def shared_fill_block_tra(blck, dim_f, dim_t, dim_t_max):
    frames_num = blck.size()[2]
    blck_tar = torch.zeros(1, dim_f, dim_t_max)
    if frames_num <= dim_t:
        start_pt = torch.randint(0, dim_t - frames_num + 1, (1,))
        blck_tar[...,start_pt:start_pt + frames_num] = blck
    else:
        start_pt = torch.randint(0, frames_num - dim_t + 1, (1,))
        blck_tar[...,:dim_t] = blck[...,start_pt:start_pt + dim_t]
    frames_num = dim_t
    return blck_tar, frames_num

def shared_fill_block_val(blck, dim_f, dim_t, dim_t_max):
    frames_num = blck.size()[2]
    blck_tar = torch.zeros(1, dim_f, dim_t_max)
    blck_tar[...,:frames_num] = blck
    return blck_tar, frames_num

def shared_fill_block_fix(blck, dim_f, dim_t, dim_t_max):
    frames_num = blck.size()[2]
    return blck, frames_num

def get_shared_fill_block(dmode):
    if   dmode.lower() == 'train': return shared_fill_block_tra
    elif dmode.lower() ==  'eval': return shared_fill_block_val
    elif dmode.lower() == 'fixed': return shared_fill_block_fix
    else: raise ValueError('Check the dmode of Dataset!')



def get_window_fn(window_fn_name):
    if   window_fn_name == 'blackman': return torch.blackman_window
    elif window_fn_name ==  'hamming': return torch.hamming_window
    else: raise ValueError('Check the window_fn_name!')

def init____spec(self, config, section):
    self.samplerate    = config.getint(section, 'samplerate')
    self.n_fft         = config.getint(section, 'n_fft')
    self.win_length    = config.getint(section, 'win_length')
    self.hop_length    = config.getint(section, 'hop_length')
    self.pad           = 0
    self.window_fn     = get_window_fn(config.get(section, 'window_fn'))
    self.dim_f         = config.getint(section, 'dim_f')
    self.dim_t         = config.getint(section, 'dim_t')
    self.dim_t_max     = config.getint(section, 'dim_t_max') + self.dim_t
    self.dim_t_shf     = config.getint(section, 'dim_t_shf')
    self.power         = config.getint(section, 'power')
    self.log           = config.getint(section, 'log')

def getitem_spec(self, idx):
    filepath = self.list_path[idx]
    sig, _   = torchaudio.load(filepath)
    wavename = filepath.stem
    cm       = torch.tensor(self.dict_cm[wavename])
    specgram = torchaudio.transforms.Spectrogram(
                    n_fft      = self.n_fft,
                    win_length = self.win_length,
                    hop_length = self.hop_length,
                    pad        = self.pad,
                    window_fn  = self.window_fn,
                    power      = self.power)(sig)
    spec_tar, frames_num = self.shared_fill_block(specgram, self.dim_f, self.dim_t, self.dim_t_max)
    if self.log == 1: spec_tar = torch.log(F.relu(spec_tar, inplace=True) + torch.finfo(torch.float32).eps)
    return wavename, spec_tar, cm, frames_num



def getitem_ceps(self, idx):
    filepath = self.list_path[idx]
    sig, _   = torchaudio.load(filepath)
    wavename = filepath.stem
    cm       = torch.tensor(self.dict_cm[wavename])
    specgram = torchaudio.transforms.Spectrogram(
                    n_fft      = self.n_fft,
                    win_length = self.win_length,
                    hop_length = self.hop_length,
                    pad        = self.pad,
                    window_fn  = self.window_fn,
                    power      = self.power)(sig)
    log_spec = torch.log(F.relu(specgram, inplace=True) + torch.finfo(torch.float32).eps)
    cepsgram = torch.tensor(dct(log_spec.numpy(), axis=-2, norm='ortho'), dtype=torch.float32)
    ceps_tar, frames_num = self.shared_fill_block(cepsgram, self.dim_f, self.dim_t, self.dim_t_max)
    return wavename, ceps_tar, cm, frames_num



def framming_w_window(y, n_fft, hop_length, win_length, window):
    """Modify from stft implementation of Librosa.

    Copyright (c) 2013--2017, librosa development team.

    Permission to use, copy, modify, and/or distribute this software for any
    purpose with or without fee is hereby granted, provided that the above
    copyright notice and this permission notice appear in all copies.

    THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
    WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
    MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
    ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
    WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
    ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
    OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE."""

    # By default, use the entire frame
    if win_length is None: win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None: hop_length = int(win_length // 4)

    # fft_window = get_window(window, win_length, fftbins=True)
    fft_window = get_window(window, win_length, fftbins=True).astype(np.float32)

    # Pad the window out to n_fft size
    fft_window = util.pad_center(fft_window, size=n_fft)

    # Reshape so that the window can be broadcast
    fft_window = fft_window.reshape((-1, 1))

    # Check audio is valid
    util.valid_audio(y)

    # Window the time series.
    y_frames = util.frame(y, frame_length=n_fft, hop_length=hop_length)

    return fft_window * y_frames

def init____adfa(self, config, section):
    init____spec(self, config, section)
    self.window_fn = config.get(section, 'window_fn')
    self.m         = adfa_arb(self.dim_f, self.n_fft).astype(np.csingle)

def init____mdfa(self, config, section):
    init____spec(self, config, section)
    self.window_fn = config.get(section, 'window_fn')
    self.m         = mdfa_arb(self.dim_f, self.samplerate, self.n_fft).astype(np.csingle)

def init_____cqa(self, config, section):
    init____spec(self, config, section)
    self.window_fn = config.get(   section, 'window_fn')
    self.base      = config.getint(section, 'base')
    self.bins      = config.getint(section, 'bins')
    self.m         = cqa_arb(self.dim_f, self.base, self.bins, self.n_fft).astype(np.csingle)

def getitem_adfa(self, idx):
    filepath = self.list_path[idx]
    sig, _   = torchaudio.load(filepath)
    wavename = filepath.stem
    cm       = torch.tensor(self.dict_cm[wavename])
    frames   = framming_w_window(y          = sig[0].numpy(), 
                                 n_fft      = self.n_fft,
                                 hop_length = self.hop_length,
                                 win_length = self.win_length,
                                 window     = self.window_fn)
    adfagram = torch.unsqueeze(torch.tensor(np.abs(np.matmul(self.m, frames)), dtype=torch.float32), 0)
    tar_gram, frames_num = self.shared_fill_block(adfagram, self.dim_f, self.dim_t, self.dim_t_max)
    if self.power > 1: tar_gram.pow_(self.power)
    if self.log  == 1: tar_gram = torch.log(F.relu(tar_gram, inplace=True) + torch.finfo(torch.float32).eps)
    return wavename, tar_gram, cm, frames_num



def stdct(y, n_fft, hop_length, win_length, window):

    framming = framming_w_window(y, n_fft, hop_length, win_length, window)

    return  dct(framming, axis=0, norm='ortho')

def init_____dct(self, config, section):
    init____spec(self, config, section)
    self.window_fn = config.get(section, 'window_fn')

def getitem_dct(self, idx):
    filepath = self.list_path[idx]
    sig, _   = torchaudio.load(filepath)
    wavename = filepath.stem
    cm       = torch.tensor(self.dict_cm[wavename])
    dct_gram = torch.unsqueeze(torch.tensor(stdct(
                                                y          = sig[0].numpy(), 
                                                n_fft      = self.n_fft,
                                                hop_length = self.hop_length,
                                                win_length = self.win_length,
                                                window     = self.window_fn), dtype=torch.float32), 0)
    dct__tar, frames_num = self.shared_fill_block(dct_gram, self.dim_f, self.dim_t, self.dim_t_max)
    if self.power > 1: dct__tar.pow_(self.power)
    if self.log  == 1: dct__tar = torch.log(F.relu(dct__tar, inplace=True) + torch.finfo(torch.float32).eps)
    return wavename, dct__tar, cm, frames_num



def init_____npy(self, config, section):
    self.dim_f     = config.getint(section, 'dim_f')
    self.dim_t     = config.getint(section, 'dim_t')
    self.dim_t_max = config.getint(section, 'dim_t_max') + self.dim_t
    self.dim_t_shf = config.getint(section, 'dim_t_shf')
    self.power     = config.getint(section, 'power')
    self.log       = config.getint(section, 'log')

def getitem_npy(self, idx):
    filepath = self.list_path[idx]
    wavename = filepath.stem
    cm       = torch.tensor(self.dict_cm[wavename])
    blck_npy = torch.unsqueeze(torch.tensor(np.load(filepath), dtype=torch.float32), 0)
    blck_tar, frames_num = self.shared_fill_block(blck_npy, self.dim_f, self.dim_t, self.dim_t_max)
    if self.power > 1: blck_npy.pow_(self.power)
    if self.log  == 1: blck_npy = torch.log(F.relu(blck_npy, inplace=True) + torch.finfo(torch.float32).eps)
    return wavename, blck_tar, cm, frames_num



class Dataset_online(Dataset):
    def __init__(self, list_path, dict_cm, config, section, dmode):

        self.list_path = list_path
        feature_name   = config.get(section, 'feature_name')
        file_extension = config.get(section, 'file_extension')

        if dict_cm == 'infer':
            filenamelist = [filepath.stem for filepath in list_path]
            self.dict_cm = dict(zip(filenamelist, list(range(len(list_path)))))
        else:
            assert len(list_path) == len(dict_cm)
            self.dict_cm = dict_cm

        if   file_extension == 'flac':
            if   feature_name[:11] == 'spectrogram': init____spec(self, config, section); self.getitem = getitem_spec
            elif feature_name[:11] == 'cepstrogram': init____spec(self, config, section); self.getitem = getitem_ceps
            elif feature_name[: 3] ==         'dct': init_____dct(self, config, section); self.getitem = getitem_dct
            elif feature_name[: 4] ==        'adfa': init____adfa(self, config, section); self.getitem = getitem_adfa
            elif feature_name[: 4] ==        'mdfa': init____mdfa(self, config, section); self.getitem = getitem_adfa
            elif feature_name[: 3] ==         'cqa': init_____cqa(self, config, section); self.getitem = getitem_adfa
            else: raise ValueError(f'please Check the feature_name of section:{section} in config.ini')
        elif file_extension == 'npy':
            if   feature_name[: 3] ==         'npy': init_____npy(self, config, section); self.getitem = getitem_npy
            else: raise ValueError(f'please Check the feature_name of section:{section} in config.ini')
        else:
            raise ValueError(f'please Check the file_extension of section:{section} in config.ini')

        self.shared_fill_block = get_shared_fill_block(dmode)

    def __len__(self):
        return len(self.list_path)

    def __getitem__(self, idx):
        wavename, feat_tar, cm, frames_num = self.getitem(self, idx)
        return wavename, feat_tar, cm, frames_num

