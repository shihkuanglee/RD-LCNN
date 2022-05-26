import os, torch, torchaudio
import torch.nn.functional as F
import numpy               as np
from torch.utils.data import Dataset

from librosa import util
from scipy.fft import dct
from scipy.signal import get_window



# ============================================================
# return dict of cms with 1 (for bona fide) and 0 (for spoof).
# ============================================================
def dict_cm_protocols_ASVspoofing_2019(path_cm_protocols):

    ndarray = np.genfromtxt(path_cm_protocols, dtype=str)
    list_filename = list()
    list_______cm = list()
    for line in ndarray:
        list_filename.append(str(line[1]))
        list_______cm.append(str(line[4]))
    dict_cm = dict(zip(list_filename, list_______cm))

    list_set__cms = list(set(list_______cm))
    list_set__cms.sort()
    assert list_set__cms == ['bonafide', 'spoof']

    list_filename = list()
    list_______cm = list()
    for key in list(dict_cm):
        list_filename.append(key)
        if dict_cm[key] == 'bonafide':
            list_______cm.append(1)
        else :
            list_______cm.append(0)
    dict_cm = dict(zip(list_filename, list_______cm))
    return dict_cm


# ===========================================
# return list of paths, dict of cms, asv data
# =============================================================================
def get_list_dict_task_online(Path_data, task, feature_folder, file_extension):

    Path_data_root = Path_data / task
    Path_cm_protocols_root =          Path_data_root / f'ASVspoof2019_{task}_cm_protocols'
    Path_cm_protocols_train = Path_cm_protocols_root / f'ASVspoof2019.{task}.cm.train.trn.txt'
    Path_cm_protocols___dev = Path_cm_protocols_root / f'ASVspoof2019.{task}.cm.dev.trl.txt'
    Path_cm_protocols__eval = Path_cm_protocols_root / f'ASVspoof2019.{task}.cm.eval.trl.txt'
    dict_cm_train = dict_cm_protocols_ASVspoofing_2019(Path_cm_protocols_train)
    dict_cm___dev = dict_cm_protocols_ASVspoofing_2019(Path_cm_protocols___dev)
    dict_cm__eval = dict_cm_protocols_ASVspoofing_2019(Path_cm_protocols__eval)

    Path_asv_scores_root =          Path_data_root / f'ASVspoof2019_{task}_asv_scores'
    Path_asv_scores_____dev = Path_asv_scores_root / f'ASVspoof2019.{task}.asv.dev.gi.trl.scores.txt'
    Path_asv_scores____eval = Path_asv_scores_root / f'ASVspoof2019.{task}.asv.eval.gi.trl.scores.txt'
    asv_data__dev = np.genfromtxt(Path_asv_scores_____dev, dtype=str)
    asv_data_eval = np.genfromtxt(Path_asv_scores____eval, dtype=str)

    Path_data_train = Path_data_root / f'ASVspoof2019_{task}_train' / feature_folder
    Path_data___dev = Path_data_root / f'ASVspoof2019_{task}_dev'   / feature_folder
    Path_data__eval = Path_data_root / f'ASVspoof2019_{task}_eval'  / feature_folder
    list_path_train = [os.path.join(Path_data_train, name + f'.{file_extension}') for name in list(dict_cm_train)]
    list_path___dev = [os.path.join(Path_data___dev, name + f'.{file_extension}') for name in list(dict_cm___dev)]
    list_path__eval = [os.path.join(Path_data__eval, name + f'.{file_extension}') for name in list(dict_cm__eval)]

    return list_path_train, dict_cm_train, \
           list_path___dev, dict_cm___dev, asv_data__dev, \
           list_path__eval, dict_cm__eval, asv_data_eval



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

def get_shared_fill_block(dmode):
    if   dmode.lower() ==  'train': return shared_fill_block_tra
    elif dmode.lower() ==   'eval': return shared_fill_block_val
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

def getitem_spec(self, idx):
    filepath = self.list_path[idx]
    sig, _   = torchaudio.load(filepath)
    wavename = filepath.split(os.sep)[-1].split('.')[0]
    cm       = torch.tensor(self.dict_cm[wavename])
    specgram = torchaudio.transforms.Spectrogram(
                    n_fft      = self.n_fft,
                    win_length = self.win_length,
                    hop_length = self.hop_length,
                    pad        = self.pad,
                    window_fn  = self.window_fn,
                    power      = self.power)(sig)
    spec_tar, frames_num = self.shared_fill_block(specgram, self.dim_f, self.dim_t, self.dim_t_max)
    return wavename, spec_tar, cm, frames_num



def getitem_ceps(self, idx):
    filepath = self.list_path[idx]
    sig, _   = torchaudio.load(filepath)
    wavename = filepath.split(os.sep)[-1].split('.')[0]
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



def stdct(y, n_fft, hop_length, win_length, window):
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
    fft_window = util.pad_center(fft_window, n_fft)

    # Reshape so that the window can be broadcast
    fft_window = fft_window.reshape((-1, 1))

    # Check audio is valid
    util.valid_audio(y)

    # Window the time series.
    y_frames = util.frame(y, frame_length=n_fft, hop_length=hop_length)

    return dct(fft_window * y_frames, axis=0, norm='ortho')

def init_____dct(self, config, section):
    init____spec(self, config, section)
    self.window_fn = config.get(section, 'window_fn')

def getitem_dct(self, idx):
    filepath = self.list_path[idx]
    sig, _   = torchaudio.load(filepath)
    wavename = filepath.split(os.sep)[-1].split('.')[0]
    cm       = torch.tensor(self.dict_cm[wavename])
    dct_gram = torch.unsqueeze(torch.tensor(stdct(
                                                y          = sig[0].numpy(), 
                                                n_fft      = self.n_fft,
                                                hop_length = self.hop_length,
                                                win_length = self.win_length,
                                                window     = self.window_fn), dtype=torch.float32), 0)
    dct__tar, frames_num = self.shared_fill_block(dct_gram, self.dim_f, self.dim_t, self.dim_t_max)
    return wavename, dct__tar, cm, frames_num



def init_____npy(self, config, section):
    self.dim_f     = config.getint(section, 'dim_f')
    self.dim_t     = config.getint(section, 'dim_t')
    self.dim_t_max = config.getint(section, 'dim_t_max') + self.dim_t
    self.dim_t_shf = config.getint(section, 'dim_t_shf')
    self.power     = config.getint(section, 'power')

def getitem_npy(self, idx):
    filepath = self.list_path[idx]
    wavename = filepath.split(os.sep)[-1].split('.')[0]
    cm       = torch.tensor(self.dict_cm[wavename])
    blck_npy = torch.unsqueeze(torch.tensor(np.load(filepath), dtype=torch.float32), 0)
    if self.power > 1: blck_npy.pow_(self.power)
    blck_tar, frames_num = self.shared_fill_block(blck_npy, self.dim_f, self.dim_t, self.dim_t_max)
    return wavename, blck_tar, cm, frames_num



class Dataset_online(Dataset):
    def __init__(self, list_path, dict_cm, config, section, dmode):

        self.list_path = list_path
        feature_name   = config.get(section, 'feature_name')
        file_extension = config.get(section, 'file_extension')

        if dict_cm == 'infer':
            filenamelist = [filepath.split(os.sep)[-1].split('.')[0] for filepath in list_path]
            self.dict_cm = dict(zip(filenamelist, list(range(len(list_path)))))
        else:
            assert len(list_path) == len(dict_cm)
            self.dict_cm = dict_cm

        if   file_extension == 'flac':
            if   feature_name[:11] == 'spectrogram': init____spec(self, config, section); self.getitem = getitem_spec
            elif feature_name[:11] == 'cepstrogram': init____spec(self, config, section); self.getitem = getitem_ceps
            elif feature_name[: 3] ==         'dct': init_____dct(self, config, section); self.getitem = getitem_dct
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

