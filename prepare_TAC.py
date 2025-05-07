import argparse, os, sys, shutil, time, torchaudio
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

from nara_wpe.utils import stft
from TAC.calc_g import tac_v8

from dataset import framming_w_window
from ADFA.adfa import adfa_arb, mdfa_arb, cqfa_arb


def write_flag():
    with open('exp-flag.txt', 'w') as f: f.write(f'1\n')

def  read_flag():
    with open('exp-flag.txt', 'r') as f: return f.read()

def get_filelist(args, dataset):
    data_root = Path(os.path.abspath(args.path_data)) / f'ASVspoof{args.year}'
    if   args.year == '2019': dataset_root = data_root / args.task
    elif args.year == '2021': dataset_root = data_root
    else: sys.exit("Invalid year argument.")
    data_source = dataset_root / f'ASVspoof{args.year}_{args.task}_{dataset}'
    filelist = [os.path.join(data_source / 'flac', line) for line in os.listdir(data_source / 'flac')]
    return data_source, filelist

def rm_mk_dir(data_target, dataset):
    print(f'Remove existing {dataset} set folder and make new one.')
    if os.path.exists(data_target) and os.path.isdir(data_target):
        shutil.rmtree(data_target)
    data_target.mkdir(exist_ok=True)

def framming(data, size):
    return framming_w_window(data, n_fft=size, hop_length=256, win_length=1024, window='blackman').T

def calc_TAC(spectra):
    prediction_filters = tac_v8(
          spectra.transpose(2, 0, 1),
          taps=16,
          delay=2,
          iterations=3,
          psd_context=0,
          statistics_mode='full')
    return prediction_filters

def calc_TAC_dataset(args, dataset):
    data_source, filelist = get_filelist(args, dataset)
    data_target = data_source / f'npy_{args.feature}'
    rm_mk_dir(data_target, dataset)

    if   args.feature ==  'TAC': pass
    elif args.feature == 'ATAC': m_adfa = adfa_arb( 513,        1024).T
    elif args.feature == 'MTAC': m_mdfa = mdfa_arb( 513, 16000, 1024).T
    elif args.feature == 'QTAC': m_cqfa = cqfa_arb( 513, 2, 1 * (513 - 1) / np.emath.logn(2, 1024 / 2), 1024).T
    else: sys.exit("Invalid feature argument. Please provide 'TAC', 'ATAC', 'MTAC', or 'QTAC'")

    print(f'Calculating {args.feature} on {args.year} {dataset} set..')
    time__total = 0
    batch__time = time.time()
    for idx in range(len(filelist)):
        data, samplerate = torchaudio.load(filelist[idx])
        if   args.feature[:3] ==  'TAC': spectra =      stft(data[0].numpy(), size=1024, shift=256)[np.newaxis,:,:]
        elif args.feature[:4] == 'ATAC': spectra = (framming(data[0].numpy(), size=1024)  @ m_adfa)[np.newaxis,:,:]
        elif args.feature[:4] == 'MTAC': spectra = (framming(data[0].numpy(), size=1024)  @ m_mdfa)[np.newaxis,:,:]
        elif args.feature[:4] == 'QTAC': spectra = (framming(data[0].numpy(), size=1024)  @ m_cqfa)[np.newaxis,:,:]
        np.save(data_target / (filelist[idx].split(os.sep)[-1].split('.')[0] + '.npy'), calc_TAC(spectra))

        if (idx + 1) % 1000 == 0:
            time_pause = 0
            if int(read_flag()) == 0: # pause program
                write_flag() # put back flag
                pause_time = time.time()
                try: input("Program paused. Press Enter to continue or Ctrl+C to quit...")
                except KeyboardInterrupt: sys.exit("\nQuitting program...")
                time_pause = time.time() - pause_time
                print(f'Paused time: {str(timedelta(seconds=int(time_pause))):>8s}')

            time__batch  = time.time() - batch__time - time_pause
            batch__time  = time.time()
            time__total += time__batch
            time_remain = (time__total / (idx + 1)) * (len(filelist) - (idx + 1))
            time_finish = datetime.now() + timedelta(seconds=time_remain)
            time__batch_str = f'batch time: {    str(timedelta(seconds=int(time__batch))):>8s}'
            time_remain_str = f'remaining time: {str(timedelta(seconds=int(time_remain))):>8s}'
            time_finish_str = f'finish time: {time_finish.strftime("%Y-%m-%d %H:%M:%S")}'
            time_str = f'{time__batch_str}, {time_remain_str}, {time_finish_str}.'
            print(f'processing: {idx + 1:>6d} / {len(filelist)}, {time_str}')

    total_time_str = str(timedelta(seconds=int(time__total)))
    print(f'Total time for calculating {args.feature} on {args.year} {dataset} set: {total_time_str:>8s}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_data", type=str, required=True, help="Specify path of ASVspoof2019 dataset directory")
    parser.add_argument("--year",      type=str, required=True, help="Specify '2019' or '2021'")
    parser.add_argument("--task",      type=str, required=True, help="Specify task of ASVspoof2019; 'LA' or 'PA'")
    parser.add_argument("--dataset",   type=str, required=True, help="Specify 'train', 'dev', 'eval', or 'all'")
    parser.add_argument("--feature",   type=str, required=True, help="Specify 'TAC', 'ATAC' or 'MTAC'")
    args = parser.parse_args()

    write_flag() # for pausing program

    if args.dataset == 'all':
        if args.year == '2019':
            datasets = ['train', 'dev', 'eval']
            for dataset in datasets:
                calc_TAC_dataset(args, dataset)
        else:
            sys.exit("Invalid year, 'all' datasets only available for year 2019.")
    elif args.year == '2019' and args.dataset in ['train', 'dev', 'eval']:
        calc_TAC_dataset(args, args.dataset)
    elif args.year == '2021' and args.dataset == 'eval':
        calc_TAC_dataset(args, args.dataset)
    else:
        sys.exit("Invalid year/dataset argument!")
