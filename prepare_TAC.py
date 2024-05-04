import argparse, os, platform, sys, librosa, time, shutil
import soundfile as sf
import     numpy as np
from pathlib import Path

from TAC.calc_g import stft, tac_v8

def filepath2npy_TAC(filepath, dir_data):
    if platform.system() == 'Darwin':
        data, samplerate = librosa.load(filepath, sr=None)
    elif platform.system() == 'Linux':
        data, samplerate = sf.read(filepath)
    else:
        sys.exit("Exiting due to unsupported operating system.")
    Obs = stft(data, size=1024, shift=256)[np.newaxis,:,:]
    G_block = tac_v8(
          Obs.transpose(2, 0, 1),
          taps=16,
          delay=2,
          iterations=3,
          psd_context=0,
          statistics_mode='full')
    filename = filepath.split(os.sep)[-1].split('.')[0] + '.npy'
    np.save(dir_data / filename, G_block)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_data", type=str, help="Specify path of ASVspoof2019 directory")
    parser.add_argument("--task",      type=str, help="Specify task of ASVspoof2019, LA or PA")
    args = parser.parse_args()

    data_root = Path(os.path.abspath(args.path_data))
    data_source_train = data_root / args.task / f'ASVspoof2019_{args.task}_train' / 'flac'
    data_source___dev = data_root / args.task / f'ASVspoof2019_{args.task}_dev'   / 'flac'
    data_source__eval = data_root / args.task / f'ASVspoof2019_{args.task}_eval'  / 'flac'
    data_target_train = data_root / args.task / f'ASVspoof2019_{args.task}_train' / 'npy_TAC'
    data_target___dev = data_root / args.task / f'ASVspoof2019_{args.task}_dev'   / 'npy_TAC'
    data_target__eval = data_root / args.task / f'ASVspoof2019_{args.task}_eval'  / 'npy_TAC'

    list_train = [os.path.join(data_source_train, line) for line in os.listdir(data_source_train)]
    list___dev = [os.path.join(data_source___dev, line) for line in os.listdir(data_source___dev)]
    list__eval = [os.path.join(data_source__eval, line) for line in os.listdir(data_source__eval)]


    ## train set
    data_target = data_target_train; filelist = list_train
    print(f'Calculating TAC on train set..')
    if os.path.exists(data_target) and os.path.isdir(data_target):
        shutil.rmtree(data_target)
    data_target.mkdir(exist_ok=True)

    start_time = time.time()
    for idx in range(len(filelist)):
        filepath2npy_TAC(filelist[idx], data_target)
        if (idx + 1) % 1000 == 0:
            elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
            print(f'processing: {idx + 1:>6d}/{len(filelist)}, elapsed time: {elapsed_time_str}')

    elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print(f'Elapsed time for calculating TAC on train set:  {elapsed_time_str}')


    ## dev set
    data_target = data_target___dev; filelist = list___dev
    print(f'Calculating TAC on dev set..')
    if os.path.exists(data_target) and os.path.isdir(data_target):
        shutil.rmtree(data_target)
    data_target.mkdir(exist_ok=True)

    start_time = time.time()
    for idx in range(len(filelist)):
        filepath2npy_TAC(filelist[idx], data_target)
        if (idx + 1) % 1000 == 0:
            elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
            print(f'processing: {idx + 1:>6d}/{len(filelist)}, elapsed time: {elapsed_time_str}')

    elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print(f'Elapsed time for calculating TAC on dev set:  {elapsed_time_str}')


    ## eval set
    data_target = data_target__eval; filelist = list__eval
    print(f'Calculating TAC on eval set..')
    if os.path.exists(data_target) and os.path.isdir(data_target):
        shutil.rmtree(data_target)
    data_target.mkdir(exist_ok=True)

    start_time = time.time()
    for idx in range(len(filelist)):
        filepath2npy_TAC(filelist[idx], data_target)
        if (idx + 1) % 1000 == 0:
            elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
            print(f'processing: {idx + 1:>6d}/{len(filelist)}, elapsed time: {elapsed_time_str}')

    elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print(f'Elapsed time for calculating TAC on eval set:  {elapsed_time_str}')
