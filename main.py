import argparse, os, sys, time, shutil
from     pathlib import Path
from collections import Counter
from    datetime import datetime, timedelta


def path_check_mkdir(path_dir):
    if os.path.exists(path_dir) and os.path.isdir(path_dir):
        shutil.rmtree(path_dir)
    path_dir.mkdir(exist_ok=True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--device",         type=str,   default='cuda',  help="Specify 'cpu' or 'cuda'")
    parser.add_argument("--cuda_vd",        type=str,   default='None',  help="CUDA_VISIBLE_DEVICES")
    parser.add_argument("--seed",           type=int,   default=999,     help="Seed value")
    parser.add_argument("--epochs",         type=int,   default=400,     help="The number of epochs")
    parser.add_argument("--esep",           type=int,   default=200,     help="The number of early stop epoch")
    parser.add_argument("--ep_eval",        type=int,   default=50,      help="The number of epoch to start evaluation")
    parser.add_argument("--bsize",          type=int,   default=10,      help="Batch size")
    parser.add_argument("--WRSns",          type=int,   default=3000,    help="Specify number of batch to active WeightedRandomSampler")
    parser.add_argument("--num_workers",    type=int,   default=0,       help="The number of workers for DataLoader")
    parser.add_argument("--lr",             type=float, default=-1,      help="Learning rate, -1 means 0.1, -2 means 0.01")
    parser.add_argument("--lr_lmbda",       type=float, default=-1,      help="The value of lr_lmbda for MultiplicativeLR, -1 means 0.1")
    parser.add_argument("--lr_step_wait",   type=int,   default=50,      help="learning rate step wait epochs")
    parser.add_argument("--opt",            type=str,   default='sgd',   help="optimizer name")
    parser.add_argument("--cp",             type=float, default='inf',   help="To clips gradient norm")
    parser.add_argument("--momentum",       type=float, default=0,       help="momentum value")
    parser.add_argument("--weight_decay",   type=float, default=0,       help="The value of weight_decay for optimizer")
    parser.add_argument("--scheduler",      type=str,   default='None',  help="scheduler name, None or MultiplicativeLR")
    parser.add_argument("--dmode_train",    type=str,   default='train', help="dmode for  train Dataset")
    parser.add_argument("--dmode___dev",    type=str,   default='eval',  help="dmode for    dev Dataset")
    parser.add_argument("--dmode__eval",    type=str,   default='eval',  help="dmode for   eval Dataset")
    parser.add_argument("--path_data",      type=str,   required=True,   help="Specify path of ASVspoof2019 directory")
    parser.add_argument("--task",           type=str,   default='PA',    help="Specify task of ASVspoof2019, LA or PA")
    parser.add_argument("--conifg_section", type=str,   required=True,   help="Specify config section for different features")
    parser.add_argument("--info",           type=str,                    help="additional message")
    args = parser.parse_args()
    # args = parser.parse_args(["--device", "cpu", "--num_workers", "0", "--path_data", "../ASVspoof2019", "--conifg_section", "TAC", "--dmode_train", "fixed", "--dmode___dev", "fixed", "--dmode__eval", "fixed"])
    # for debugging

    import   configparser
    config = configparser.ConfigParser()
    config.read('config.ini')

    dim_f          = config.getint(args.conifg_section, 'dim_f')
    dim_t          = config.getint(args.conifg_section, 'dim_t')
    feature_folder = config.get(   args.conifg_section, 'feature_folder')
    file_extension = config.get(   args.conifg_section, 'file_extension')

    from dataset import get_list_dict_task_online as get_list_dict
    list_path_train, dict_cm_train, \
    list_path___dev, dict_cm___dev, asv_data__dev, \
    list_path__eval, dict_cm__eval, asv_data_eval= \
    get_list_dict(Path(args.path_data), args.task, feature_folder, file_extension)
    from dataset import Dataset_online as Data_set
    dataset_tra = Data_set(list_path_train, dict_cm_train, config, args.conifg_section, args.dmode_train)
    dataset_dev = Data_set(list_path___dev, dict_cm___dev, config, args.conifg_section, args.dmode___dev)
    dataset_eva = Data_set(list_path__eval, dict_cm__eval, config, args.conifg_section, args.dmode__eval)


    # ##################
    import torch, random
    import torch.backends.cudnn as cudnn
    import torch.nn as nn
    import numpy    as np
    from torch.utils.data import DataLoader
    device = torch.device(args.device)
    if args.device == 'cuda' and args.cuda_vd != 'None':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_vd
    random.seed(      int(args.seed))
    np.random.seed(   int(args.seed))
    torch.manual_seed(int(args.seed))
    cudnn.deterministic = True


    class__counts_train = Counter(dict_cm_train.values())
    counts__class_train = [class__counts_train[False], class__counts_train[True]]
    class_weights_train = [1 - (x / class__counts_train.total()) for x in counts__class_train]
    if args.WRSns == None:
        Dloader_tra = DataLoader(dataset_tra, batch_size=args.bsize, shuffle=True, num_workers=args.num_workers, collate_fn=None)
        nl_criterion_train = nn.CrossEntropyLoss(torch.tensor(class_weights_train, dtype=torch.float32)).to(device)
    else:
        weights_sampler_train = [class_weights_train[value] for value in dict_cm_train.values()]
        from torch.utils.data import WeightedRandomSampler
        Sampler_tra = WeightedRandomSampler(weights_sampler_train, args.bsize * args.WRSns)
        Dloader_tra = DataLoader(dataset_tra, batch_size=args.bsize, sampler=Sampler_tra, num_workers=args.num_workers, collate_fn=None)
        nl_criterion_train = nn.CrossEntropyLoss().to(device)


    class__counts___dev = Counter(dict_cm___dev.values())
    counts__class___dev = [class__counts___dev[False], class__counts___dev[True]]
    class_weights___dev = [1 - (x / class__counts___dev.total()) for x in counts__class___dev]
    Dloader_dev = DataLoader(dataset_dev, batch_size=args.bsize, shuffle=False, num_workers=args.num_workers, collate_fn=None)
    nl_criterion___dev = nn.CrossEntropyLoss(torch.tensor(class_weights___dev, dtype=torch.float32)).to(device)


    nl_criterion = nn.CrossEntropyLoss().to(device)
    Dloader_eva = DataLoader(dataset_eva, batch_size=args.bsize, shuffle=False, num_workers=args.num_workers, collate_fn=None)


    # #################################
    from model import T45_LCNN as Model
    model = Model(data_shape=[dim_f, dim_t], LDO_p1=0.75, LDO_p2=0.00).to(device)

    lr =     10 ** float(args.lr)
    clip =   10 ** float(args.cp)
    momentum     = float(args.momentum)
    weight_decay = float(args.weight_decay)
    if   args.opt.lower() ==  'sgd': optimizer = torch.optim.SGD( model.parameters(), lr=lr, momentum=momentum,  weight_decay=weight_decay)
    elif args.opt.lower() == 'adam': optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay, amsgrad=False)
    else: raise ValueError('Check the optimizer name')


    # ############################################################################################
    save_folder_name = f'exp__{datetime.now().strftime("%Y-%m%d-%H%M-%S")}__{args.conifg_section}'
    save_folder_Path = Path(os.getcwd()) / save_folder_name
    path_check_mkdir(save_folder_Path)

    save_folder_modl_Path = save_folder_Path / 'modl'
    save_folder_name_Path = save_folder_Path / 'name'
    save_folder_scrs_Path = save_folder_Path / 'scrs'
    save_folder_cmky_Path = save_folder_Path / 'cmky'
    path_check_mkdir(save_folder_modl_Path)
    path_check_mkdir(save_folder_name_Path)
    path_check_mkdir(save_folder_scrs_Path)
    path_check_mkdir(save_folder_cmky_Path)


    # #################################################
    hist_tra_losses = np.array(list(), dtype=np.single)
    hist_dev_losses = np.array(list(), dtype=np.single)
    hist_eva_losses = np.array(list(), dtype=np.single)
    hist_tra_eer_cm = np.array(list(), dtype=np.single)
    hist_dev_eer_cm = np.array(list(), dtype=np.single)
    hist_eva_eer_cm = np.array(list(), dtype=np.single)

    with open(save_folder_Path / 'flag.txt', 'w') as f: f.write(f'1\n')
    with open(save_folder_Path / 'hist.txt', 'w') as f:
        f.write(f"\n  Running command:\n  {' '.join(sys.argv)}\n")
        f.write(f'\n  len(dataset_tra): {len(dataset_tra):>6d}')
        f.write(f'\n  len(dataset_dev): {len(dataset_dev):>6d}')
        f.write(f'\n  len(dataset_eva): {len(dataset_eva):>6d}')
        f.write(f'\n\n{"save_folder_name":>20s}: {save_folder_name}\n\n')
        for key, value in args.__dict__.items():
            f.write(f"{key:>20}: {value}\n")
        f.write(f'\n')


    # ############################################
    if   args.scheduler == 'None': lr_current = lr
    elif args.scheduler == 'MultiplicativeLR':
        lmbda = lambda epoch: 10 ** float(args.lr_lmbda) if ((epoch % int(args.lr_step_wait) == 0) & (epoch > 0)) else 1
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)
    else: raise parser.error('Check the scheduler name!')

    from train_eval_infer import spf_det_train, spf_det_eval
    from em.em_2019 import get_eer, get_tDCF
    for ep in range(args.epochs):
        with open(save_folder_Path / 'flag.txt', 'r') as f: flag = int(f.read())
        if ep == args.esep: break # end program
        elif flag == 0:         # pause program
            with open(save_folder_Path / 'flag.txt', 'w') as f: f.write(f'1\n') # put back flag
            try: input("Program paused. Press Enter to continue or Ctrl+C to quit...")
            except KeyboardInterrupt: sys.exit("\nQuitting program...")
        elif flag >= 2: args.esep = flag # renew esep
        else: pass


        model.train()
        time_start = time.time()
        loss_train_avg, name_train, scrs_train, cmky_train = spf_det_train(model, device, Dloader_tra, nl_criterion_train, optimizer, clip)
        time_train = time.time() - time_start
        np.save(save_folder_name_Path / f'name_train__ep_{ep:03d}', name_train)
        np.save(save_folder_scrs_Path / f'scrs_train__ep_{ep:03d}', scrs_train)
        np.save(save_folder_cmky_Path / f'cmky_train__ep_{ep:03d}', cmky_train)
        bona__cm_train = scrs_train[cmky_train == 'bonafide']
        spoof_cm_train = scrs_train[cmky_train == 'spoof']
        eer___cm_train = get_eer(bona__cm_train, spoof_cm_train)[0] * 100

        torch.save(model.state_dict(), save_folder_modl_Path / f'modl__ep_{ep:03d}.pt')
        if args.scheduler == 'MultiplicativeLR':
            lr_current = scheduler.get_last_lr()[0]
            scheduler.step()

        if eer___cm_train == 0: str_first = f'ep:{ep:>3d} lr:{lr_current:>7.5f}  train EER:{            " 0":<6s}'
        else:                   str_first = f'ep:{ep:>3d} lr:{lr_current:>7.5f}  train EER:{eer___cm_train:>6.3f}'
        sys.stdout.write(f'\r{str_first}  time-train:{str(timedelta(seconds=int(time_train))):>8s}')


        model.eval()
        time_start = time.time()
        loss___dev_avg, name___dev, scrs___dev, cmky___dev = spf_det_eval(model, device, Dloader_dev, nl_criterion___dev)
        time_dev = time.time() - time_start
        np.save(save_folder_name_Path / f'name___dev__ep_{ep:03d}', name___dev)
        np.save(save_folder_scrs_Path / f'scrs___dev__ep_{ep:03d}', scrs___dev)
        np.save(save_folder_cmky_Path / f'cmky___dev__ep_{ep:03d}', cmky___dev)
        bona__cm___dev = scrs___dev[cmky___dev == 'bonafide']
        spoof_cm___dev = scrs___dev[cmky___dev == 'spoof']
        min_tDCF___dev = get_tDCF(asv_data__dev, bona__cm___dev, spoof_cm___dev)
        eer___cm___dev = get_eer(bona__cm___dev, spoof_cm___dev)[0] * 100

        if min_tDCF___dev == 0: str_first_tdcf = f'{str_first}   dev tDCF EER:{            " 0":<6s}'
        else:                   str_first_tdcf = f'{str_first}   dev tDCF EER:{min_tDCF___dev:>6.3f}'
        if eer___cm___dev == 0: str_secnd      = f'{str_first_tdcf}{            " 0":<6s}'
        else:                   str_secnd      = f'{str_first_tdcf}{eer___cm___dev:>6.3f}'
        sys.stdout.write(f'\r{str_secnd}  time-dev:{str(timedelta(seconds=int(time_dev))):>8s}')

        hist_tra_losses = np.append(hist_tra_losses, loss_train_avg)
        hist_dev_losses = np.append(hist_dev_losses, loss___dev_avg)
        hist_tra_eer_cm = np.append(hist_tra_eer_cm, eer___cm_train)
        hist_dev_eer_cm = np.append(hist_dev_eer_cm, eer___cm___dev)

        if (ep < args.ep_eval) | (eer___cm___dev > hist_dev_eer_cm.min()): # pass evaluation
            time_eval = 0
            loss__eval_avg = 1
            eer___cm__eval = 50
            str_third = f'{str_secnd}'
        else:
            time_start = time.time()
            loss__eval_avg, name__eval, scrs__eval, cmky__eval = spf_det_eval(model, device, Dloader_eva, nl_criterion)
            time_eval = time.time() - time_start
            np.save(save_folder_name_Path / f'name__eval__ep_{ep:03d}', name__eval)
            np.save(save_folder_scrs_Path / f'scrs__eval__ep_{ep:03d}', scrs__eval)
            np.save(save_folder_cmky_Path / f'cmky__eval__ep_{ep:03d}', cmky__eval)
            bona__cm__eval = scrs__eval[cmky__eval == 'bonafide']
            spoof_cm__eval = scrs__eval[cmky__eval == 'spoof']
            min_tDCF__eval = get_tDCF(asv_data_eval, bona__cm__eval, spoof_cm__eval)
            eer___cm__eval = get_eer(bona__cm__eval, spoof_cm__eval)[0] * 100

            if min_tDCF__eval == 0: str_secnd_tdcf = f'{str_secnd}   eval tDCF EER:{            " 0":<6s}'
            else:                   str_secnd_tdcf = f'{str_secnd}   eval tDCF EER:{min_tDCF__eval:>6.3f}'
            if eer___cm__eval == 0: str_third      = f'{str_secnd_tdcf}{            " 0":<6s}'
            else:                   str_third      = f'{str_secnd_tdcf}{eer___cm__eval:>6.3f} '

        hist_eva_losses = np.append(hist_eva_losses, loss__eval_avg)
        hist_eva_eer_cm = np.append(hist_eva_eer_cm, eer___cm__eval)

        np.save(save_folder_Path / 'hist_tra_losses', hist_tra_losses)
        np.save(save_folder_Path / 'hist_dev_losses', hist_dev_losses)
        np.save(save_folder_Path / 'hist_eva_losses', hist_eva_losses)
        np.save(save_folder_Path / 'hist_tra_eer_cm', hist_tra_eer_cm)
        np.save(save_folder_Path / 'hist_dev_eer_cm', hist_dev_eer_cm)
        np.save(save_folder_Path / 'hist_eva_eer_cm', hist_eva_eer_cm)

        eer_dev_argmin = hist_dev_eer_cm.argmin()
        if hist_dev_eer_cm[eer_dev_argmin] == 0:
            eer_dev_zero_idx = np.where(hist_dev_eer_cm == 0)[0]
            eer_dev_argmin = eer_dev_zero_idx[hist_dev_losses[eer_dev_zero_idx].argmin()]
        if hist_dev_eer_cm[eer_dev_argmin] == 0: amin_de_de = f'{                             " 0":<6s}'
        else:                                    amin_de_de = f'{hist_dev_eer_cm[eer_dev_argmin]:>6.3f}'
        if hist_eva_eer_cm[eer_dev_argmin] == 0: amin_de_ee = f'{                             " 0":<6s}'
        else:                                    amin_de_ee = f'{hist_eva_eer_cm[eer_dev_argmin]:>6.3f}'
        time_all_str = str(timedelta(seconds=int(time_train + time_dev + time_eval)))
        str_final = f'{str_third}   time-all:{time_all_str:>8s}   min dev EER:{amin_de_de} at ep:{eer_dev_argmin:>3d}, eval EER:{amin_de_ee}'

        sys.stdout.write(f'\r{str_final}\n')
        with open(save_folder_Path / 'hist.txt', 'a') as f: f.write(f'{str_final}\n')
