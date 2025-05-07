import torch
import numpy as np


def spf_det_train(model, device, Dloader, nl_criterion, optimizer, clip):

    loss_train = list()
    name_train = list()
    scrs_train = list()
    cmky_train = list()

    for i, (wavename, tf_block, cm, frames_num) in enumerate(Dloader):
        frames_ovr = frames_num.numpy() - Dloader.dataset.dim_t
        if frames_ovr.max() > 0:
            shift_num = - (- frames_ovr // Dloader.dataset.dim_t_shf)
            for num in range(shift_num.max() + 1):
                spec_tmp = tf_block.to(device)[...,:Dloader.dataset.dim_t]
                if num == 0:
                    scr_output = model(spec_tmp)
                    scr_output_tmp = scr_output[:,1].cpu().detach().numpy()
                    loss = nl_criterion(scr_output, cm.to(device))
                else:
                    for j in range(len(shift_num)):
                        if shift_num[j] == num:
                            start_points = num * Dloader.dataset.dim_t_shf
                            end___points = start_points + Dloader.dataset.dim_t
                            spec_tmp[j] = tf_block.to(device)[j,...,start_points:end___points]
                    scr_output = model(spec_tmp)
                    for j in range(len(shift_num)):
                        if shift_num[j] == num:
                            scr_output_tmp[j] += scr_output[j,1].cpu().detach().numpy()
                    loss += nl_criterion(scr_output, cm.to(device))
            loss /= (shift_num.max() + 1)
            for j in range(len(shift_num)):
                if shift_num[j] > 0:
                    scr_output_tmp[j] /= (shift_num[j] + 1)
        else:
            scr_output = model(tf_block.to(device)[...,:Dloader.dataset.dim_t])
            scr_output_tmp = scr_output[:,1].cpu().detach().numpy()
            loss = nl_criterion(scr_output, cm.to(device))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        for j in range(tf_block.size(0)):
            loss_train.append(loss.item())
            name_train.append(wavename[j])
            scrs_train.append(scr_output_tmp[j])
            if cm[j]: cmky_train.append('bonafide')
            else:     cmky_train.append(   'spoof')

    loss_train_avg = sum(loss_train)/len(loss_train)

    return loss_train_avg, np.array(name_train), np.array(scrs_train), np.array(cmky_train)


def spf_det_eval(model, device, Dloader, nl_criterion):

    loss__eval = list()
    name__eval = list()
    scrs__eval = list()
    cmky__eval = list()

    with torch.no_grad():
        for i, (wavename, tf_block, cm, frames_num) in enumerate(Dloader):
            frames_ovr = frames_num.numpy() - Dloader.dataset.dim_t
            if frames_ovr.max() > 0:
                shift_num = - (- frames_ovr // Dloader.dataset.dim_t_shf)
                for num in range(shift_num.max() + 1):
                    spec_tmp = tf_block.to(device)[...,:Dloader.dataset.dim_t]
                    if num == 0:
                        scr_output = model(spec_tmp)
                        scr_output_tmp = scr_output[:,1].cpu().detach().numpy()
                        loss = nl_criterion(scr_output, cm.to(device))
                    else:
                        for j in range(len(shift_num)):
                            if shift_num[j] == num:
                                start_points = num * Dloader.dataset.dim_t_shf
                                end___points = start_points + Dloader.dataset.dim_t
                                spec_tmp[j] = tf_block.to(device)[j,...,start_points:end___points]
                        scr_output = model(spec_tmp)
                        for j in range(len(shift_num)):
                            if shift_num[j] == num:
                                scr_output_tmp[j] += scr_output[j,1].cpu().detach().numpy()
                        loss += nl_criterion(scr_output, cm.to(device))
                loss /= (shift_num.max() + 1)
                for j in range(len(shift_num)):
                    if shift_num[j] > 0:
                        scr_output_tmp[j] /= (shift_num[j] + 1)
            else:
                scr_output = model(tf_block.to(device)[...,:Dloader.dataset.dim_t])
                scr_output_tmp = scr_output[:,1].cpu().detach().numpy()
                loss = nl_criterion(scr_output, cm.to(device))

            for j in range(tf_block.size(0)):
                loss__eval.append(loss.item())
                name__eval.append(wavename[j])
                scrs__eval.append(scr_output_tmp[j])
                if cm[j]: cmky__eval.append('bonafide')
                else:     cmky__eval.append(   'spoof')

    loss__eval_avg = sum(loss__eval)/len(loss__eval)

    return loss__eval_avg, np.array(name__eval), np.array(scrs__eval), np.array(cmky__eval)


def spf_det_infer(model, device, Dloader):

    name_infer = list()
    scrs_infer = list()

    with torch.no_grad():
        for i, (wavename, tf_block, cm, frames_num) in enumerate(Dloader):
            frames_ovr = frames_num.numpy() - Dloader.dataset.dim_t
            if frames_ovr.max() > 0:
                shift_num = - (- frames_ovr // Dloader.dataset.dim_t_shf)
                for num in range(shift_num.max() + 1):
                    spec_tmp = tf_block.to(device)[...,:Dloader.dataset.dim_t]
                    if num == 0:
                        scr_output = model(spec_tmp)
                        scr_output_tmp = scr_output[:,1].cpu().detach().numpy()
                    else:
                        for j in range(len(shift_num)):
                            if shift_num[j] == num:
                                start_points = num * Dloader.dataset.dim_t_shf
                                end___points = start_points + Dloader.dataset.dim_t
                                spec_tmp[j] = tf_block.to(device)[j,...,start_points:end___points]
                        scr_output = model(spec_tmp)
                        for j in range(len(shift_num)):
                            if shift_num[j] == num:
                                scr_output_tmp[j] += scr_output[j,1].cpu().detach().numpy()
                for j in range(len(shift_num)):
                    if shift_num[j] > 0:
                        scr_output_tmp[j] /= (shift_num[j] + 1)
            else:
                scr_output = model(tf_block.to(device)[...,:Dloader.dataset.dim_t])
                scr_output_tmp = scr_output[:,1].cpu().detach().numpy()

            for j in range(tf_block.size(0)):
                name_infer.append(wavename[j])
                scrs_infer.append(scr_output_tmp[j])

    return np.array(name_infer), np.array(scrs_infer)

