"""
Utility functions for training, evaluation, and Fourier-domain operations.
"""
import random
import os
import errno
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio


def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def partition_list(case_num, train_num, sample_per_case, random_seed,
                   train=True):
    sample_list = list(range(0, case_num))
    random.seed(random_seed)
    random.shuffle(sample_list)
    if train:
        data_case_list = sample_list[:train_num]
    else:
        data_case_list = sample_list[train_num:]
    data_list = [
        case * sample_per_case + i
        for case in data_case_list
        for i in range(sample_per_case)
    ]
    return data_list


def circular_mask(input, sz, r):
    N = input.shape[0]
    x, y = np.meshgrid(np.arange(sz), np.arange(sz))
    d = np.sqrt((x - sz // 2) ** 2 + (y - sz // 2) ** 2)
    m = d < r
    m = np.expand_dims(m, axis=0)
    m = np.repeat(m, N, axis=0)
    return input * m


def get_mean_ssim(ref_stack, pred_stack, normalize_method=None,
                  data_range=1400):
    ssim_list = []
    for i in range(ref_stack.shape[0]):
        ref = ref_stack[i]
        pred = pred_stack[i]
        if normalize_method is None:
            pred[pred < 0] = 0
        elif normalize_method == 'Shift':
            pred = pred - np.min(pred)
        max_v = np.max(ref)
        ref = ref / max_v
        pred = pred / max_v
        ssim_pred = ssim(ref, pred, data_range=1)
        if np.isnan(ssim_pred):
            ssim_pred = 0
        ssim_list.append(ssim_pred)
    return sum(ssim_list) / len(ssim_list)


def get_mean_psnr(ref_stack, pred_stack, data_range=4090):
    psnr = np.zeros((ref_stack.shape[0],))
    for i in range(ref_stack.shape[0]):
        ref_max = np.max(ref_stack[i])
        ref_norm = ref_stack[i] / ref_max
        pred_norm = pred_stack[i] / ref_max
        psnr[i] = peak_signal_noise_ratio(ref_norm, pred_norm, data_range=1)
        if np.isnan(psnr[i]):
            psnr[i] = 0
    return np.mean(psnr)


def get_mean_rmse(ref_stack, pred_stack):
    rmse = np.zeros((ref_stack.shape[0],))
    for i in range(ref_stack.shape[0]):
        ref_max = np.max(ref_stack[i])
        ref_norm = ref_stack[i] / ref_max
        pred_norm = pred_stack[i] / ref_max
        rmse[i] = np.sqrt(np.mean((ref_norm - pred_norm) ** 2))
    return np.mean(rmse)


def clip_grad_norm(optimizer, max_norm, norm_type=2):
    for group in optimizer.param_groups:
        torch.nn.utils.clip_grad_norm_(group['params'], max_norm, norm_type)


class Fourier_Utils:
    @staticmethod
    def fft2d(x):
        return torch.fft.fftshift(torch.fft.fft2(x))

    @staticmethod
    def ifft2d(x):
        return torch.fft.ifft2(torch.fft.ifftshift(x))

    @staticmethod
    def get_amplitude(x):
        return torch.abs(Fourier_Utils.fft2d(x))

    @staticmethod
    def get_phase(x):
        return torch.angle(Fourier_Utils.fft2d(x))

    @staticmethod
    def normalize_amplitude(amp):
        amp_log = torch.log(amp + 1)
        min_v = amp_log.amin(dim=[2, 3], keepdim=True)
        max_v = amp_log.amax(dim=[2, 3], keepdim=True)
        return (amp_log - min_v) / (max_v - min_v + 1e-8)

    @staticmethod
    def phase_to_sincos(phase):
        return torch.cat([torch.sin(phase), torch.cos(phase)], dim=1)

    @staticmethod
    def sincos_to_phase(sincos):
        C2 = sincos.shape[1]
        assert C2 % 2 == 0, "Channel dim must be even"
        C = C2 // 2
        return torch.atan2(sincos[:, :C], sincos[:, C:])

    @staticmethod
    def get_phase_and_amp(x):
        fft = Fourier_Utils.fft2d(x)
        amp = torch.abs(fft)
        amp_norm = Fourier_Utils.normalize_amplitude(amp)
        phase = torch.angle(fft)
        phase_sincos = Fourier_Utils.phase_to_sincos(phase)
        return phase_sincos, amp_norm

    @staticmethod
    def get_phase_amp_loss(pred, target):
        pred_phase, pred_amp = Fourier_Utils.get_phase_and_amp(pred)
        target_phase, target_amp = Fourier_Utils.get_phase_and_amp(target)
        phase_loss = torch.mean(torch.abs(pred_phase - target_phase))
        amp_loss = torch.mean(torch.abs(pred_amp - target_amp))
        return phase_loss, amp_loss

    @staticmethod
    def get_fourier_loss(pred, target):
        pred_f = Fourier_Utils.fft2d(pred)
        target_f = Fourier_Utils.fft2d(target)
        return torch.mean(torch.abs(pred_f - target_f))
