import torch
import numpy as np
import math
import warnings


def check_bfloat16_support():
    # check if cuda version supports bfloat16
    device_capable = torch.cuda.is_bf16_supported()
    supported_devices = ['H100', 'Ada', '3090', '3080', '4080', '4090']
    device_can_utilize = np.any(
        [device in torch.cuda.get_device_name() for device in supported_devices])
    return device_capable and device_can_utilize


class Scheduler():
    SUPPORTED_SCHEDULERS = ['linear', 'cosine', 'none']
    
    def get_lr(schedule, *args, **kwargs):
        """ metric is a string for the function """

        if schedule not in Scheduler.SUPPORTED_SCHEDULERS:
            e_msg = f"Scheduler {schedule} not supported. " + \
                    f"Supported schedulers are {Scheduler.SUPPORTED_SCHEDULERS}"
            raise ValueError(e_msg)
        
        return getattr(Scheduler, schedule)(*args, **kwargs)
    
    @staticmethod
    def linear(curr_step, max_step, min_lr_factor):
        current_progress = curr_step / max_step
        gain = 1 - current_progress * (1 - min_lr_factor)
        return gain
    
    @staticmethod
    def cosine(curr_step, max_step, min_lr_factor):
        current_progress = curr_step / max_step
        gain = min_lr_factor + (1 - min_lr_factor) * \
            (1 + math.cos(math.pi * current_progress)) / 2
        return gain

    @staticmethod
    def none():
        return 1.0