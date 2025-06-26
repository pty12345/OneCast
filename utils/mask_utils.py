import math
import random
import torch
from torch import nn

def get_loss_weight(t, mask, min_val=0.3):
    return 1 - (1 - mask) * ((1 - t) * (1 - min_val))[:, None]

def mask_or_random_replace_tokens(ts_tokens, mask_id, config, mask_schedule, is_train=True):
    batch_size, seq_len = ts_tokens.shape

    if not is_train and config.training.get("eval_mask_ratios", None):
        mask_prob = random.choices(config.training.eval_mask_ratios, k=batch_size)
        mask_prob = torch.tensor(mask_prob, device=ts_tokens.device)
    else:
        # Sample a random timestep for each image
        timesteps = torch.rand(batch_size, device=ts_tokens.device)
        # Sample a random mask probability for each image using timestep and cosine schedule
        mask_prob = mask_schedule(timesteps)
        mask_prob = mask_prob.clip(config.training.min_masking_rate)

    # creat a random mask for each image
    num_token_masked = (seq_len * mask_prob).round().clamp(min=1)

    mask_contiguous_region_prob = config.training.get("mask_contiguous_region_prob", None)

    if mask_contiguous_region_prob is None:
        mask_contiguous_region = False
    else:
        mask_contiguous_region = random.random() < mask_contiguous_region_prob

    batch_randperm = torch.rand(batch_size, seq_len, device=ts_tokens.device).argsort(dim=-1)
    mask = batch_randperm < num_token_masked.unsqueeze(-1)

    # mask using mask_id
    input_ids = torch.where(mask, mask_id, ts_tokens)
    # mask time series and create input and labels
    # if config.training.get("noise_type", "mask"):
    #     input_ids = torch.where(mask, mask_id, ts_tokens)
    # elif config.training.get("noise_type", "random_replace"):
    #     # sample random tokens from the vocabulary
    #     random_tokens = torch.randint_like(
    #         ts_tokens, low=0, high=config.model.codebook_size, device=ts_tokens.device
    #     )
    #     input_ids = torch.where(mask, random_tokens, ts_tokens)
    # else:
    #     raise ValueError(f"noise_type {config.training.noise_type} not supported")

    if (
            config.training.get("predict_all_tokens", False)
            or config.training.get("noise_type", "mask") == "random_replace"
    ):
        labels = ts_tokens
        loss_weight = get_loss_weight(mask_prob, mask.long())
    else:
        labels = torch.where(mask, ts_tokens, -100)
        loss_weight = None

    return input_ids, labels, loss_weight, mask_prob