import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import functools

import vilt.modules.losses as L 
from torchvision.transforms import ToPILImage

def save_predictions(ret, writer):
    batch_size = len(ret['pacs_label'])

    for i in range(batch_size):
        targ = ret['pacs_label'][i]
        writer.write(f'{targ} ')

        for j in ret['pacs_logits'][i]:
            writer.write(f'{j} ')

        writer.write('\n')


def coop_finetune(pl_module, batch):
    config = pl_module.hparams.config

    image = batch['image'][0]
    label = torch.tensor(batch['label']).to(pl_module.device)

    logits = pl_module.coop_model(image)

    loss = F.cross_entropy(logits, label)

    phase = "train" if pl_module.training else "val"
    task = config['task']
    ret = {
        "loss": loss,
        f"{task}_logits": logits,
        f"{task}_label": label,
    }
    accuracy = getattr(pl_module, f"{phase}_{task}_accuracy")(
        logits, label
    )
    total_loss = getattr(pl_module, f"{phase}_{task}_loss")(loss)
    pl_module.log(f"{task}/{phase}/loss", total_loss)
    pl_module.log(f"{task}/{phase}/accuracy", accuracy)

    return ret


def prompt_finetune(pl_module, batch):
    config = pl_module.hparams.config

    image = batch['image'][0]
    label = torch.tensor(batch['label']).to(pl_module.device)

    text_features = pl_module.text_features.to(pl_module.device)
    image_features = pl_module.clip.encode_image(image)

    image_features_norm = image_features/image_features.norm(dim=-1, keepdim=True)
    text_features_norm = text_features/text_features.norm(dim=-1, keepdim=True)

    logit_scale = pl_module.clip.logit_scale.exp()
    logits = image_features_norm @ text_features_norm.T * logit_scale

    loss = F.cross_entropy(logits, label)

    phase = "train" if pl_module.training else "val"
    task = config['task']
    ret = {
        "loss": loss,
        f"{task}_logits": logits,
        f"{task}_label": label,
    }
    accuracy = getattr(pl_module, f"{phase}_{task}_accuracy")(
        logits, label
    )
    total_loss = getattr(pl_module, f"{phase}_{task}_loss")(loss)
    pl_module.log(f"{task}/{phase}/loss", total_loss)
    pl_module.log(f"{task}/{phase}/accuracy", accuracy)

    return ret


def promptless_finetune(pl_module, batch):
    config = pl_module.hparams.config

    image = batch['image'][0]
    label = torch.tensor(batch['label']).to(pl_module.device)

    image_features = pl_module.clip.encode_image(image)

    logits = pl_module.cls_head(image_features)

    loss = F.cross_entropy(logits, label)

    phase = "train" if pl_module.training else "val"
    task = config['task']
    ret = {
        "loss": loss,
        f"{task}_logits": logits,
        f"{task}_label": label,
    }
    accuracy = getattr(pl_module, f"{phase}_{task}_accuracy")(
        logits, label
    )
    total_loss = getattr(pl_module, f"{phase}_{task}_loss")(loss)
    pl_module.log(f"{task}/{phase}/loss", total_loss)
    pl_module.log(f"{task}/{phase}/accuracy", accuracy)

    return ret

def prompt_proreg(pl_module, batch):
    config = pl_module.hparams.config

    image = batch['image'][0]
    label = torch.tensor(batch['label']).to(pl_module.device)

    text_features = pl_module.text_features.to(pl_module.device)
    image_features = pl_module.clip.encode_image(image)

    image_features_norm = image_features/image_features.norm(dim=-1, keepdim=True)
    text_features_norm = text_features/text_features.norm(dim=-1, keepdim=True)

    logit_scale = pl_module.clip.logit_scale.exp()
    logits = image_features_norm @ text_features_norm.T * logit_scale

    if config["finetune_mode"] == 'prompt_proreg':
        with torch.no_grad():
            text_features = pl_module.prompt_weight.to(pl_module.device)
            image_features = pl_module.prompt_model.encode_image(image)
            image_features_norm = image_features/image_features.norm(dim=-1, keepdim=True)
            text_features_norm = text_features/text_features.norm(dim=-1, keepdim=True)
            logits_tea = image_features_norm @ text_features_norm.T *  pl_module.prompt_model.logit_scale.exp()
        if config['kd_loss'] ==  'xERM':
            loss = L.xERM(logits, logits_tea, label, alpha=config['alpha'], gamma=config['gamma'], T=config['T'])
        else:
            loss = L.knowledge_distill(logits, logits_tea, label, alpha=config['alpha'], T=config['T'])

    phase = "train" if pl_module.training else "val"
    task = config['task']
    ret = {
        "loss": loss,
        f"{task}_logits": logits,
        f"{task}_label": label,
    }
    accuracy = getattr(pl_module, f"{phase}_{task}_accuracy")(
        logits, label
    )
    total_loss = getattr(pl_module, f"{phase}_{task}_loss")(loss)
    pl_module.log(f"{task}/{phase}/loss", total_loss)
    pl_module.log(f"{task}/{phase}/accuracy", accuracy)

    return ret

def ensemble_test(pl_module, batch):
    config = pl_module.hparams.config

    image = batch['image'][0]
    label = torch.tensor(batch['label']).to(pl_module.device)

    text_features = pl_module.text_features.to(pl_module.device)
    image_features = pl_module.clip.encode_image(image)

    image_features_norm = image_features/image_features.norm(dim=-1, keepdim=True)
    text_features_norm = text_features/text_features.norm(dim=-1, keepdim=True)

    logit_scale = pl_module.clip.logit_scale.exp()
    logits = image_features_norm @ text_features_norm.T * logit_scale

    with torch.no_grad():
        text_features = pl_module.prompt_weight.to(pl_module.device)
        image_features = pl_module.prompt_model.encode_image(image)
        image_features_norm = image_features/image_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features/text_features.norm(dim=-1, keepdim=True)
        logits_tea = image_features_norm @ text_features_norm.T *  pl_module.prompt_model.logit_scale.exp()


    alpha = config['alpha']

    logits = (1 - alpha) * logits + alpha * logits_tea
    phase = "train" if pl_module.training else "val"
    task = config['task']
    ret = {
        f"{task}_logits": logits,
        f"{task}_label": label,
    }
    accuracy = getattr(pl_module, f"{phase}_{task}_accuracy")(
        logits, label
    )
    pl_module.log(f"{task}/{phase}/accuracy", accuracy)

    return ret


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()
