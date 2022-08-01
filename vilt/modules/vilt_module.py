import torch
import torch.nn as nn
import pytorch_lightning as pl
import vilt.modules.vision_transformer as vit
from vilt.transforms.utils import inception_unnormalize
from PIL import Image

from vilt.modules import objectives, vilt_utils, losses, heads
from torchvision.transforms import ToPILImage
import json
import numpy as np
import clip
import torch.nn.functional as F


class ViLTransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
               
        self.save_hyperparameters()
        self.clip, _ = clip.load(config['backbone_name'], 'cpu')
        for p in self.clip.parameters(): 
            p.data = p.data.float()
            if self.hparams.config['finetune_mode'] == 'linear_probe':
                p.requires_grad = False

        classes = vilt_utils.read_list(config["classname_path"])
        
        text_inputs = torch.cat([clip.tokenize(f"{config['prompt_template']} {c}.") for c in classes])
        text_features = self.clip.encode_text(text_inputs)
        self.text_features = nn.Parameter(text_features)

        vilt_utils.set_metrics(self)
        
        if self.hparams.config["finetune_mode"] == 'prompt_proreg' or self.hparams.config['finetune_mode'] == 'ensemble_test':
            self.prompt_model, _ = clip.load(config['backbone_name'], 'cpu')
            for p in self.prompt_model.parameters(): 
                p.data = p.data.float()
                p.requires_grad = False
            self.prompt_weight = self.clip.encode_text(text_inputs)

        if self.hparams.config['finetune_mode'] == 'coop_finetune':
            # using CoOp setting, context word number 16
            self.coop_model = heads.CoOp(model=self.clip, allclasses=classes, num_context_word=16)   

        if self.hparams.config['finetune_mode'] == 'promptless_finetune':
            hs = 512
            vs = self.hparams.config["label_size"] 
            self.cls_head = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, vs),
            )
            self.cls_head.apply(objectives.init_weights)


        if self.hparams.config["load_path"] != "":
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)
            print(f'===> load model from {self.hparams.config["load_path"]}') 

        if config['save_result']:
            self.writer = open("result/result.txt", "w")

    def forward(self, batch):
        if self.hparams.config["finetune_mode"] == 'prompt_finetune':
            ret = objectives.prompt_finetune(self, batch)
        elif self.hparams.config["finetune_mode"] == 'prompt_proreg':
            ret = objectives.prompt_proreg(self, batch)
        elif self.hparams.config["finetune_mode"] == 'coop_finetune':
            ret = objectives.coop_finetune(self, batch)
        elif self.hparams.config['finetune_mode'] == 'promptless_finetune':
            ret = objectives.promptless_finetune(self, batch)
        elif self.hparams.config['finetune_mode'] == 'ensemble_test':
            ret = objectives.ensemble_test(self, batch)


        if self.hparams.config['save_result']:
            objectives.save_predictions(ret, self.writer)

        return ret

    def training_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])

        return total_loss

    def training_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        ret = dict()

        return ret

    def test_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)
        if self.hparams.config['save_result']:
            self.writer.close()
            
    def configure_optimizers(self):
        return vilt_utils.set_schedule(self)
