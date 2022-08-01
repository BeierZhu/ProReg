def forward_deprecated(self, batch):
    image = batch['image'][0]
    label = torch.tensor(batch['label']).to(self.device)

    text_features = self.text_features.to(self.device)

    image_features = self.clip.encode_image(image)
    image_features_norm = image_features/image_features.norm(dim=-1, keepdim=True)
    text_features_norm = text_features/text_features.norm(dim=-1, keepdim=True)

    logit_scale = self.clip.logit_scale.exp()
    logits = image_features_norm @ text_features_norm.T * logit_scale

    if self.hparams.config["finetune_mode"] == 'prompt_proreg':
        with torch.no_grad():
            text_features = self.prompt_weight.to(self.device)
            image_features = self.prompt_model.encode_image(image)
            image_features_norm = image_features/image_features.norm(dim=-1, keepdim=True)
            text_features_norm = text_features/text_features.norm(dim=-1, keepdim=True)
            logits_tea = image_features_norm @ text_features_norm.T *  self.prompt_model.logit_scale.exp()
        if self.hparams.config['kd_loss'] ==  'xERM':
            loss = losses.xERM(logits, logits_tea, label, alpha=1, gamma=0.5, T=1)
        else:
            loss = losses.knowledge_distill(logits, logits_tea, label, alpha=0.5, T=1)
    else:
        loss = F.cross_entropy(logits, label)

    phase = "train" if self.training else "val"
    task = 'nico'
    ret = {
        "loss": loss,
        f"{task}_logits": logits,
        f"{task}_label": label,
    }
    accuracy = getattr(self, f"{phase}_{task}_accuracy")(
        logits, label
    )
    total_loss = getattr(self, f"{phase}_{task}_loss")(loss)
    self.log(f"{task}/{phase}/loss", total_loss)
    self.log(f"{task}/{phase}/accuracy", accuracy)

    return ret