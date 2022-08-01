import torch
import torch.nn as nn
import clip

class CoOp(nn.Module):
    def __init__(self, model, allclasses, num_context_word=1):
        super(CoOp, self).__init__()
        self.num_context_word = num_context_word
        self.model = model.eval()
        # freeze all paramter in model
        for param in self.model.parameters():
            param.requires_grad = False

        with torch.no_grad():
            self.class_text_inputs = torch.cat([clip.tokenize(f"{c}") for c in allclasses])
            self.class_embedding = model.token_embedding(self.class_text_inputs).type(model.dtype)
            self.pre_embedding = self.class_embedding[:,:1,:]
            self.post_embedding = self.class_embedding[:,1:-self.num_context_word,:]
            self.pre_embedding.requires_grad = False
            self.post_embedding.requires_grad = False

        self.context_embedding = nn.Parameter(torch.randn(self.class_embedding.size(0), 
                                                          num_context_word, 
                                                          self.class_embedding.size(2)).type(model.dtype) / self.class_embedding.size(2) ** 0.5)
 
    def encode_text(self):
        pre_embedding = self.pre_embedding.to(self.context_embedding.device)
        post_embedding = self.post_embedding.to(self.context_embedding.device)

        x = torch.cat((pre_embedding, self.context_embedding, post_embedding),dim=1)
        x = x + self.model.positional_embedding.type(self.model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x).type(self.model.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), self.class_text_inputs.argmax(dim=-1)+self.num_context_word] @ self.model.text_projection

        return x

    def forward(self, image):
        text_features = self.encode_text()
        image_features = self.model.encode_image(image)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()

        return logits_per_image