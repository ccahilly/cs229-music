import torch.nn as nn
import torch
from transformers import T5ForConditionalGeneration, ClapModel, EncoderDecoderCache

class ClapT5Model(nn.Module):
    def __init__(self, device="cpu", clap_model=None, t5_model=None, frozen=False):
        super(ClapT5Model, self).__init__()
        self.device = device
        self.frozen = frozen

        self.clap_model = clap_model or ClapModel.from_pretrained("laion/larger_clap_music").to(device)
        self.t5_model = t5_model or T5ForConditionalGeneration.from_pretrained("t5-small").to(device)

        if frozen:
            for param in self.clap_model.parameters():
                param.requires_grad = False

    def forward(self, batch):
        # Extract inputs
        inputs = batch["inputs"].to(self.device)
        inputs["input_features"] = inputs["input_features"].squeeze(1)
        labels = batch["labels"].to(self.device)
        decoder_attention_mask = batch["decoder_attention_mask"].to(self.device)

        # Extract embeddings from CLAP
        if self.frozen:
            with torch.no_grad():
                clap_outputs = self.clap_model.get_audio_features(**inputs)
        else:
            clap_outputs = self.clap_model.get_audio_features(**inputs)

        # Pass embeddings to T5
        outputs = self.t5_model(
            inputs_embeds=clap_outputs.unsqueeze(1),
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )
        return outputs