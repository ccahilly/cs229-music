import torch.nn as nn
import torch
from transformers import T5ForConditionalGeneration, Wav2Vec2Model

class Wav2Vec2T5Model(nn.Module):
    def __init__(self, device="cpu", wav2vec2_model=None, t5_model=None, frozen=False):
        super(Wav2Vec2T5Model, self).__init__()
        self.device = device
        self.frozen = frozen

        self.wav2vec2_model = wav2vec2_model or Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(self.device)
        self.t5_model = t5_model or T5ForConditionalGeneration.from_pretrained("t5-small").to(self.device)

        self.reduction_layer = nn.Linear(self.wav2vec2_model.config.hidden_size, self.t5_model.config.d_model).to(self.device)
        
        if self.frozen:
            for param in self.wav2vec2_model.parameters():
                param.requires_grad = False

    def forward(self, batch):
        # Extract inputs
        input_values = batch["input_values"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        decoder_attention_mask = batch["decoder_attention_mask"].to(self.device)

        # Extract embeddings from Wav2Vec2
        if self.frozen:
            with torch.no_grad():
                wav2vec_outputs = self.wav2vec2_model(input_values, attention_mask=attention_mask)
        else:
            wav2vec_outputs = self.wav2vec2_model(input_values, attention_mask=attention_mask)
        
        audio_embeddings = wav2vec_outputs.last_hidden_state
        reduced_embeddings = self.reduction_layer(audio_embeddings)

        # Pass embeddings to T5
        outputs = self.t5_model(
            inputs_embeds=reduced_embeddings,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )
        return outputs