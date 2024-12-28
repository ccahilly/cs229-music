import torch.nn as nn
import torch
from transformers import T5ForConditionalGeneration, AutoModel

class MertT5Model(nn.Module):
    def __init__(self, device="cpu", mert_model=None, t5_model=None, frozen=False):
        super(MertT5Model, self).__init__()
        self.device = device
        self.frozen = frozen

        self.mert_model = mert_model or AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True).to(self.device)
        self.t5_model = t5_model or T5ForConditionalGeneration.from_pretrained("t5-small").to(self.device)

        self.aggregator = nn.Conv1d(in_channels=13, out_channels=1, kernel_size=1).to(self.device)
        self.reduction_layer = nn.Linear(768, self.t5_model.config.d_model).to(self.device)
        
        if self.frozen:
            for param in self.mert_model.parameters():
                param.requires_grad = False

    def forward(self, batch):
        # Extract inputs
        inputs = batch["inputs"].to(self.device)
        inputs["input_values"] = inputs["input_values"].squeeze(1)
        labels = batch["labels"].to(self.device)
        decoder_attention_mask = batch["decoder_attention_mask"].to(self.device)

        # Extract embeddings from MERT
        if self.frozen:
            with torch.no_grad():
                mert_outputs = self.mert_model(inputs["input_values"], output_hidden_states=True)
        else:
            mert_outputs = self.mert_model(inputs["input_values"], output_hidden_states=True)
        
        all_layer_hidden_states = torch.stack(mert_outputs.hidden_states).squeeze()

        current_batch_size = all_layer_hidden_states.size(1)  # Dynamically fetch batch size
        print(f"Current batch size: {current_batch_size}")
        seq_len = inputs["input_values"].size(1)  # Dynamically fetch sequence length
        feature_dim = all_layer_hidden_states.size(-1) // seq_len  # Compute feature size
        print(f"Sequence length: {seq_len}, Feature dimension: {feature_dim}")
        
        combined_dim = all_layer_hidden_states.view(current_batch_size, 13, -1) # [batch_size, layers, time_steps * features]

        # Apply Conv1d for learnable aggregation
        aggregated_embedding = self.aggregator(combined_dim)  # [batch_size, 1, time_steps * features]

        # Uncombine the last dimension back into time_steps and features
        aggregated_embedding = aggregated_embedding.view(current_batch_size, 749, 768)  # [batch_size, time_steps, features]

        # Reduce embeddings
        reduced_embeddings = self.reduction_layer(aggregated_embedding)

        # Pass embeddings to T5
        outputs = self.t5_model(
            inputs_embeds=reduced_embeddings,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )
        return outputs
    
    def inference(self, batch, tokenizer, max_length=50):
        """Run inference to generate captions."""
        with torch.no_grad():
            # Extract inputs
            inputs = batch["inputs"].to(self.device)
            inputs["input_values"] = inputs["input_values"].squeeze(1)

            # Extract embeddings from MERT
            mert_outputs = self.mert_model(inputs["input_values"], output_hidden_states=True)

            all_layer_hidden_states = torch.stack(mert_outputs.hidden_states).squeeze()
            current_batch_size = all_layer_hidden_states.size(1)  # Dynamically fetch batch size

            combined_dim = all_layer_hidden_states.view(current_batch_size, 13, -1)

            # Aggregate embeddings
            aggregated_embedding = self.aggregator(combined_dim)
            aggregated_embedding = aggregated_embedding.view(current_batch_size, 749, 768)

            # Reduce embeddings
            reduced_embeddings = self.reduction_layer(aggregated_embedding)

            # Generate predictions
            outputs = self.t5_model.generate(
                inputs_embeds=reduced_embeddings,
                max_length=max_length,
                num_beams=5,  # Beam search for diversity
                early_stopping=True
            )

            # Decode predictions
            predictions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        
            return predictions