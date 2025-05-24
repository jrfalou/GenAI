import torch
from torch import nn
from typing import Optional
from dataclasses import dataclass

from transformers import T5ForConditionalGeneration, AutoConfig, AutoModelForSequenceClassification
from transformers.modeling_outputs import Seq2SeqLMOutput
from trl import AutoModelForSeq2SeqLMWithValueHead
from trl.trainer.utils import first_true_indices


@dataclass
class ValueModelOutput(Seq2SeqLMOutput):
    """
    Base class for sequence-to-sequence language model outputs with an additional value head output.

    Inherits from:
        transformers.Seq2SeqLMOutput

    Added Fields:
        value (torch.FloatTensor of shape (batch_size, sequence_length)):
            Scalar value predictions from a value head, typically used in reinforcement learning
            or reward modeling settings (e.g., PPO, DPO, RLAIF).

    This structure is useful when you need both the usual seq2seq outputs (e.g. logits, attentions)
    and a per-token or per-sequence value estimate.
    """
    value: Optional[torch.FloatTensor] = None
    hidden_states: Optional[torch.FloatTensor] = None
    sequences: Optional[torch.FloatTensor] = None


def extract_input_and_output_segments(tensor, pad_token_id=0):
    """
    Extracts first and second segments (with preserved padding) from a 2D tensor.
    Assumes all sequences are already padded to the same length.
    Fully vectorized and GPU-compatible.

    Returns:
        Tuple[Tensor, Tensor]: input_seqs (B, input_len), output_seqs (B, output_len)
    """
    B, T = tensor.shape
    device = tensor.device

    # Detect segment starts
    non_pad = (tensor != pad_token_id).int()
    changes = torch.cat([torch.zeros(B, 1, dtype=torch.int, device=device), non_pad], dim=1)
    starts = ((changes[:, 1:] - changes[:, :-1]) == 1).nonzero(as_tuple=False)  # (N, 2)

    # Gather first and second start indices (assumes ≥2 segments per row)
    counts = torch.bincount(starts[:, 0], minlength=B)
    assert (counts >= 2).all(), "Each row must have at least two segments"

    row_offsets = torch.zeros(B, dtype=torch.long, device=device)
    unique_rows, row_counts = torch.unique_consecutive(starts[:, 0], return_counts=True)
    row_offsets[unique_rows] = row_counts.cumsum(0) - row_counts

    first_starts = starts[row_offsets, 1]
    second_starts = starts[row_offsets + 1, 1]

    # Infer lengths from the first row that has padding in output
    input_len = (second_starts - first_starts).min().item()

    # True output_len is fixed and visible in the batch — so we can just infer it by checking
    # how many tokens are left in the output segment including padding
    lengths = [(tensor[b, s:] != pad_token_id).nonzero(as_tuple=False).shape[0]
               for b, s in zip(range(B), second_starts)]
    output_len = max(lengths)

    # Build batched indices
    idx_in = torch.arange(input_len, device=device).unsqueeze(0)
    idx_out = torch.arange(output_len, device=device).unsqueeze(0)
    batch_idx = torch.arange(B, device=device).unsqueeze(1)

    input_idx = first_starts.unsqueeze(1) + idx_in
    output_idx = second_starts.unsqueeze(1) + idx_out

    # Clamp indices for safety (if a sequence is shorter than inferred length)
    input_idx = input_idx.clamp(max=T - 1)
    output_idx = output_idx.clamp(max=T - 1)

    input_seqs = tensor[batch_idx, input_idx]
    output_seqs = tensor[batch_idx, output_idx]

    return input_seqs, output_seqs


class MyT5WithOverrides(T5ForConditionalGeneration):
    @classmethod
    def from_pretrained_custom(cls, model_name, **kwargs):
        # Load config
        config = AutoConfig.from_pretrained(model_name, **kwargs)

        # Instantiate model and convert to dtype
        model = cls(config)
        
        # Load state dict from base model (same dtype)
        base_model = T5ForConditionalGeneration.from_pretrained(model_name, **kwargs)
        model.load_state_dict(base_model.state_dict())
        model = model.to(kwargs.get('torch_dtype', torch.float32))
        return model

    def forward(self, *args, **kwargs):
        if 'input_ids' in kwargs:
            inputs_, outputs_ = extract_input_and_output_segments(
                kwargs['input_ids'], pad_token_id=self.config.pad_token_id)
            kwargs['labels'] = outputs_
        kwargs.pop('position_ids', None)
        output = super().forward(*args, **kwargs)
        if 'input_ids' in kwargs:
            output.logits = torch.cat(
                [inputs_.unsqueeze(-1).expand(-1, -1, output.logits.shape[2]), output.logits], dim=1)
            hidden_states_ = tuple([
                torch.cat([inputs_.unsqueeze(-1).expand(-1, -1, s.shape[2]), s], dim=1)
                for s in output.decoder_hidden_states])
        else:
            hidden_states_ = output.decoder_hidden_states
        return ValueModelOutput(
            loss=output.loss,
            logits=output.logits,
            past_key_values=output.past_key_values,
            decoder_hidden_states=output.decoder_hidden_states,
            decoder_attentions=output.decoder_attentions,
            cross_attentions=output.cross_attentions,
            encoder_last_hidden_state=output.encoder_last_hidden_state,
            encoder_hidden_states=output.encoder_hidden_states,
            encoder_attentions=output.encoder_attentions,
            value=None,
            hidden_states=hidden_states_)


class ModelWrapper(nn.Module):
    base_model_prefix = "_self_as_base"

    def __init__(self, model):
        super().__init__()
        self.model = model  # registered in self._modules["model"]

        # sequence lengths are meant to be used when the score function is called by trl ppo_trainer
        # as it is not expecting a classification head
        self.sequence_lengths = None

    def __getattr__(self, name):
        if name == self.base_model_prefix:
            return self
        return getattr(self._modules["model"], name)

    def forward(self, *args, **kwargs):
        if 'input_ids' in kwargs:
            inputs_, labels = extract_input_and_output_segments(
                kwargs['input_ids'], pad_token_id=0)
            kwargs['attention_mask'] = labels != 0
            kwargs['position_ids'] = kwargs['attention_mask'].cumsum(1) - kwargs['attention_mask'].long()  # exclusive cumsum
            kwargs['input_ids'] = torch.masked_fill(labels, ~kwargs['attention_mask'], 0)
            self.sequence_lengths = inputs_.shape[1] + first_true_indices(labels == 0) - 1
        kwargs.pop("use_cache", None)
        output = self._modules["model"](*args, **kwargs)
        # TODO reshape outputs
        return output

    def score(self, encoder_last_hidden_states):
        r"""
        wrapper function to be used in ppo_training
        Args:
            decoder_hidden_states (`torch.Tensor`):
                The hidden states of the decoder. This is the output of the decoder.
        """
        # Step 2: Get [CLS] token embedding (first token)
        cls_hidden = encoder_last_hidden_states[:, 0, :]  # shape: [batch_size, hidden_size]

        # Step 3: Pass through the classification head
        classifier = self._modules['model'].classifier

        # Manual forward pass
        x = classifier.dense(cls_hidden)
        x = torch.tanh(x)
        x = classifier.dropout(x)
        output = classifier.out_proj(x)
        
        # sequence lengths are meant to be used when the score function is called by trl ppo_trainer
        # as it is not expecting a classification head
        # score will be computed as reward_logits[torch.arange(reward_logits.size(0), device=reward_logits.device),
        # sequence_lengths,].squeeze(-1)
        if self.sequence_lengths is not None:
            output = (
                torch.arange(self.sequence_lengths.max() + 1, device=output.device)[None, :]
                >= self.sequence_lengths[:, None]) * output[:, 0][:, None]
            self.sequence_lengths = None
        return output

    @classmethod
    def from_pretrained_custom(cls, model_name, **kwargs):
        return cls(AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs))


class MyAutoModelForSeq2SeqLMWithValueHead(AutoModelForSeq2SeqLMWithValueHead):
    r"""
    A seq2seq model with a value head in addition to the language model head.
    This class inherits from `~trl.PreTrainedModelWrapper` and wraps a
    `transformers.PreTrainedModel` class. The wrapper class supports classic functions
    such as `from_pretrained` and `push_to_hub` and also provides some additional
    functionalities such as `generate`.

    Args:
        pretrained_model (`transformers.PreTrainedModel`):
            The model to wrap. It should be a causal language model such as GPT2.
            or any model mapped inside the `AutoModelForSeq2SeqLM` class.
        kwargs:
            Additional keyword arguments passed along to the `ValueHead` class.
    """
    def __init__(self, pretrained_model, **kwargs):
        super().__init__(pretrained_model, **kwargs)

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        return_past_key_values=False,
        **kwargs,
    ):
        kwargs["past_key_values"] = past_key_values
        if self.is_peft_model and self.pretrained_model.active_peft_config.peft_type == "PREFIX_TUNING":
            kwargs.pop("past_key_values", None)

        kwargs.pop('output_hidden_states', None)
        base_model_output = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,  # We force the model to output hidden states
            **kwargs,
        )

        last_hidden_state = base_model_output.decoder_hidden_states[-1]
        lm_logits = base_model_output.logits
        loss = base_model_output.loss

        value = self.v_head(last_hidden_state).squeeze(-1)

        # force upcast in fp32 if logits are in half-precision
        if lm_logits.dtype != torch.float32:
            lm_logits = lm_logits.float()

        if return_past_key_values:
            return (lm_logits, loss, value, base_model_output.past_key_values)
        else:
            # in the ppo_trainer script, logits is expected to also contain inputs logits, even though they're discarded next
            # as values don't matter but only shape does, let's prepend inputs_ to the logits
            return ValueModelOutput(
                loss=loss,
                logits=lm_logits,
                past_key_values=base_model_output.past_key_values,
                decoder_hidden_states=base_model_output.decoder_hidden_states,
                decoder_attentions=base_model_output.decoder_attentions,
                cross_attentions=base_model_output.cross_attentions,
                encoder_last_hidden_state=base_model_output.encoder_last_hidden_state,
                encoder_hidden_states=base_model_output.encoder_hidden_states,
                encoder_attentions=base_model_output.encoder_attentions,
                value=value,
            )

    def generate(self, *args, **kwargs):
        r"""
        We call `generate` on the wrapped model.
        """
        output = self.pretrained_model.generate(*args, **kwargs)
        if 'input_ids' in kwargs:
            try:
                inputs_, labels = extract_input_and_output_segments(kwargs['input_ids'], pad_token_id=0)
            except AssertionError as e:
                inputs_ = kwargs['input_ids']
        if isinstance(output, torch.Tensor):
            output = ValueModelOutput(sequences=output)
        output.sequences = torch.cat([inputs_, output.sequences[:, 1:]], dim=1)
        return output
        
    def score(self, decoder_hidden_states):
        r"""
        wrapper function to be used in ppo_training
        Args:
            decoder_hidden_states (`torch.Tensor`):
                The hidden states of the decoder. This is the output of the decoder.
        """
        return self.v_head(decoder_hidden_states).squeeze(-1)
