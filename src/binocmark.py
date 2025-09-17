import torch
from functools import cached_property
import scipy.stats
from transformers import AutoModel, AutoConfig
import torch
from peft import PeftModel
import torch.nn.functional as F

class BinocMark:
    def __init__(self, model, performer_lora_path, observer_lora_path, tokenizer):
        self.model = model
        self.observer_lora_path = observer_lora_path
        self.performer_lora_path = performer_lora_path
        self.tokenizer = tokenizer
        self.binocmodel = PeftModel.from_pretrained(model, observer_lora_path, 'observer')
        self.binocmodel.load_adapter(model_id=performer_lora_path, adapter_name='performer')
        self.binocmodel.eval()

    @torch.no_grad()
    def score_text_batch(self, text_batch):
        inputs = self.tokenizer(text_batch, return_tensors="pt", padding=True).to(self.model.device)
        self.binocmodel.set_adapter('performer')
        performer_outputs = self.binocmodel(inputs.input_ids, attention_mask=inputs.attention_mask)
        self.binocmodel.set_adapter('observer')
        observer_outputs = self.binocmodel(inputs.input_ids, attention_mask=inputs.attention_mask)
        encodings = {"input_ids": inputs.input_ids, "attention_mask": inputs.attention_mask}
        ppl = self.log_ppl(encodings, performer_outputs.logits)
        xppl = self.log_xppl(observer_outputs.logits, performer_outputs.logits, encodings)
        binoc_scores = ppl / xppl
        return binoc_scores.float()

    def log_ppl(self, encoding,
               logits: torch.Tensor,
               temperature: float = 1.0):
        logits = (logits[..., :-1, :] / temperature).contiguous()      # (B, T-1, V)
        labels = encoding['input_ids'][..., 1:].contiguous()           # (B, T-1)
        attn   = encoding['attention_mask'][..., 1:].contiguous()      # (B, T-1)

        # Token-level negative log likelihood
        nll = F.cross_entropy(
            input=logits.transpose(1, 2),  # (B, V, T-1)
            target=labels,
            reduction='none'
        )

        # Masked average
        agg = (nll * attn).sum(dim=1) / attn.sum(dim=1)
        return agg  # (B,)



    def log_xppl(self, observer_logits: torch.Tensor,
            performer_logits: torch.Tensor,
            encoding,
            temperature: float = 1.0):
        """
        Compute log-XPPL = average over tokens of <MO_probs, log MP_probs>.
        """

        o = (observer_logits[..., :-1, :] / temperature).contiguous()
        q = (performer_logits[..., :-1, :] / temperature).contiguous()
        attn = encoding['attention_mask'][..., 1:].contiguous()        # (B, T-1)

        # MO probabilities and MP log-probs
        mo_probs = F.softmax(o, dim=-1)       # (B, T-1, V)
        mp_logp  = F.log_softmax(q, dim=-1)   # (B, T-1, V)

        # Token-wise inner product
        token_scores = (mo_probs * mp_logp).sum(dim=-1)  # (B, T-1)

        # Masked average
        agg = (token_scores * attn).sum(dim=1) / attn.sum(dim=1)
        return -agg  # (B,)
    