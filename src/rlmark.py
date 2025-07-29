from src.rl_watermark.model_utils import RewardModel
from transformers import AutoModel, AutoConfig
import torch


class RLMark:
    def __init__(self, rl_model_path, tokenizer):
        model_config = AutoConfig.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
        for key in ('dropout', 'attention_dropout', 'hidden_dropout', 'activation_dropout'):
            if hasattr(model_config, key):
                setattr(model_config, key, 0.0)

        detector = AutoModel.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", config=model_config, device_map="auto")
        detector = detector.train()

        self.device = detector.device
        reward_model = RewardModel(detector, tokenizer)

        reward_model.load_state_dict(torch.load(
            rl_model_path + "/reward_model.ckpt"))
        reward_model = reward_model.to(self.device)
        self.reward_model = reward_model
    
        self.tokenizer = tokenizer

    @torch.no_grad()
    def score_text_batch(self, text_batch):
        inputs = self.tokenizer(text_batch, return_tensors="pt", padding=True).to(self.device)
        human_scores = self.reward_model.forward_value(
            inputs.input_ids, inputs.attention_mask, prompt_length=1)
        return human_scores["chosen_end_scores"].cpu()
