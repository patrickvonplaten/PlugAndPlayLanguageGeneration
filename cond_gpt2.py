#!/usr/bin/env python3
import ipdb  # noqa: F401
from transformers import GPT2LMHeadModel
# from transformers import GPT2LMHeadModel, GPT2Config
from attribute_controller import UnconditionedAttributeController, TopicAtttributeController
import torch
import torch.nn.functional as F
from hf_lg_utils import top_k_top_p_filtering


class CondGPT2LMHeadModel(object):

    def __init__(self, model, attribute_controller, config):
        self.model = model
        self.attribute_controller = attribute_controller
        ipdb.set_trace()
        self.fuse_with_real_probs = config['fuse_with_real_probs']
        self.gm_scale = config['gm_scale']
        self.temperature = config['temperature']
        self.top_k = config['top_k']
        self.top_p = config['top_p']
        self.init_model()

    def init_model(self):
        self.model.to('cpu')
        self.model.eval()
        for param in self.model.parameters():  # Freeze model weights
            param.requires_grad = False

    @classmethod
    def from_conditioned_on_topic(cls, model_size, tokenizer, config):
        model = GPT2LMHeadModel.from_pretrained(model_size)
        ipdb.set_trace()
        attribute_controller = TopicAtttributeController(tokenizer, config['attribute_controller'])
        return cls(model, attribute_controller, config['lm_model'])

    @classmethod
    def from_unconditioned(cls, model_size, config):
        model = GPT2LMHeadModel.from_pretrained(model_size)
        attribute_controller = UnconditionedAttributeController()
        return cls(model, attribute_controller, config['lm_model'])

    def __call__(self, input_tokens, past_key_value_states):
        conditioned_past_key_value_embed = self.attribute_controller(self.model, input_tokens, past_key_value_states)
        tokens_logits, key_value_embeds = self.model(input_tokens, past=conditioned_past_key_value_embed)
        last_token_logits = torch.squeeze(tokens_logits[:, -1, :]).detach()
        heated_last_token_logits = last_token_logits / self.temperature
        token_probs = F.softmax(heated_last_token_logits, dim=-1)

        if self.fuse_with_real_probs:
            orig_token_logits, _ = self.model(input_tokens, past=past_key_value_states)
            orig_last_token_logits = torch.squeeze(orig_token_logits[:, -1, :])
            heated_orig_token_logits = orig_last_token_logits / self.temperature
            orig_token_probs = F.softmax(heated_orig_token_logits, dim=-1)
            token_probs = (token_probs ** self.gm_scale) * (orig_token_probs ** (1 - self.gm_scale))

        filtered_next_token_probs = top_k_top_p_filtering(token_probs, top_k=self.top_k, top_p=self.top_p, filter_value=0.0)

        return filtered_next_token_probs, key_value_embeds
