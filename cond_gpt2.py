#!/usr/bin/env python3
import ipdb  # noqa: F401
from transformers import GPT2LMHeadModel, GPT2Config
from attribute_controller import UnconditionedAttributeController, TopicAtttributeController


class CondGPT2LMHeadModel(object):

    def __init__(self, model, attribute_controller):
        self.model = model
        self.init_model()
        self.attribute_controller = attribute_controller

    def init_model(self):
        self.model.to('cpu')
        self.model.eval()
        for param in self.model.parameters():  # Freeze GPT-2 weights
            param.requires_grad = False

    @classmethod
    def from_conditioned_on_topic(cls, model_size, topic, tokenizer):
        config = GPT2Config.from_pretrained(model_size)
        config.__dict__['output_hidden_states'] = True
        model = GPT2LMHeadModel.from_pretrained(model_size, config=config)
        attribute_controller = TopicAtttributeController(topic, tokenizer)
        return cls(model, attribute_controller)

    @classmethod
    def from_unconditioned(cls, model_size):
        model = GPT2LMHeadModel.from_pretrained(model_size)
        attribute_controller = UnconditionedAttributeController()
        return cls(model, attribute_controller)

    def __call__(self, input_tokens, past_key_value_embeds):
        conditioned_past_key_value_embed = self.attribute_controller(self.model, input_tokens, past_key_value_embeds)
        tokens_logits, key_value_embeds = self.model(input_tokens, past=conditioned_past_key_value_embed)
        return tokens_logits, key_value_embeds
