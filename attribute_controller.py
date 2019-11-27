#!/usr/bin/env python3
import ipdb  # noqa: F401
import torch
import torch.nn.functional as F
import numpy as np
from operator import add
from hf_lg_utils import np_to_var
from attribute_losses import BagOfWordAttributeLoss

BOW_LOSS = 'bow'
SENTIMENT_LOSS = 'sentiment'
CLICKBAIT_LOSS = 'clickbait'


class AtttributeController(object):

    def __init__(self, tokenizer, config):
        self.attribute_losses_config = config['attribute_losses']
        self.attribute_losses = self.create_attribute_losses(self.attribute_losses_config, tokenizer)
        self.window_len = config['window_len']
        self.kl_scale = config['kl_scale']
        self.num_grad_steps = config['num_grad_steps']
        self.step_size = config['step_size']
        self.gamma = config['gamma']
        self.small_const = 1e-15

    def create_attribute_losses(self, losses_config, tokenizer):
        attribute_losses = []
        for loss_config in losses_config:
            loss_type = loss_config['type']
            if loss_type == BOW_LOSS:
                attribute_losses.append(BagOfWordAttributeLoss(loss_config['topic'], loss_config['weight'], tokenizer))
            elif loss_type == SENTIMENT_LOSS:
                NotImplementedError('SENTIMENT_LOSS not implemented yet')
            elif loss_type == CLICKBAIT_LOSS:
                NotImplementedError('CLICKBAIT_LOSS not implemented yet')
            else:
                raise ValueError('{} does not exist'.format(loss_type))
        return attribute_losses

    def perturb_past_key_value_states(self, model, prev_input_token, past_key_value_states, orig_logits, grad_norms):

        decay_mask = get_decay_mask(self.small_const, self.window_len)
        curr_seq_len = get_seq_len(past_key_value_states)
        window_mask = get_window_mask(past_key_value_states, curr_seq_len, decay_mask, self.window_len)
        past_key_value_states_orig = map_key_value_states(get_random_state, past_key_value_states)

        for _ in range(self.num_grad_steps):
            past_key_value_states_perturb = map_key_value_states(map_to_var, past_key_value_states_orig, requires_grad=True)
            loss = self.get_total_loss(model, prev_input_token, past_key_value_states_perturb, past_key_value_states, orig_logits)
            loss.backward()
            grad_norms = self.get_grad_norms(past_key_value_states_perturb, grad_norms, window_mask)
            grads = map_key_value_states(compute_grad, past_key_value_states_perturb, mask=window_mask, grad_norms=grad_norms, gamma=self.gamma, step_size=self.step_size)

            past_key_value_states_orig = list(map(add, grads, past_key_value_states_orig))
            map_key_value_states(set_grad_to_zero, past_key_value_states_perturb)
            past_key_value_states = map_key_value_states(detach_state, past_key_value_states)

        past_key_value_states_perturb = map_key_value_states(map_to_var, past_key_value_states_orig, requires_grad=False)
        comb_past_key_value_states_perturb = list(map(add, past_key_value_states, past_key_value_states_perturb))
        return comb_past_key_value_states_perturb, grad_norms

    def get_total_loss(self, model, prev_input_token, past_key_value_states_perturb, past_key_value_states, orig_logits):
        comb_past_key_value_states_perturb = list(map(add, past_key_value_states, past_key_value_states_perturb))
        next_logit_perturb, _ = model(prev_input_token, past=comb_past_key_value_states_perturb)
        next_probs_perturb = F.softmax(next_logit_perturb, dim=1)  # probabs

        loss = 0
        for attribute_loss in self.attribute_losses:
            loss += attribute_loss(next_probs_perturb)
        if self.kl_scale > 0.0:
            loss += self.get_kullback_leib_div_loss(orig_logits, next_probs_perturb)
        return loss

    def get_kullback_leib_div_loss(self, orig_logits, next_probs_perturb):
        orig_probs = (F.softmax(orig_logits, dim=1))
        orig_probs = orig_probs + self.small_const * (orig_probs < self.small_const).type(torch.FloatTensor).detach()  # leave x = x + a. using x += a leads to problems with gradient flow through in-place operations. # noqa: E501
        correction = self.small_const * (next_probs_perturb < self.small_const).type(torch.FloatTensor).detach()
        corrected_next_probs_perturb = next_probs_perturb + correction.detach()
        kullback_leib_log_term = (corrected_next_probs_perturb / orig_probs).log()
        return self.kl_scale * ((corrected_next_probs_perturb * kullback_leib_log_term).sum())

    def get_grad_norms(self, key_value_states, grad_norms, mask):
        if grad_norms is None:
            grad_norms = map_key_value_states(init_grad_norm, key_value_states, mask=mask, grad_norms=grad_norms, small_const=self.small_const)
        else:
            grad_norms = map_key_value_states(update_grad_norm, key_value_states, mask=mask, grad_norms=grad_norms)
        return grad_norms

    def __call__(self, model, prev_input_token, past_key_value_states=None, grad_norms=None):

        if past_key_value_states is None:
            past_input_tokens = prev_input_token[:, :-1]
            _, past_key_value_states = model(past_input_tokens)

        prev_input_token = prev_input_token[:, -1]
        orig_logits, _ = model(prev_input_token, past_key_value_states)

        perturb_past_key_value_states, grad_norms = self.perturb_past_key_value_states(model=model,
                prev_input_token=prev_input_token,
                past_key_value_states=past_key_value_states,
                orig_logits=orig_logits, grad_norms=grad_norms)

        return perturb_past_key_value_states


def map_key_value_states(mapping_fn, key_value_states, **kwargs):
    return [mapping_fn(state, state_idx, **kwargs) for state_idx, state in enumerate(key_value_states)]


def set_grad_to_zero(state, state_idx):
    state.grad.data.zero_()


def detach_state(state, state_idx):
    return state.detach()


def map_to_var(state, state_idx, requires_grad):
    return np_to_var(state, requires_grad=requires_grad)


def update_grad_norm(state, state_idx, mask, grad_norms):
    curr_grad_norms = torch.norm(state.grad * mask)
    prev_grad_norms = grad_norms[state_idx]
    return torch.max(curr_grad_norms, prev_grad_norms)


def init_grad_norm(state, state_idx, mask, grad_norms, small_const):
    return torch.norm(state.grad * mask) + small_const


def compute_grad(state, state_idx, mask, grad_norms, gamma, step_size):
    normed_grad = state.grad / grad_norms[state_idx] ** gamma
    return -step_size * (normed_grad * mask).data.numpy()


def get_random_state(state, state_idx):
    return np.random.uniform(0.0, 0.0, state.shape).astype('float32')


def get_seq_len(key_value_states):
    return key_value_states[0].shape[3]


def get_decay_mask(small_const, window_len):
    return torch.arange(0., 1.0 + small_const, 1.0 / (window_len))[1:]


def get_window_mask(key_value_states, curr_seq_len, decay_mask, window_len):
    if curr_seq_len <= window_len:
        return torch.ones_like(key_value_states[0])

    key_val_zeros = torch.zeros(get_key_value_states_shape(key_value_states, curr_seq_len - window_len))
    key_val_ones = torch.ones(get_key_value_states_shape(key_value_states, window_len))
    key_val_ones_masked = apply_mask(key_val_ones, decay_mask)
    return torch.cat((key_val_ones_masked, key_val_zeros), dim=-2)


def apply_mask(key_value_states, mask):
    key_value_states_reverse = mask * key_value_states.permute(0, 1, 2, 4, 3)
    return key_value_states_reverse.permute(0, 1, 2, 4, 3)


def get_key_value_states_shape(key_value_states, seq_len):
    return tuple(key_value_states[0].shape[:-2]) + tuple([seq_len]) + tuple([key_value_states[0].shape[-1]])
