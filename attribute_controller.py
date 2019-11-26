#!/usr/bin/env python3
from utils.file import read_in_file
import ipdb  # noqa: F401
import torch
import torch.nn.functional as F
import numpy as np
from operator import add
from hf_lg_utils import np_to_var


class AttributeController(object):

    def __init__(self):
        if type(self) is AttributeController:
            raise Exception('AbstractClass "AttributeController" cannot be instanteniated')

    def __call__(self, model, input_tokens, key_value_states):
        raise NotImplementedError('Subclass must override __call__()')


class UnconditionedAttributeController(AttributeController):

    def __init__(self):
        super(UnconditionedAttributeController, self).__init__()

    def __call__(self, model, input_tokens, key_value_states):
        return key_value_states,


class TopicAtttributeController(AttributeController):

    def __init__(self, tokenizer, config):
        super(TopicAtttributeController, self).__init__()
        self.tokenizer = tokenizer
        self.topic = config['topic']
        self.bag_of_words_file_path = './wordlists/{}.txt'.format(self.topic)
        self.window_len = config['window_len']
        self.kl_scale = config['kl_scale']
        self.num_grad_steps = config['num_grad_steps']
        self.step_size = config['step_size']
        self.gamma = config['gamma']
        self.grad_norms = None
        self.small_const = 1e-15
        self.bag_of_words_tensor = self.read_in_bag_of_words_to_tensor()

    def read_in_bag_of_words_to_tensor(self):
        vocab_size = len(self.tokenizer)
        bag_of_words = read_in_file(self.bag_of_words_file_path)
        tokenized_bag_of_words = [self.tokenizer.encode(word) for word in bag_of_words]
        one_token_bag_of_words = list(filter(lambda tokens: len(tokens) <= 1, tokenized_bag_of_words))
        one_token_bag_of_words_tensor = torch.tensor(one_token_bag_of_words)

        one_hot_bag_of_words_targets = torch.zeros(len(one_token_bag_of_words), vocab_size)
        one_hot_bag_of_words_targets.scatter_(1, one_token_bag_of_words_tensor, 1)
        return one_hot_bag_of_words_targets

    def perturb_past_key_value_states(self, model, prev_input_token, past_key_value_states, curr_probs, grad_norms):

        decay_mask = self.get_decay_mask()
        curr_seq_len = get_seq_len(past_key_value_states)
        window_mask = get_window_mask(past_key_value_states, curr_seq_len, decay_mask, self.window_len)
        past_key_value_states_orig = map_key_value_states(get_random_state, past_key_value_states)

        for _ in range(self.num_grad_steps):
            past_key_value_states_perturb = map_key_value_states(map_to_var, past_key_value_states_orig, requires_grad=True)
            loss = self.get_loss(model, prev_input_token, past_key_value_states_perturb, past_key_value_states, curr_probs)
            loss.backward()
            grad_norms = self.get_grad_norms(past_key_value_states_perturb, grad_norms, window_mask)
            grads = map_key_value_states(compute_grad, past_key_value_states_perturb, mask=window_mask, grad_norms=grad_norms, gamma=self.gamma, step_size=self.step_size)

            past_key_value_states_orig = list(map(add, grads, past_key_value_states_orig))
            map_key_value_states(set_grad_to_zero, past_key_value_states_perturb)
            past_key_value_states = map_key_value_states(detach_state, past_key_value_states)

        past_key_value_states_perturb = map_key_value_states(map_to_var, past_key_value_states_orig, requires_grad=False)
        comb_past_key_value_states_perturb = list(map(add, past_key_value_states, past_key_value_states_perturb))
        return comb_past_key_value_states_perturb, grad_norms

    def get_decay_mask(self):
        return torch.arange(0., 1.0 + self.small_const, 1.0 / (self.window_len))[1:]

    def get_loss(self, model, prev_input_token, past_key_value_states_perturb, past_key_value_states, curr_probs):

        comb_past_key_value_states_perturb = list(map(add, past_key_value_states, past_key_value_states_perturb))
        next_logit, next_key_value_states = model(prev_input_token, past=comb_past_key_value_states_perturb)
        next_probabilities = F.softmax(next_logit, dim=1)

        loss = self.get_bag_of_words_loss(next_probabilities)
        if self.kl_scale > 0.0:
            loss += self.get_kullback_leib_div_loss(curr_probs, next_probabilities)
        return loss

    def get_grad_norms(self, key_value_states, grad_norms, mask):
        if grad_norms is None:
            grad_norms = map_key_value_states(init_grad_norm, key_value_states, mask=mask, grad_norms=grad_norms, small_const=self.small_const)
        else:
            grad_norms = map_key_value_states(update_grad_norm, key_value_states, mask=mask, grad_norms=grad_norms)
        return grad_norms

    def get_bag_of_words_loss(self, probabilities):
        topic_word_logits = torch.mm(probabilities, torch.t(self.bag_of_words_tensor))  # get probabilities of words included in bag of words for topic
        loss_bag_of_words = -torch.log(torch.sum(topic_word_logits))
        return loss_bag_of_words

    def get_kullback_leib_div_loss(self, curr_probabilities, next_probabilities):
        next_curr_probabilities = (F.softmax(curr_probabilities, dim=1))
        next_curr_probabilities = next_curr_probabilities + self.small_const * (next_probabilities <= self.small_const).type(torch.FloatTensor).detach()  # leave x = x + a. using x += a leads to problems with gradient flow through in-place operations. # noqa: E501
        correction = self.small_const * (next_probabilities <= self.small_const).type(torch.FloatTensor).detach()
        corrected_next_probabilities = next_probabilities + correction.detach()
        kullback_leib__log_term = (corrected_next_probabilities / next_curr_probabilities).log()
        return self.kl_scale * ((corrected_next_probabilities * kullback_leib__log_term).sum())

    def __call__(self, model, prev_input_token, past_key_value_states=None, grad_norms=None):

        if past_key_value_states is None:
            past_input_tokens = prev_input_token[:, :-1]
            _, past_key_value_states = model(past_input_tokens)

        prev_input_token = prev_input_token[:, -1]
        curr_probs, _ = model(prev_input_token, past_key_value_states)
#        torch.autograd.set_detect_anomaly(True)

        perturb_past_key_value_states, grad_norms = self.perturb_past_key_value_states(model=model,
                prev_input_token=prev_input_token,
                past_key_value_states=past_key_value_states,
                curr_probs=curr_probs, grad_norms=grad_norms)

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
    return step_size * (normed_grad * mask).data.numpy()


def get_random_state(state, state_idx):
    return np.random.uniform(0.0, 0.0, state.shape).astype('float32')


def get_seq_len(key_value_states):
    return key_value_states[0].shape[3]


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
