#!/usr/bin/env python3
from utils.file import read_in_file
import ipdb  # noqa: F401
import torch
import torch.nn.functional as F
import numpy as np
from operator import add
from torch.autograd import Variable


def torch_to_var(tensor, requires_grad=False, volatile=False):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor, requires_grad=requires_grad, volatile=volatile)


def np_to_var(np_array, requires_grad=False, volatile=False):
    tensor = torch.from_numpy(np_array)
    return torch_to_var(tensor, requires_grad=requires_grad, volatile=volatile)


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

    def __init__(self, topic, tokenizer):
        super(TopicAtttributeController, self).__init__()
        self.bag_of_words_file_path = './wordlists/{}.txt'.format(topic)
        self.tokenizer = tokenizer
        self.bag_of_words_tensor = self.read_in_bag_of_words_to_tensor()
        self.do_pertub = True
        self.grad_norms = None
        self.window_length = 5
        self.kl_scale = 0.1
        self.num_iterations = 3
        self.small_const = 1e-15
        self.step_size = 0.03
        self.gamma = 1.5

    def read_in_bag_of_words_to_tensor(self):
        vocab_size = len(self.tokenizer)
        bag_of_words = read_in_file(self.bag_of_words_file_path)
        tokenized_bag_of_words = [self.tokenizer.encode(word) for word in bag_of_words]
        one_token_bag_of_words = list(filter(lambda tokens: len(tokens) <= 1, tokenized_bag_of_words))
        one_token_bag_of_words_tensor = torch.tensor(one_token_bag_of_words)

        one_hot_bag_of_words_targets = torch.zeros(len(one_token_bag_of_words), vocab_size)
        one_hot_bag_of_words_targets.scatter_(1, one_token_bag_of_words_tensor, 1)
        return one_hot_bag_of_words_targets

    def perturb_past_key_value_states(self, model, prev_input_token, past_key_value_states, curr_probs):

        decay_mask = torch.arange(0., 1.0 + self.small_const, 1.0 / (self.window_length))[1:]
        curr_len = past_key_value_states[0].shape[3]

        if curr_len <= self.window_length:
            window_mask = torch.ones_like(past_key_value_states[0])
        else:
            window_mask = self.get_window_mask(past_key_value_states, curr_len, decay_mask)

#        print('Shape prev_input_token: {}'.format(prev_input_token.shape))
#        print('Shape past_key_value_states: {}'.format(past_key_value_states[0].shape))
#        print('Shape curr_probs: {}'.format(curr_probs.shape))

        past_key_value_states_perturb_orig = [(np.random.uniform(0.0, 0.0, tensor.shape).astype('float32')) for tensor in past_key_value_states]

        for iter_idx in range(self.num_iterations):
            past_key_value_states_perturb = [np_to_var(np_array, requires_grad=True) for np_array in past_key_value_states_perturb_orig]
            comb_past_key_value_states_perturb = list(map(add, past_key_value_states, past_key_value_states_perturb))

            next_logit, next_key_value_states = model(prev_input_token, past=comb_past_key_value_states_perturb)
            next_probabilities = F.softmax(next_logit, dim=1)

            loss = self.get_bag_of_words_loss(next_probabilities)
            if self.kl_scale > 0.0:
                loss += self.get_kullback_leib_div_loss(curr_probs, next_probabilities)
            loss.backward()

            self.grad_norms = self.get_initial_grad_norm(past_key_value_states_perturb, window_mask) if self.grad_norms is None else self.update_grad_norm(self.grad_norms, past_key_value_states_perturb, window_mask)  # noqa: E501
            grad = self.get_grad(self.grad_norms, past_key_value_states_perturb, window_mask)
            past_key_value_states_perturb_orig = list(map(add, grad, past_key_value_states_perturb_orig))
            self.set_grads_to_zero(past_key_value_states_perturb)
            past_key_value_states = [key_value_state.detach() for key_value_state in past_key_value_states]

#        past_key_value_states_perturb = [np_to_var(np_array, requires_grad=True) for np_array in past_key_value_states_perturb_orig]
        past_key_value_states_perturb = [np_to_var(np_array) for np_array in past_key_value_states_perturb_orig]
        comb_past_key_value_states_perturb = list(map(add, past_key_value_states, past_key_value_states_perturb))
        return comb_past_key_value_states_perturb

    def get_window_mask(self, past_key_value_states, curr_len, decay_mask):
        ones_key_val_shape = tuple(past_key_value_states[0].shape[:-2]) + tuple([self.window_length]) + tuple([past_key_value_states[0].shape[-1]])

        zeros_key_val_shape = tuple(past_key_value_states[0].shape[:-2]) + tuple([curr_len - self.window_length]) + tuple([past_key_value_states[0].shape[-1]])

        ones_mask = torch.ones(ones_key_val_shape)
        ones_mask = decay_mask * ones_mask.permute(0, 1, 2, 4, 3)
        ones_mask = ones_mask.permute(0, 1, 2, 4, 3)
        return torch.cat((ones_mask, torch.zeros(zeros_key_val_shape)), dim=-2)

    def set_grads_to_zero(self, key_value_states):
        for key_value_state in key_value_states:
            key_value_state.grad.data.zero_()

    def update_grad_norm(self, grad_norms, key_value_states, window_mask):
        return [torch.max(grad_norms[index], torch.norm(key_value_state.grad * window_mask)) for index, key_value_state in enumerate(key_value_states)]

    def get_initial_grad_norm(self, key_value_states, window_mask):
        return [(torch.norm(key_value_state.grad * window_mask) + self.small_const) for index, key_value_state in enumerate(key_value_states)]

    def get_grad(self, grad_norms, key_value_states, window_mask):
        return [-self.step_size * (key_value_state.grad * window_mask / grad_norms[index] ** self.gamma).data.numpy() for index, key_value_state in enumerate(key_value_states)]

    def get_kullback_leib_div_loss(self, curr_probabilities, next_probabilities):
        next_curr_probabilities = (F.softmax(curr_probabilities, dim=1))

        next_curr_probabilities = next_curr_probabilities + self.small_const * (next_probabilities <= self.small_const).type(torch.FloatTensor).detach()  # leave x = x + a. using x += a leads to problems with gradient flow through in-place operations. # noqa: E501
        correction = self.small_const * (next_probabilities <= self.small_const).type(torch.FloatTensor).detach()
        corrected_next_probabilities = next_probabilities + correction.detach()
        kullback_leib__log_term = (corrected_next_probabilities / next_curr_probabilities).log()
        return self.kl_scale * ((corrected_next_probabilities * kullback_leib__log_term).sum())

    def get_bag_of_words_loss(self, probabilities):
        topic_word_logits = torch.mm(probabilities, torch.t(self.bag_of_words_tensor))  # get probabilities of words included in bag of words for topic
        loss_bag_of_words = -torch.log(torch.sum(topic_word_logits))
        return loss_bag_of_words

    def __call__(self, model, prev_input_token, past_key_value_states):

        if past_key_value_states is None:
            past_input_tokens = prev_input_token[:, :-1]
            _, past_key_value_states = model(past_input_tokens)

        prev_input_token = prev_input_token[:, -1]
        curr_probs, _ = model(prev_input_token, past_key_value_states)
        torch.autograd.set_detect_anomaly(True)

        if self.do_pertub:
            perturb_past_key_value_states = self.perturb_past_key_value_states(model=model,
                    prev_input_token=prev_input_token,
                    past_key_value_states=past_key_value_states,
                    curr_probs=curr_probs)

        return perturb_past_key_value_states
