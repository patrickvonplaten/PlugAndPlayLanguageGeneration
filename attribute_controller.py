#!/usr/bin/env python3
from utils.file import read_in_file
import ipdb
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
        return key_value_states


class TopicAtttributeController(AttributeController):

    bag_of_words_files = {
        'science': './wordlists/science.txt',
    }

    def __init__(self, topic, tokenizer, do_pertub=True, window_length=5, gm_scale=0.8, kl_scale=0.4, num_iterations=1):
        super(TopicAtttributeController, self).__init__()
        self.topic = topic
        self.tokenizer = tokenizer
        self.bag_of_words_tensor = self.read_in_bag_of_words_to_tensor()
        self.do_pertub = do_pertub
        self.window_length = window_length
        self.gm_scale = gm_scale
        self.kl_scale = kl_scale
        self.num_iterations = num_iterations
        self.small_const = 1e-15

    def read_in_bag_of_words_to_tensor(self):
        vocab_size = len(self.tokenizer)
        bag_of_words_file_path = TopicAtttributeController.bag_of_words_files[self.topic]
        bag_of_words = read_in_file(bag_of_words_file_path)
        tokenized_bag_of_words = [self.tokenizer.encode(word) for word in bag_of_words]
        one_token_bag_of_words = list(filter(lambda tokens: len(tokens) <= 1, tokenized_bag_of_words))
        one_token_bag_of_words_tensor = torch.tensor(one_token_bag_of_words)

        one_hot_bag_of_words_targets = torch.zeros(len(one_token_bag_of_words), vocab_size)
        one_hot_bag_of_words_targets.scatter_(1, one_token_bag_of_words_tensor, 1)
        return one_hot_bag_of_words_targets

    def perturb_past_key_value_states(self, model, prev_input_token, past_key_value_states, curr_key_value_states, curr_probs, grad_norms):

        past_key_value_states_perturb_orig = [(np.random.uniform(0.0, 0.0, tensor.shape).astype('float32')) for tensor in past_key_value_states]

        for iter_idx in range(self.num_iterations):
            past_key_value_states_perturb = [np_to_var(np_array, requires_grad=True) for np_array in past_key_value_states_perturb_orig]
            comb_past_key_value_states_perturb = list(map(add, past_key_value_states, past_key_value_states_perturb))

            next_logits, next_key_value_states, _ = model(prev_input_token, past=comb_past_key_value_states_perturb)
            next_logits_last_token = next_logits[:, -1, :]
            next_probabilities = F.softmax(next_logits_last_token, dim=1)
            loss = self.get_bag_of_words_loss(next_probabilities)
            if self.kl_scale > 0.0:
                loss += self.get_kullback_leib_div_loss(curr_probs, next_probabilities)
            loss.backward()

            grad_norms = self.get_initial_grad_norm(past_key_value_states_perturb, window_mask) if grad_norms is None else self.update_grad_norm(grad_norms, past_key_value_states_perturb, window_mask)
            grad = self.get_grad(grad_norms, key_value_states, window_mask)
            past_key_value_states_perturb_orig = list(map(ad, grad, past_key_value_states_perturb_orig))
            self.set_grads_to_zero(past_key_value_states_perturb)
            past_key_value_states = [key_value_state.detach() for key_value_state in past_key_value_states]

        past_

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
        ipdb.set_trace()
        next_curr_probabilities = (F.softmax(curr_probabilities[:, -1, :], dim=-1))
        next_curr_probabilities += self.small_const * (next_probabilities <= self.small_const).type(torch.FloatTensor).detach()
        correction = self.small_const * (next_probabilities <= self.small_const).type(torch.FloatTensor).detach()
        corrected_next_probabilities = next_probabilities + correction
        kullback_leib__log_term = (corrected_next_probabilities / next_curr_probabilities).log()
        return self.kl_scale * ((corrected_next_probabilities * kullback_leib__log_term).sum())

    def get_bag_of_words_loss(self, probabilities):
        topic_word_logits = torch.mm(probabilities, torch.t(self.bag_of_words_tensor))  # get probabilities of words included in bag of words for topic
        loss_bag_of_words = -torch.log(torch.sum(topic_word_logits))
        return loss_bag_of_words

    def __call__(self, model, input_tokens, past_key_value_states):
        grad_norms = None
#        if prev_key_value_states is None:

        past_input_tokens = input_tokens[:, :-1]
        prev_input_token = input_tokens[:, -1]
        _, past_key_value_states, _ = model(past_input_tokens)
        curr_probs, curr_key_value_states, _ = model(input_tokens)

        if self.do_pertub:
            self.perturb_past_key_value_states(model=model, prev_input_token=prev_input_token,
                    past_key_value_states=past_key_value_states,
                    curr_probs=curr_probs, curr_key_value_states=curr_key_value_states,
                    grad_norms=grad_norms)
