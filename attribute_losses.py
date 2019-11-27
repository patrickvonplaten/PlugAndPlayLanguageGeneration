import torch
from hf_lg_utils import read_in_file


class AttributeLoss(object):

    def __init_(self):
        if type(self) is AttributeLoss:
            raise Exception('AbstractClass "AttributeLoss" cannot be instanteniated')

    def __call__(self, probs):
        # TODO: might need more input than just the probs
        raise NotImplementedError('Subclass must override __call__()')


class BagOfWordAttributeLoss(AttributeLoss):

    def __init__(self, topic, weight, tokenizer):
        self.topic = topic
        self.tokenizer = tokenizer
        self.weight = weight
        self.bag_of_words_file_path = './wordlists/{}.txt'.format(self.topic)
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

    def __call__(self, probs):
        topic_word_logits = torch.mm(probs, torch.t(self.bag_of_words_tensor))  # get probs of words included in bag of words for topic
        loss_bag_of_words = -torch.log(torch.sum(topic_word_logits))
        return self.weight * loss_bag_of_words
