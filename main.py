#!/usr/bin/env python3
import ipdb  # noqa: F401
import random  # noqa: F401
import numpy as np  # noqa: F401
from argparse import ArgumentParser  # noqa: F401
import torch
from transformers import GPT2Tokenizer
from cond_gpt2 import CondGPT2LMHeadModel


def main(args):
    input_words = args.input_words
    num_words_to_generate = args.num_words_to_generate
    model_size = args.model_size
    topic = args.topic
#    seed = args.seed

#    torch.manual_seed(seed)

    tokenizer = GPT2Tokenizer.from_pretrained(model_size)
    cond_model = CondGPT2LMHeadModel.from_conditioned_on_topic(model_size, topic, tokenizer)

    tokenized_input_words = create_tokenized_input_words(tokenizer, input_words)
    generated_tokens = generate_tokens_auto_reg(cond_model, tokenized_input_words, num_words_to_generate, tokenizer)
    generated_words = tokenizer.decode(generated_tokens, clean_up_tokenization_spaces=True, skip_special_tokens=True)
    print(generated_words)


def create_tokenized_input_words(tokenizer, input_words):
    tokenized_input_words = torch.tensor(tokenizer.encode(input_words), dtype=torch.long).unsqueeze(0)  # Batch size 1
    return torch.cat((torch.tensor([[tokenizer.eos_token_id]]), tokenized_input_words), dim=1)  # Add <EOS> to begin of sentence


def generate_tokens_auto_reg(cond_model, context_tokens, num_words_to_generate, tokenizer):
    key_value_embeds = None
    final_tokens = context_tokens[0].tolist()
    input_token = context_tokens
    for word_pos in range(num_words_to_generate):
        tokens_probs, key_value_embeds = cond_model(input_token, key_value_embeds)
        sampled_next_token = torch.multinomial(tokens_probs, num_samples=1)
        input_token = sampled_next_token.unsqueeze(0)
        final_tokens += input_token[0].tolist()
        if (word_pos) % 5 == 0:
            print(tokenizer.decode(final_tokens, clean_up_tokenization_spaces=True, skip_special_tokens=True))
    return final_tokens


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_words', type=str, default='The president')
    parser.add_argument('--num_words_to_generate', type=int, default=50)
    parser.add_argument('--model_size', type=str, default='gpt2-medium')
    parser.add_argument('--lm_generation_type', type=str, default='unconditioned')
    parser.add_argument('--topic', type=str, default='science')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    main(args)
