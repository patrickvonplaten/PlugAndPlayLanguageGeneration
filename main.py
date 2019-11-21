#!/usr/bin/env python3
import ipdb  # noqa: F401
import random  # noqa: F401
import numpy as np  # noqa: F401
from argparse import ArgumentParser  # noqa: F401
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from cond_gpt2 import CondGPT2LMHeadModel
from hf_lg_utils import top_k_top_p_filtering


def main(args):
    input_words = args.input_words
    num_words_to_generate = args.num_words_to_generate
    model_size = args.model_size
    seed = args.seed
    topic = args.topic

    torch.manual_seed(seed)

    tokenizer = GPT2Tokenizer.from_pretrained(model_size)
    cond_model = CondGPT2LMHeadModel.from_conditioned_on_topic(model_size, topic, tokenizer)

    tokenized_input_words = create_tokenized_input_words(tokenizer, input_words)
    generated_tokens = generate_tokens_auto_reg(cond_model, tokenized_input_words, num_words_to_generate)
    generated_words = tokenizer.decode(generated_tokens, clean_up_tokenization_spaces=True, skip_special_tokens=True)
    print(generated_words)


def create_tokenized_input_words(tokenizer, input_words):
    tokenized_input_words = torch.tensor(tokenizer.encode(input_words), dtype=torch.long).unsqueeze(0)  # Batch size 1
    return torch.cat((torch.tensor([[tokenizer.eos_token_id]]), tokenized_input_words), dim=1)  # Add <EOS> to begin of sentence


def generate_tokens_auto_reg(cond_model, input_tokens, num_words_to_generate, temperature=0.7, top_k=0, top_p=0.9):
    key_value_embeds = None
    final_tokens = input_tokens[0].tolist()
    for word_pos in range(num_words_to_generate):
        tokens_logits, key_value_embeds = cond_model(input_tokens, key_value_embeds)
        heated_next_token_logits = tokens_logits[0, -1, :] / temperature
        filtered_next_token_logits = top_k_top_p_filtering(heated_next_token_logits, top_k=top_k, top_p=top_p)
        sampled_next_token = torch.multinomial(F.softmax(filtered_next_token_logits, dim=-1), num_samples=1)
        input_tokens = sampled_next_token.unsqueeze(0)
        final_tokens += input_tokens[0].tolist()
    return final_tokens


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_words', type=str, default='Fabio lives in Cologne and')
    parser.add_argument('--num_words_to_generate', type=int, default=100)
    parser.add_argument('--model_size', type=str, default='gpt2-medium')
    parser.add_argument('--lm_generation_type', type=str, default='unconditioned')
    parser.add_argument('--topic', type=str, default='science')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    main(args)
