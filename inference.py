import numpy as np

import torch
from torch.utils.data import DataLoader, random_split

from transformers import BloomTokenizerFast, BloomForCausalLM, AutoTokenizer

from datasets import load_dataset

from tqdm.auto import tqdm

from functools import partial
from collections import defaultdict

from shutil import copyfile


def build_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/rugpt3large_based_on_gpt2", padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_model():
    return BloomForCausalLM.from_pretrained("bigscience/bloom-560m")


def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False


if __name__ == '__main__':
    tokenizer = build_tokenizer()
    model = build_model()
    # print(model.transformer.word_embeddings)
    model.transformer.word_embeddings = torch.nn.Embedding(tokenizer.vocab_size + 1, 1024)
    model.transformer.word_embeddings_layernorm = torch.nn.LayerNorm([1024], eps=1e-5, elementwise_affine=True)
    snapshot = torch.load('last_snapshot.tar', map_location='cpu')
    model.load_state_dict(snapshot['model'])
    freeze_model(model)
    model.eval()
    
    text_promt = 'Меня зовут Анна. Я работаю учителем в школе. Сегодня на уроке'
    inputs = tokenizer(text_promt, padding=True, return_tensors='pt', max_length=64, truncation=True)
    tokens = inputs['input_ids'][0]
    for token in tokens:
        token = token.item()
        print(token, tokenizer.decode(token))
    outputs = model(**inputs)
    logits = outputs.logits
    print(tokenizer.vocab_size)
    print(logits.shape)
    print(logits.argmax())
    logits = logits[0, -1, :tokenizer.vocab_size + 1]
    print(logits.shape)
    pred_token = logits.argmax()
    print(pred_token)
    print(tokenizer.decode(pred_token.item()))
    print(tokenizer.decode(9140))


    print(inputs.keys())
    print(inputs['input_ids'].shape)
    print(inputs['input_ids'])

    # print(pred_token)
    # print(type(pred_token))
    # print(pred_token.unsqueeze(0).unsqueeze(0))
    pred_token = pred_token.unsqueeze(0).unsqueeze(0)

    print(torch.cat([inputs['input_ids'], pred_token], dim=1))
    print(tokenizer.decode(tokens))
    # print(torch.cat([input['input_ids'], torch.LongTensor(pred_token.item())], dim=1))
    # print(torch.cat([inputs['input_ids'], torch.LongTensor([[pred_token.item]]), dim=1))
    # print(type(outputs))
    # print(text_tokens)
    print(inputs['attention_mask'])
    print(inputs['attention_mask'].shape)
    print(torch.LongTensor([[1]]))
    import numpy as np
    for i in range(10):
        outputs = model(**inputs)
        logits = outputs.logits
        logits = logits[0, -1, :tokenizer.vocab_size + 1]
        # pred_token = logits.argmax()
        print(torch.nn.functional.softmax(logits).sum())
        pred_token = np.random.choice(len(logits), p=torch.nn.functional.softmax(logits).numpy())
        # pred_token = pred_token.unsqueeze(0).unsqueeze(0)


        inputs['input_ids'] = torch.cat([inputs['input_ids'], torch.LongTensor([[pred_token]])], dim=1)
        inputs['attention_mask'] = torch.cat([inputs['attention_mask'], torch.LongTensor([[1]])], dim=1)

    tokens = tokens = inputs['input_ids'][0]
    print(tokenizer.decode(tokens))
