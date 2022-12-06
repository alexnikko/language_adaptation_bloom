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


def collate_fn(data, tokenizer):
    texts = [x['text'] for x in data]
    inputs = tokenizer(texts, padding=True, return_tensors='pt', max_length=64, truncation=True)
    inputs['labels'] = torch.where(inputs['input_ids'] == 50257, -100, inputs['input_ids'])
    return inputs


def get_pred_token(logits, greedy=False):
    if greedy:
        pred_token = logits.argmax()  # greedy decoding
    else:
        pred_token = np.random.choice(len(logits), p=torch.nn.functional.softmax(logits, dim=-1).numpy())  # sampling
    return pred_token


if __name__ == '__main__':
    tokenizer = build_tokenizer()
    model = build_model()

    original = False
    # original = True
    if not original:
        model.transformer.word_embeddings = torch.nn.Embedding(tokenizer.vocab_size + 1, 1024)
        model.transformer.word_embeddings_layernorm = torch.nn.LayerNorm([1024], eps=1e-5, elementwise_affine=True)
        model.lm_head = torch.nn.Linear(1024, tokenizer.vocab_size + 1, bias=False)
        snapshot = torch.load('train_results/lm_head_learnable/last_snapshot.tar', map_location='cpu')
        model.load_state_dict(snapshot['model'])
    freeze_model(model)
    model.eval()

    print(f'Bloom model with original weights = {original} has been loaded.')
    

    # text_promt = 'Меня зовут Анна. Я работаю учителем в школе. Сегодня на уроке'
    # inputs = tokenizer(text_promt, padding=True, return_tensors='pt', max_length=64, truncation=True)
    # n_tokens_to_generate = 10
    # for i in tqdm(range(n_tokens_to_generate)):
    #     outputs = model(**inputs)
    #     logits = outputs.logits
    #     logits = logits[0, -1]

    #     pred_token = get_pred_token(logits)

    #     inputs['input_ids'] = torch.cat([inputs['input_ids'], torch.LongTensor([[pred_token]])], dim=1)
    #     inputs['attention_mask'] = torch.cat([inputs['attention_mask'], torch.LongTensor([[1]])], dim=1)

    # tokens = inputs['input_ids'][0]
    # print(tokenizer.decode(tokens))

    # evaluation
    num_workers = 4
    batch_size = 16
    n_steps_per_epoch = 128
    seed, buffer_size = 111, 1024
    # dataset = load_dataset('oscar', "unshuffled_deduplicated_ru", split='train', streaming=True)
    dataset = load_dataset('mc4', 'ru', split='validation', streaming=True)
    dataset = dataset.shuffle(seed, buffer_size=buffer_size)
    dataset = dataset.with_format("torch")
    # dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
    #                         collate_fn=partial(collate_fn, tokenizer=tokenizer), pin_memory=True)
    # history = defaultdict(list)
    # losses = []
    # perplexity_values = []
    # device = 'cuda:0'
    # model.to(device)
    # for i, batch in tqdm(enumerate(dataloader, start=1), total=n_steps_per_epoch):
    #     batch = {key: value.to(device) for key, value in batch.items()}
    #     outputs = model(**batch)

    #     loss = outputs.loss

    #     losses.append(loss.item())
    #     perplexity_values.append(torch.exp(loss).item())

    #     if i == n_steps_per_epoch:
    #         break
        

    # print()
    # print(f'Val loss = {np.mean(losses)}')
    # print(f'Val perplexity = {np.mean(perplexity_values)}')
    # print()


    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    batch = next(iter(loader))
    print(batch['text'][0])
    assert False
    text_promt = 'Меня зовут Анна. Я работаю учителем в школе. Сегодня на уроке'
    inputs = tokenizer(text_promt, padding=True, return_tensors='pt', max_length=64, truncation=True)
    n_tokens_to_generate = 10
    for i in tqdm(range(n_tokens_to_generate)):
        outputs = model(**inputs)
        logits = outputs.logits
        logits = logits[0, -1]

        pred_token = get_pred_token(logits)

        inputs['input_ids'] = torch.cat([inputs['input_ids'], torch.LongTensor([[pred_token]])], dim=1)
        inputs['attention_mask'] = torch.cat([inputs['attention_mask'], torch.LongTensor([[1]])], dim=1)

    tokens = inputs['input_ids'][0]
    print(tokenizer.decode(tokens))