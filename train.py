import numpy as np

import torch
from torch.utils.data import DataLoader, random_split

from transformers import BloomTokenizerFast, BloomForCausalLM, AutoTokenizer

from datasets import load_dataset

from tqdm.auto import tqdm

from functools import partial
from collections import defaultdict

import os
from shutil import copyfile, rmtree


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


def main():
    tokenizer = build_tokenizer()
    model = build_model()

    num_workers = 4
    batch_size = 48
    n_steps_per_epoch = 1024 # 10_000
    seed, buffer_size = 42, 1024
    dataset = load_dataset('oscar', "unshuffled_deduplicated_ru", split='train', streaming=True)
    dataset = dataset.shuffle(seed, buffer_size=buffer_size)
    dataset = dataset.with_format("torch")
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                            collate_fn=partial(collate_fn, tokenizer=tokenizer), pin_memory=True)

    freeze_model(model)
    model.transformer.word_embeddings = torch.nn.Embedding(tokenizer.vocab_size + 1, 1024)
    model.transformer.word_embeddings_layernorm = torch.nn.LayerNorm([1024], eps=1e-5, elementwise_affine=True)
    model.lm_head = torch.nn.Linear(1024, tokenizer.vocab_size + 1, bias=False)

    device = 'cuda:0'

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    n_epochs = 300
    history = defaultdict(list)

    # # try to overfit
    # batch = next(iter(dataloader))
    # batch = {key: value.to(device) for key, value in batch.items()}
    # losses = []
    # perplexity_values = []
    # for _ in tqdm(range(100)):
    #     outputs = model(**batch)
    #     loss = outputs.loss
    #     loss.backward()
    #     optimizer.step()
    #     optimizer.zero_grad()

    #     losses.append(loss.item())
    #     perplexity_values.append(torch.exp(loss).item())

    #     print(f'Loss = {losses[-1]}')
    #     print(f'Perplexity = {perplexity_values[-1]}')
    # assert False

    start_epoch = 0
    # resume = True
    resume = False
    exp_name = 'lm_head_learnable'
    savedir = os.path.join('train_results', exp_name)
    if not resume and os.path.exists(savedir):
        ans = input(f'Savedir = {savedir} exists. Do you want to rewrite it? Y/n: ').lower()
        if ans == 'y':
            rmtree(savedir)
    os.makedirs(savedir)
    last_snapshot_name = os.path.join(savedir, 'last_snapshot.tar')
    if resume:
        assert os.path.exists(last_snapshot_name)

    if resume:
        snapshot = torch.load(last_snapshot_name, map_location='cpu')
        model.load_state_dict(snapshot['model'])
        optimizer.load_state_dict(snapshot['optimizer'])
        history = snapshot['history']
        start_epoch = len(history['train_loss'])
        copyfile(last_snapshot_name, os.path.join(savedir, f'snapshot_epoch_{str(start_epoch).zfill(4)}.tar'))
    for epoch in tqdm(range(start_epoch, n_epochs)):
        dataset.set_epoch(epoch)
        losses = []
        perplexity_values = []
        model.train()
        for i, batch in tqdm(enumerate(dataloader, start=1), total=n_steps_per_epoch):
            optimizer.zero_grad()

            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            perplexity_values.append(torch.exp(loss).item())

            if i == n_steps_per_epoch:
                break

        print()
        print(f'Train loss = {np.mean(losses)}')
        print(f'Train perplexity = {np.mean(perplexity_values)}')
        print()

        history['train_loss'].append(np.mean(losses))
        history['train_perplexity'].append(np.mean(perplexity_values))

        snapshot = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'history': history
        }

        # path = 'last_snapshot.tar'
        path = last_snapshot_name
        torch.save(snapshot, path)

        # losses = []
        # perplexity_values = []
        # model.eval()
        # with torch.inference_mode():
        #     for batch in tqdm(val_loader):
        #         batch = {key: value.to(device) for key, value in batch.items()}
        #         outputs = model(**batch)
        #
        #         loss = outputs.loss
        #
        #         losses.append(loss.item())
        #         perplexity_values.append(torch.exp(loss).item())
        #
        # print()
        # print(f'Val loss = {np.mean(losses)}')
        # print(f'Val perplexity = {np.mean(perplexity_values)}')
        # print()


if __name__ == '__main__':
    main()
