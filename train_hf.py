import numpy as np

import torch
from torch.utils.data import DataLoader, random_split

from transformers import BloomTokenizerFast, BloomForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling

from datasets import load_dataset

from tqdm.auto import tqdm

from functools import partial
from collections import defaultdict

import os
from shutil import copyfile, rmtree


def build_tokenizer(padding_side='right', truncation_side='right'):
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="sberbank-ai/rugpt3large_based_on_gpt2",
        padding_side=padding_side,
        truncation_side=truncation_side
    )
    tokenizer.pad_token = tokenizer.eos_token
    print('EOS TOKEN ID', tokenizer.eos_token_id)
    print(f'Tokenizer vocab size: {tokenizer.vocab_size}')
    # tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m", padding_side='right')
    # print('BOS token: ', tokenizer.bos_token, tokenizer.bos_token_id)
    # print('EOS token: ', tokenizer.eos_token, tokenizer.eos_token_id)
    # print('PAD token: ', tokenizer.pad_token, tokenizer.pad_token_id)
    # print('UNK token: ', tokenizer.unk_token, tokenizer.unk_token_id)
    # print('SEP token: ', tokenizer.sep_token, tokenizer.sep_token_id)
    return tokenizer


def build_model(n_tokens, pretrained_model_name_or_path='bigscience/bloom-560m'):
    model = BloomForCausalLM.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path)
    print_desc(model)
    freeze_model(model)
    model.transformer.word_embeddings = torch.nn.Embedding(n_tokens, 1024)
    model.lm_head = torch.nn.Linear(1024, n_tokens, bias=False)
    model.lm_head.weight = model.transformer.word_embeddings.weight
    print_desc(model)
    return model


def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False


def print_desc(model):
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f'''
    Total number of parameters = {n_params / 10 ** 6:.3f} M.
    Total number of trainable parameters = {n_trainable_params / 10 ** 6:.3f} M.
    Percentage of trainable parameters = {n_trainable_params / n_params * 100:.2f}%
    '''
    )


def collate_fn(data, tokenizer):
    texts = [x['text'] for x in data]
    inputs = tokenizer(texts, padding='max_length', return_tensors='pt', max_length=256, truncation=True)
    inputs['labels'] = torch.where(torch.eq(inputs['input_ids'], tokenizer.pad_token_id), -100, inputs['input_ids'])
    return inputs


def build_data(tokenizer, batch_size=8, num_workers=4, seed=42, buffer_size=1024):
    dataset = load_dataset('oscar', "unshuffled_deduplicated_ru", split='train', streaming=True)
    dataset = dataset.shuffle(seed, buffer_size=buffer_size)
    dataset = dataset.with_format("torch")
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                            collate_fn=partial(collate_fn, tokenizer=tokenizer), pin_memory=True)
    return dataloader


def main():
    tokenizer = build_tokenizer()
    model = build_model(n_tokens=tokenizer.vocab_size + 1)  # vocab + pad

    # text = 'Привет, как дела?'

    # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    # inputs = tokenizer(text, return_tensors="pt", max_length=16, padding='max_length')
    # inputs = tokenizer(text, return_tensors="pt")
    # inputs = data_collator.torch_call(["Hello, my dog is cute"])
    # print(inputs)
    # assert False
    # for x in dir(tokenizer):
    #     if x.startswith('__'):
    #         continue
    #     print(x)
    # assert False
    # print(inputs)
    # outputs = model(**inputs, labels=inputs["input_ids"])
    # outputs = model(**inputs, labels=torch.where(torch.eq(inputs['input_ids'], tokenizer.pad_token_id), -100, inputs['input_ids']))
    # print([tokenizer.decode(input_id) for input_id in inputs['input_ids'][0]])
    # print(''.join([tokenizer.decode(input_id) for input_id in inputs['input_ids'][0]]))
    # print(outputs.loss)

    # print(outputs.logits.shape)
    # print(outputs.logits[0, -8:, :5])

    # print(model.__dict__)

    n_steps_per_epoch = 1024
    dataloader = build_data(tokenizer)

    device = 'cuda:0'

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    n_epochs = 300
    history = defaultdict(list)

    # try to overfit
    # batch = next(iter(dataloader))
    # batch = {key: value.to(device) for key, value in batch.items()}
    # losses = []
    # perplexity_values = []
    # for _ in tqdm(range(10_000)):
    #     outputs = model(**batch)
    #     loss = outputs.loss
    #     loss.backward()
    #     optimizer.step()
    #     optimizer.zero_grad()
    #
    #     losses.append(loss.item())
    #     perplexity_values.append(torch.exp(loss).item())
    #
    #     print(f'Loss = {losses[-1]}')
    #     print(f'Perplexity = {perplexity_values[-1]}')
    # assert False

    start_epoch = 0
    # resume = True
    resume = False
    exp_name = 'test_exp_0'
    savedir = os.path.join('train_results', exp_name)
    last_snapshot_name = os.path.join(savedir, 'last_snapshot.tar')
    if not resume and os.path.exists(savedir):
        ans = input(f'Savedir = {savedir} exists. Do you want to rewrite it? Y/n: ').lower()
        if ans == 'y':
            rmtree(savedir)
    os.makedirs(savedir)
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
        dataloader.dataset.set_epoch(epoch)
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
