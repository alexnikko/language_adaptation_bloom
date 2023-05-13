import numpy as np

import torch
from torch.utils.data import DataLoader, random_split

from transformers import BloomTokenizerFast, BloomForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, \
    get_scheduler
from cosine_scheduler_with_warmup import CosineAnnealingWarmupRestarts

from datasets import load_dataset

from tqdm.auto import tqdm

from functools import partial
from collections import defaultdict

import os
from shutil import copyfile, rmtree

import argparse

from petals import DistributedBloomForCausalLM

from termcolor import colored


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


def build_model(n_tokens, pretrained_model_name_or_path='bigscience/bloom-7b1-petals', tied_weights=True):
    # model = BloomForCausalLM.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path)
    # allowed_servers = [
    #    '12D3KooWG1kUacmYwvAwMSvbunDc1G6BBdQqhueqGa3r49kKQWg9',
    #    '12D3KooWPB7ExVxsAeyfh2fCW8Z8anFZ3NnF9cMMgoF14giziobf'
    # ]
    allowed_servers = ['12D3KooWEawWeRy4sD31VnYhMKh3Gqr5Bj3LjLLZwZUaynGgipAT']
    # allowed_servers = ['12D3KooWKBFvrQ7e5MA5aWBMURe4XLQXYYcenjUHCN9Mx6brRKJw']
    pre_seq_len = 16
    tuning_mode = 'ptune'
    model = DistributedBloomForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        pre_seq_len=pre_seq_len,
        tuning_mode=tuning_mode,
        allowed_servers=allowed_servers,
    )  # , allowed_servers=allowed_servers)
    hidden_size = 4096
    if pretrained_model_name_or_path in ['bigscience/bloom-petals', 'bigscience/bloomz-petals']:
        hidden_size = 14336

    model.transformer.word_embeddings = torch.nn.Embedding(n_tokens, hidden_size)
    model.lm_head = torch.nn.Linear(hidden_size, n_tokens, bias=False)
    if tied_weights:
        model.lm_head.weight = model.transformer.word_embeddings.weight
    model.transformer.word_embeddings_layernorm = torch.nn.LayerNorm(hidden_size)
    model.transformer.ln_f = torch.nn.LayerNorm(hidden_size)

    unfreeze_model(model)
    print_desc(model)
    return model


def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_model(model):
    for p in model.parameters():
        p.requires_grad = True


def print_desc(model):
    print(model)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f'''
    Total number of parameters = {n_params / 10 ** 6:.3f} M.
    Total number of trainable parameters = {n_trainable_params / 10 ** 6:.3f} M.
    Percentage of trainable parameters = {n_trainable_params / n_params * 100:.2f}%
    '''
    )

    req_grad_dict = {name: param.requires_grad for name, param in model.named_parameters()}
    for key, value in model.state_dict().items():
        color = 'green' if req_grad_dict[key] else 'red'
        print(f'''{colored(key, color)}
                            shape: {colored(value.shape, attrs=['bold'])}
                            num_params: {colored(str(value.numel() / 10 ** 6) + ' M', attrs=['bold'])}
                            size: {colored(str(value.numel() * value.element_size() / 2 ** 20) + ' MB', attrs=['bold'])}
                            dtype: {colored(value.dtype, attrs=['bold'])}
                            requires_grad = {colored(req_grad_dict[key], attrs=['bold'])}''')


def collate_fn(data, tokenizer, max_length=256):
    texts = [x['text'] for x in data]
    inputs = tokenizer(texts, padding='max_length', return_tensors='pt', max_length=max_length, truncation=True)
    inputs['labels'] = torch.where(torch.eq(inputs['input_ids'], tokenizer.pad_token_id), -100, inputs['input_ids'])
    return inputs


def build_data(tokenizer, max_length=256, batch_size=8, num_workers=4, seed=42, buffer_size=1024):
    dataset = load_dataset('oscar', 'unshuffled_deduplicated_ru', split='train', streaming=True)
    dataset = dataset.shuffle(seed, buffer_size=buffer_size)
    dataset = dataset.with_format('torch')
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                            collate_fn=partial(collate_fn, tokenizer=tokenizer, max_length=max_length),
                            pin_memory=True)
    return dataloader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', help='resume training from last checkpoint', action='store_true')
    parser.add_argument('--not_tied', help='create lm head with its own trainable weight matrix', action='store_true')
    parser.add_argument('--big', help='use bloom-176B', action='store_true')
    parser.add_argument('--exp_name', help='name of the experiment', type=str)
    args = parser.parse_args()
    args.tied = not args.not_tied
    for k, v in vars(args).items():
        print(f'{k} = {v}')

    tokenizer = build_tokenizer()
    model_name = 'bigscience/bloom-petals'
    if not args.big:
        model_name = 'bigscience/bloom-7b1-petals'
    model = build_model(n_tokens=tokenizer.vocab_size + 1, pretrained_model_name_or_path=model_name,
                        tied_weights=args.tied)  # vocab + pad

    n_epochs = 300
    n_steps_per_epoch = 32
    batch_size = 32
    max_length = 32
    num_workers = 16
    tokenizer.model_max_length = max_length
    dataloader = build_data(tokenizer, batch_size=batch_size, max_length=max_length, num_workers=num_workers)
    print(f'''
    BATCH_SIZE = {batch_size}
    MAX_SEQ_LEN = {max_length}
    NUM_WORKERS = {num_workers}
    ''')

    device = 'cuda:0'
    learning_rate = 6e-5
    # learning_rate = 3e-4
    min_lr = 6e-6
    beta_1 = 0.9
    beta_2 = 0.95
    weight_decay = 1e-1

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), betas=(beta_1, beta_2), eps=1e-8, lr=learning_rate,
                                  weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler()

    history = defaultdict(list)

    # # try to overfit
    # batch = next(iter(dataloader))
    # batch = {key: value.to(device) for key, value in batch.items()}
    # batch['attention_mask'] = None
    # losses = []
    # perplexity_values = []
    # for _ in tqdm(range(1_000)):
    #     cur_lr = optimizer.param_groups[0]['lr']
    #     with torch.autocast(device_type='cuda', dtype=torch.float16):
    #         outputs = model(**batch)
    #         loss = outputs.loss
    #         # loss = loss_fn(output, target)
    #     # with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
    #     #     outputs = model(**batch)
    #     # loss = model(data)
    #     # outputs = model(**batch)
    #     # loss = outputs.loss
    #     # loss.backward()
    #     scaler.scale(loss).backward()
    #     scaler.unscale_(optimizer)
    #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
    #     # optimizer.step()
    #     scaler.step(optimizer)
    #     optimizer.zero_grad()
    #     scaler.update()
    #     # lr_scheduler.step()
    #
    #     losses.append(loss.item())
    #     perplexity_values.append(torch.exp(loss).item())
    #
    #     print(f'Loss = {losses[-1]}')
    #     print(f'Perplexity = {perplexity_values[-1]}')
    #     print(f'Lr = {cur_lr}')
    # assert False

    start_epoch = 0
    exp_name = args.exp_name
    savedir = os.path.join('train_results', exp_name)
    last_snapshot_name = os.path.join(savedir, 'last_snapshot.tar')
    if not args.resume:
        if os.path.exists(savedir):
            ans = input(f'Savedir = {savedir} exists. Do you want to rewrite it? Y/n: ').lower()
            if ans == 'y':
                rmtree(savedir)
            else:
                exit(1)
        os.makedirs(savedir)
    else:
        assert os.path.exists(last_snapshot_name)

    if args.resume:
        snapshot = torch.load(last_snapshot_name, map_location='cpu')
        model.transformer.word_embeddings.load_state_dict(snapshot['model'])
        if args.not_tied:
            model.lm_head.load_state_dict(snapshot['lm_head'])
        # optimizer.load_state_dict(snapshot['optimizer'])
        history = snapshot['history']
        start_epoch = len(history['train_loss'])
        copyfile(last_snapshot_name, os.path.join(savedir, f'snapshot_epoch_{str(start_epoch).zfill(4)}.tar'))

    lr_scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                                 first_cycle_steps=n_steps_per_epoch * (n_epochs - start_epoch),
                                                 max_lr=learning_rate, min_lr=min_lr,
                                                 warmup_steps=n_steps_per_epoch * 2)
    for epoch in tqdm(range(start_epoch, n_epochs)):
        dataloader.dataset.set_epoch(epoch)
        losses = []
        perplexity_values = []
        lrs = []
        model.train()
        for i, batch in tqdm(enumerate(dataloader, start=1), total=n_steps_per_epoch):
            optimizer.zero_grad()
            lr_scheduler.step()
            cur_lr = optimizer.param_groups[0]['lr']

            batch['attention_mask'] = None
            batch = {key: value.to(device) for key, value in batch.items() if key != 'attention_mask'}
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(**batch)
                loss = outputs.loss
            # loss.backward()
            # optimizer.step()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            scaler.step(optimizer)
            scaler.update()

            losses.append(loss.item())
            perplexity_values.append(torch.exp(loss).item())
            lrs.append(cur_lr)

            if i == n_steps_per_epoch:
                break

        print()
        print(f'Train loss = {np.mean(losses)}')
        print(f'Train perplexity = {np.mean(perplexity_values)}')
        print(f'Current learning rate = {lrs[-1]}')
        print()

        history['train_loss'].append(np.mean(losses))
        history['train_perplexity'].append(np.mean(perplexity_values))
        history['lr'].extend(lrs)

        snapshot = {
            # 'model': model.transformer.word_embeddings.state_dict(),
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'history': history
        }
        # if args.not_tied:
        #     snapshot['lm_head'] = model.lm_head.state_dict()
        path = last_snapshot_name
        torch.save(snapshot, path)


if __name__ == '__main__':
    main()
