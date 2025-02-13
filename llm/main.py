import torch

with open('wizard_of_oz.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(set(text))

string_to_int = {c: i for i, c in enumerate(chars)}
int_to_string = {i: c for i, c in enumerate(chars)}

encode = lambda s:  [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

