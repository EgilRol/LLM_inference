from safetensors import safe_open
import torch

# discover actual files from index for robustness
import json
with open('assets/llama3/model.safetensors.index.json', 'r') as f:
    index = json.load(f)
weight_map = index['weight_map']
embed_file = weight_map['model.embed_tokens.weight']
lm_file = weight_map['lm_head.weight']
print('embed_file:', embed_file)
print('lm_file:', lm_file)

with safe_open(f'assets/llama3/{embed_file}', framework='pt') as f:
    embed = f.get_tensor('model.embed_tokens.weight')
with safe_open(f'assets/llama3/{lm_file}', framework='pt') as f:
    lm = f.get_tensor('lm_head.weight')

print('embed shape:', tuple(embed.shape), 'dtype:', embed.dtype)
print('lm shape   :', tuple(lm.shape), 'dtype:', lm.dtype)
print('torch.equal:', torch.equal(embed, lm))
print('allclose    :', torch.allclose(embed, lm, atol=0, rtol=0))

if embed.shape == lm.shape:
    diff = (embed.to(torch.float32) - lm.to(torch.float32)).abs()
    print('max_abs_diff:', float(diff.max().item()))
    nz = int((diff != 0).sum().item())
    print('num_diff    :', nz)
    if nz:
        idx = int(diff.view(-1).argmax().item())
        row = idx // embed.shape[1]
        col = idx % embed.shape[1]
        print('worst_idx   :', (row, col))
        print('embed_val   :', float(embed[row, col].to(torch.float32).item()))
        print('lm_val      :', float(lm[row, col].to(torch.float32).item()))

