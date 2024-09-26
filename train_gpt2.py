from dataclasses import dataclass
import math
import torch
import torch.backends
import torch.backends.mps
import torch.nn as nn
from torch.nn import functional as F

#---------------------------------------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch 
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd) 
        # output projection 
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a bias more of a mask, but folloing the OpenAI/HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimentionality (n_embd)
        # calculate query, key, values for all heads in batch and move head to be the batch
        # nh is the number of heads and hs is the head size and C is the number of channels (C = nh * hs)
        # e.g. in GPT-2 (124M), n_head = 12, hs = 64, so nh*hs = 768 channels in the transformer 
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim = 2) # one big tensor to three tensor 
        
        k = k.view(B, T, self.n_head, C// self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C// self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C// self.n_head).transpose(1, 2) # (B, nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, : T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # output projection 
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear( 4 * config.n_embd ,  config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
    def forward(self, x ):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12 
    n_head: int = 12
    n_embd: int = 768




class GPT(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight shering scheme
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weight)

    def _init_weight(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std = (2 + self.config.n_layer) **-0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    
    def forward(self, idx, targets=None):
        # idx has a shape (B, T) batch size, block_size/context length
        B, T = idx.shape
        assert T <= self.config.block_size, f"Cannaot forward of length {T}, block size {self.config.block_size}  "
        # forward the token and position embeddings
        pos = torch.arange(0,T, dtype=torch.long, device=idx.device) # has shape T
        pos_emb = self.transformer.wpe(pos) # postional embeding of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb # pos_embed brodcust to (B, T, n_embd) from (T, n_embd) then sum with tok_emb
        # Final size is (B, T, n_embd)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1) )
        return logits, loss




    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
#------------------------------------------------------------------------------
import tiktoken
class DataLoaderLight:
    def __init__(self, B, T) -> None:
        self.B = B
        self.T = T
        with open("input.txt", 'r', encoding="utf-8") as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f'LOaded {len(self.tokens)} tokens')
        print(f'1 epoch = {len(self.tokens)// (B*T) } batches')

        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B*T + 1]
        x = buf[: -1].view(B, T)
        y = buf [1:].view(B,T)
        self.current_position += B * T
        if self.current_position + (B * T) + 1 > len(self.tokens):
            self.current_position = 0
        return x, y
    

# auto detect gpu

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'

print(f'Using device {device}')


torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)
#device = 'cpu'
# Read it and inspect it 

train_loader = DataLoaderLight(B=4, T=32)

model =  GPT(GPTConfig())
model.to(device)
#logits, loss = model(x, y)

optimizer = torch.optim.AdamW(model.parameters())

for i in range(50):
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    print(f"Step {i}, loss {loss.item()}")

    




print(loss)
import sys; sys.exit(0)


number_return_sequences = 5
max_length = 30

model = GPT.from_pretrained('gpt2')
#model = GPT(GPTConfig())
model.eval()
model.to(device)



# generate right now x is (B, T) where B = 5, T = 8

torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x) # B, T, vocab_size
        logits = logits[:, -1, :] # (B, vocab-size)
        # got the probabilities
        probs = F.softmax(logits, dim=-1 )
        # get prbablities
        topk_probs, topk_indicies =  torch.topk(probs, 50, dim=-1) # (5, 50)
        # Select top-k proababilits 
        ix = torch.multinomial(topk_probs, 1) #(B, 1)
        # gather the corresponding indecies
        xcol = torch.gather(topk_indicies, -1, ix) # (B, 1)
        x = torch.cat((x, xcol), dim=1)

# print inducidual raws of output 
for i in range(number_return_sequences):
    tokens = x [ i , :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)


# Continue from 1h:22 min