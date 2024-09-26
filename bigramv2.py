import torch 
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # how many independent sequence will we process in parallel
block_size = 50 # what is the maximum context length for prediction?
max_iters = 5000
eval_interval = 500 
learning_rate = 4e-3
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_layers = 20
n_head = 60
n_embed = 600
dropout = 0.2

#--------------------

torch.manual_seed(1337)

# Read it and inspect it 
with open("input.txt", 'r', encoding="utf-8") as f:
    text = f.read()

#Here are all unique charecters that occur in the text 
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create a mapping from charecter to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # encoder takes string of char and return a list of int
decode = lambda l: ''.join([itos[i] for i in l]) # decoder takes a list of int and return a string of chars

data = torch.tensor(encode(text), dtype=torch.long)
# Let's now split the dataset into train and validation sets
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    #Generate a small batch of data of inputs x and target y
    data = train_data if  split =='train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i: i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size+1] for i in ix])
    return x, y

def count_parameters(model: nn.Module) -> float:
    total_params = sum(p.numel() for p in model.parameters())
    return total_params / 1_000_000  # Convert to millions


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            X, Y = X.to(device), Y.to(device)
            logits, loss = model(X,Y)
            losses [k] = loss.item()
        out[split] = losses.mean()
        model.train()
    return out


class GroupedConvLastDim(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super(GroupedConvLastDim, self).__init__()
        # Define a 1D convolution layer with grouped convolutions
        self.conv1d = nn.Conv1d(in_channels=in_channels, 
                                out_channels=out_channels, 
                                kernel_size=1, 
                                groups=groups, bias=False)  # Set groups to reduce the number of parameters

    def forward(self, x):
        # x is of shape (B, T, C)
        
        # Permute the input tensor to (B, C, T) for Conv1d
        x = x.permute(0, 2, 1)  # (B, C, T)
        
        # Apply the grouped convolution along the last dimension
        x = self.conv1d(x)  # (B, head_size, T)
        
        # Permute back to (B, T, head_size)
        x = x.permute(0, 2, 1)  # (B, T, head_size)
        
        return x
    
class Feedforward(nn.Module):
    """ A simple linear layer followed by a non-linearity """
    def __init__(self, n_embed) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    
class LayerNorm1d(nn.Module): # (used to be BatchNorm1d)

  def __init__(self, dim, eps=1e-5, momentum=0.1):
    super().__init__()
    self.eps = eps
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)

  def __call__(self, x):
    # calculate the forward pass
    xmean = x.mean(1, keepdim=True) # batch mean
    xvar = x.var(1, keepdim=True) # batch variance
    xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
    self.out = self.gamma * xhat + self.beta
    return self.out

  def parameters(self):
    return [self.gamma, self.beta]
  


class Head(nn.Module):
    """ ONe head size of self attention """
    def __init__(self, head_size) -> None:
        super().__init__()

        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) #(B, T, 16)
        q = self.query(x) #(B, T, 16) 

        wei = q @ k.transpose(-2, -1) * C**-0.5# (B, T , 16 ) @ (B, 16, T) -->  (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T]==0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out
    
class HeadC(nn.Module):
    """ ONe head size of self attention """
    def __init__(self, head_size) -> None:
        super().__init__()

        self.key = GroupedConvLastDim(in_channels=n_embed, out_channels=head_size, groups=head_size)
        self.query = GroupedConvLastDim(in_channels=n_embed, out_channels=head_size, groups=head_size)
        self.value = GroupedConvLastDim(in_channels=n_embed, out_channels=head_size, groups=head_size)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) #(B, T, 16)
        q = self.query(x) #(B, T, 16) 

        wei = q @ k.transpose(-2, -1) * C**-0.5# (B, T , 16 ) @ (B, 16, T) -->  (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T]==0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out
    

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parellel """
    def __init__(self, num_heads, head_size) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):

        out =  torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(out)
        out = self.proj(out)
        return out

class Block(nn.Module):
    def __init__(self, n_embed, n_head) -> None:
        super().__init__()
        head_size  = n_embed// n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = Feedforward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 =  nn.LayerNorm(n_embed)


    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class Bigramlanguagemodel(nn.Module):
    def __init__(self):
        super().__init__()
        #each token directly reads off the logits for the next token from a lookup table
        self.token_embdding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embeding_table = nn.Embedding(block_size, n_embed) #postional embedding
        self.blocks = nn.Sequential(
            *[Block(n_embed, n_head=n_head) for _ in range(n_layers)]            
        )
        self.ln_f = nn.LayerNorm(n_embed)

        self.lm_head = nn.Linear(n_embed, vocab_size)
        
    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and target are both (B,T) tensor of integer
       
        token_emb = self.token_embdding_table(idx) # (B, T, C)
        pos_emb = self.position_embeding_table(torch.arange(T, device=device)) # (T, C)
        x = token_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # apply the head
        x = self.ln_f(x) 

        logits = self.lm_head(x) #(B,T,vocab_size)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view (B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the curren context
        
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx [:, -block_size:] 
            # get the predictions
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] # becomes (B,C)
            # apply softmax to get probabilitis
            probs = F.softmax(logits, dim=-1) # (B,C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim = 1) # ((B, T+1)
        return idx
        
model = Bigramlanguagemodel()
m = model.to(device)
total_params = count_parameters(model)
print(f"Total parameters: {total_params}")
# Create an optimizer


optimizer = torch.optim.Adam(m.parameters(), lr=1e-3)


batch_size = 32
for iter in range (max_iters):

    # Every once in a while evlauate the loss on the train and val set
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: train loss {losses['train']:.4f} val loss {losses['val']:.4f}")
    xb, yb = get_batch("train")
    xb, yb = xb.to(device), yb.to(device)
    #evaluate loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
print(decode(m.generate(idx = torch.ones((1,1), dtype=torch.long, device=device), max_new_tokens=200)[0].tolist()))

total_params = count_parameters(model)
print(f"Total parameters: {total_params}")