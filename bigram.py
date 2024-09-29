import torch 
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequence will we process in parallel
block_size = 8 # what is the maximum context length for prediction?
max_iters = 3000
eval_interval = 300 
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

#---------------------------------#

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


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X,Y)
            losses [k] = loss.item()
        out[split] = losses.mean()
        model.train()
    return out
    
class Bigramlanguagemodel(nn.Module):
    def __init__(self):
        super().__init__()
        #each token directly reads off the logits for the next token from a lookup table
        self.token_embdding_table = nn.Embedding(vocab_size, vocab_size)
    def forward(self, idx, targets=None):
        # idx and target are both (B,T) tensor of integer
        logits = self.token_embdding_table(idx) # (B, T, C)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view (B*T, C)
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the curren context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            logits = logits[:, -1, :] # becomes (B,C)
            # apply softmax to get probabilitis
            probs = F.softmax(logits, dim=-1) # (B,C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim = 1) # ((B, T+1)
        return idx
        
model = Bigramlanguagemodel()
m = model.to(device)

# Create an optimizer
optimizer = torch.optim.Adam(m.parameters(), lr=1e-3)


batch_size = 32
for iter in range (max_iters):

    # Every once in a while evlauate the loss on the train and val set
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: train loss {losses['train']:.4f} val loss {losses['val']:.4f}")
    xb, yb = get_batch("train")
    #evaluate loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
print(decode(m.generate(idx = torch.zeros((1,1), dtype=torch.long), max_new_tokens=500)[0].tolist()))
    