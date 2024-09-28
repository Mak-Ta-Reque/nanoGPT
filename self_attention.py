import torch.nn  as nn
import torch.nn.functional as F
import torch
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
    
class Head(nn.Module):
    """ ONe head size of self attention """
    def __init__(self, head_size, n_embd, block_size) -> None:
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) #(B, T, 16)
        q = self.query(x) #(B, T, 16) 
        v = self.value(x)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return y
    
class HeadC(nn.Module):
    """ ONe head size of self attention """
    def __init__(self, head_size, n_embd, block_size) -> None:
        super().__init__()

        self.key = GroupedConvLastDim(in_channels=n_embd, out_channels=head_size, groups=1)
        self.query = GroupedConvLastDim(in_channels=n_embd, out_channels=head_size, groups=1)
        self.value = GroupedConvLastDim(in_channels=n_embd, out_channels=head_size, groups=1)
        self.register_buffer('bias', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) #(B, T, 16)
        q = self.query(x) #(B, T, 16) 

        #wei = q @ k.transpose(-2, -1) * C**-0.5# (B, T , 16 ) @ (B, 16, T) -->  (B, T, T)
        #wei = wei.masked_fill(self.bias[:T, :T]==0, float('-inf'))
        #wei = F.softmax(wei, dim=-1)
        #wei = self.dropout(wei)
        v = self.value(x)
        #out = wei @ v
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return y
    
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parellel """
    def __init__(self, config) -> None:
        super().__init__()
        n_head = config.n_head
        n_embd = config.n_embd
        head_size = n_embd // n_head
        block_size = config.block_size
        self.heads = nn.ModuleList([HeadC(head_size, n_embd, block_size ) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):

        out =  torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(out)
        out = self.proj(out)
        return out