class MultiAttention(nn.Module):
    def __init__(self,causal,cov_emb:int = 32,bias:bool = False,n_head:int = 2,factor:int = 1,dropout:float = 0.3,mlp=False):
        super().__init__()

        self.causal = causal

        if mlp:
            self.c_attn_q  = nn.ModuleDict({feature:MLP(cov_emb, cov_emb,factor=factor,dropout=dropout, bias=bias) for feature in self.causal})
            self.c_attn_k = nn.ModuleDict({feature: MLP(cov_emb, cov_emb, bias=bias,factor=factor,dropout=dropout) for feature in self.causal})
            self.c_attn_v = nn.ModuleDict({feature:MLP(cov_emb, cov_emb,factor=factor,dropout=dropout, bias=bias) for feature in self.causal})
        else:
            self.c_attn_q  = nn.ModuleDict({feature:nn.Linear(cov_emb, cov_emb,bias=bias) for feature in self.causal})
            self.c_attn_k1 = nn.ModuleDict({feature:nn.Linear(cov_emb, cov_emb,bias=bias) for feature in self.causal})
            self.c_attn_k2 = nn.ModuleDict({feature:nn.Linear(cov_emb, cov_emb,bias=bias) for feature in self.causal})
            # self.c_attn_k = nn.Linear(cov_emb, cov_emb, bias=bias)
            self.c_attn_v1 = nn.ModuleDict({feature:nn.Linear(cov_emb,cov_emb,bias=bias) for feature in self.causal})
            self.c_attn_v2 = nn.ModuleDict({feature:nn.Linear(cov_emb,cov_emb,bias=bias) for feature in self.causal})


        self.c_proj = nn.ModuleDict({feature:nn.Linear(cov_emb, cov_emb, bias=bias) for feature in self.causal})

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = cov_emb
        self.dropout = dropout

        self.ln_att = LayerNorm(cov_emb, bias=bias)
        self.mlp  = MLP(cov_emb,cov_emb,factor=factor,dropout=dropout,bias=bias)

    def forward(self,causal_decoder,past_kv):

        queries = {feature: self.c_attn_q[feature](causal_decoder[feature]) for feature in self.causal}
        queries = torch.stack(tuple(queries.values()),dim=1)

        keys1 = {feature: self.c_attn_k1[feature](past_kv) for feature in self.causal}
        keys1 = torch.stack(tuple(keys1.values()),dim=1)
        keys2 = {feature: self.c_attn_k2[feature](causal_decoder[feature]) for feature in self.causal}
        keys2 = torch.stack(tuple(keys2.values()),dim=1)

        keys = torch.cat((keys1,keys2),dim=2)

        # keys = self.c_attn_k(past_kv).unsqueeze(1)
        values1 = {feature: self.c_attn_v1[feature](past_kv) for feature in self.causal}
        values1 = torch.stack(tuple(values1.values()),dim=1)
        values2 = {feature: self.c_attn_v2[feature](causal_decoder[feature]) for feature in self.causal}
        values2 = torch.stack(tuple(values2.values()),dim=1)

        values = torch.cat((values1,values2),dim=2)

        B, c, T, C = keys.size()
        keys = keys.view(B, c, T, self.n_head, C // self.n_head).transpose(2, 3) # (B,f, nh, T, hs)

        B, c, T, C = values.size()
        values = values.view(B, c, T, self.n_head, C // self.n_head).transpose(2, 3) 

        B, c, T, C = queries.size()
        queries = queries.view(B, c, T, self.n_head, C // self.n_head).transpose(2, 3)

        att = (queries @ keys.transpose(-2, -1)) * (1.0 / math.sqrt(keys.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ values 
        y = y.transpose(1, 2).contiguous().view(B,c, T, C) 
        for i,feature in enumerate(self.causal):
            causal_decoder[feature] = causal_decoder[feature] + self.resid_dropout(self.c_proj[feature](y[:,i])) 
            causal_decoder[feature] + causal_decoder[feature] + self.mlp(self.ln_att(causal_decoder[feature]))

        return causal_decoder
    