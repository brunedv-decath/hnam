
from copy import copy
from typing import Dict, List, Tuple, Union

from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn
from torchmetrics import Metric as LightningMetric

from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from pytorch_forecasting.metrics import MAE, MAPE, MASE, RMSE, SMAPE, MultiHorizonMetric, MultiLoss, QuantileLoss
from pytorch_forecasting.models.base_model import BaseModelWithCovariates
from pytorch_forecasting.models.nn import LSTM, MultiEmbedding
from dataclasses import dataclass
import torch.nn.functional as F
import math


class Projection(nn.Module):

    def __init__(self,in_size,out_size,bias=True):
        super().__init__()
        self.c_fc = nn.Linear(in_size, out_size, bias=bias)

    def forward(self, x):
        x = self.c_fc(x)
        return x
    
class MLP(nn.Module):
    def __init__(self,in_size,out_size,factor=2, dropout=0.3, bias=False):
        super().__init__()

        self.w1 = nn.Linear(in_size,out_size*factor, bias=bias)
        self.w2 = nn.Linear(out_size*factor,out_size, bias=bias)
        self.w3 = nn.Linear(in_size,out_size*factor,bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
    
     

class MLP(nn.Module):

    def __init__(self,in_size,out_size,factor=2,dropout=0.3,bias=True):
        super().__init__()
        self.c_fc    = nn.Linear(in_size, factor * out_size, bias=bias)
        self.c_proj  = nn.Linear(factor * out_size, out_size, bias=bias)
        self.dropout = dropout
        if self.dropout:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        if self.dropout:
            x = self.dropout(x)
        return x
    
    
class Trend(nn.Module):
    """multi head attention trend block where each head is for one quantile, maybe through a bias as well"""

    def __init__(self,cov_emb,trend_emb,factor=2,output_size=1, bias=False,dropout=0.3):
        super().__init__()

        self.trend_emb = trend_emb
        self.output_size = output_size


        self.q_proj = nn.Linear(cov_emb, trend_emb * output_size, bias=bias)
        self.kv_proj = nn.Linear(cov_emb, trend_emb * output_size * 2, bias=bias)
        # self.q_proj = MLP(cov_emb, trend_emb * output_size,factor=factor, bias=bias)
        # self.kv_proj = MLP(cov_emb, trend_emb * output_size * 2,factor=factor,bias=bias)

        self.attn_dropout = nn.Dropout(0.3)
        self.resid_dropout = nn.Dropout(0.3)

        self.bias_proj = MLP(trend_emb * output_size,output_size,factor=1,dropout=dropout,bias=bias)

    def forward(self, trend_q, trend_kv, trend_normed):

        Bq, Tq, Cq = trend_q.size()
        Bkv, Tkv, Ckv = trend_kv.size()

        q = self.q_proj(trend_q)
        kv = self.kv_proj(trend_kv)

        k,v  = kv.split(self.trend_emb*self.output_size, dim=2)

        k = k.view(Bkv, Tkv, self.output_size, self.trend_emb).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(Bq, Tq, self.output_size, self.trend_emb ).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(Bq, Tkv, self.output_size, self.trend_emb).transpose(1, 2) # (B, nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        bias = att @ v
        bias = bias.transpose(1, 2).contiguous().view(Bq, Tq,self.trend_emb*self.output_size)
        bias = self.resid_dropout(bias)
        bias = self.bias_proj(bias)
        trend = (att @ trend_normed.unsqueeze(1)).squeeze(3).transpose(1,2) + bias

        return trend, att

    
class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    


class Attention(nn.Module):

    def __init__(self,cov_emb,n_head,dropout,bias):
        super().__init__()
        # key, query, value projections for all heads, but in a batch
        # self.c_attn_q  = nn.Linear(cov_emb, cov_emb, bias=bias)
        # self.c_attn_k = nn.Linear(cov_emb, cov_emb, bias=bias)
        # self.c_attn_v = nn.Linear(cov_emb, cov_emb, bias=bias)
        self.c_attn_q  = MLP(cov_emb, cov_emb, bias=bias)
        self.c_attn_k = MLP(cov_emb, cov_emb, bias=bias)
        self.c_attn_v = MLP(cov_emb, cov_emb, bias=bias)
        # output projection
        self.c_proj = nn.Linear(cov_emb, cov_emb, bias=bias)
        # regularization

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = cov_emb
        self.dropout = dropout

        # flash attention make GPU go brrrrr but support is only in PyTorch nightly and still a bit scary
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and self.dropout == 0.0

    def forward(self,q,k,v):

        q = self.c_attn_q(q)
        k = self.c_attn_k(k)
        v = self.c_attn_v(v) 

        B, T, C = k.size()
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        B, T, C = v.size()
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        B, T, C = q.size()
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=False)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head output_size side by side
        y = self.resid_dropout(self.c_proj(y))
        return y
    

class TemporalConvolutionalLayer(nn.Module):
    def __init__(self, cov_emb, dropout_rate=0.1):
        super(TemporalConvolutionalLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels=cov_emb, out_channels=cov_emb*2,
                              kernel_size=3, padding=0)  # No padding here, handled in forward()
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.GELU()
        self.proj_back = nn.Linear(cov_emb*2,cov_emb)

    def forward(self, x):
        if x.dim() == 4:
            resize = True
            b,f,t,c = x.size()
            x = x.reshape(b*f,t,c)
        else:
            resize = False
        x = x.transpose(-1,-2)
        pad = (1,1)  # Padding only in time dimension (left, right)
        x = F.pad(x, pad, 'replicate')
        x = self.conv(x)
        x = self.dropout(self.activation(x))
        x = x.transpose(-1,-2)
        x = self.proj_back(x)
        if resize:
            x = x.reshape(b,f,t,c)
        
        return x
    
class PastAttention(nn.Module):
    def __init__(self,cov_emb:int = 32,bias:bool = False,n_head:int = 2,factor:int = 1,dropout:float = 0.3,att_proj='linear'):
        super().__init__()

        if att_proj == 'linear':
            self.c_attn_q  = nn.ModuleDict({feature:nn.Linear(cov_emb, cov_emb,bias=bias) for feature in self.causal})
            self.c_attn_k = nn.ModuleDict({feature:nn.Linear(cov_emb, cov_emb,bias=bias) for feature in self.causal})
            # self.c_attn_k = nn.Linear(cov_emb, cov_emb, bias=bias)
            self.c_attn_v = nn.ModuleDict({feature:nn.Linear(cov_emb,cov_emb,bias=bias) for feature in self.causal})

        elif att_proj == 'mlp':
            self.c_attn_q  = nn.ModuleDict({feature:MLP(cov_emb, cov_emb,factor=factor,dropout=dropout, bias=bias) for feature in self.causal})
            self.c_attn_k = nn.ModuleDict({feature: MLP(cov_emb, cov_emb, bias=bias,factor=factor,dropout=dropout) for feature in self.causal})
            self.c_attn_v = nn.ModuleDict({feature:MLP(cov_emb, cov_emb,factor=factor,dropout=dropout, bias=bias) for feature in self.causal})

        elif att_proj == 'cnn':
            self.c_attn_q  = TemporalConvolutionalLayer(cov_emb, dropout_rate=dropout)
            self.c_attn_k = TemporalConvolutionalLayer(cov_emb, dropout_rate=dropout)
            self.c_attn_v =TemporalConvolutionalLayer(cov_emb, dropout_rate=dropout)

        self.c_proj = nn.Linear(cov_emb, cov_emb, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = cov_emb
        self.dropout = dropout

        self.ln_att = LayerNorm(cov_emb, bias=bias)
        self.mlp  = MLP(cov_emb,cov_emb,factor=factor,dropout=dropout,bias=bias)    # also for each feature maybe?

    def forward(self,x):

        q = self.c_attn_k(x)
        k = self.c_attn_k(x)
        v = self.c_attn_v(x)

        B, T, C = k.size()
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        B, T, C = v.size()
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        B, T, C = q.size()
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)


        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head output_size side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))

        return y
    
class FutureAttention(nn.Module):
    def __init__(self,causal,cov_emb:int = 32,bias:bool = False,n_head:int = 2,factor:int = 1,dropout:float = 0.3,att_proj='linear'):
        super().__init__()

        self.causal = causal

        if att_proj == 'linear':
            self.c_attn_q  = nn.ModuleDict({feature:nn.Linear(cov_emb, cov_emb,bias=bias) for feature in self.causal})
            self.c_attn_k = nn.ModuleDict({feature:nn.Linear(cov_emb, cov_emb,bias=bias) for feature in self.causal})
            # self.c_attn_k = nn.Linear(cov_emb, cov_emb, bias=bias)
            self.c_attn_v = nn.ModuleDict({feature:nn.Linear(cov_emb,cov_emb,bias=bias) for feature in self.causal})

        elif att_proj == 'mlp':
            self.c_attn_q  = nn.ModuleDict({feature:MLP(cov_emb, cov_emb,factor=factor,dropout=dropout, bias=bias) for feature in self.causal})
            self.c_attn_k = nn.ModuleDict({feature: MLP(cov_emb, cov_emb, bias=bias,factor=factor,dropout=dropout) for feature in self.causal})
            self.c_attn_v = nn.ModuleDict({feature:MLP(cov_emb, cov_emb,factor=factor,dropout=dropout, bias=bias) for feature in self.causal})

        elif att_proj == 'cnn':
            self.c_attn_q  = TemporalConvolutionalLayer(cov_emb, dropout_rate=dropout)
            self.c_attn_k = TemporalConvolutionalLayer(cov_emb, dropout_rate=dropout)
            self.c_attn_v =TemporalConvolutionalLayer(cov_emb, dropout_rate=dropout)

        self.c_proj = nn.ModuleDict({feature:nn.Linear(cov_emb, cov_emb, bias=bias) for feature in self.causal})

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = cov_emb
        self.dropout = dropout

        self.ln_att = LayerNorm(cov_emb, bias=bias)
        self.mlp  = MLP(cov_emb,cov_emb,factor=factor,dropout=dropout,bias=bias)    # also for each feature maybe?

    def forward(self,causal_decoder):

        # queries = {feature: self.c_attn_q[feature](causal_decoder[feature]) for feature in self.causal}
        x = torch.stack(tuple(causal_decoder.values()),dim=1)
        queries = self.c_attn_q(x)
        # keys = {feature: self.c_attn_k[feature](past_kv) for feature in self.causal}
        # keys = torch.stack(tuple(past_kv.values()),dim=1)
        keys = self.c_attn_k(x)
        # keys = self.c_attn_k(past_kv).unsqueeze(1)
        # values = {feature: self.c_attn_v[feature](past_kv) for feature in self.causal}
        # values = torch.stack(tuple(past_kv.values()),dim=1)
        values = self.c_attn_v(x)

        B, c, T, C = keys.size()
        keys = keys.view(B, c, T, self.n_head, C // self.n_head).transpose(2, 3) # (BatchSize,Features,TimeSteps,NumberHeads,Channels)

        B, c, T, C = values.size()
        values = values.view(B, c, T, self.n_head, C // self.n_head).transpose(2, 3) 

        B, c, T, C = queries.size()
        queries = queries.view(B, c, T, self.n_head, C // self.n_head).transpose(2, 3)

        y = torch.nn.functional.scaled_dot_product_attention(queries, keys, values, attn_mask=None, dropout_p=self.dropout, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B,c, T, C) 
        for i,feature in enumerate(self.causal):
            causal_decoder[feature] = causal_decoder[feature] + self.resid_dropout(self.c_proj[feature](y[:,i])) 
            causal_decoder[feature] = causal_decoder[feature] + self.mlp(self.ln_att(causal_decoder[feature]))

        return causal_decoder

    
class MultiAttention(nn.Module):
    def __init__(self,causal,cov_emb:int = 32,bias:bool = False,n_head:int = 2,factor:int = 1,dropout:float = 0.3,att_proj='linear'):
        super().__init__()

        self.causal = causal

        if att_proj == 'linear':
            self.c_attn_q  = nn.ModuleDict({feature:nn.Linear(cov_emb, cov_emb,bias=bias) for feature in self.causal})
            self.c_attn_k = nn.ModuleDict({feature:nn.Linear(cov_emb, cov_emb,bias=bias) for feature in self.causal})
            # self.c_attn_k = nn.Linear(cov_emb, cov_emb, bias=bias)
            self.c_attn_v = nn.ModuleDict({feature:nn.Linear(cov_emb,cov_emb,bias=bias) for feature in self.causal})

        elif att_proj == 'mlp':
            self.c_attn_q  = nn.ModuleDict({feature:MLP(cov_emb, cov_emb,factor=factor,dropout=dropout, bias=bias) for feature in self.causal})
            self.c_attn_k = nn.ModuleDict({feature: MLP(cov_emb, cov_emb, bias=bias,factor=factor,dropout=dropout) for feature in self.causal})
            self.c_attn_v = nn.ModuleDict({feature:MLP(cov_emb, cov_emb,factor=factor,dropout=dropout, bias=bias) for feature in self.causal})

        elif att_proj == 'cnn':
            # self.c_attn_q  = nn.ModuleDict({feature:TemporalConvolutionalLayer(cov_emb, dropout_rate=dropout) for feature in self.causal})
            # self.c_attn_k = nn.ModuleDict({feature:TemporalConvolutionalLayer(cov_emb, dropout_rate=dropout) for feature in self.causal})
            # self.c_attn_v = nn.ModuleDict({feature:TemporalConvolutionalLayer(cov_emb, dropout_rate=dropout) for feature in self.causal})
            self.c_attn_q  = TemporalConvolutionalLayer(cov_emb, dropout_rate=dropout)
            self.c_attn_k = TemporalConvolutionalLayer(cov_emb, dropout_rate=dropout)
            self.c_attn_v =TemporalConvolutionalLayer(cov_emb, dropout_rate=dropout)

        self.c_proj = nn.ModuleDict({feature:nn.Linear(cov_emb, cov_emb, bias=bias) for feature in self.causal})

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = cov_emb
        self.dropout = dropout

        self.ln_att = LayerNorm(cov_emb, bias=bias)
        self.mlp  = MLP(cov_emb,cov_emb,factor=factor,dropout=dropout,bias=bias)    # also for each feature maybe?

    def forward(self,causal_decoder,past_kv):

        # queries = {feature: self.c_attn_q[feature](causal_decoder[feature]) for feature in self.causal}
        queries = torch.stack(tuple(causal_decoder.values()),dim=1)
        queries = self.c_attn_q(queries)
        # keys = {feature: self.c_attn_k[feature](past_kv) for feature in self.causal}
        # keys = torch.stack(tuple(past_kv.values()),dim=1)
        keys = self.c_attn_k(past_kv.unsqueeze(1))
        # keys = self.c_attn_k(past_kv).unsqueeze(1)
        # values = {feature: self.c_attn_v[feature](past_kv) for feature in self.causal}
        # values = torch.stack(tuple(past_kv.values()),dim=1)
        values = self.c_attn_v(past_kv.unsqueeze(1))

        B, c, T, C = keys.size()
        keys = keys.view(B, c, T, self.n_head, C // self.n_head).transpose(2, 3) # (BatchSize,Features,TimeSteps,NumberHeads,Channels)

        B, c, T, C = values.size()
        values = values.view(B, c, T, self.n_head, C // self.n_head).transpose(2, 3) 

        B, c, T, C = queries.size()
        queries = queries.view(B, c, T, self.n_head, C // self.n_head).transpose(2, 3)

        y = torch.nn.functional.scaled_dot_product_attention(queries, keys, values, attn_mask=None, dropout_p=self.dropout, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B,c, T, C) 
        for i,feature in enumerate(self.causal):
            causal_decoder[feature] = causal_decoder[feature] + self.resid_dropout(self.c_proj[feature](y[:,i])) 
            causal_decoder[feature] = causal_decoder[feature] + self.mlp(self.ln_att(causal_decoder[feature]))

        return causal_decoder
    
    
class Block(nn.Module):

    def __init__(self,cov_emb,n_head=2,dropout=0.3,factor=2,bias=True):
        super().__init__()
        assert cov_emb % n_head == 0, "n_embd must be divisible by n_head"
        self.ln_1 = LayerNorm(cov_emb, bias=bias)
        self.ln_2 = LayerNorm(cov_emb, bias=bias)
        self.ln_3 = LayerNorm(cov_emb, bias=bias)
        self.attn = Attention(cov_emb,n_head,dropout,bias)
        self.mlp  = MLP(cov_emb,cov_emb,factor=factor,dropout=dropout,bias=bias)


    def forward(self, q,kv):
        q = self.ln_1(q)
        kv = self.ln_2(kv)
        x = q + self.attn(q,kv,kv)
        x = x + self.mlp(self.ln_3(x))
        return x
    

class VSN(nn.Module):
    def __init__(self,cov_emb,variables):
        super().__init__()

        self.gru_layer = nn.GRU(cov_emb,hidden_size=cov_emb,batch_first=True)
        self.proj_back = nn.Linear(cov_emb,len(variables))

    def forward(self,x):

        x,h = self.gru_layer(x)
        x = self.proj_back(x)[:,-1,:].unsqueeze(1).softmax(dim=-1)
        return x

class HNAM(BaseModelWithCovariates):
    def __init__(
        self,
        output_size: Union[int, List[int]] = 7,
        loss: MultiHorizonMetric = None,
        max_encoder_length: int = 10,
        static_categoricals: List[str] = [],
        static_reals: List[str] = [],
        time_varying_categoricals_encoder: List[str] = [],
        time_varying_categoricals_decoder: List[str] = [],
        categorical_groups: Dict[str, List[str]] = {},
        time_varying_reals_encoder: List[str] = [],
        time_varying_reals_decoder: List[str] = [],
        x_reals: List[str] = [],
        x_categoricals: List[str] = [],
        hidden_continuous_size: int = 4,
        hidden_continuous_sizes: Dict[str, int] = {},
        embedding_sizes: Dict[str, Tuple[int, int]] = {},
        embedding_paddings: List[str] = [],
        embedding_labels: Dict[str, np.ndarray] = {},
        learning_rate: float = 1e-3,
        log_interval: Union[int, float] = -1,
        log_val_interval: Union[int, float] = None,
        log_gradient_flow: bool = False,
        reduce_on_plateau_patience: int = 1000,
        monotone_constaints: Dict[str, int] = {},
        share_single_variable_networks: bool = False,
        logging_metrics: nn.ModuleList = None,
        trend_emb: int = 32,
        cov_emb: int = 32,
        cov_heads: int = 2,
        dropout: float = 0.1,
        factor: int = 2,
        bias: bool = False,
        trend_query: List[str] = [],
        base = [],
        causal = [],
        attention = True,
        att_proj = 'linear',
        rescale = True,
        **kwargs,
    ):
       
        if logging_metrics is None:
            logging_metrics = nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE()])

        embedding_sizes = {k:(v[0],cov_emb) for k,v in embedding_sizes.items()}

        self.save_hyperparameters()

        # store loss function separately as it is a module
        assert isinstance(loss, LightningMetric), "Loss has to be a PyTorch Lightning `Metric`"
        super().__init__(loss=loss, logging_metrics=logging_metrics, **kwargs)


        output_size = self.hparams.output_size
        static_cats = self.hparams.static_categoricals
        cats_dec = self.hparams.time_varying_categoricals_decoder
        trend_emb = self.hparams.trend_emb
        cov_emb = self.hparams.cov_emb
        cov_heads = self.hparams.cov_heads
        dropout = self.hparams.dropout
        bias = self.hparams.bias

        # processing inputs
        # embeddings
        self.input_embeddings = MultiEmbedding(
            embedding_sizes=self.hparams.embedding_sizes,
            categorical_groups=self.hparams.categorical_groups,
            embedding_paddings=self.hparams.embedding_paddings,
            x_categoricals=self.hparams.x_categoricals,
            max_embedding_size=self.hparams.cov_emb
        )

        # continuous variable processing
        self.prescalers = nn.ModuleDict(
            {
                name: nn.Linear(1, self.hparams.cov_emb)
                for name in self.reals
            }
        )
        # self.vsn = VSN(cov_emb,variables=self.hparams.x_reals+self.hparams.x_categoricals)

        self.trend_ln = LayerNorm(cov_emb,bias=bias)
        self.causal_lns = nn.ModuleDict({feature:LayerNorm(cov_emb, bias=bias) for feature in self.hparams.causal})
        self.past_ln = LayerNorm(cov_emb,bias=bias)
        self.post_ln = LayerNorm(cov_emb,bias=bias)

        self.trend_block = Trend(cov_emb=cov_emb,
                                 trend_emb = trend_emb,
                                 output_size=output_size,
                                 dropout = dropout,
                                 factor = 1)

        if self.hparams.attention:
            self.past_attention = PastAttention(cov_emb=cov_emb,bias=bias,n_head=cov_heads,factor=1,dropout=dropout,att_proj=self.hparams.att_proj)
            self.multi_attention = MultiAttention(causal = self.hparams.causal,cov_emb=cov_emb,bias=bias,n_head=cov_heads,factor=1,dropout=dropout,att_proj=self.hparams.att_proj)
            self.future_attention = FutureAttention(causal = self.hparams.causal,cov_emb=cov_emb,bias=bias,n_head=cov_heads,factor=1,dropout=dropout,att_proj=self.hparams.att_proj)

        # self.causal_mlps_pre = nn.ModuleDict({feature:MLP(cov_emb,cov_emb,factor=factor,dropout=dropout,bias=bias) for feature in self.hparams.causal})
        # self.causal_mlps_post = nn.ModuleDict({feature:MLP(cov_emb,cov_emb,factor=factor,dropout=dropout,bias=bias) for feature in self.hparams.causal})
        self.causal_projections = nn.ModuleDict({feature:Projection(cov_emb,(self.hparams['embedding_sizes'].get(feature,(2,None))[0]-1)*output_size,bias=True) for feature in self.hparams.causal})

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        input dimensions: n_samples x time x variables
        """
        # DEFINING SHAPES
        encoder_lengths = x["encoder_lengths"]
        decoder_lengths = x["decoder_lengths"]
        target_scale = x["target_scale"]
        max_encoder_length = int(encoder_lengths.max())

        # DEFINING FEATURES
        reals =    self.hparams.x_reals
        cats = self.hparams.x_categoricals
        static_cats = self.hparams.static_categoricals
        static_reals = self.hparams.static_reals
        reals_enc = self.hparams.time_varying_reals_encoder
        cats_enc = self.hparams.time_varying_categoricals_encoder
        reals_dec =  self.hparams.time_varying_reals_decoder
        cats_dec = self.hparams.time_varying_categoricals_decoder
        target = [self.hparams.x_reals[-1]]

        ### ASSEMBLING INPUTS AS DICTIONARIES AND SCALING
        # RAW TENSORS
        x_cat = torch.cat([x["encoder_cat"], x["decoder_cat"]], dim=1)          # concatenate in time dimension
        x_cont = torch.cat([x["encoder_cont"], x["decoder_cont"]], dim=1)       # concatenate in time dimension

        # RAW DICTIONARIES
        x_cat_normal = {name: x_cat[..., i].unsqueeze(-1) for i, name in enumerate(cats)}
        x_cont_normal = {name: x_cont[..., i].unsqueeze(-1) for i, name in enumerate(reals)}
        x_normal = {**x_cat_normal,**x_cont_normal}

        # SCALED DICTIONARIES
        x_cat_encoded = self.input_embeddings(x_cat)
        x_cont_encoded = {feature: self.prescalers[feature](x_cont[..., idx].unsqueeze(-1))
                        for idx, feature in enumerate(reals)}
        x_encoded = {**x_cat_encoded,**x_cont_encoded}

        # TREND BLOCK
        trend_query = self.dtf(x_encoded,self.hparams.trend_query,stack=True).sum(axis=2)[:,max_encoder_length:,:]
        trend_query = self.trend_ln(trend_query)
        past_key_values = self.dtf(x_encoded,cats+reals,stack=True).sum(axis=2)[:,:max_encoder_length,:]
        past_key_values = self.past_ln(past_key_values)
        trend_normal = self.dtf(x_normal,target,stack=True).sum(axis=2)[:,:max_encoder_length,:]
        x_level, attention = self.trend_block(trend_query,past_key_values,trend_normal)

        # FINDING COVARIATE COEFFICIENTS
        # assemble tensors that only 'know' the cov it self and those lower in the interaction hierarchy + base information
        causal_decoder = {}
        for i,feature in enumerate(self.hparams.causal):
            causal_decoder[feature] = self.dtf(x_encoded,self.hparams.causal[:i+1]+self.hparams.base,stack=True).sum(axis=2)[:,max_encoder_length:,:]
            causal_decoder[feature] = self.causal_lns[feature](causal_decoder[feature])
            # causal_decoder[c] = self.causal_mlps_pre[c](causal_decoder[c])

        if self.hparams.attention:
            past_key_values = self.past_attention(past_key_values)
            causal_decoder = self.multi_attention(causal_decoder,past_key_values)
            causal_decoder = self.future_attention(causal_decoder)

            # causal_decoder = self.multi_attention2(causal_decoder,past_key_values)

        for feature in self.hparams.causal:
            # causal_decoder[feature] = self.post_ln(self.causal_mlps_post[feature](causal_decoder[feature]))
            causal_decoder[feature] = self.causal_projections[feature](causal_decoder[feature])  
            causal_decoder[feature] = torch.stack(causal_decoder[feature].chunk(self.hparams.output_size,dim=-1),dim=1)
    
        output = x_level.clone()

        cats_dec = [c for c in self.hparams.causal if c in cats_dec]
        cat_effects = {}
        for c in cats_dec:

            n_classes = self.hparams['embedding_sizes'][c][0]
            one_hot_cat = torch.nn.functional.one_hot(x_normal[c],n_classes)
            one_hot_cat = one_hot_cat.transpose(1,2)     # unsqueeze is already done in one hot 
            one_hot_cat = one_hot_cat[:,:,max_encoder_length:,1:]   # only decoder, sparse one hot
            cat_effect =  causal_decoder[c] * one_hot_cat
            cat_effects[c] = cat_effect.sum(dim=-1).transpose(1,2) # back to batch x time x output_size(quantiles)
            output += cat_effects[c]


        reals_dec = [c for c in self.hparams.causal if c in reals_dec]
        real_effects = {}
        for c in reals_dec:
            real_normal = x_normal[c][:,max_encoder_length:,:]
            real_effects[c] = causal_decoder[c].squeeze(-1).transpose(1,2) * real_normal
            output += real_effects[c]

        if self.hparams.rescale:
            prediction = self.transform_output(output, target_scale=target_scale)
            x_level = self.transform_output(x_level, target_scale=target_scale)
            cat_dict = {feature:o * target_scale[:,1].view(o.size(0),1,1) for feature,o in cat_effects.items()}
            if len(real_effects) > 0:
                cont_dict = {feature:o * target_scale[:,1].view(o.size(0),1,1) for feature,o in real_effects.items()}
            else:
                cont_dict = {}
        else:
            prediction = output
            cat_dict = cat_effects
            if len(real_effects) > 0:
                cont_dict = real_effects
            else:
                cont_dict = {}


        return self.to_network_output(
                    prediction= prediction,
                    x_level=x_level,
                    attention=attention,
                    **cat_dict,
                    **cont_dict)

    def expand_static_context(self, context, timesteps):
        """
        add time dimension to static context
        """
        return context[:, None].expand(-1, timesteps, -1)


    @classmethod
    def from_dataset(
        cls,
        dataset: TimeSeriesDataSet,
        **kwargs,
    ):
        
        default_kwargs = dict(  cov_emb = 32,
                trend_emb = 32,
                cov_heads = 2,
                dropout  = 0.3,
                factor = 2,
                bias = False,
                loss = QuantileLoss(),
                attention = True,
                att_proj = 'linear',
                trend_query  = ['time_idx'])
        
        kwargs = {**default_kwargs,**kwargs}
        kwargs['output_size'] = len(kwargs['loss'].quantiles)
        

        # add maximum encoder length
        # update defaults
        new_kwargs = copy(kwargs)
        new_kwargs["max_encoder_length"] = dataset.max_encoder_length
        new_kwargs.update(cls.deduce_default_output_parameters(dataset, kwargs, QuantileLoss()))
        # create class and return
        return super().from_dataset(
            dataset, **new_kwargs
        )
    

    def rescale_on(self):
        self.hparams.rescale = True

    def rescale_off(self):
        self.hparams.rescale = False

    def get_emb_size(self,features):
        cat_features = [feature for feature in features if feature in self.input_embeddings.keys()]
        real_features = [feature for feature in features if feature in self.prescalers]
        cats = sum([self.input_embeddings[feature].weight.shape[-1] for feature in cat_features])
        reals = sum([self.prescalers[feature].weight.shape[0] for feature in real_features])
        return cats+reals

    def dtf(self,dict,keys,stack=False):
        if not stack:
            return torch.cat([dict[key] for key in keys],dim=-1)
        else:
            return torch.stack([dict[key] for key in keys],dim=-2)



    