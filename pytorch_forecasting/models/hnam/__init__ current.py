
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

    def __init__(self,cov_emb,trend_emb,output_size=1, bias=False):
        super().__init__()

        self.ln_q = LayerNorm(cov_emb, bias=bias)
        self.ln_k = LayerNorm(cov_emb, bias=bias)
        self.query_proj = nn.Linear(cov_emb, trend_emb, bias=False)
        self.key_proj = nn.Linear(cov_emb, trend_emb, bias=False)

        self.linear1 = nn.Linear(1,16)
        self.linear2 = nn.Linear(16,output_size)

    def forward(self, trend_query, trend_key, trend_values):
        trend_query = self.ln_q(trend_query)
        trend_key = self.ln_k(trend_key)

        trend_query = self.query_proj(trend_query)
        trend_key = self.key_proj(trend_key)

        attention_scores = torch.matmul(trend_query, trend_key.transpose(-2, -1))
        attention_scores = attention_scores / (trend_key.size(-1) ** 0.5)
        attention = F.softmax(attention_scores, dim=-1)

        attention = self.linear2(F.gelu(self.linear1(attention.unsqueeze(-1)))).permute(0,3,1,2)
        x_level = torch.matmul(attention, trend_values.unsqueeze(1)).squeeze(-1).transpose(1,2)
        
        return x_level, attention
    
    
    
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
        self.c_attn_q  = nn.Linear(cov_emb, cov_emb, bias=bias)
        self.c_attn_k = nn.Linear(cov_emb, cov_emb, bias=bias)
        self.c_attn_v = nn.Linear(cov_emb, cov_emb, bias=bias)
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

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        dec_len = q.size(1)
        enc_len = k.size(1)

        q = self.c_attn_q(q)
        k = self.c_attn_k(k)
        v = self.c_attn_v(v)  # is this even necessary

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

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
    
class ApplyAttention(nn.Module):

    def __init__(self, cov_emb,factor,dropout,bias):
        super().__init__()
        self.ln = LayerNorm(cov_emb, bias=bias)
        self.proj_v = nn.Linear(cov_emb, cov_emb, bias=bias)
        self.proj_back = MLP(cov_emb, cov_emb, factor=factor, dropout=dropout, bias=bias)

    def forward(self, v, att):
        _, _, C = v.size()
        B, h, T, T_all = att.size()
        v = self.ln(v)
        v = self.proj_v(v)
        v = v = v.view(B, T_all, h, C // h).transpose(1, 2)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj_back(y)
        return y
    

class AttentionValues(nn.Module):

    def __init__(self,cov_emb,n_head,bias):
        super().__init__()

        self.ln_1 = LayerNorm(cov_emb, bias=bias)
        self.ln_2 = LayerNorm(cov_emb, bias=bias)
        self.c_attn_q  = nn.Linear(cov_emb, 1 * cov_emb, bias=bias)
        self.c_attn_k = nn.Linear(cov_emb, 1 * cov_emb, bias=bias)

        # self.c_attn_q  = MLP(cov_emb, 1 * cov_emb, bias=bias)
        # self.c_attn_k = MLP(cov_emb, 1 * cov_emb, bias=bias)

        self.n_head = n_head
        self.n_embd = cov_emb

    def forward(self,q,k):

        q = self.ln_1(q)    
        k = self.ln_2(k)

        q = self.c_attn_q(q)
        k = self.c_attn_k(k)

        B, T, C = k.size()
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        B, T, C = q.size()
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)

        return att
    
class Block(nn.Module):

    def __init__(self,cov_emb,n_head=2,dropout=0.3,factor=2,bias=True):
        super().__init__()
        assert cov_emb % n_head == 0, "n_embd must be divisible by n_head"
        self.ln_1 = LayerNorm(cov_emb, bias=bias)
        self.ln_2 = LayerNorm(cov_emb, bias=bias)
        self.ln_3 = LayerNorm(cov_emb, bias=bias)
        self.ln_4 = LayerNorm(cov_emb, bias=bias)
        self.attn = Attention(cov_emb,n_head,dropout,bias)
        self.mlp  = MLP(cov_emb,cov_emb,factor=factor,dropout=dropout,bias=bias)


    def forward(self, q,kv):
        q = self.ln_1(q)
        kv = self.ln_2(kv)
        x = self.attn(q,kv,kv) + q
        x = x + self.mlp(self.ln_3(x))
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
        trend_query = self.hparams.trend_query


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

        self.trend_query = static_cats + static_reals + trend_query

        self.trend_block = Trend(cov_emb=cov_emb,
                                 trend_emb = trend_emb,
                                 output_size=output_size)

        if self.hparams.attention:
            self.attention_values = AttentionValues(cov_emb=cov_emb,n_head=cov_heads,bias=bias)
            self.attention_apply = ApplyAttention(cov_emb=cov_emb,bias=bias,factor=factor,dropout=dropout) 


        self.causal_mlps = nn.ModuleDict({feature:MLP(cov_emb,cov_emb,factor=factor,dropout=dropout,bias=bias) for feature in self.hparams.causal})
        self.causal_projections = nn.ModuleDict({feature:Projection(cov_emb,(self.hparams['embedding_sizes'].get(feature,(2,None))[0]-1)*output_size,bias=bias) for feature in self.hparams.causal})


        self.n_classes = [self.hparams.embedding_sizes[feature][0] for feature in cats_dec if feature in self.hparams.causal]
        self.n_classes_sum = sum(self.n_classes)
        self.n_classes_sum_sparse = self.n_classes_sum - len(self.n_classes) #+ 1

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
        trend_query = self.dtf(x_encoded,self.trend_query,stack=True).sum(axis=2)[:,max_encoder_length:,:]
        trend_key = self.dtf(x_encoded,cats+reals,stack=True).sum(axis=2)[:,:max_encoder_length,:]
        trend_values = self.dtf(x_normal,target,stack=True).sum(axis=2)[:,:max_encoder_length,:]
        x_level,attention = self.trend_block(trend_query,trend_key,trend_values)


        # BASE INFORMATION
        base_summed = self.dtf(x_encoded,self.hparams.base,stack=True).sum(axis=2)


        # FINDING COVARIATE COEFFICIENTS
        # assemble tensors that only 'know' the cov it self and those lower in the interaction hierarchy + base information
        causal_info = {}
        for i in range(len(self.hparams.causal)):
            causal_info[self.hparams.causal[i]] = self.dtf(x_encoded,self.hparams.causal[:i+1],stack=True).sum(axis=2) + base_summed

        # truncate only to decoder
        causal_decoder = {}
        for c in self.hparams.causal:
            causal_decoder[c] = causal_info[c][:,max_encoder_length:,:]

        # for temporal enrichment, prepare a tensor with everything we know about the past
        # which will be passed as keys and values to the attention block
        if self.hparams.attention:
            complete_past = self.dtf(x_encoded,reals+cats,stack=True).sum(axis=2)
            known_past = complete_past[:,:max_encoder_length,:]
            known_future = self.dtf(x_encoded,reals_dec+cats_dec+static_reals+static_cats,stack=True).sum(axis=2)[:,max_encoder_length:,:]
            known_all = torch.cat([known_past,known_future],dim=1)
            known_all = known_past

            attention_values = self.attention_values(known_future,known_all)


        for c in self.hparams.causal:
            if self.hparams.attention:
                # known_cov = torch.cat([known_past,causal_decoder[c]],dim=1)
                known_cov = known_past
                causal_decoder[c] += self.attention_apply(known_cov,attention_values) 

            causal_decoder[c] = self.causal_projections[c](self.causal_mlps[c](causal_decoder[c]))  # here we would insert attention
            causal_decoder[c] = torch.stack(causal_decoder[c].chunk(self.hparams.output_size,dim=-1),dim=1)
    
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
            real_normal = x_normal[c].unsqueeze(1)
            real_normal = real_normal[:,:,max_encoder_length:,:]
            real_effect = causal_decoder[c] * real_normal
            real_effects[c] = real_effect.squeeze(3).transpose(1,2) # back to batch x time x output_size(quantiles)
            output += real_effects[c]

        prediction = self.transform_output(output, target_scale=target_scale)

        x_level = self.transform_output(x_level, target_scale=target_scale)
        cat_dict = {feature:o * target_scale[:,1].view(o.size(0),1,1) for feature,o in cat_effects.items()}
        if len(real_effects) > 0:
            cont_dict = {feature:o * target_scale[:,1].view(o.size(0),1,1) for feature,o in real_effects.items()}
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
                trend_query  = ['time_idx','weekofyear'])
        
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



    