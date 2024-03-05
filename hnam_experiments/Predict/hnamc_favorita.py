# %%
import numpy as np
import pandas as pd
import random
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from dataclasses import dataclass
import lightning.pytorch as pl

import torch
from torch import nn
from pytorch_forecasting import TimeSeriesDataSet,QuantileLoss,EncoderNormalizer,GroupNormalizer,DeepAR,TemporalFusionTransformer,HNAM,NaNLabelEncoder,SMAPE,RMSE,MAE,MAPE
from lightning.pytorch.callbacks import ModelCheckpoint

from typing import Dict, List, Tuple, Union
from copy import copy
import os
import io

np.random.seed(0)
torch.manual_seed(0)
### ALL SETTINGS


if torch.cuda.is_available():
    ACC = 'gpu'                                     # accelerator gpu (on hi perf cluster) or cpu (local on mac)
    NW = int(os.environ.get('OMP_NUM_THREADS',0))   # number of threads (cores) 
    torch.set_float32_matmul_precision('medium')
else:
    ACC = 'cpu'
    NW = 0
print(f'Using {NW} threads')

LIMIT_TS = False # limit number of time series

LR = 1e-3   # learning rate
WD = 1e-2   # weight decay
BS = 256    # batch size
RLRP = 10   # reduce on plateau patience
EPOCHS = 300 # maximum number of epochs 
FINE_TUNE = 100 # fine tune epochs
PATIENCE = 30 # patience for early stopping
LONG_MODEL = 'HNAM' 
SHORT_MODEL = 'hnam'  # for saving results pickle
PTFC_MODEL = HNAM
LOSS = RMSE(quantiles=[0.5])


holiday = ['national_holiday','regional_holiday','local_holiday']
MODEL_KWARGS = dict(base=           ['time_idx','relative_time_idx','art','pos','doy_sine','doy_cosine'],
                    causal=         ['weekday','onpromotion'] + holiday,
                    trend_query  =  ['time_idx','relative_time_idx','art','pos','doy_sine','doy_cosine'],
                    cov_emb=32,trend_emb=8,factor=4,attention=True,att_proj='cnn')

# data
DATASET = 'Favorita'

os.makedirs(f'../Models/{DATASET}/{LONG_MODEL}', exist_ok=True)
# dataset
pred_length = 14
val_length = 30
enc_len = 30



df = pd.read_pickle(f'../Processed/{DATASET}/{DATASET.lower()}_data.pkl')

if LIMIT_TS:
    sel = list(df.groupby('time_series')['sales'].sum().sort_values(ascending=False).index)[:LIMIT_TS]
    df = df[df['time_series'].isin(sel)]

time_idxs = df['time_idx'].unique()

ds_kwargs = dict(
    static_categoricals=['art','pos','cluster'],
    time_varying_unknown_reals=['sales','dcoilwtico'],
    time_varying_known_reals=['time_idx','doy_sine','doy_cosine'],
    time_varying_known_categoricals=['weekday','onpromotion','national_holiday','regional_holiday','local_holiday'],
    target_normalizer = GroupNormalizer(groups=['time_series']),
    categorical_encoders = {
                        'national_holiday': NaNLabelEncoder(add_nan=False).fit(df['national_holiday']),
                        'regional_holiday': NaNLabelEncoder(add_nan=False).fit(df['regional_holiday']),
                        'local_holiday': NaNLabelEncoder(add_nan=False).fit(df['local_holiday'])})

with open('../Config/dates.txt') as f:
    lines = f.read().splitlines()
    lines = [l for l in lines if l]
keys = ['Walmart','Retail','Favorita']
test_dates = dict(zip(keys,[[],[],[]]))
for l in lines:
    if l in keys:
        key = l
    else:
        test_dates[key].append(l)
test_dates = pd.Series(test_dates[DATASET])

def add_month(date):   ### NEW
    if date.month == 12:
        return date.replace(year=date.year+1, month=1, day=1)
    else:
        return date.replace(month=date.month+1, day=1)
    
test_end_dates = test_dates.apply(lambda x: pd.to_datetime(x)).apply(add_month)

dtot = df[['time_idx','date']].drop_duplicates().set_index('date')['time_idx']
dtot = dtot.asfreq('D').bfill().to_dict()

ttod = {v:k for k,v in dtot.items()}

train_ends = test_dates.map(dtot) - 1  # from test_start to train_end so minus 1
train_ends = train_ends.to_list()

with open(f"../Models/{DATASET}/{LONG_MODEL}/best_model.txt", "r") as f:
    lines = f.readlines()
best_models = dict()
for line in lines:
    line = line.strip()
    if '.0' in line:
        line = line.split('.')[0]
    if line.isnumeric():
        key = int(float(line))
    else:
        best_models[key] = line
best_models

all_results = pd.DataFrame()
for test_period in range(len(test_dates)):
        
    train_end = train_ends[test_period]
    test_end_date = test_end_dates[test_period]
    best_model_path = best_models[int(train_end)]

    df_train = df[df['time_idx'] <= train_end - val_length]
    df_val = df[(df['time_idx'] > train_end - val_length - enc_len) & (df['time_idx'] <= train_end)]
    df_test = df[(df['time_idx'] > train_end - enc_len) & (df['date'] < test_end_date)]  # changed for plotting
    ds_train = TimeSeriesDataSet(df,
                            target='sales',
                            group_ids=['time_series'],
                            time_idx='time_idx',
                            min_encoder_length=enc_len,
                            max_encoder_length=enc_len,
                            min_prediction_length=pred_length,
                            max_prediction_length=pred_length,
                            time_varying_unknown_categoricals=[],
                            add_relative_time_idx=True,
                            allow_missing_timesteps=True,
                            scale_target=True,
                            **ds_kwargs
                            )

    ds_val = TimeSeriesDataSet.from_dataset(ds_train, df_val, stop_randomization=True,scale_target=True)
    ds_test = TimeSeriesDataSet.from_dataset(ds_train, df_test, stop_randomization=True,scale_target=False)
                            
    dl_train = ds_train.to_dataloader(train=True,num_workers = NW)
    dl_val = ds_val.to_dataloader(train=False,num_workers = NW)
    dl_test = ds_test.to_dataloader(train=False,num_workers = NW)
    checkpoint = torch.load(best_model_path, map_location=torch.device('cpu'))
    checkpoint['hyper_parameters']['loss'] = RMSE()
    checkpoint['hyper_parameters']['logging_metrics'] = torch.nn.ModuleList([SMAPE(),RMSE(),MAE(),MAPE()])
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    buffer.seek(0)
    model = PTFC_MODEL.load_from_checkpoint(buffer)
    model.rescale_on()

    raw = model.predict(dl_test, mode="raw", return_x=True, return_index=True,trainer_kwargs=dict(accelerator="cpu"))
    index = raw.index
    preds = raw.output.prediction
    dec_len = preds.shape[1]
    n_quantiles = preds.shape[-1]
    covs = list(raw.output.keys()[1:])
    quantiles = None

    preds_df = pd.DataFrame(index.values.repeat(dec_len * n_quantiles, axis=0),columns=index.columns)
    preds_df = preds_df.assign(h=np.tile(np.repeat(np.arange(1,1+dec_len),n_quantiles),len(preds_df)//(dec_len*n_quantiles)))
    preds_df = preds_df.assign(q=np.tile(np.arange(n_quantiles),len(preds_df)//n_quantiles))
    preds_df['pred_idx'] = preds_df['time_idx'] + preds_df['h'] - 1
    if quantiles is not None:
        preds_df['q'] = preds_df['q'].map({i:q for i,q in enumerate(quantiles)})
    elif preds_df.q.nunique() == 1:
        preds_df = preds_df.drop(columns=['q'])
    preds_df['pred'] = preds.flatten()
    # preds_df['true'] = raw.x['decoder_target'].flatten()

    for k in covs:
        if k == 'attention':
            continue
            for step in range(raw.output[k].shape[2]):
                preds_df[f'att_{-raw.output[k].shape[2]+step}'] = raw.output[k][:,:,step].flatten()
        else:
            preds_df['effect_'+k] = raw.output[k].flatten()

    all_results = pd.concat([all_results,preds_df])

all_results.to_pickle(f'../Evaluation/{DATASET}/{SHORT_MODEL}.pkl')