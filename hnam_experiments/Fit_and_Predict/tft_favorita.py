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
from pytorch_forecasting import TimeSeriesDataSet,QuantileLoss,EncoderNormalizer,GroupNormalizer,DeepAR,TemporalFusionTransformer,HNAM,NaNLabelEncoder,SMAPE,RMSE
from lightning.pytorch.callbacks import ModelCheckpoint

from typing import Dict, List, Tuple, Union
from copy import copy
import os

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
LONG_MODEL = 'TFT' 
SHORT_MODEL = 'tft'  # for saving results pickle
PTFC_MODEL = TemporalFusionTransformer
LOSS = RMSE(quantiles=[0.5])



MODEL_KWARGS = dict(hidden_size=32)

# data
DATASET = 'Favorita'
holiday = ['national_holiday','regional_holiday','local_holiday']
os.makedirs(f'../Models/{DATASET}/{LONG_MODEL}', exist_ok=True)
# dataset
pred_length = 14
val_length = 30
test_length = 30
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

def add_month(date):   
    if date.month == 12:
        return date.replace(year=date.year+1, month=1, day=1)
    else:
        return date.replace(month=date.month+1, day=1)
    
test_end_dates = test_dates.apply(lambda x: pd.to_datetime(x)).apply(add_month)

dtot = df[['time_idx','date']].drop_duplicates().set_index('date')['time_idx']
dtot = dtot.asfreq('D').bfill().to_dict()

ttod = {k:v for k,v in dtot.items()}

train_ends = test_dates.map(dtot) - 1  # from test_start to train_end so minus 1
train_ends = train_ends.to_list()

all_results = pd.DataFrame()

first_run = True
for train_end,test_end_date in zip(train_ends,test_end_dates):

    df_train = df[df['time_idx'] <= train_end - val_length]
    df_val = df[(df['time_idx'] > train_end - val_length - enc_len) & (df['time_idx'] <= train_end)]
    df_test = df[(df['time_idx'] > train_end - enc_len) & (df['date'] < test_end_date)]
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

    if first_run:
        model = PTFC_MODEL.from_dataset(ds_train,loss=LOSS,learning_rate=LR,weight_decay=WD,**MODEL_KWARGS,reduce_on_plateau_patience=RLRP)
    else:
        model = PTFC_MODEL.load_from_checkpoint(best_model_path)

    model.rescale_off()


    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename='model-{epoch:02d}-{val_loss:.2f}',  # Save the model with the epoch number and the validation metric
        save_top_k=1,  # Save only the best model
        mode='min',  # The best model is the one with the lowest validation SMAPE
        auto_insert_metric_name=False,  # Prevent prepending monitored metric name to the filename
    )

    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=PATIENCE,
        verbose=True,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs= EPOCHS if first_run else FINE_TUNE,  # The number of epochs is 0 if it is not the first run
        accelerator=ACC,
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback, checkpoint_callback],  # Add the checkpoint_callback to the list of callbacks
        default_root_dir=f'../Models/{DATASET}/{LONG_MODEL}/',  # The path where the runs and models will be saved
    )

    if not first_run:
        # Validate the model on the validation data loader
        validation_results = trainer.validate(model, dataloaders=dl_val)
        initial_val_loss = validation_results[0]['val_loss']
        print(f'Initial Validation Loss: {initial_val_loss}')
        
        # Update the checkpoint callback's best score with the initial validation loss
        checkpoint_callback.best_model_score = torch.tensor(initial_val_loss)
        checkpoint_callback.best_model_path = os.path.join(
            checkpoint_callback.dirpath, 
            f"model-initial-val_loss={initial_val_loss:.2f}.ckpt"
        )
        
        trainer.save_checkpoint(checkpoint_callback.best_model_path)

    trainer.fit(model, dl_train,val_dataloaders=dl_val)

    best_model_path = trainer.checkpoint_callback.best_model_path

    # write best model path
    with open(f"../Models/{DATASET}/{LONG_MODEL}/best_model.txt", "a") as f:
        f.write(str(train_end)+'\n')
        f.write(best_model_path+'\n')

    model = PTFC_MODEL.load_from_checkpoint(best_model_path)
    model.rescale_on()

    output = model.predict(dl_test, mode="quantiles", return_x=True, return_index=True,trainer_kwargs=dict(accelerator=ACC))
    index = output.index
    preds = output.output
    dec_len = preds.shape[1]
    n_quantiles = len(model.loss.quantiles)
    assert n_quantiles == preds.shape[-1] 
    quantiles = model.loss.quantiles

    preds_df = pd.DataFrame(index.values.repeat(dec_len * n_quantiles, axis=0),columns=index.columns)
    preds_df = preds_df.assign(h=np.tile(np.repeat(np.arange(1,1+dec_len),n_quantiles),len(preds_df)//(dec_len*n_quantiles)))
    preds_df = preds_df.assign(q=np.tile(np.arange(n_quantiles),len(preds_df)//n_quantiles))
    preds_df['pred_idx'] = preds_df['time_idx'] + preds_df['h'] - 1
    preds_df['q'] = preds_df['q'].map({i:q for i,q in enumerate(quantiles)})
    preds_df['pred'] = preds.flatten().cpu()
    preds_df['true'] = torch.repeat_interleave(output.x['decoder_target'].flatten().cpu(),n_quantiles)

    results = preds_df[['time_idx','time_series','h','pred_idx','pred','true']]
    all_results = pd.concat([all_results,results],axis=0)
    first_run = False
    
all_results.to_pickle(f'../Evaluation/{DATASET}/{SHORT_MODEL}.pkl')