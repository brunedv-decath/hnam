import pandas as pd
import numpy as np
from statsforecast.models import AutoARIMA, AutoETS
from joblib import Parallel, delayed
import multiprocessing

LIMIT = False
DATASET = 'Walmart'
SEASON_LENGTH = 7
HORIZON = 14
MODEL = 'arima'
QUANTILES = [0.5]

holiday = ['Sporting', 'Cultural', 'National']
COVS = ['relprice','snap']+holiday
CATS = ['snap']+holiday

conflevels = [int(abs(100 - (100-sl*100)*2)) for sl in QUANTILES]
clevel = [str(int(abs(100 - (100-sl*100)*2))) for sl in QUANTILES]
clevel_r = []
for q,s in zip(QUANTILES,clevel):
    prefix = 'lo-' if q < 0.5 else 'hi-'
    clevel_r.append(prefix+s)
replacer = dict(zip(clevel_r,QUANTILES))
conflevels = list(set(conflevels))
data = pd.read_pickle(f'../Processed/{DATASET}/{DATASET.lower()}_data.pkl')

data = data[['time_series','date','time_idx','sales']+COVS]
data = pd.get_dummies(data,columns=CATS,drop_first=True)

covs = list(data.columns[4:])

time_series = data['time_series'].unique()
data = data.query('time_series in @time_series')
data = data[['time_series','date','time_idx','sales']+covs]

if LIMIT:
    time_series = time_series[:LIMIT]
    data = data.query('time_series in @time_series')
    data = data[data.time_idx > 1500]

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

test_dates = pd.to_datetime(pd.Series(test_dates[DATASET]))

dtot = data[['time_idx','date']].drop_duplicates().set_index('date')['time_idx']
dtot = dtot.to_dict()

ttod = {v:k for k,v in dtot.items()}

train_ends = test_dates.map(dtot) - 1  # from test_start to train_end so minus 1
train_ends = train_ends.to_list()
def add_month(date):   ### NEW
    if date.month == 12:
        return date.replace(year=date.year+1, month=1, day=1)
    else:
        return date.replace(month=date.month+1, day=1)
    
test_end_dates = test_dates.apply(lambda x: pd.to_datetime(x)).apply(add_month)
test_ends = test_end_dates.map(dtot).to_list()
valid_dates = pd.to_datetime(pd.Series(data.date.unique()).sort_values().reset_index(drop=True))

# Define the function to run forecast for a single time series
def forecast_time_series(time_series_i):
    all_preds_ts = pd.DataFrame()

    for first_test,end_test in zip(test_dates,test_end_dates):

        past_data = data.query('date < @first_test & time_series == @time_series_i').set_index('date').asfreq('D').fillna(0)
        future_covs = data.query('date >= @first_test & date <= @end_test & time_series == @time_series_i').set_index('date').asfreq('D').fillna(0)
        max_pred = first_test + pd.Timedelta(days=13)

        if HORIZON == 12:
            past_data = past_data[past_data.index.dayofweek != 6]
            future_covs = future_covs[future_covs.index.dayofweek != 6]

        if MODEL == 'arima':
            model =  AutoARIMA(seasonal=True,season_length=SEASON_LENGTH, allowdrift=True,allowmean=True,num_cores = 1)
        elif MODEL == 'ets':
            model = AutoETS(season_length = SEASON_LENGTH)
                    
        model.fit(past_data['sales'].values,past_data[covs].values if MODEL == 'arima' else None)

        while max_pred < end_test:
            preds = model.forward(y = past_data['sales'].values,
                            X = past_data[covs].values if MODEL == 'arima' else None,
                            X_future = future_covs[covs].values[:HORIZON] if MODEL == 'arima' else None,
                            h = HORIZON,
                            level= conflevels)

            preds = pd.DataFrame(preds).rename(columns=replacer)[QUANTILES].melt(value_name='pred',var_name='q')
            preds['time_idx'] = dtot[first_test]

            if HORIZON == 12:
                dates = pd.date_range(start=first_test, end=first_test+pd.Timedelta(days=13))
                dates = dates[dates.dayofweek != 6]
            else:
                dates = pd.date_range(start=first_test, end=first_test+pd.Timedelta(days=HORIZON-1))

            preds['pred_date'] = dates
            preds = preds[preds['pred_date'].isin(valid_dates.values)]
            preds['pred_idx'] = preds['pred_date'].map(dtot)
            preds['h'] = preds['pred_idx'] - preds['time_idx'] + 1
            preds['time_series'] = time_series_i
            all_preds_ts = pd.concat([all_preds_ts,preds])


            first_test = first_test + pd.Timedelta(days=1)
            while first_test not in valid_dates.values:
                first_test = first_test + pd.Timedelta(days=1)

            past_data = data.query('date < @first_test & time_series == @time_series_i').set_index('date').asfreq('D').fillna(0)
            future_covs = data.query('date >= @first_test & date <= @end_test & time_series == @time_series_i').set_index('date')[covs].asfreq('D').fillna(0)
            if HORIZON == 12:
                past_data = past_data[past_data.index.dayofweek != 6]
                future_covs = future_covs[future_covs.index.dayofweek != 6]
            max_pred = first_test + pd.Timedelta(days=13)
            

    return all_preds_ts

# Run the forecasts in parallel
num_cores = multiprocessing.cpu_count()
results = Parallel(n_jobs=num_cores)(delayed(forecast_time_series)(ts) for ts in time_series)

# Concatenate all results
all_results = pd.concat(results, axis=0)
all_results = all_results[['time_series','time_idx','pred_idx','pred_date','h','q','pred']]
all_results.to_pickle(f'../Evaluation/{DATASET}/{MODEL.lower()}.pkl')