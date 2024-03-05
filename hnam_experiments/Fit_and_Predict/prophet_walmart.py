import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing
from prophet import Prophet

LIMIT = False
DATASET = 'Walmart'
holiday = ['Sporting', 'Cultural', 'National']
COVS = ['relprice','snap']+holiday
CATS = ['snap']+holiday
WEEKLENGTH = 7
HORIZON = 14
MODEL = 'prophet'

data = pd.read_pickle(f'../Processed/{DATASET.capitalize()}/{DATASET.lower()}_data.pkl')

time_series = data['time_series'].unique()
if LIMIT:
    time_series = time_series[:LIMIT]
    data = data.query('time_series in @time_series')
    data = data[data.time_idx > 1500]

data = data[['time_series','date','time_idx','sales']+COVS]
data = data.rename(columns={'date':'ds','sales':'y'})
data = pd.get_dummies(data,columns=CATS,drop_first=True)

covs = list(data.columns[4:])


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

dtot = data[['time_idx','ds']].drop_duplicates().set_index('ds')['time_idx']
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
valid_dates = pd.to_datetime(pd.Series(data.ds.unique()).sort_values().reset_index(drop=True))


def forecast_time_series(time_series_i):
    all_preds_ts = pd.DataFrame()
    for first_test,end_test in zip(test_dates,test_end_dates):

        past_data = data.query('ds < @first_test & time_series == @time_series_i')
        future_covs = data.query('ds >= @first_test & ds <= @end_test & time_series == @time_series_i')


        max_pred = first_test + pd.Timedelta(days=13)

        model_add = Prophet(yearly_seasonality=True, weekly_seasonality=True, seasonality_mode='additive')
        model_mul = Prophet(yearly_seasonality=True, weekly_seasonality=True, seasonality_mode='multiplicative')
        for cov in covs:
            model_add.add_regressor(cov)
            model_mul.add_regressor(cov)
        model_add.fit(past_data)
        model_mul.fit(past_data)
        pred_add = model_add.predict(past_data)
        pred_mul = model_mul.predict(past_data)

        # calculate rmse
        rmse_add = np.sqrt(((past_data['y'] - pred_add['yhat'])**2).mean())
        rmse_mul = np.sqrt(((past_data['y'] - pred_mul['yhat'])**2).mean())
        # select model
        if rmse_add.mean() < rmse_mul.mean():
            modeltype = 'add'
        else:
            modeltype = 'mul'
       
        while max_pred < end_test:

            if modeltype == 'add':
                model = Prophet(yearly_seasonality=True, weekly_seasonality=True, seasonality_mode='additive')
            else:
                model = Prophet(yearly_seasonality=True, weekly_seasonality=True, seasonality_mode='multiplicative')


            model.fit(past_data)
            forecast = model.make_future_dataframe(periods=30, include_history=False, freq='D')
            forecast = forecast[forecast.ds <= max_pred]
            forecast = forecast[forecast.ds.isin(valid_dates.values)]
            forecast = forecast.merge(future_covs, on='ds', how='left')
            forecast = model.predict(forecast)
            forecast = forecast[['ds', 'yhat']]
            forecast.columns = ['pred_date', 'pred']
            forecast['pred_idx'] = forecast['pred_date'].map(dtot)
            forecast['time_idx'] = dtot[first_test]
            forecast['h'] = forecast['pred_idx'] - forecast['time_idx'] + 1
            forecast['time_series'] = time_series_i
            forecast['q'] = 0.5 # could expand for probabilistic forecasts
            all_preds_ts = pd.concat([all_preds_ts,forecast])

            #### PREPARE NEXT LOOP

            first_test = first_test + pd.Timedelta(days=1)
            while first_test not in valid_dates.values:
                first_test = first_test + pd.Timedelta(days=1)

            past_data = data.query('ds < @first_test & time_series == @time_series_i')
            future_covs = data.query('ds >= @first_test & ds <= @end_test & time_series == @time_series_i')[covs+['ds']] 

            max_pred = first_test + pd.Timedelta(days=13)
            print(max_pred)
            
    return all_preds_ts

# Run the forecasts in parallel
num_cores = multiprocessing.cpu_count()
results = Parallel(n_jobs=num_cores)(delayed(forecast_time_series)(ts) for ts in time_series)

# Concatenate all results
all_preds = pd.concat(results, axis=0)
all_preds = all_preds[['time_series','time_idx','pred_idx','pred_date','h','q','pred']]
all_preds.to_pickle(f'../Evaluation/{DATASET}/{MODEL.lower()}.pkl')