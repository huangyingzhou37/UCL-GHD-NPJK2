# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 23:19:06 2021

@author: Yingzhou Huang
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import json
from scipy.signal import argrelextrema
import pickle
import os

SAVE_FILE=False

""" raw data """
data_raw = pd.read_csv('https://oxcgrtportal.azurewebsites.net/api/CSVDownload', 
                       index_col=None, parse_dates=['Date'])

data = data_raw[['CountryName','CountryCode','Jurisdiction','Date',\
                 'ConfirmedCases','ConfirmedDeaths','StringencyIndex',\
                 'GovernmentResponseIndex','ContainmentHealthIndex',\
                 'C6_Stay at home requirements']] 

data = data.loc[data['Jurisdiction']=='NAT_TOTAL']
all_countries = list(set(data['CountryName']))

with open('country_params.json') as json_file:
    country_params = json.load(json_file)

""" population data """
with open('F:/Google Drive/covid/population.json') as json_data: # source: wiki
    pop = json.load(json_data)
    json_data.close()
   
all_ctry_w_pop = []
for k in all_countries: 
    if k in pop.keys():
        all_ctry_w_pop.append(k)
    # else: 
    #     print(k)


def plot_2_series(data, x_col, y_col):
    ax1 = data[y_col].plot()
    ax2 = ax1.twinx()
    ax2.spines['right'].set_position(('axes', 1.0))
    data[x_col].plot(ax=ax2,color='DarkOrange')


def plot_peaks(df, x_rate=False, y_rate=False, title=''):
    if y_rate: 
        df[y_col] = df[y_col] / df[y_col].shift(1) - 1
    if x_rate: 
        df[x_rate] = df[x_rate] / df[x_rate].shift(1) - 1
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(df.index, df[y_col], color='SteelBlue')
    ax2.plot(df.index, df[x_col], color='DarkOrange')
    ax1.set_ylabel(y_col, color='SteelBlue')
    ax2.set_ylabel(x_col, color='DarkOrange')
    ax1.scatter(df.index, df['max'], c='g')
    plt.title(title)
    # plt.show()
    if SAVE_FILE: 
        plt.savefig('charts/%s & %s/peaks_%s.png' % (x_col, y_col, title), figsize=(10,8))
    else:
        plt.show()
    

# x_col = 'GovernmentResponseIndex'
# x_col = 'C6_Stay at home requirements'
x_col = 'StringencyIndex'
# y_col = 'ConfirmedDeaths'
y_col = 'ConfirmedCases'
y_col_rate = y_col+'_rate'

chart_folder = os.path.join("charts", "%s & %s" % (x_col, y_col))
if not os.path.exists(chart_folder):
    os.makedirs(chart_folder)

y_max_mapping = {
    "ConfirmedCases": 100,
    "ConfirmedDeaths": 10
    }


results_all = {}
countries_no_data = []
countries_bad_results = []

def run_analysis():
    df_all = pd.DataFrame()
    for c in all_ctry_w_pop:
        # c='United Kingdom' c='Gambia'
        print(c)
        results_c = {}
        POLICY_LAG = country_params[c]['policy_lag'] # weeks
        BEFORE_PEAK = country_params[c]['before_peak']
        AFTER_PEAK = country_params[c]['after_peak']
        df = data.loc[data['CountryName']==c]
        df = df [['Date', x_col, y_col]]
        df = df.set_index('Date')
        df[y_col] = df[y_col].diff()
        
        df_smooth = df.copy()
        df_smooth[y_col] = df_smooth[y_col].rolling(7).mean()
        
        # plot Stringency vs new cases
        # plot_2_series(df_smooth, x_col, y_col)
        # df_smooth.dropna().plot.scatter(x="StringencyIndex", y="ConfirmedDeaths")
        # df_smooth['YearMon'] = df_smooth.index.strftime('%Y-%m')
        
        df_avg = df_smooth.resample("W").mean()
        df_peak = df_avg.copy()
        # df_avg['max'] = df_avg[y_col][(df_avg[y_col].shift(1) < df_avg[y_col]) & (df_avg[y_col].shift(-1) < df_avg[y_col])]
        df_peak[y_col_rate] = df_peak[y_col] / df_peak[y_col].shift(1) - 1
        df_peak[y_col+'_lead'] = df_peak.shift(-AFTER_PEAK)[y_col]
        # plot_2_series(df_peak, y_col_rate, y_col)
        df_peak['max'] = df_avg.iloc[argrelextrema(df_avg[y_col].values, np.greater, order=4)[0]][y_col]
        df_peak['max'] = df_peak.loc[df_peak['max'] > y_max_mapping[y_col]]['max']
        df_peak['max'] = df_peak.loc[df_peak[y_col+'_lead']<df_peak[y_col]*(1 - .03 * AFTER_PEAK)]['max'] # 4 weeks after the peak should < 70% of peak cases e.g. c='Finland'
        
        
        plot_peaks(df_peak.loc[df_peak.index < '2021-02-01'], y_rate=False, title=''+c)
        
        df_target = df_peak.copy()[[x_col,y_col, 'max']]
        df_target[y_col] = df_target[y_col] / pop[c] # divide new cases by population
        df_target[y_col_rate] = df_target[y_col] / df_target[y_col].shift(1) - 1
        
        df_target[x_col+'_lag'] = df_target[x_col].shift(POLICY_LAG) # shift policy forward by x weeks, in order to compare with new cases/deaths
    
        def shift_and_get(df_target, df_nearby, i):
            df_near_peak = df_target.copy()
            df_near_peak['max'] = df_near_peak['max'].shift(i)
            df_near_peak = df_near_peak.dropna()[[x_col+'_lag', y_col_rate]]
            df_nearby = pd.concat([df_nearby, df_near_peak])
            return df_nearby
            
        df_nearby = pd.DataFrame()
        df_peak_points = df_target.dropna()[[x_col+'_lag', y_col_rate]]
        df_nearby = pd.concat([df_nearby, df_peak_points])
        for i in range(1,BEFORE_PEAK+1):
            df_nearby = shift_and_get(df_target, df_nearby, -i)
        for i in range(1,AFTER_PEAK+1):
            df_nearby = shift_and_get(df_target, df_nearby, i)
        crr = df_nearby.corr().iloc[0,1]
        results_c['corr'] = crr
        results_c['policy_lag'] = POLICY_LAG
        results_c['before_peak'] = BEFORE_PEAK
        results_c['after_peak'] = AFTER_PEAK
        results_c['df_nearby'] = df_nearby
        
        if results_c['df_nearby'] is None: 
            countries_no_data.append(c)
        else:
            results_all[c] = results_c
            fig = results_c['df_nearby'].plot.scatter(x=x_col+'_lag', y=y_col_rate, title='%s corr=%.3f' % (c, results_c['corr'])).get_figure()
            fig.savefig('charts/%s & %s/corr_%s.png' % (x_col, y_col,c))
            results_c['df_nearby']['Country'] = c
            df_all = pd.concat([df_all, results_c['df_nearby']])

run_analysis()

# with open('charts/results_all.pickle', 'wb') as f:
#     pickle.dump(results_all, f)


with open('results_all.pickle', 'rb') as f:
    results_all = pickle.load(f)
    
""" plot ranking """
ranking_dict = []
for country in results_all:
    p = results_all[country]
    corr = p['corr']
    ranking_dict.append({'country': country, 'corr': corr})
df_ranking = pd.DataFrame(ranking_dict)
df_ranking.sort_values('corr', inplace=True)
df_ranking = df_ranking.reset_index()[['country', 'corr']]

print("Median:{0:.3f}".format(df_ranking['corr'].median()))
print("Mean  :{0:.3f}".format(df_ranking['corr'].mean()))
print("90% Q :{0:.3f}".format(df_ranking['corr'].quantile(0.9)))
print("75% Q :{0:.3f}".format(df_ranking['corr'].quantile(0.75)))
print("25% Q :{0:.3f}".format(df_ranking['corr'].quantile(0.25)))
print("10% Q :{0:.3f}".format(df_ranking['corr'].quantile(0.1)))

import matplotlib.patches as patches
fig, ax = plt.subplots(figsize=(32,8), facecolor='white', dpi= 80)
ax.vlines(x=df_ranking.index, ymin=0, ymax=df_ranking['corr'], color='firebrick', alpha=0.8, linewidth=6)
# Annotate Text
# for i, cty in enumerate(df_ranking['corr']):
#     ax.text(i, cty+0.5, round(cty, 1), horizontalalignment='center')
# Title, Label, Ticks and Ylim
ax.set_title('Correlation ranking', fontdict={'size':14})
ax.set(ylabel='Correlation', ylim=(-1, 0))
plt.xticks(df_ranking.index, df_ranking.country, rotation=45, horizontalalignment='right', fontsize=10)

# Add patches to color the X axis labels
# p1 = patches.Rectangle((.829, -0.005), width=.071, height=.13, alpha=.1, facecolor='red', transform=fig.transFigure)
# p2 = patches.Rectangle((.124, -0.005), width=.7055, height=.13, alpha=.1, facecolor='green', transform=fig.transFigure)
# fig.add_artist(p1)
# fig.add_artist(p2)
plt.show()
