# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 23:09:28 2021

@author: Yingzhou Huang
"""


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import json
from linearmodels import PooledOLS
import statsmodels.api as sm

data_raw = pd.read_csv('owid-covid-data.csv', index_col=None, parse_dates=['date'])

with open('population.json') as json_data: # source: wiki
    pop = json.load(json_data)
    json_data.close()


X_cols = ['people_fully_vaccinated_per_hundred', 'stringency_index', 
            'population_density', 'median_age', 'aged_70_older', 'gdp_per_capita', 
            'cardiovasc_death_rate', 'diabetes_prevalence', 
            'hospital_beds_per_thousand'] # , 'female_smokers', 'male_smokers', 'extreme_poverty','handwashing_facilities', human_development_index, 'life_expectancy'


X_cols = ['people_fully_vaccinated_per_hundred', 'population_density','aged_65_older', 'gdp_per_capita','cardiovasc_death_rate', 'diabetes_prevalence','hospital_beds_per_thousand'] 
# X_cols = ['people_fully_vaccinated_per_hundred','stringency_index']
df_data_all = data_raw[['location', 'date', 'new_cases_smoothed', 'new_deaths_smoothed'] + X_cols]


X_variables = X_cols + ['year_month']
Y_variable = 'pop_adj_covid_deaths'

# df_data_all['infection_death_rate'] = df_data_all['new_deaths_smoothed'] / df_data_all['new_cases_smoothed']
# df_data_all['smokers'] = df_data_all['male_smokers'] + df_data_all['female_smokers']


df_data_2021 = df_data_all.copy()
   
# for lct in df_data_2021.location.unique():
#     df_data_2021.loc[df_data_2021.location==lct,Y_variable] = \
#         df_data_2021.loc[df_data_2021.location==lct]['new_deaths_smoothed'] \
#             / df_data_2021.loc[df_data_2021.location==lct]['new_cases_smoothed'].shift(19)

# for lct in df_data_2021.location.unique():
#     df_data_2021.loc[df_data_2021.location==lct,'stringency_index'] = \
#         df_data_2021.loc[df_data_2021.location==lct]['stringency_index'].shift(19)



for lct in df_data_2021.location.unique():
    df_data_2021.loc[df_data_2021.location==lct,'people_fully_vaccinated_per_hundred'] = \
        df_data_2021.loc[df_data_2021.location==lct]['people_fully_vaccinated_per_hundred'].shift(19)

for col in X_cols: 
    for lct in df_data_2021.location.unique():
        df_data_2021.loc[df_data_2021.location==lct,col] = \
            df_data_2021.loc[df_data_2021.location==lct][col].ffill()
            
df_data_2021 = df_data_2021.loc[df_data_2021.date>'2021-02-01']

df_data_2021['year_month'] = df_data_2021['date'].dt.strftime('%Y-%m')

dataset = df_data_2021.set_index(['location', 'date'])
dates = dataset.index.get_level_values('date').to_list()
dataset['date'] = pd.Categorical(dates)
dataset['location'] = dataset.index.get_level_values('location')
dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
dataset['people_fully_vaccinated_per_hundred'].fillna(0, inplace=True)
dataset = dataset.dropna()

for ctry in dataset['location'].unique():
    if ctry not in pop:
        print(ctry)
pop['Czechia'] = pop['Czech Republic']
pop['Kyrgyzstan'] = pop['Kyrgyz Republic']
pop['Slovakia'] = pop['Slovak Republic']
pop['Timor'] = pop['Timor-Leste']

for lct in dataset.location.unique():
    if lct in pop:
        dataset.loc[dataset.location==lct,Y_variable] = \
            dataset.loc[dataset.location==lct]['new_deaths_smoothed'] / pop[lct] * 1000
    else: 
        dataset.loc[dataset.location==lct,Y_variable] = np.nan

dataset = dataset.dropna()


# exog = sm.tools.tools.add_constant(dataset[X_variables])
# endog = dataset[Y_variable]
# # random effects model
# model_re = RandomEffects(endog, exog) 
# re_res = model_re.fit() 
# print(re_res)

# add months dummy variables
# dataset['year_month'] = dataset['date'].dt.strftime('%Y-%m')


""" assumptions:
(1) Linearity, 
(2) Exogeneity (Hausman-Test), 
(3a) Homoskedasticity (White-Test and Breusch-Pagan-Test) 
(3b) Non-autocorrelation (Durbin-Watson-Test), 
(4) Independent variables are not Stochastic and 
(5) No Multicolinearity.
"""
# Perform PooledOLS
exog = sm.tools.tools.add_constant(dataset[X_variables])
endog = dataset[Y_variable]
mod = PooledOLS(endog, exog)
pooledOLS_res = mod.fit(cov_type='clustered', cluster_entity=True)

# Store values for checking homoskedasticity graphically
fittedvals_pooled_OLS = pooledOLS_res.predict().fitted_values
residuals_pooled_OLS = pooledOLS_res.resids
# Homoskedasticity check
# a) Residuals-Plot for growing Variance Detection
fig, ax = plt.subplots()
ax.scatter(fittedvals_pooled_OLS, residuals_pooled_OLS, color = 'blue')
ax.axhline(0, color = 'r', ls = '--')
ax.set_xlabel('Predicted Values', fontsize = 15)
ax.set_ylabel('Residuals', fontsize = 15)
ax.set_title('Homoskedasticity Test', fontsize = 18)
plt.show() # shows growing variance hence heteroskedasticity

# b) White-Test
from statsmodels.stats.diagnostic import het_white, het_breuschpagan
pooled_OLS_dataset = pd.concat([dataset, residuals_pooled_OLS], axis=1)
pooled_OLS_dataset = pooled_OLS_dataset.drop(['date'], axis=1).fillna(0)
exog = sm.tools.tools.add_constant(dataset[X_cols]).fillna(0)
white_test_results = het_white(pooled_OLS_dataset['residual'], exog)
labels = ['LM-Stat', 'LM p-val', 'F-Stat', 'F p-val'] 
print(dict(zip(labels, white_test_results)))
# 3A.3 Breusch-Pagan-Test
breusch_pagan_test_results = het_breuschpagan(pooled_OLS_dataset['residual'], exog)
labels = ['LM-Stat', 'LM p-val', 'F-Stat', 'F p-val'] 
print(dict(zip(labels, breusch_pagan_test_results)))

# 3.B Non-Autocorrelation
# Durbin-Watson-Test
""" The Durbin-Watson-Test will have one output between 0 – 4. 
The mean (= 2) would indicate that there is no autocorrelation identified, 
0 – 2 means positive autocorrelation (the nearer to zero the higher the correlation), 
and 2 – 4 means negative autocorrelation (the nearer to four the higher the correlation)
"""
from statsmodels.stats.stattools import durbin_watson
durbin_watson_test_results = durbin_watson(pooled_OLS_dataset['residual']) 
print(durbin_watson_test_results) # 0.1499983534207501: strong positive autocorrelation


"""Perform FE- and RE-model"""
from linearmodels import PanelOLS
from linearmodels import RandomEffects
exog = sm.tools.tools.add_constant(dataset[X_variables])
endog = dataset[Y_variable]
# random effects model
model_re = RandomEffects(endog, exog) 
re_res = model_re.fit() 
print(re_res)
# fixed effects model
model_fe = PanelOLS(endog, exog, entity_effects = True) 
fe_res = model_fe.fit() 
#print results
print(fe_res)

"""RE or FE? Hausman-Test
p very small -> null hypothesis is rejected, so have endogeneity, so choose FE
"""
import numpy.linalg as la
from scipy import stats
import numpy as np
def hausman(fe, re):
    b = fe.params
    B = re.params
    v_b = fe.cov
    v_B = re.cov
    df = b[np.abs(b) < 1e8].size
    chi2 = np.dot((b - B).T, la.inv(v_b - v_B).dot(b - B)) 
    pval = stats.chi2.sf(chi2, df)
    return chi2, df, pval
hausman_results = hausman(fe_res, re_res) 
print('chi-Squared: ' + str(hausman_results[0]))
print('degrees of freedom: ' + str(hausman_results[1]))
print('p-Value: ' + str(hausman_results[2]))



