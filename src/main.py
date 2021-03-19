
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 16:06:56 2021

@author: Cate
"""

from ICC import ICC21
import pandas as pd
import os
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, train_test_split

source_folder = r'../data/'
df_E = pd.read_excel(os.path.join(source_folder, 'Myo_radiomics.xlsx'), sheet_name='Foglio1')
df_GL = pd.read_excel(os.path.join(source_folder, 'Myo_radiomics_GL.xlsx'), sheet_name = 'Foglio1')

ctd = df_E.columns[0:23] #controllare il numero fino a quella da eliminare
df_E = df_E.drop(columns=ctd)
df_GL = df_GL.drop(columns=ctd)

#Riproducibilità delle features (alla fine elimina le non riproducibili da df_E) 
cases = np.array(df_GL['Case'])
df_E_cut = df_E.loc[df_E['Case'].isin(cases)]
df_GL = df_GL.drop(columns='Case')
cols_rep = df_GL.columns
print('Original number of features: ', len(cols_rep))
for col_rep in cols_rep:
    icc = ICC21(df_E_cut[col_rep],df_GL[col_rep])
    if (abs(icc) <= 0.80):
        del(df_E[col_rep])
print('Reproducible features: ', len(df_E.columns)-1) #-1 per la colonna Case

#%% Outcome loading
df_myo = pd.read_excel(os.path.join(source_folder, 'TAmyocarditis_01-03-2021.xlsx'), sheet_name='PazFea')
outcome = df_myo['Scar Post']
del(df_E['Case'])
df = df_E
scaler = preprocessing.StandardScaler()
df_sc = scaler.fit_transform(df)
df_sc = pd.DataFrame(df_sc, columns = df_E.columns)

#%% Intercorrelate
corr = df_sc.corr().abs()
corr = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
to_drop = [column for column in corr.columns if any(corr[column]>0.8)]
df_sc = df_sc.drop(columns=to_drop)
print('No high intercorrelations (r>0.8): ', len(df_sc.columns))

#%% Train-test split
X_train, X_test, y_train, y_test = train_test_split(df_sc, outcome, test_size = 0.15, random_state = 42)
X_train = pd.DataFrame(X_train, columns = df_sc.columns)
X_test = pd.DataFrame(X_test, columns = df_sc.columns)
X_train = pd.DataFrame(X_train, columns = df_sc.columns)
X_test = pd.DataFrame(X_test, columns = df_sc.columns)




