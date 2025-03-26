import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sklearn

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from xgboost import XGBRegressor

def create_table(df, region, field):
    
    df_m = []
    keys = list(df.keys())
    
    for i in range(len(keys)):
        df_m.append(df[keys[i]].dropna(subset=[region + '_naam'])[[region + '_naam', region + '_' + field]])
        df_m[i].columns = ['naam', keys[i]] 
        if i == 0:
            table = df_m[0]
        else:
            table = pd.merge(table, df_m[i], how='outer', on='naam')
    
    table.loc[:, table.columns != 'naam'] = table.loc[:, table.columns != 'naam'].fillna(0).astype(int)

    return table


def create_pct_table(df, region):

    df_m = []
    keys = list(df.keys())
    
    with warnings.catch_warnings():
        
        warnings.simplefilter("ignore")
        
        for i in range(len(keys)):
            df_m.append(df[keys[i]].dropna(subset=[region + '_naam'])[[region + '_naam', region + '_aantal']])
            df_m[i].columns = ['naam', keys[i]] 
            df_m[i][keys[i]] = 100*df_m[i][keys[i]]/(df_m[i][keys[i]].sum())
            if i == 0:
                table = df_m[0]
            else:
                table = pd.merge(table, df_m[i], how='outer', on='naam')
        
        table.loc[:, table.columns != 'naam'] = table.loc[:, table.columns != 'naam'].fillna(0)

    return table


def plot_name(df, names):

    plt.figure(figsize=(9, 4))
    years_list = np.arange(1995, 2024, 1)
    #years_list = list(df.keys())

    for name in names:
        if df['naam'].isin([name]).any():
            
            row = df[df['naam'] == name].drop(['naam','1995-2023'], axis= 1).values.flatten()
            plt.plot(years_list.flatten(), np.ma.masked_where(row == 0, row), marker='o', markerfacecolor='red',linestyle='-', ms=4, lw=1.3, label=name);

        else:
            print('The name ' + name + ' does not appear in the database you selected.')

    plt.grid(alpha=0.7);
    plt.xticks(fontsize=8);
    plt.yticks(fontsize=8);
    plt.gca().set_ylim(bottom=0);
    plt.xlim(1994, 2024);
    plt.legend();


def regplot_poly(df, naam, d):

    keys = list(df.keys())[1:]

    row = pd.DataFrame(df[df['naam'] == naam].iloc[0,1:]).reset_index()
    row.columns = ['jaar', 'aantal']
    row['jaar'] = np.arange(0, df.shape[1]-1, 1)
    
    y = row['aantal']
    X = row.drop('aantal', axis=1)
        
    polynomial_converter = PolynomialFeatures(degree=d, include_bias=True)
    polynomial_converter.fit(X)
    X_poly = polynomial_converter.transform(X)
    
    model = LinearRegression()
    model.fit(X_poly, y)
    y_pred = model.predict(X_poly)

    plt.figure(figsize=(10,3))
    plt.plot(keys, y, linestyle='--', linewidth=0.8, color='k', marker='o', ms=3);
    plt.plot(keys, y_pred, lw=1.5);
    plt.grid(ls='--');
    plt.xticks(fontsize=8, rotation=45);
    plt.yticks(fontsize=8);

    RMSE = round(mean_squared_error(y, y_pred)**(1/2),3)
    print(f'RMSE: {RMSE}')
    print('')


def regplot_gaussian(df, naam):

    keys = list(df.keys())[1:]

    row = pd.DataFrame(df[df['naam'] == naam].iloc[0,1:]).reset_index()
    row.columns = ['jaar', 'aantal']
    row['jaar'] = np.arange(0, df.shape[1]-1, 1)

    y = row['aantal']
    X = row.drop('aantal', axis=1)
    
    def gaussian(x, b, a, mu, sigma):
        return b + a * np.exp(-((x - mu)**2) / (2 * sigma**2))

    params, _ = curve_fit(gaussian, row['jaar'], row['aantal'], p0=[0, 250, 28, 1])

    b_fit, a_fit, mu_fit, sigma_fit = params

    y_pred = b_fit + a_fit*np.exp(-((X['jaar'] - mu_fit)**2) / (2*sigma_fit**2))

    plt.figure(figsize=(10,3))
    plt.plot(keys, y, linestyle='--', linewidth=0.8, color='k', marker='o', ms=3);
    plt.plot(keys, y_pred, lw=1.5);
    plt.grid(ls='--');
    plt.xticks(fontsize=8, rotation=45);
    plt.yticks(fontsize=8);

    RMSE = round(mean_squared_error(y, y_pred)**(1/2),3)
    print(f'RMSE: {RMSE}')
    print('')


def regplot_lag(df, naam, n_lags):

    keys = list(df.keys())[1:]

    row = pd.DataFrame(df[df['naam'] == naam].iloc[0,1:]).reset_index()
    row.columns = ['jaar', 'aantal']
    row['jaar'] = np.arange(0, df.shape[1]-1, 1)

    for i in range(n_lags):
        
        row[f'lag_{i+1}'] = row['aantal'].shift(i+1)

    pd.set_option('future.no_silent_downcasting', True)
    row.bfill(inplace=True) 
    row.infer_objects(copy=False)
    
    y = row['aantal']
    X = row.drop(['jaar', 'aantal'], axis=1)
        
    polynomial_converter = PolynomialFeatures(degree=1, include_bias=True)
    polynomial_converter.fit(X)
    X_poly = polynomial_converter.transform(X)
    
    model = LinearRegression()
    model.fit(X_poly, y)
    y_pred = model.predict(X_poly)

    plt.figure(figsize=(10,3))
    plt.plot(keys, y, linestyle='--', linewidth=0.8, color='k', marker='o', ms=3);
    plt.plot(keys, y_pred, lw=1.5);
    plt.grid(ls='--');
    plt.xticks(fontsize=8, rotation=45);
    plt.yticks(fontsize=8);

    RMSE = round(mean_squared_error(y, y_pred)**(1/2),3)
    print(f'RMSE: {RMSE}')
    print('')


def regplot_poly_lag(df, naam, d, n_lags):

    keys = list(df.keys())[1:]

    row = pd.DataFrame(df[df['naam'] == naam].iloc[0,1:]).reset_index()
    row.columns = ['jaar', 'aantal']
    row['jaar'] = np.arange(0, df.shape[1]-1, 1)
    
    y = row['aantal']
    X = row.drop('aantal', axis=1)
        
    polynomial_converter = PolynomialFeatures(degree=d, include_bias=True)
    polynomial_converter.fit(X)
    X_features = polynomial_converter.transform(X)

    pd.set_option('future.no_silent_downcasting', True)
    
    for i in range(n_lags):

        y_hat = y.shift(i+1)
        y_hat.bfill(inplace=True) 
        y_hat.infer_objects(copy=False)
        y_hat = np.array(y_hat).reshape(-1, 1)
        
        X_features = np.concatenate((X_features, y_hat), axis=1)
    
    model = LinearRegression()
    model.fit(X_features, y)
    y_pred = model.predict(X_features)

    plt.figure(figsize=(10,3))
    plt.plot(keys, y, linestyle='--', linewidth=0.8, color='k', marker='o', ms=3);
    plt.plot(keys, y_pred, lw=1.5);
    plt.grid(ls='--');
    plt.xticks(fontsize=8, rotation=45);
    plt.yticks(fontsize=8);

    RMSE = round(mean_squared_error(y, y_pred)**(1/2),3)
    print(f'RMSE: {RMSE}')
    print('')


def regplot_fourier_lag(df, naam, periods, n_lags):

    keys = list(df.keys())[1:]

    row = pd.DataFrame(df[df['naam'] == naam].iloc[0,1:]).reset_index()
    row.columns = ['jaar', 'aantal']
    row['jaar'] = np.arange(0, df.shape[1]-1, 1)
    
    y = row['aantal']
    X = row.drop('aantal', axis=1)

    fourier_terms = np.zeros((len(df.iloc[0,:])-1, 2*len(periods)))
    
    for i in range(len(df.iloc[0,:])-1):
        for k in range(len(periods)):

            fourier_terms[i,2*k] = np.sin(i/((len(df.iloc[0,:])-2)/periods[k]) *2*np.pi)
            fourier_terms[i,2*k+1] = np.cos(i/((len(df.iloc[0,:])-2)/periods[k]) *2*np.pi)     

    X_features = np.concatenate((X, fourier_terms), axis=1)

    pd.set_option('future.no_silent_downcasting', True)
    
    for i in range(n_lags):

        y_hat = y.shift(i+1)
        y_hat.bfill(inplace=True) 
        y_hat.infer_objects(copy=False)
        y_hat = np.array(y_hat).reshape(-1, 1)
        
        X_features = np.concatenate((X_features, y_hat), axis=1)

    model = LinearRegression()
    model.fit(X_features, y)
    y_pred = model.predict(X_features)

    plt.figure(figsize=(10,3))
    plt.plot(keys, y, linestyle='--', linewidth=0.8, color='k', marker='o', ms=3);
    plt.plot(keys, y_pred, lw=1.5);
    plt.grid(ls='--');
    plt.xticks(fontsize=8, rotation=45);
    plt.yticks(fontsize=8);

    RMSE = round(mean_squared_error(y, y_pred)**(1/2),3)
    print(f'RMSE: {RMSE}')
    print('')


def regplot_poly_fourier_lag(df, naam, d, periods, n_lags):

    keys = list(df.keys())[1:]

    row = pd.DataFrame(df[df['naam'] == naam].iloc[0,1:]).reset_index()
    row.columns = ['jaar', 'aantal']
    row['jaar'] = np.arange(0, df.shape[1]-1, 1)
    
    y = row['aantal']
    X = row.drop('aantal', axis=1)

    polynomial_converter = PolynomialFeatures(degree=d, include_bias=True)
    polynomial_converter.fit(X)
    X_features = polynomial_converter.transform(X)

    fourier_terms = np.zeros((len(df.iloc[0,:])-1, 2*len(periods)))
    
    for i in range(len(df.iloc[0,:])-1):
        for k in range(len(periods)):

            fourier_terms[i,2*k] = np.sin(i/((len(df.iloc[0,:])-2)/periods[k]) *2*np.pi)
            fourier_terms[i,2*k+1] = np.cos(i/((len(df.iloc[0,:])-2)/periods[k]) *2*np.pi)     

    X_features = np.concatenate((X_features, fourier_terms), axis=1)

    pd.set_option('future.no_silent_downcasting', True)
    
    for i in range(n_lags):

        y_hat = y.shift(i+1)
        y_hat.bfill(inplace=True) 
        y_hat.infer_objects(copy=False)
        y_hat = np.array(y_hat).reshape(-1, 1)
        
        X_features = np.concatenate((X_features, y_hat), axis=1)

    model = LinearRegression()
    model.fit(X_features, y)
    y_pred = model.predict(X_features)

    plt.figure(figsize=(10,3))
    plt.plot(keys, y, linestyle='--', linewidth=0.8, color='k', marker='o', ms=3);
    plt.plot(keys, y_pred, lw=1.5);
    plt.grid(ls='--');
    plt.xticks(fontsize=8, rotation=45);
    plt.yticks(fontsize=8);

    RMSE = round(mean_squared_error(y, y_pred)**(1/2),3)
    print(f'RMSE: {RMSE}')
    print('')


def errorplot_poly(df, d):

    RMSE = np.zeros(len(d))
    RMSE_STD = np.zeros(len(d))
    RMSE_point = np.zeros(len(df))
    
    for j in range(len(d)):
        for l in range(len(df)):
                
            row = pd.DataFrame(df.iloc[l,1:]).reset_index()
            row.columns = ['jaar', 'aantal']
            row['jaar'] = np.arange(0, df.shape[1]-1, 1)
        
            y = row['aantal']
            X = row.drop('aantal', axis=1)
            
            polynomial_converter = PolynomialFeatures(degree=d[j], include_bias=True)
            polynomial_converter.fit(X)
            X_features = polynomial_converter.transform(X)

            model = LinearRegression()
            model.fit(X_features, y)
            y_pred = model.predict(X_features)

            RMSE[j] += 1/len(df)*mean_squared_error(y, y_pred)**(1/2)
            RMSE_point[l] = mean_squared_error(y, y_pred)**(1/2) 

        RMSE_STD[j] = RMSE_point.std()
        
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,4));
    
    axes[0].plot(d, RMSE)
    axes[0].set_xlabel('polynomial degree');
    axes[0].set_title('average RMSE', fontsize=10);
    axes[0].tick_params(axis='both', labelsize=8);
    axes[0].grid(alpha=0.7);
    
    axes[1].plot(d, RMSE_STD)
    axes[1].set_xlabel('polynomial degree');
    axes[1].set_title('standard deviation of RMSE', fontsize=10);
    axes[1].tick_params(axis='both', labelsize=8);
    axes[1].grid(alpha=0.7); 
    
    fig.subplots_adjust(wspace=0.3,hspace=0.3)
    
    return RMSE, RMSE_STD


def errorplot_lag(df, lags):

    RMSE = np.zeros(len(lags))
    RMSE_STD = np.zeros(len(lags))
    RMSE_point = np.zeros(len(df))
    
    for j in range(len(lags)):
        for l in range(len(df)):
            
            row = pd.DataFrame(df.iloc[l,1:]).reset_index()
            row.columns = ['jaar', 'aantal']
            row['jaar'] = np.arange(0, df.shape[1]-1, 1)
        
            y = row['aantal']
            X_features = row.drop('aantal', axis=1)

            pd.set_option('future.no_silent_downcasting', True)
            
            for i in range(lags[j]):
    
                y_hat = y.shift(i+1)
                y_hat.bfill(inplace=True) 
                y_hat.infer_objects(copy=False)
                y_hat = np.array(y_hat).reshape(-1, 1)
                
                X_features = np.concatenate((X_features, y_hat), axis=1)

            model = LinearRegression()
            model.fit(X_features, y)
            y_pred = model.predict(X_features)

            RMSE[j] += 1/len(df)*mean_squared_error(y, y_pred)**(1/2)
            RMSE_point[l] = mean_squared_error(y, y_pred)**(1/2) 

        RMSE_STD[j] = RMSE_point.std()
            
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,4));
    
    axes[0].plot(lags, RMSE)
    axes[0].set_xlabel('lags');
    axes[0].set_title('average RMSE', fontsize=10);
    axes[0].tick_params(axis='both', labelsize=8);
    axes[0].grid(alpha=0.7);
    
    axes[1].plot(lags, RMSE_STD)
    axes[1].set_xlabel('lags');
    axes[1].set_title('standard deviation of RMSE', fontsize=10);
    axes[1].tick_params(axis='both', labelsize=8);
    axes[1].grid(alpha=0.7); 
    
    fig.subplots_adjust(wspace=0.3,hspace=0.3)

    return RMSE, RMSE_STD


def errorplot_fourier(df, periods):

    RMSE = np.zeros(len(periods))
    RMSE_STD = np.zeros(len(periods))
    RMSE_point = np.zeros(len(df))                        
                             
    for j in range(len(periods)):

        fourier_terms = np.zeros((len(df.iloc[0,:])-1, 2*(j+1)))
    
        for i in range(len(df.iloc[0,:])-1):
            for k in range(j+1):
    
                fourier_terms[i,2*k] = np.sin(i/((len(df.iloc[0,:])-2)/periods[k]) *2*np.pi)
                fourier_terms[i,2*k+1] = np.cos(i/((len(df.iloc[0,:])-2)/periods[k]) *2*np.pi)     

        for l in range(len(df)):
            
            row = pd.DataFrame(df.iloc[l,1:]).reset_index()
            row.columns = ['jaar', 'aantal']
            row['jaar'] = np.arange(0, df.shape[1]-1, 1)
        
            y = row['aantal']
            X = row.drop('aantal', axis=1)
            
            X_features = np.concatenate((X, fourier_terms), axis=1)

            model = LinearRegression()
            model.fit(X_features, y)
            y_pred = model.predict(X_features)

            RMSE[j] += 1/len(df)*mean_squared_error(y, y_pred)**(1/2)
            RMSE_point[l] = mean_squared_error(y, y_pred)**(1/2) 

        RMSE_STD[j] = RMSE_point.std()

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,4));

    axes[0].plot(periods, RMSE)
    axes[0].set_xlabel('periods');
    axes[0].set_title('average RMSE', fontsize=10);
    axes[0].tick_params(axis='both', labelsize=8);
    axes[0].grid(alpha=0.7);
    
    axes[1].plot(periods, RMSE_STD)
    axes[1].set_xlabel('periods');
    axes[1].set_title('standard deviation of RMSE', fontsize=10);
    axes[1].tick_params(axis='both', labelsize=8);
    axes[1].grid(alpha=0.7); 
    
    fig.subplots_adjust(wspace=0.3,hspace=0.3)

    return RMSE, RMSE_STD


def errorplot_poly_lag(df, d, lags):

    RMSE = np.zeros((len(d), len(lags)))
    RMSE_STD = np.zeros((len(d), len(lags)))
    RMSE_point = np.zeros(len(df))
    
    for j in range(len(d)):
        for k in range(len(lags)):

            for l in range(len(df)):
                
                row = pd.DataFrame(df.iloc[l,1:]).reset_index()
                row.columns = ['jaar', 'aantal']
                row['jaar'] = np.arange(0, df.shape[1]-1, 1)
            
                y = row['aantal']
                X = row.drop('aantal', axis=1)
                
                polynomial_converter = PolynomialFeatures(degree=d[j], include_bias=True)
                polynomial_converter.fit(X)
                X_features = polynomial_converter.transform(X)

                pd.set_option('future.no_silent_downcasting', True)  

                for i in range(lags[k]):
        
                    y_hat = y.shift(i+1)
                    y_hat.bfill(inplace=True) 
                    y_hat.infer_objects(copy=False)
                    y_hat = np.array(y_hat).reshape(-1, 1)
                    
                    X_features = np.concatenate((X_features, y_hat), axis=1)

                model = LinearRegression()
                model.fit(X_features, y)
                y_pred = model.predict(X_features)

                RMSE[j,k] += 1/len(df)*mean_squared_error(y, y_pred)**(1/2)
                RMSE_point[l] = mean_squared_error(y, y_pred)**(1/2) 

            RMSE_STD[j,k] = RMSE_point.std()
            
    sns.heatmap(RMSE, cmap='coolwarm', xticklabels=lags, yticklabels=d);
    plt.xlabel('lags');
    plt.ylabel('polynomial degree');
    plt.title('Average RMSE');
    
    return RMSE, RMSE_STD


def errorplot_fourier_lag(df, periods, lags):

    RMSE = np.zeros((len(periods), len(lags)))
    RMSE_STD = np.zeros((len(periods), len(lags)))
    RMSE_point = np.zeros(len(df))                        
                             
    for j in range(len(periods)):

        fourier_terms = np.zeros((len(df.iloc[0,:])-1, 2*(j+1)))
    
        for i in range(len(df.iloc[0,:])-1):
            for k in range(j+1):
    
                fourier_terms[i,2*k] = np.sin(i/((len(df.iloc[0,:])-2)/periods[k]) *2*np.pi)
                fourier_terms[i,2*k+1] = np.cos(i/((len(df.iloc[0,:])-2)/periods[k]) *2*np.pi)     
                
        for k in range(len(lags)):

            for l in range(len(df)):
                
                row = pd.DataFrame(df.iloc[l,1:]).reset_index()
                row.columns = ['jaar', 'aantal']
                row['jaar'] = np.arange(0, df.shape[1]-1, 1)
            
                y = row['aantal']
                X = row.drop('aantal', axis=1)
                
                X_features = np.concatenate((X, fourier_terms), axis=1)
                
                pd.set_option('future.no_silent_downcasting', True)
                
                for i in range(lags[k]):
        
                    y_hat = y.shift(i+1)
                    y_hat.bfill(inplace=True) 
                    y_hat.infer_objects(copy=False)
                    y_hat = np.array(y_hat).reshape(-1, 1)
                    
                    X_features = np.concatenate((X_features, y_hat), axis=1)
        
                model = LinearRegression()
                model.fit(X_features, y)
                y_pred = model.predict(X_features)

                RMSE[j,k] += 1/len(df)*mean_squared_error(y, y_pred)**(1/2)
                RMSE_point[l] = mean_squared_error(y, y_pred)**(1/2) 

            RMSE_STD[j,k] = RMSE_point.std()
            
    sns.heatmap(RMSE, cmap='coolwarm', xticklabels=lags, yticklabels=periods);
    plt.xlabel('lags');
    plt.ylabel('periods');
    plt.title('Average RMSE');
    
    return RMSE, RMSE_STD


def errorplot_poly_fourier_lag(df, d, periods, lags):

    RMSE = np.zeros((len(d), len(periods), len(lags)))
    RMSE_STD = np.zeros((len(d), len(periods), len(lags)))
    RMSE_point = np.zeros(len(df))                        
                             
    for m in range(len(d)):

        polynomial_converter = PolynomialFeatures(degree=d[m], include_bias=True)
        
        for j in range(len(periods)):
    
            fourier_terms = np.zeros((len(df.iloc[0,:])-1, 2*(j+1)))
        
            for i in range(len(df.iloc[0,:])-1):
                for k in range(j+1):
        
                    fourier_terms[i,2*k] = np.sin(i/((len(df.iloc[0,:])-2)/periods[k]) *2*np.pi)
                    fourier_terms[i,2*k+1] = np.cos(i/((len(df.iloc[0,:])-2)/periods[k]) *2*np.pi)     
                    
            for k in range(len(lags)):
                for l in range(len(df)):
                    
                    row = pd.DataFrame(df.iloc[l,1:]).reset_index()
                    row.columns = ['jaar', 'aantal']
                    row['jaar'] = np.arange(0, df.shape[1]-1, 1)
                
                    y = row['aantal']
                    X = row.drop('aantal', axis=1)
    
                    polynomial_converter.fit(X)
                    X_features = polynomial_converter.transform(X)
                    X_features = np.concatenate((X_features, fourier_terms), axis=1)
                    
                    pd.set_option('future.no_silent_downcasting', True)        
                    
                    for i in range(lags[k]):
            
                        y_hat = y.shift(i+1)
                        y_hat.bfill(inplace=True) 
                        y_hat.infer_objects(copy=False)
                        y_hat = np.array(y_hat).reshape(-1, 1)
                        
                        X_features = np.concatenate((X_features, y_hat), axis=1)
    
                    model = LinearRegression()
                    model.fit(X_features, y)
                    y_pred = model.predict(X_features)
    
                    RMSE[m,j,k] += 1/len(df)*mean_squared_error(y, y_pred)**(1/2)
                    RMSE_point[l] = mean_squared_error(y, y_pred)**(1/2) 
    
                RMSE_STD[m,j,k] = RMSE_point.std()
    
    return RMSE, RMSE_STD


def predict_pfl_recur(df, naam, steps, d, periods, lags):

    row = pd.DataFrame(df[df['naam'] == naam].iloc[0,1:]).reset_index()
    row.columns = ['jaar', 'aantal']
    N = len(row)
    
    X_0 = np.arange(0, N+steps, 1).reshape(-1, 1)

    y = pd.Series([int(0)] * int(N+steps)) 
    y[:N] = row['aantal'][:N].astype(y.dtype)
    
    polynomial_converter = PolynomialFeatures(degree=d, include_bias=True)
    fourier_terms = np.zeros((N+steps, 2*len(periods)))

    polynomial_converter.fit(X_0)
    X_features = polynomial_converter.transform(X_0)
        
    for i in range(N+steps):
    
        for k in range(len(periods)):

            fourier_terms[i,2*k] = np.sin(i/((N-1)/periods[k]) *2*np.pi)
            fourier_terms[i,2*k+1] = np.cos(i/((N-1)/periods[k]) *2*np.pi)  

    X_features = np.concatenate((X_features, fourier_terms), axis=1)

    pd.set_option('future.no_silent_downcasting', True)        

    for i in range(lags):

        y_hat = y.shift(i+1)
        y_hat.bfill(inplace=True) 
        y_hat.infer_objects(copy=False)
        y_hat = np.array(y_hat).reshape(-1, 1)

        X_features = np.concatenate((X_features, y_hat), axis=1)

    X = X_features[:-steps]
    model = LinearRegression()
    model.fit(X, y[:N])  
    
    y_pred = pd.Series([int(0)] * int(N+steps)) 
    y_pred[:N] = model.predict(X).astype(y.dtype)
    
    for s in range(steps):
        
        y_pred[N+s] = model.predict(X_features[N+s].reshape(1, -1)).astype(y.dtype)[0]

        for t in range(steps - 1 - s):
            X_features[N+s+t+1, -lags+t] = y_pred[N+s]

    return y[:N], y_pred


def plot_pfl_recur(df, naam, steps, d, periods, lags):

    keys = list(df.keys())[1:]

    row = pd.DataFrame(df[df['naam'] == naam].iloc[0,1:]).reset_index()
    row.columns = ['jaar', 'aantal']
    row['jaar'] = np.arange(0, df.shape[1]-1, 1)
    y = row['aantal']
    X = row.drop('aantal', axis=1)
    
    y_0, y_pred = predict_pfl_recur(df.iloc[:,:-steps], naam, steps, d, periods, lags)

    x1 = keys[:-steps]
    y1 = y_pred[:-steps]

    x2 = keys[-steps:]
    y2 = y_pred[-steps:]

    plt.figure(figsize=(10,3))
    plt.plot(keys, y, linestyle='--', linewidth=0.8, color='k', marker='o', ms=3);
    plt.plot(x1, y1, lw=1.5, color='blue');
    plt.plot(x2, y2, lw=1.5, color='red');
    plt.plot([x1[-1], x2[0]], [y1.iloc[-1], y2.iloc[0]], color='red')
    plt.grid(ls='--');
    plt.xticks(fontsize=8, rotation=45);
    plt.yticks(fontsize=8);

    RMSE = round(mean_squared_error(y2, y[-steps:])**(1/2),3)
    print(f'RMSE: {RMSE}')
    print('')


def predict_pfl_dir(df, naam, steps, d, periods, lags):

    row = pd.DataFrame(df[df['naam'] == naam].iloc[0,1:]).reset_index()
    row.columns = ['jaar', 'aantal']
    N = len(row)
    
    X_0 = np.arange(0, N+steps, 1).reshape(-1, 1)

    y = pd.Series([int(0)] * int(N+steps)) 
    y[:N] = row['aantal'][:N].astype(y.dtype)
    
    polynomial_converter = PolynomialFeatures(degree=d, include_bias=True)
    fourier_terms = np.zeros((N+steps, 2*len(periods)))

    polynomial_converter.fit(X_0)
    X_features = polynomial_converter.transform(X_0)
        
    for i in range(N+steps):
    
        for k in range(len(periods)):

            fourier_terms[i,2*k] = np.sin(i/((N-1)/periods[k]) *2*np.pi)
            fourier_terms[i,2*k+1] = np.cos(i/((N-1)/periods[k]) *2*np.pi)  
 

    X_features = np.concatenate((X_features, fourier_terms), axis=1)

    pd.set_option('future.no_silent_downcasting', True)        

    for i in range(lags):

        y_hat = y.shift(i+1)
        y_hat.bfill(inplace=True) 
        y_hat.infer_objects(copy=False)
        y_hat = np.array(y_hat).reshape(-1, 1)

        X_features = np.concatenate((X_features, y_hat), axis=1)
    
    y_pred = pd.Series([int(0)] * int(N+steps)) 
    
    for s in range(steps):

        X = X_features[:N+s]
        model = LinearRegression()
        model.fit(X, y[:N+s])  

        if s == 0:
            y_pred[:N] = model.predict(X).astype(y.dtype)

        y_pred[N+s] = model.predict(X_features[N+s].reshape(1, -1)).astype(y.dtype)[0]
        y[N+s] = model.predict(X_features[N+s].reshape(1, -1)).astype(y.dtype)[0]
        
        for t in range(steps - 1 - s):
            X_features[N+s+t+1, -lags+t] = y_pred[N+s]

    return y[:N], y_pred


def predict_pfl_dir_boosted(df, naam, steps, d, periods, lags):

    row = pd.DataFrame(df[df['naam'] == naam].iloc[0,1:]).reset_index()
    row.columns = ['jaar', 'aantal']
    N = len(row)
    
    X_0 = np.arange(0, N+steps, 1).reshape(-1, 1)

    y = pd.Series([int(0)] * int(N+steps)) 
    y_pred = pd.Series([int(0)] * int(N+steps)) 

    y[:N] = row['aantal'][:N].astype(y.dtype)
    
    polynomial_converter = PolynomialFeatures(degree=d, include_bias=True)
    fourier_terms = np.zeros((N+steps, 2*len(periods)))

    polynomial_converter.fit(X_0)
    X_features = polynomial_converter.transform(X_0)
        
    for i in range(N+steps):
    
        for k in range(len(periods)):

            fourier_terms[i,2*k] = np.sin(i/((N-1)/periods[k]) *2*np.pi)
            fourier_terms[i,2*k+1] = np.cos(i/((N-1)/periods[k]) *2*np.pi)  
 
    X_features = np.concatenate((X_features, fourier_terms), axis=1)

    pd.set_option('future.no_silent_downcasting', True)        

    for i in range(lags):

        y_hat = y.shift(i+1)
        y_hat.bfill(inplace=True) 
        y_hat.infer_objects(copy=False)
        y_hat = np.array(y_hat).reshape(-1, 1)

        X_features = np.concatenate((X_features, y_hat), axis=1)
    
    for s in range(steps):

        X = X_features[:N+s]
        
        model = LinearRegression()
        model.fit(X, y[:N+s]) 

        y_fit = model.predict(X).astype(y.dtype)
        y_resid = y[:N+s] - y_fit
        
        xgb = XGBRegressor()
        xgb.fit(X, y_resid)

        if s == 0:
             y_pred[:N] = y_fit + xgb.predict(X).astype(y.dtype)

        y_pred[N+s] = model.predict(X_features[N+s].reshape(1, -1)).astype(y.dtype)[0] + xgb.predict(X_features[N+s].reshape(1,-1)).astype(y.dtype)[0]
        y[N+s] = model.predict(X_features[N+s].reshape(1, -1)).astype(y.dtype)[0] + xgb.predict(X_features[N+s].reshape(1,-1)).astype(y.dtype)[0]
        
        for t in range(steps - 1 - s):
            X_features[N+s+t+1, -lags+t] = y_pred[N+s]

    return y[:N], y_pred


def plot_pfl_dir(df, naam, steps, d, periods, lags):

    keys = list(df.keys())[1:]

    row = pd.DataFrame(df[df['naam'] == naam].iloc[0,1:]).reset_index()
    row.columns = ['jaar', 'aantal']
    row['jaar'] = np.arange(0, df.shape[1]-1, 1)
    y = row['aantal']
    X = row.drop('aantal', axis=1)
    
    y_0, y_pred = predict_pfl_dir(df.iloc[:,:-steps], naam, steps, d, periods, lags)

    x1 = keys[:-steps]
    y1 = y_pred[:-steps]

    x2 = keys[-steps:]
    y2 = y_pred[-steps:]

    plt.figure(figsize=(10,3))
    plt.plot(keys, y, linestyle='--', linewidth=0.8, color='k', marker='o', ms=3);
    plt.plot(x1, y1, lw=1.5, color='blue');
    plt.plot(x2, y2, lw=1.5, color='red');
    plt.plot([x1[-1], x2[0]], [y1.iloc[-1], y2.iloc[0]], color='red')
    plt.grid(ls='--');
    plt.xticks(fontsize=8, rotation=45);
    plt.yticks(fontsize=8);

    RMSE = round(mean_squared_error(y2, y[-steps:])**(1/2),3)
    print(f'RMSE: {RMSE}')
    print('')


def plot_pfl_dir_boosted(df, naam, steps, d, periods, lags):

    keys = list(df.keys())[1:]

    row = pd.DataFrame(df[df['naam'] == naam].iloc[0,1:]).reset_index()
    row.columns = ['jaar', 'aantal']
    row['jaar'] = np.arange(0, df.shape[1]-1, 1)
    y = row['aantal']
    X = row.drop('aantal', axis=1)
    
    y_0, y_pred = predict_pfl_dir_boosted(df.iloc[:,:-steps], naam, steps, d, periods, lags)

    x1 = keys[:-steps]
    y1 = y_pred[:-steps]

    x2 = keys[-steps:]
    y2 = y_pred[-steps:]

    plt.figure(figsize=(10,3))
    plt.plot(keys, y, linestyle='--', linewidth=0.8, color='k', marker='o', ms=3);
    plt.plot(x1, y1, lw=1.5, color='blue');
    plt.plot(x2, y2, lw=1.5, color='red');
    plt.plot([x1[-1], x2[0]], [y1.iloc[-1], y2.iloc[0]], color='red')
    plt.grid(ls='--');
    plt.xticks(fontsize=8, rotation=45);
    plt.yticks(fontsize=8);

    RMSE = round(mean_squared_error(y2, y[-steps:])**(1/2),3)
    print(f'RMSE: {RMSE}')
    print('')