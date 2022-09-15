import datetime
import os

import numpy as np
import pandas as pd

import feature_engine.missing_data_imputers as mdi
from feature_engine import categorical_encoders as ce
from feature_engine import variable_transformers as vt

from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn import linear_model
from sklearn import model_selection

from azure.storage.blob import ContainerClient

start_time = datetime.datetime.now()

connection_string = os.environ.get('STORAGE_ACCOUNT_KEY')
container_name = 'auto'

data_path = 'data/'
data_file = 'auto.xlsx'

model_path = 'models/'
model_name = f"auto_model_{start_time.strftime('%Y-%m-%d--%H-%M-%S')}.pickle"

def read_input_data(data_path, data_file, connection_string, container_name):
    ''' Downloads data file from container, import its data to a
        pandas DataFrame and return it.
        
        Data file must be a CSV,
        container URL and key must be from Azure Blob Storage.
    '''
    
    # instaciate container client 
    container_client = ContainerClient.from_connection_string(
                                        conn_str= connection_string,
                                        container_name= container_name)
    
    # create a handle to the blob file
    blob_client = container_client.get_blob_client(blob = data_path + data_file)
    
    # download blob to local
    with open(data_path + data_file, 'wb') as my_file:
        blob_data = blob_client.download_blob()
        blob_data.readinto(my_file)
        
    # close used handle
    blob_client.close()
    container_client.close()
    
    # read file to DataFrame
    dataframe = pd.read_excel(data_path + data_file)
    
    return dataframe

def save_model(model, model_path, model_name, connection_string, container_name):
    ''' First, serialize model object to local disk in a pickle file,
        then, upload to Azure Blob Storage
    '''
    
    model.to_pickle(model_path + model_name)
    
    # instaciate container client 
    container_client = ContainerClient.from_connection_string(
                                        conn_str= connection_string,
                                        container_name= container_name)
    
    # create a handle to the blob file
    blob_client = container_client.get_blob_client(blob = model_path + model_name)
    
    # if check if file exists before upload it
    if not blob_client.exists():
        with open(model_path + model_name, 'rb') as my_file:
            blob_client.upload_blob(my_file)
            
    
    # close used handle
    blob_client.close()
    container_client.close()
    print('Upload completed!')

def run_train(data_path, data_file, model_path, model_name,
            connection_string, container_name):
    '''Run the complete training pipeline:
        Download and import input data
        Create calculated features
        Optimize model parameters
        Fit and assess mod
        Select model champion
        Serialize model to local disk
        Upload model do cloud
    '''
    
    # donwlod and import input data
    auto_df = read_input_data(data_path, data_file, connection_string, container_name)
    
    # create calculated features
    auto_df['cylinders_displacement'] = auto_df['cylinders'] * auto_df['displacement']
    auto_df['specif_torque'] = auto_df['horsepower'] / auto_df['cylinders_displacement']
    auto_df['fake_torque'] = auto_df['weight'] / auto_df['acceleration']
    
    # define features types
    target = 'mpg' # Milhas por galão
    num_vars = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 
                'year', 'cylinders_displacement', 'specif_torque', 'fake_torque']
    cat_vars = ['origin']
    auto_df[cat_vars] = auto_df[cat_vars].astype(str)
    
    # spli train and test subsets
    X_train, X_test, y_train, y_test = model_selection.train_test_split( auto_df[num_vars+cat_vars],
                                                                     auto_df[target],
                                                                     random_state=1992,
                                                                     test_size=0.25)
    
    #setup pipeline
    log = vt.LogTransformer(variables=num_vars) # Define o transformador do transformação logaritmica
    onehot = ce.OneHotCategoricalEncoder(variables=cat_vars, drop_last=True) # Cria Dummys
    model = linear_model.Lasso() # Definição do modelo

    full_pipeline = Pipeline( steps=[
        ("log", log),
        ("onehot", onehot),
        ('model', model) ] )

    param_grid = { 'model__alpha':[0.0167, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 0.8, 1], # linspace
                   'model__normalize':[True],
                   'model__random_state':[1992]}

    search = model_selection.GridSearchCV(full_pipeline,
                                              param_grid,
                                              cv=5,
                                              n_jobs=-1,
                                              scoring='neg_root_mean_squared_error')

    search.fit(X_train, y_train) # Executa o treinamento!!

    best_model = search.best_estimator_
    
    #input results to pandas dataset
    cv_result = pd.DataFrame(search.cv_results_) # Pega resultdos do grid
    cv_result = cv_result.sort_values(by='mean_test_score', ascending = False,)
    
    # Verificando erro na base de teste
    y_test_pred = best_model.predict(X_test)
    root_mean_squadred_erro = metrics.mean_squared_error( y_test, y_test_pred) ** (1/2)
    #print( "Raiz do Erro Quadrático Médio:", root_mean_squadred_erro)
    
    # refit model
    best_model.fit( auto_df[num_vars+cat_vars], auto_df[target] )
    
    # create pandas.Series with champion model
    model_s = pd.Series( {"cat_vars":cat_vars,
                      "num_vars":num_vars,
                      "fit_vars": X_train.columns.tolist(),
                      "model":best_model,
                      "rmse":root_mean_squadred_erro} )
    
    save_model(model_s, model_path, model_name, connection_string, container_name)

if __name__ == '__main__':
    run_train(data_path, data_file, model_path, model_name,
            connection_string, container_name)

    elapsed_time = datetime.datetime.now() - start_time
    print(f'Full process complete in: {elapsed_time.total_seconds()} seconds')