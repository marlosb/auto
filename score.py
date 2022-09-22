import datetime
import os

from azure.storage.blob import ContainerClient
import pandas as pd
from sklearn import linear_model

start_time = datetime.datetime.now()
connection_string = os.environ.get('STORAGE_ACCOUNT_KEY')
container_name = 'auto'
model_path = 'models/'

def get_model(model_path, connection_string, container_name):
    ''' Function to list all files in the model_path directory,
        download the most recent one (assumes timestamp on name)
        and import the model
    '''

    # instaciate container client 
    container_client = ContainerClient.from_connection_string(
                                        conn_str= connection_string,
                                        container_name= container_name)
    
    # List all files on model_path
    models_list = []
    for model in container_client.list_blobs(name_starts_with=model_path):
        models_list.append(model['name'])
        
    # Sort all models and get last one
    models_list.sort()
    last_model_name = models_list[-1]
    
    # create a handle to the blob file
    blob_client = container_client.get_blob_client(blob = last_model_name)
    
    # download blob to local
    with open(last_model_name, 'wb') as my_file:
        blob_data = blob_client.download_blob()
        blob_data.readinto(my_file)
    
    # closes all handles
    blob_client.close()
    container_client.close()
    
    model = pd.read_pickle(last_model_name)
    
    return model, last_model_name

def get_score(param_dict, model_package):
    ''' Receive all features and score it
    '''
    
    # Extract used objects
    model = model_package.model
    columns = model_package.fit_vars
    
    # create calculated features
    if 'cylinders_displacement' in columns:
        param_dict['cylinders_displacement'] = (
                   param_dict['cylinders'] * param_dict['displacement'])
    if 'specif_torque' in columns:
        param_dict['cylinders_displacement'] = (
                   param_dict['cylinders'] * param_dict['displacement'])
        param_dict['specif_torque'] = (
        param_dict['horsepower'] / param_dict['cylinders_displacement'])
    if 'fake_torque' in columns:
        param_dict['fake_torque'] = (
            param_dict['weight'] / param_dict['acceleration'])
        
    # score 
    score = model.predict(pd.DataFrame.from_dict({1: param_dict}, 
                                                 orient='index'))
    
    return score

if __name__ == '__main__':
    # create test case
    input_dict = {'cylinders' : 8,
              'displacement': 320,
              'horsepower': 150,
              'weight': 3459,
              'acceleration': 11.0,
              'year': 70,
              'origin': 1}

    model_package, model_version = get_model(model_path, 
                                            connection_string, 
                                            container_name)

    result = get_score(input_dict, model_package)

    elapsed_time = datetime.datetime.now() - start_time

    result_dict = {'start timne': start_time,
               'model version': model_version,
               'input data': input_dict,
               'predicted score': result[0],
               'scoring time': elapsed_time.total_seconds()}

    print(result_dict)