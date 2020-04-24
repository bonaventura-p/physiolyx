# [START functions_HARclassifier]

import pandas as pd
import numpy as np

import datetime 

import os
import tempfile
from scipy import stats
import tensorflow as tf

from google.cloud import storage

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from tensorflow.keras import Model, initializers, optimizers, regularizers
from tensorflow.keras.layers import Input, Dense, Conv1D, Dropout, LSTM, TimeDistributed
from tensorflow.keras.callbacks import ModelCheckpoint


# We keep model as global variable so we don't have to reload it in case of warm invocations
model = None

class SimpleLSTM(Model):
  def __init__(self):
    super(SimpleLSTM, self).__init__()
    self.r1 = LSTM(32,input_shape=(timeSteps, n_features), activation='tanh', kernel_regularizer=regularizers.l2(0.02), return_sequences = True)
    self.r2 = LSTM(32, activation='tanh', kernel_regularizer=regularizers.l2(0.02),  return_sequences = False)
    self.sm = Dense(n_classes, activation='softmax')
    
  def call(self, x):
    x = self.r1(x)
    x = self.r2(x)
    
    return self.sm(x)



def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.get_blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print('Blob {} downloaded to {}.'.format(
        source_blob_name,
        destination_file_name))
    
def handler(bucket, name, ta):
    global model

    # Model load which only happens during cold starts
    if model is None:
        download_blob(BUCKETNAME, 'model_ep04_val0.25.index', '/tmp/model_ep04_val0.25.index')
        download_blob(BUCKETNAME, 'model_ep04_val0.25.data-00000-of-00001', '/tmp/model_ep04_val0.25.data-00000-of-00001')
        model = SimpleLSTM()
        model.load_weights('/tmp/model_ep04_val0.25')
    
    input_np = PREPROCESSINPUT
    predictions = model.call(input_np)
    print(predictions)
    
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        
    return class_names[numpy.argmax(predictions)]



def monitorDataHARLambda(data, context):
    """Background Cloud Function to be triggered by Cloud Storage.
       This generic function logs relevant data when a file is changed.

    Args:
        data (dict): The Cloud Functions event payload.
        context (google.cloud.functions.Context): Metadata of triggering event.
    Returns:
        None; the output is written to Stackdriver Logging
    """
    print('Event ID: {}'.format(context.event_id))
    print('Event type: {}'.format(context.event_type))
    print('Bucket: {}'.format(data['bucket']))
    print('File: {}'.format(data['name']))

    bucket = client.get_bucket(data['bucket'])
    blob = bucket.get_blob(data['name'])

    table_data = tableReader(blob)

    table_data = tableProcess(table_data, data)

    ###8. KDE.py TBD

    ### 9. HAR.py TBD

    csvUploader(table_data, data)


colslab = ['index','scene_index','time','ms_lastline','head_posx',"head_posy","head_posz","head_rotx","head_roty","head_rotz",
        "right_posx","right_posy","right_posz","right_rotx","right_roty","right_rotz",
        "left_posx","left_posy","left_posz","left_rotx","left_roty","left_rotz",'timedel','action']

download_blob(data['bucket'],data['name'],'file.txt')
table = tableReader('file.txt',cols=colslab)


#from analytics.parsing import analyze_session
#analytics is a folder, parsing is .py, analyse_session is def








# [END functions_HARclassifier]