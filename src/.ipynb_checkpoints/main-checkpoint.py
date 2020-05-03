# [START ]

import numpy as np
import pandas as pd
from google.cloud import storage
import datetime
import os
import tempfile
from scipy import stats
import tensorflow as tf

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from tensorflow.keras import Model, initializers, optimizers, regularizers
from tensorflow.keras.layers import Dense, Conv1D, Dropout, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint


import helpers.analytics
import helpers.filemanager
import helpers.model



client = storage.Client()

#add fix for warm invocations


def monitorDataLambda(data, context):
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

    blobDownloader(data['bucket'], data['name'], 'file.txt')
    
    table_data = tableReader('file.txt', cols=None)

    table_data = tableProcessor(table_data, data, serve= True)

    ###8. KDE.py TBD

    ### 9. HAR.py TBD
    table_data['predAction'] = predWrapper(table_data, bucket=data['bucket'],timeSteps=72)


    ### 10. export as csv to out-bucket
    table_data.to_csv('/tmp/test.csv', header=True, index=False) #temp becomes the infile
    
    blobUploader('physio-out-bucket', '/tmp/test.csv', str(data['name'][:22]+'.csv'))


    
# [END ]




