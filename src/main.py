# [START ]

import pandas as pd #to_csv
from google.cloud import storage

import sys

sys.path.append('./src')

from helpers.analytics import tableProcessor, predWrapper
from helpers.filemanager import blobDownloader, blobUploader, tableReader



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

    print('Bucket: {}'.format(data['bucket']))
    print('File: {}'.format(data['name']))

    blobDownloader(data['bucket'], data['name'], '/tmp/file.txt')

    table_data = tableReader('/tmp/file.txt', cols=None)

    table_data = tableProcessor(table_data, data, serve= True)

    ###add KDE

    table_data['predAction'] = predWrapper(table_data, bucket=data['bucket'],timeSteps=72)

    table_data.to_csv('/tmp/test.csv', header=True, index=False) #temp becomes the infile

    buckets_dict = {'physio-bucket':'physio-out-bucket','test-physio-bucket':'test-physio-out-bucket'}

    blobUploader(buckets_dict[data['bucket']], '/tmp/test.csv', str(data['name'][:22]+'.csv'))



# [END ]
