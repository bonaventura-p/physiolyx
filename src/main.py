# [START ]

import pandas as pd
from google.cloud import storage

from helpers.analytics import tableProcessor, predWrapper
from helpers.filemanager import blobDownloader, blobUploader, tableReader


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




