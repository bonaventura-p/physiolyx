# [START functions_helloworld_storage_generic]

import pandas as pd
from google.cloud import storage

client = storage.Client()


def hello_gcs_generic(data, context):
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

    cols = ['index', 'scene_index', 'time', 'ms_lastline', 'head_posx', "head_posy", "head_posz", "head_rotx",
            "head_roty", "head_rotz","right_posx", "right_posy", "right_posz", "right_rotx", "right_roty", "right_rotz",
            "left_posx", "left_posy", "left_posz", "left_rotx", "left_roty", "left_rotz", ]

    bucket = client.get_bucket(data['bucket'])
    blob = bucket.get_blob(data['name'])

    with open("/tmp/my-file.txt", "wb") as file_obj:
        blob.download_to_file(file_obj)

    table_data = pd.read_table("/tmp/my-file.txt", sep=',', header=0, names=cols)

    table_data.to_csv('/tmp/test.csv', header=True, index= False)

    outbucket = client.get_bucket("physio-out-bucket")
    outname = data['name'][:22] + ".csv"
    outblob = outbucket.blob(outname)

    with open("/tmp/test.csv", "rb") as outfile:
        outblob.upload_from_file(outfile)

# [END functions_helloworld_storage_generic]


