# [START functions_monitorDataLambda]

import pandas as pd
from google.cloud import storage
import datetime
import numpy as np

client = storage.Client()


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

    bucket = client.get_bucket(data['bucket'])
    blob = bucket.get_blob(data['name'])

    table_data = tableReader(blob)

    table_data = tableProcess(table_data, data)

    ###8. KDE.py TBD

    ### 9. HAR.py TBD

    csvUploader(table_data, data)



def rotRescaler(val, D=360):
    '''add descr'''
    return np.where(val > D / 2, val - D, val)


def valDivider(table, k=1000):
    '''add descr'''

    return table / k


def metscoreCalc(scene, table):
    '''add descr'''
    #0 - Menu
    #1 - Gym(goalkeeping)
    #2 - TouchObject
    #3 - Fruit picking
    #4 - Gym_TimeTrials(goalkeeping time trials)
    #5 - Calibration
    #6 - Sorting
    metscore_dict = {0: 1, 1: 2.5, 2: 2.5, 3: 3, 4: 2.5, 5: 1, 6: 2}

    metscore = []
    for row in range(len(table)):
        metscore.append(metscore_dict[table[scene][row]] / 4320) #60*72

    return metscore

def tableReader(blob):
    '''add descr'''

    cols = ['index', 'scene_index', 'time', 'ms_lastline', 'head_posx', "head_posy", "head_posz", "head_rotx",
            "head_roty", "head_rotz", "right_posx", "right_posy", "right_posz", "right_rotx", "right_roty",
            "right_rotz", "left_posx", "left_posy", "left_posz", "left_rotx", "left_roty", "left_rotz", ]

    with open("/tmp/my-file.txt", "wb") as file_obj:
        blob.download_to_file(file_obj)

    table = pd.read_table("/tmp/my-file.txt", sep=',', header=0, names=cols)

    return table


def tableProcess(table, data):
    '''add descr'''
    '''data(dict): The Cloud Functions event payload.'''

    ### 2. divide /1000 ###
    table.loc[:, 'head_posx':] = valDivider(table.loc[:, 'head_posx':])

    ### 3. format time ###
    table['time'] = pd.to_datetime(table['time'], format="%H:%M:%S").dt.time

    ### 4. create date ###
    table['date'] = pd.to_datetime(data['name'][12:22], format = "%d-%m-%Y")

    ### 5. create seconds ###
    timedelta = pd.to_timedelta(table.time.astype(str))
    diff = timedelta.diff().fillna(pd.Timedelta(seconds=0)) / 1e9
    table['seconds'] = np.cumsum(diff).astype(int)

    ### 6. rescale rotations ###
    cols = ["head_rotx", "head_roty", "head_rotz", "right_rotx", "right_roty", "right_rotz",
            "left_rotx", "left_roty", "left_rotz"]

    for col in cols:
        table[str(col + "_n")] = rotRescaler(table[col])

    ### 7. compute MET score at frame level ###
    table['met_score'] = metscoreCalc('scene_index',table)

    return table


def csvUploader(table, data, bucket="physio-out-bucket"):
    '''data (dict): cloud functions event payload'''

    table.to_csv('/tmp/test.csv', header=True, index=False)

    outbucket = client.get_bucket(bucket)
    outname = data['name'][:22] + ".csv"
    outblob = outbucket.blob(outname)

    with open("/tmp/test.csv", "rb") as outfile:
        outblob.upload_from_file(outfile)

# [END functions_monitorDataLambda]
