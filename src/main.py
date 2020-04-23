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

#table_data.loc[table_data.scene_index==key, 'scene_index'] = scene_dict[key]


def sceneDict(table):
    '''add descr'''
    scene_dict = {0: 'Menu', 1: 'Goalkeeping', 2: 'Touch object', 3: 'Fruit picking', 4: 'Gym time trials',
                  5: 'Calibration', 6: 'Sorting'}

    for key in scene_dict.keys():
        table.loc[table.scene_index == key, 'scene_index'] = scene_dict[key]

    return table['scene_index']

def rotRescaler(val, D=360):
    '''add descr'''
    return (np.where(val > D / 2, -1*(val - D), 0),np.where(val > D / 2, 0, val))


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

def tableReader(blob, cols):
    '''add descr'''

    if cols is None:
        cols = ['index', 'scene_index', 'time', 'ms_lastline', 'head_posx', "head_posy", "head_posz", "head_rotx",
            "head_roty", "head_rotz", "right_posx", "right_posy", "right_posz", "right_rotx", "right_roty",
            "right_rotz", "left_posx", "left_posy", "left_posz", "left_rotx", "left_roty", "left_rotz"]

    with open("/tmp/my-file.txt", "wb") as file_obj:
        blob.download_to_file(file_obj)

    table = pd.read_table("/tmp/my-file.txt", sep=',', header=0, names=cols)

    return table


def tableProcess(table, data):
    '''add descr'''
    '''data(dict): The Cloud Functions event payload.'''

    ### 2. divide /1000 ###
    table.loc[:, 4:22] = valDivider(table.loc[:, 4:22])

    ### 3. format time ###
    table['time'] = pd.to_datetime(table['time'], format="%H:%M:%S").dt.time #do we really need this column?

    ### 4. create date ###
    table['date'] = pd.to_datetime(data['name'][12:22], format = "%d-%m-%Y")

    ### 5. create seconds ###
    #on hold for now
    #timedelta = pd.to_timedelta(table.time.astype(str))
    #diff = timedelta.diff().fillna(pd.Timedelta(seconds=0)) / 1e9
    #table['seconds'] = np.cumsum(diff).astype(int)

    ### 6. rescale rotations ###
    cols = ["head_rotx", "head_roty", "head_rotz", "right_rotx", "right_roty", "right_rotz",
            "left_rotx", "left_roty", "left_rotz"]

    for col in cols:
        table[str(col + "_n")], table[col] = rotRescaler(table[col])

    ### 7. compute MET score at frame level ###
    table['met_score'] = metscoreCalc('scene_index', table)

    ### 8. replace scene_index with string names ###
    table['scene_index'] = sceneDict(table)

    ### 9. create id column ###
    table['id'] = 1 #in production this should come from the quest

    ### 10. drop unused columns ###
    table = table.drop('index ms_lastline'.split(), axis=1)

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
