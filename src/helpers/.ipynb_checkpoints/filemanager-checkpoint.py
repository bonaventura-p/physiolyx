#add descr

from google.cloud import storage
import pandas as pd


def blobDownloader(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.get_blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print('Blob {} downloaded to {}.'.format(
        source_blob_name,
        destination_file_name))
    
    
def blobUploader(bucket, infile, outfile):
    '''add descr'''
    
    client = storage.Client()
    outbucket = client.get_bucket(bucket)
    outblob = outbucket.blob(outfile)
    
    with open(infile, "rb") as out:
        outblob.upload_from_file(out)
    
    print('File {} uploaded to {} as {}.'.format(
        infile, bucket, outfile))

                   
def tableReader(file, cols):
    '''add descr'''
    if cols is None:
        cols = ['index', 'scene_index', 'time', 'ms_lastline', 'head_posx', "head_posy", "head_posz", "head_rotx",
            "head_roty", "head_rotz", "right_posx", "right_posy", "right_posz", "right_rotx", "right_roty",
            "right_rotz", "left_posx", "left_posy", "left_posz", "left_rotx", "left_roty", "left_rotz"]

    table = pd.read_table(file, sep=',', header=0, names=cols)

    return table

def colBlank (df,col='action'):
    '''is there any non empty?'''
    return bool(df[col].any())