import os
os.chdir("/Users/bonaventurapacileo/Documents/IS-DS/VR/physiolyx/src")

#os.getcwd()

from GCSmanage import upload_blob

#import subprocess
#subprocess.call(['/Users/bonaventurapacileo/Documents/IS-DS/VR/keys/gcp-setup.sh'])
id1=['monitorData 26-02-2020_id1.txt',
'monitorData 27-02-2020_id1.txt',
'monitorData 31-01-2020_id1.txt',
'monitorData 06-03-2020_id1.txt',
'monitorData 03-03-2020_id1.txt']

#[upload_blob("physio-bucket", "/Users/bonaventurapacileo/Documents/IS-DS/VR/data/"+id, id) for id in id1]

###########
id1 = ['monitorDataLabel 20-03-2020.txt']
[upload_blob("physio-bucket", "/Users/bonaventurapacileo/Documents/IS-DS/VR/"+id, id) for id in id1]
