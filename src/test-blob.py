import os
os.chdir("/Users/bonaventurapacileo/Documents/IS-DS/VR/physiolyx/src")

#os.getcwd()

from GCSmanage import upload_blob

#import subprocess
#subprocess.call(['/Users/bonaventurapacileo/Documents/IS-DS/VR/keys/gcp-setup.sh'])

upload_blob("physio-bucket", "/Users/bonaventurapacileo/Documents/IS-DS/VR/data/monitorData 18-03-2020.txt", "monitorData 18-03-2020.txt")

###########
