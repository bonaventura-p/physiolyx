import os
os.chdir("/Users/bonaventurapacileo/Documents/IS-DS/VR/physiolyx/src")

#os.getcwd()

from GCSmanage import upload_blob

#import subprocess
#subprocess.call(['/Users/bonaventurapacileo/Documents/IS-DS/VR/keys/gcp-setup.sh'])

upload_blob("physio-bucket", "/Users/bonaventurapacileo/Documents/IS-DS/VR/data/monitorData 04-03-2020.txt", "data/monitorData 04-03-2020.txt")

###########

#######################
###1#####column names##
#######################
cols = ['index','scene_index','time','ms_lastline','head_posx',"head_posy","head_posz","head_rotx","head_roty","head_rotz",
        "right_posx","right_posy","right_posz","right_rotx","right_roty","right_rotz",
        "left_posx","left_posy","left_posz","left_rotx","left_roty","left_rotz",]

alldata = pd.read_table('data/monitorData 18-03-2020.txt', sep=',', header=0, names=cols)

#########################################################
###1.1#####restrict to gym (maybe to all but scene 0) ##
####################################################
gym = alldata[alldata.scene_index==1]

#######################
###2#####divide /1000 ##
#######################
alldata.loc[:,'head_posx':] = alldata.loc[:,'head_posx':]/1000



#######################
###3#####format time dat ##
#######################
gym['time'] = pd.to_datetime(gym['time'], format="%H:%M:%S").dt.time
#gym = gym[gym.time >= datetime.time(15,7,37)]


#########################################################
###4#####define new rot ##
####################################################
cols = ["head_rotx","head_roty","head_rotz","right_rotx","right_roty","right_rotz",
        "left_rotx","left_roty","left_rotz"]

for col in cols:
    #print(col)
    gym[str(col+"_n")] = gym[col]
    gym.loc[gym[str(col+"_n")]>180,str(col+"_n")] = gym.loc[gym[col]>180,col] - 360


############################
###5#####create seconds ##
#############################
timedelta = pd.to_timedelta(gym.time.astype(str))
diff = timedelta.diff().fillna(pd.Timedelta(seconds=0))/1e9
gym['seconds'] = np.cumsum(diff).astype(int)

#########################################################
###6#####compute MET score at frame level ##
####################################################
##create dictionary scene/met


#########################################################
# 7 and 8 on other files######
####################################################

#########################################################
###9#####upload back to bucket ##
####################################################
