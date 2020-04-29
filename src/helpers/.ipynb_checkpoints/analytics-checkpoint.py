#check imports and imports from filemanager

import pandas as pd
import numpy as np
from scipy import stats
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint


from helpers.model import DeepLSTM, DeepConvLSTM
from helpers.filemanager import colBlank, blobDownloader


def sceneDict(table):
    '''add descr'''
    scene_dict = {0: 'Menu', 1: 'Goalkeeping', 2: 'Touch object', 3: 'Fruit picking', 4: 'Gym time trials',
                  5: 'Calibration', 6: 'Sorting'}

    for key in scene_dict.keys():
        table.loc[table.scene_index == key, 'scene_index'] = scene_dict[key]

    return table['scene_index']


def rotRescaler(val, D=360, serve=True):
    '''add descr'''
    if serve:
        return (np.where(val > D / 2, -1*(val - D), 0),np.where(val > D / 2, 0, val))
    else:
        return (np.where(val > D / 2, (val - D), val))

    
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

def tableProcessor(table, data, serve= True):
    '''add descr'''
    '''data(dict): The Cloud Functions event payload.'''

    ### 2. divide /1000 ###
    colDivider = ['head_posx',"head_posy","head_posz","head_rotx","head_roty","head_rotz",
        "right_posx","right_posy","right_posz","right_rotx","right_roty","right_rotz",
        "left_posx","left_posy","left_posz","left_rotx","left_roty","left_rotz"]
    
    table.loc[:, colDivider] = valDivider(table.loc[:, colDivider])
    
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
        if serve:
            table[str(col + "_n")], table[col] = rotRescaler(table[col], serve=serve)
        else:
            table[col] = rotRescaler(table[col], serve=serve)

    ### 7. compute MET score at frame level ###
    table['met_score'] = metscoreCalc('scene_index', table)

    ### 8. replace scene_index with string names ###
    table['scene_index'] = sceneDict(table)

    ### 9. create id column ###
    table['id'] = 1 #in production this should come from the quest

    ### 10. drop unused columns ###
    table = table.drop('index ms_lastline'.split(), axis=1)

    return table


def featureProcessor(df, cols, timeSteps=72, step=14, serve=True):
    '''preprocessing. WORKS ONE MARKER AT THE TIME. WATCH OUT FOR OUTPUTNAMES'''

    if cols is None:
        cols = ["head_rotx","head_roty","head_rotz"]
        
    features= len(cols)
    segments = []
    coldict = {}

    #fill 0's
    if [colBlank(df, col=c) for c in cols]:
        print('Missing features, filling with 0\'s')
        df = df.fillna(0)
        
    #normalise data otherwise loss= nan
    for col in cols:
        df[col] = (df[col] - df[col].min())/(df[col].max()-df[col].min())

    #we reshape into 3d arrays of length equal to timesteps. final df is= (N*timesteps*features)
    if serve:
        
        for i in range(0, len(df), timeSteps):
            
            for col in cols:

                coldict[str(col[5:])] = df[col].values[i: i + timeSteps]
                segments.append(coldict[str(col[5:])])

    else:     
        
        for i in range(0, len(df) - timeSteps, step):
            
            for col in cols:

                coldict[str(col[5:])] = df[col].values[i: i + timeSteps]
                segments.append(coldict[str(col[5:])])
   
    reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, timeSteps, features)

    print('Preprocessing completed')

    return reshaped_segments


def labelProcessor(df, timeSteps=72, step=14, method='last'):
    '''add descr'''
    
    labels = []
    
    if colBlank(df, col='action'):
        print('Missing labels, filling with 0\'s')
        df = df.fillna(0)
    
    df['action'] = df.action.fillna(0)
    
    for i in range(0, len(df) - timeSteps, step):
        
        if method == 'max':
            label = stats.mode(df['action'][i:i + timeSteps])[0][0]
        else:
            label = df['action'].iloc[i + timeSteps]
        
        labels.append(label)
    
    labels = pd.get_dummies(labels)
    truelabels= pd.get_dummies(labels).idxmax(1)
    labels = np.asarray(labels, dtype = np.float32)


    return labels, truelabels


def reshuffler(df, div=70):
    '''add descr'''
    return np.random.shuffle(df.values.reshape(-1,int(np.floor(df.shape[0]/div)),df.shape[1]))

    
def modelFitter(X,y, model, epochs=1, batch_size=12):
    '''removed .hdf5 from filename'''
    
    # define loss and optimizer
    adam = optimizers.Adam(learning_rate=0.001) #(ordo learning_rate=0.01, decay=0.9)

    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['categorical_accuracy']) #sparse categorical crossentropy if labels not one-hot encoded
    model.save_weights('model.h5')
    model.load_weights('model.h5')
    
    callbacks = ModelCheckpoint('model_ep{epoch:02d}_val{val_categorical_accuracy:.2f}', monitor='val_categorical_accuracy', verbose=0,
                                save_best_only=True, save_weights_only=True, mode='max')
    history = model.fit(X,y, validation_split = 0.2, batch_size = 12,
                epochs = epochs, callbacks = [callbacks])
    
    return history



def actionPredictor(bucket, df, timeSteps=72, model=None):

    # Model load which only happens during cold starts
    if model is None:
        blobDownloader(bucket, 'model_ep04_val0.25.index', '/tmp/model_ep04_val0.25.index')
        blobDownloader(bucket, 'model_ep04_val0.25.data-00000-of-00001', '/tmp/model_ep04_val0.25.data-00000-of-00001')
        
        model = DeepLSTM(timeSteps=72, n_features=3, n_classes=10)
        model.load_weights('/tmp/model_ep04_val0.25')
    
    class_names=['RROT', 'EXTE', 'LROT', 'FLEX', 'RBEN', 'LBEN', '3DLEX', '3DREX',
       '3DLFL', '3DRFL']
    
    actionPred = []
    
    for r in range(df.shape[0]):  
        pred = model.call(df[r:r+1])
        actionPred.append(class_names[np.argmax(pred)])

    return actionPred


def predWrapper(df,bucket, timeSteps=72, model=None):
        '''add descr'''
                    
        Xpred = featureProcessor(df, timeSteps=timeSteps, cols=None)
        ypred = actionPredictor(bucket, Xpred, timeSteps=timeSteps, model=model)
        
        #reshape predicted actions into table shape
        action = [ypred[val] for val in range(len(ypred)) for _ in range(timeSteps)]
        [action.append(ypred[len(ypred)-1]) for _ in range(1,1+len(df)-len(action))]

        return action

