#add descr
from tensorflow.keras import Model, initializers, optimizers, regularizers
from tensorflow.keras.layers import Dense, Conv1D, Dropout, LSTM


class SimpleLSTM(Model):
  def __init__(self):
    super(SimpleLSTM, self).__init__()
    self.r1 = LSTM(32,input_shape=(timeSteps, n_features), activation='tanh', kernel_regularizer=regularizers.l2(0.02), return_sequences = True)
    self.r2 = LSTM(32, activation='tanh', kernel_regularizer=regularizers.l2(0.02),  return_sequences = False)
    self.sm = Dense(n_classes, activation='softmax')
    
  def call(self, x):
    x = self.r1(x)
    x = self.r2(x)
    
    return self.sm(x)


class DeepConvLSTM(Model):
  def __init__(self):
    super(DeepConvLSTM, self).__init__()
    self.c1 = Conv1D(8, 1,input_shape=(timeSteps, n_features), kernel_regularizer=regularizers.l2(0.02), activation='relu', kernel_initializer='orthogonal') #ordo filters=64, kernel_size = 5
    self.c2 = Conv1D(8, 3,kernel_regularizer=regularizers.l2(0.02), activation='relu', kernel_initializer='orthogonal')
    #self.c3 = Conv1D(8, 3,kernel_regularizer=regularizers.l2(0.02), activation='relu', kernel_initializer='orthogonal')
    #self.c4 = Conv1D(8, 3, activation='relu', kernel_initializer='orthogonal')
    self.do1 = Dropout(0.5)
    self.r1 = LSTM(16, activation='tanh', kernel_regularizer=regularizers.l2(0.02), return_sequences = True) #ordo cells=128
    self.do2 = Dropout(0.5)
    self.r2 = LSTM(16, activation='tanh', kernel_regularizer=regularizers.l2(0.02),  return_sequences = False)
    self.sm = Dense(n_classes, activation='softmax')
    
  def call(self, x):
    x = self.c1(x)
    x = self.c2(x)
    #x = self.c3(x)
    #x = self.c4(x)
    x = self.do1(x)
    x = self.r1(x)
    x = self.do2(x)
    x = self.r2(x)
    
    return self.sm(x)



