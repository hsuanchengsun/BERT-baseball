

# ### Import Module

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
from numpy.random import permutation
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense,Activation, LSTM, Dropout, TimeDistributed, Flatten
from keras.models import load_model
from keras import optimizers
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.callbacks import EarlyStopping


# ### Load Data

# In[2]:


x_train_load = pd.read_csv('x_training_origin.csv', delimiter=',')
y_train_load = pd.read_csv('y_training_origin.csv', delimiter=',')

x_val_load = pd.read_csv('x_validation_origin.csv', delimiter=',')
y_val_load = pd.read_csv('y_validation_origin.csv', delimiter=',')

x_test_load = pd.read_csv('x_2018.csv', delimiter=',')
y_test_load = pd.read_csv('y_2018.csv', delimiter=',')

numTrainSize = x_train_load.shape[0]
numValidatSize = x_val_load.shape[0]
numTestSize = x_test_load.shape[0]

numVar = x_train_load.shape[1]

print("training data and variables")
print(numTrainSize, numVar)
print("There are 109 variables, which equals to 4 data information and 21 variables of 1 year for total 5 years")
#describe variables
print("validating data used for test")
print(numValidatSize)
print("data used for prediction")
print(numTestSize)


# In[3]:


x_train_load.sample(5)


# In[4]:


y_train_load.sample(5)


# In[5]:


x_val_load.sample(5)


# In[6]:


y_val_load.sample(5)


# In[7]:


x_test_load.sample(5)


# In[8]:


y_test_load.sample(5)


# ### Separating and normalizing input data

# In[9]:
column = ['FirstYear_TB', 'SecondYear_TB', 'ThirdYear_TB', 'FourthYear_TB', 'FifthYear_TB']
x_train_load.drop(columns = column, inplace = True)
x_val_load.drop(columns = column, inplace = True)
x_test_load.drop(columns = column, inplace = True)

x_train_sep = x_train_load.iloc[:,4:]
x_val_sep = x_val_load.iloc[:,4:]
x_test_sep = x_test_load


# In[10]:


# (x - x.min) / (x.max - x.min)

x_train_norm = (x_train_sep - x_train_sep.min())  / (x_train_sep.max() - x_train_sep.min())
x_val_norm = (x_val_sep - x_val_sep.min())  / (x_val_sep.max() - x_val_sep.min())
x_test_norm = (x_test_sep - x_test_sep.min())  / (x_test_sep.max() - x_test_sep.min())


# In[11]:


x_test_norm.sample(5)


# In[12]:


#reshape to (n, 5, 21)

x_train_reshape = np.reshape(x_train_norm.values, (numTrainSize, 5, 20))
x_val_reshape = np.reshape(x_val_norm.values, (numValidatSize, 5, 20))
x_test_reshape = np.reshape(x_test_norm.values, (numTestSize, 5, 20))


# ### Separating output data

# In[13]:


y_train = y_train_load[y_train_load.columns[3:]]
y_val = y_val_load[y_val_load.columns[3:]]
y_test = y_test_load


# In[14]:


y_test.sample(5)


# ### Classify

# In[15]:


y_train_TY = y_train.values
for k in range(numTrainSize):
    y_train_TY[k] = np.floor(y_train_TY[k]/5)# each interval is 5 HR


# In[16]:


y_val_TY = y_val.values
for k in range(numValidatSize):
    y_val_TY[k] = np.floor(y_val_TY[k]/5)# each interval is 5 HR


# In[17]:


y_test_TY = y_test.values
for k in range(numTestSize):
    y_test_TY[k] = np.floor(y_test_TY[k]/5)# each interval is 5 HR


# In[18]:


y_train_TY.shape


# In[19]:


#bins = np.arange(0, 7,1)
#plt.hist(y_train_TY, bins = bins, alpha = 0.8)
#plt.show()


# In[20]:


#bins = np.arange(0, 7,1)
#plt.hist(y_val_TY, bins = bins, alpha = 0.8)
#plt.show()


# In[21]:


#bins = np.arange(0, 7,1)
#plt.hist(y_test_TY, bins = bins, alpha = 0.8)
#plt.show()



# In[20]:


#one hot

y_train_cat = np_utils.to_categorical(y_train_TY, 12)
y_val_cat = np_utils.to_categorical(y_val_TY, 12)
y_test_cat = np_utils.to_categorical(y_test_TY, 12)


# In[21]:


y_train_cat


# ### Permutation

# In[22]:


perm_train = permutation(numTrainSize)
x_train_reshape = x_train_reshape[perm_train]
y_train_cat = y_train_cat[perm_train]

perm_val = permutation(numValidatSize)
x_val_reshape = x_val_reshape[perm_val]
y_val_cat = y_val_cat[perm_val]

perm_test = permutation(numTestSize)
x_test_reshape = x_test_reshape[perm_test]
y_test_cat = y_test_cat[perm_test]


# In[23]:


x_train = np.concatenate((x_train_reshape, x_val_reshape))
y_train = np.concatenate((y_train_cat, y_val_cat))


# In[25]:


y_train.shape


# ### Models

# In[17]:


#model_1: LSTM128x2, Droupout_0.25, TimeDistributed_10, FC1024, Droupout_0.25, FC128, FC12

model_1 = Sequential()
model_1.add(LSTM(128, input_shape=(5, 20), return_sequences=True))
model_1.add(LSTM(128, return_sequences=True))
model_1.add(Dropout(rate=0.25))
model_1.add(TimeDistributed(Dense(10)))
model_1.add(Flatten())
model_1.add(Dense(1024, activation="relu"))
model_1.add(Dropout(rate=0.25))
model_1.add(Dense(128, activation="relu"))
model_1.add(Dense(12, activation='softmax'))
adam = optimizers.Adam(lr = 0.000001)
model_1.compile(loss = "categorical_crossentropy", optimizer=adam, metrics=["accuracy"])
model_1.summary()


# In[32]:


#model_2: LSTM128x2, Droupout_0.25, FC1024, Droupout_0.25, FC128, FC12

model_2 = Sequential()
model_2.add(LSTM(128, input_shape=(5, 20), return_sequences=True))
model_2.add(LSTM(128, return_sequences=True))
model_2.add(Dropout(rate=0.25))
model_2.add(Flatten())
model_2.add(Dense(1024, activation="relu"))
model_2.add(Dropout(rate=0.25))
model_2.add(Dense(128, activation="relu"))
model_2.add(Dense(12, activation='softmax'))
adam = optimizers.Adam(lr = 0.000001)
model_2.compile(loss = "categorical_crossentropy", optimizer=adam, metrics=["accuracy"])
model_2.summary()


# In[18]:


#model_3: LSTM128x1, Droupout_0.25, TimeDistributed_10, FC1024, Droupout_0.25, FC128, FC12

model_3 = Sequential()
model_3.add(LSTM(128, input_shape=(5, 20), return_sequences=True))
model_3.add(Dropout(rate=0.25))
model_3.add(TimeDistributed(Dense(10)))
model_3.add(Flatten())
model_3.add(Dense(1024, activation="relu"))
model_3.add(Dropout(rate=0.25))
model_3.add(Dense(128, activation="relu"))
model_3.add(Dense(12, activation='softmax'))
adam = optimizers.Adam(lr = 0.000001)
model_3.compile(loss = "categorical_crossentropy", optimizer=adam, metrics=["accuracy"])
model_3.summary()


# In[43]:


#model_4: LSTM128x2, Droupout_0.25, FC1024, Droupout_0.25, FC12

model_4 = Sequential()
model_4.add(LSTM(128, input_shape=(5, 20), return_sequences=True))
model_4.add(LSTM(128, return_sequences=True))
model_4.add(Dropout(rate=0.25))
model_4.add(Flatten())
model_4.add(Dense(1024, activation="relu"))
model_4.add(Dropout(rate=0.25))
model_4.add(Dense(12, activation='softmax'))
adam = optimizers.Adam(lr = 0.000001)
model_4.compile(loss = "categorical_crossentropy", optimizer=adam, metrics=["accuracy"])
model_4.summary()


# In[36]:


#model_5: LSTM128x2, BN, TimeDistributed_10, FC1024, BN, FC128, FC12

model_5 = Sequential()
model_5.add(LSTM(128, input_shape=(5, 20), return_sequences=True))
model_5.add(LSTM(128, return_sequences=True))
model_5.add(BatchNormalization())
model_5.add(TimeDistributed(Dense(10)))
model_5.add(Flatten())
model_5.add(Dense(1024, activation="relu"))
model_5.add(BatchNormalization())
model_5.add(Dense(128, activation="relu"))
model_5.add(Dense(12, activation='softmax'))
adam = optimizers.Adam(lr = 0.0000001)
model_5.compile(loss = "categorical_crossentropy", optimizer=adam, metrics=["accuracy"])
model_5.summary()


# In[38]:


#model_6: LSTM32, FC1024, FC12

model_6 = Sequential()
model_6.add(LSTM(32, input_shape=(5, 20)))
model_6.add(Dense(1024, activation="relu"))
model_6.add(Dense(12,activation='softmax'))
adam = optimizers.Adam(lr = 0.000001)
model_6.compile(loss = "categorical_crossentropy", optimizer=adam, metrics=["accuracy"])
model_6.summary()


# In[39]:


#model_7: LSTM64x2, Droupout_0.25, FC1024, Droupout_0.25, FC128, FC12

model_7 = Sequential()
model_7.add(LSTM(64, input_shape=(5, 20), return_sequences=True))
model_7.add(LSTM(64, return_sequences=True))
model_7.add(Dropout(rate=0.25))
model_7.add(Flatten())
model_7.add(Dense(1024, activation="relu"))
model_7.add(Dropout(rate=0.25))
model_7.add(Dense(128, activation="relu"))
model_7.add(Dense(12, activation='softmax'))
adam = optimizers.Adam(lr = 0.000001)
model_7.compile(loss = "categorical_crossentropy", optimizer=adam, metrics=["accuracy"])
model_7.summary()


# In[40]:


#model_8: LSTM128x2, Droupout_0.25, FC512, Droupout_0.25, FC64, FC12

model_8 = Sequential()
model_8.add(LSTM(128, input_shape=(5, 20), return_sequences=True))
model_8.add(LSTM(128, return_sequences=True))
model_8.add(Dropout(rate=0.25))
model_8.add(Flatten())
model_8.add(Dense(512, activation="relu"))
model_8.add(Dropout(rate=0.25))
model_8.add(Dense(64, activation="relu"))
model_8.add(Dense(12, activation='softmax'))
adam = optimizers.Adam(lr = 0.000001)
model_8.compile(loss = "categorical_crossentropy", optimizer=adam, metrics=["accuracy"])
model_8.summary()


# In[26]:


#model_9 LSTM128, FC12

model_9 = Sequential()
model_9.add(LSTM(128, input_shape=(5, 20)))
model_9.add(Dense(12,activation='softmax'))
adam = optimizers.Adam(lr = 0.000001)
model_9.compile(loss = "categorical_crossentropy", optimizer=adam, metrics=["accuracy"])
model_9.summary()


# In[30]:


#model_10: LSTM128, FC1024, BN, FC12

model_10 = Sequential()
model_10.add(LSTM(128, input_shape=(5, 20)))
model_10.add(Dense(1024, activation="relu"))
model_10.add(BatchNormalization())
model_10.add(Dense(12,activation='softmax'))
adam = optimizers.Adam(lr = 0.000001)
model_10.compile(loss = "categorical_crossentropy", optimizer=adam, metrics=["accuracy"])
model_10.summary()


# ### Training
early_stopping = EarlyStopping(monitor='val_loss',patience=3)
# In[27]:


model_1_50_stat = model_1.fit(x_train, y_train, 
                              batch_size=14, 
                              epochs=20000, 
                              validation_data=(x_test_reshape, y_test_cat),
                              callbacks=[early_stopping]
                             )


# In[28]:


model_2_50_stat = model_2.fit(x_train, y_train, 
                              batch_size=14, 
                              epochs=20000, 
                              validation_data=(x_test_reshape, y_test_cat),
                              callbacks=[early_stopping]
                             )


# In[20]:


model_3_50_stat = model_3.fit(x_train, y_train, 
                              batch_size=14, 
                              epochs=20000, 
                              validation_data=(x_test_reshape, y_test_cat),
                              callbacks=[early_stopping]
                             )


# In[30]:


model_4_50_stat = model_4.fit(x_train, y_train, 
                              batch_size=14, 
                              epochs=20000, 
                              validation_data=(x_test_reshape, y_test_cat),
                              callbacks=[early_stopping]
                             )


# In[31]:


model_5_50_stat = model_5.fit(x_train, y_train, 
                              batch_size=14, 
                              epochs=20000, 
                              validation_data=(x_test_reshape, y_test_cat),
                              callbacks=[early_stopping]
                             )


# In[32]:


model_6_50_stat = model_6.fit(x_train, y_train, 
                              batch_size=14, 
                              epochs=20000, 
                              validation_data=(x_test_reshape, y_test_cat),
                              callbacks=[early_stopping]
                             )


# In[33]:


model_7_50_stat = model_7.fit(x_train, y_train, 
                              batch_size=14, 
                              epochs=20000, 
                              validation_data=(x_test_reshape, y_test_cat),
                              callbacks=[early_stopping]
                             )


# In[34]:


model_8_50_stat = model_8.fit(x_train, y_train, 
                              batch_size=14, 
                              epochs=20000, 
                              validation_data=(x_test_reshape, y_test_cat),
                              callbacks=[early_stopping]
                             )


# In[27]:


model_9_50_stat = model_9.fit(x_train, y_train, 
                              batch_size=14, 
                              epochs=20000, 
                              validation_data=(x_test_reshape, y_test_cat),
                              callbacks=[early_stopping]
                             )


# In[31]:


model_10_50_stat = model_10.fit(x_train, y_train, 
                              batch_size=14, 
                              epochs=20000, 
                              validation_data=(x_test_reshape, y_test_cat),
                              callbacks=[early_stopping]
                             )


# In[62]:


pd.concat([pd.DataFrame(np.array(model_1_50_stat.history['val_acc'])),
           pd.DataFrame(np.array(model_2_50_stat.history['val_acc'])),
           pd.DataFrame(np.array(model_3_50_stat.history['val_acc'])),
           pd.DataFrame(np.array(model_4_50_stat.history['val_acc'])),
           pd.DataFrame(np.array(model_5_50_stat.history['val_acc'])),
           pd.DataFrame(np.array(model_6_50_stat.history['val_acc'])),
           pd.DataFrame(np.array(model_7_50_stat.history['val_acc'])),
           pd.DataFrame(np.array(model_8_50_stat.history['val_acc'])),
           pd.DataFrame(np.array(model_9_50_stat.history['val_acc'])),
           pd.DataFrame(np.array(model_10_50_stat.history['val_acc']))],axis=1).to_csv("val_acc.csv")


# ### Analysis

# In[35]:


# summarize history for accuracy
plt.figure(figsize=(14,8))
plt.plot(model_1_50_stat.history['acc'])
plt.plot(model_2_50_stat.history['acc'])
plt.plot(model_3_50_stat.history['acc'])
plt.plot(model_4_50_stat.history['acc'])
plt.plot(model_5_50_stat.history['acc'])
plt.plot(model_6_50_stat.history['acc'])
plt.plot(model_7_50_stat.history['acc'])
plt.plot(model_8_50_stat.history['acc'])
plt.plot(model_9_50_stat.history['acc'])
plt.plot(model_10_50_stat.history['acc'])


plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['model_1', 'model_2', 'model_3', 'model_4', 'model_5', 'model_6', 'model_7', 'model_8', 'model_9', 'model_10'], loc=(1.04,0.5))
plt.savefig('training_acc.png')
plt.close()


# In[38]:


# summarize history for test accuracy
plt.figure(figsize=(14,8))
plt.plot(model_1_50_stat.history['val_acc'])
plt.plot(model_2_50_stat.history['val_acc'])
plt.plot(model_3_50_stat.history['val_acc'])
plt.plot(model_4_50_stat.history['val_acc'])
plt.plot(model_5_50_stat.history['val_acc'])
plt.plot(model_6_50_stat.history['val_acc'])
plt.plot(model_7_50_stat.history['val_acc'])
plt.plot(model_8_50_stat.history['val_acc'])
plt.plot(model_9_50_stat.history['val_acc'])
plt.plot(model_10_50_stat.history['val_acc'])

plt.title('model validation accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['model_1', 'model_2', 'model_3', 'model_4', 'model_5', 'model_6', 'model_7', 'model_8', 'model_9', 'model_10'], loc=(1.04,0.5))
plt.savefig('validating_acc.png')
plt.close()


# In[37]:


# summarize history for loss
plt.figure(figsize=(14,8))
plt.plot(model_1_50_stat.history['loss'])
plt.plot(model_2_50_stat.history['loss'])
plt.plot(model_3_50_stat.history['loss'])
plt.plot(model_4_50_stat.history['loss'])
plt.plot(model_5_50_stat.history['loss'])
plt.plot(model_6_50_stat.history['loss'])
plt.plot(model_7_50_stat.history['loss'])
plt.plot(model_8_50_stat.history['loss'])
plt.plot(model_9_50_stat.history['loss'])
plt.plot(model_10_50_stat.history['loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['model_1', 'model_2', 'model_3', 'model_4', 'model_5', 'model_6', 'model_7', 'model_8', 'model_9', 'model_10'], loc=(1.04,0.5))
plt.savefig('training_loss.png')
plt.close()


# In[39]:


# summarize history for test loss
plt.figure(figsize=(14,8))
plt.plot(model_1_50_stat.history['val_loss'])
plt.plot(model_2_50_stat.history['val_loss'])
plt.plot(model_3_50_stat.history['val_loss'])
plt.plot(model_4_50_stat.history['val_loss'])
plt.plot(model_5_50_stat.history['val_loss'])
plt.plot(model_6_50_stat.history['val_loss'])
plt.plot(model_7_50_stat.history['val_loss'])
plt.plot(model_8_50_stat.history['val_loss'])
plt.plot(model_9_50_stat.history['val_loss'])
plt.plot(model_10_50_stat.history['val_loss'])

plt.title('model validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['model_1', 'model_2', 'model_3', 'model_4', 'model_5', 'model_6', 'model_7', 'model_8', 'model_9', 'model_10'], loc=(1.04,0.5))
plt.savefig('validating_loss.png')
plt.close()


# ### testing

# In[52]:


result_val_model_1 = model_1.predict(x_test_reshape)
pd.DataFrame(result_val_model_1).to_csv("result_test_model_1.csv")
result_val_model_2 = model_2.predict(x_test_reshape)
pd.DataFrame(result_val_model_2).to_csv("result_test_model_2.csv")
result_val_model_3 = model_3.predict(x_test_reshape)
pd.DataFrame(result_val_model_3).to_csv("result_test_model_3.csv")
result_val_model_4 = model_4.predict(x_test_reshape)
pd.DataFrame(result_val_model_4).to_csv("result_test_model_4.csv")
result_val_model_5 = model_5.predict(x_test_reshape)
pd.DataFrame(result_val_model_5).to_csv("result_test_model_5.csv")
result_val_model_6 = model_6.predict(x_test_reshape)
pd.DataFrame(result_val_model_6).to_csv("result_test_model_6.csv")
result_val_model_7 = model_7.predict(x_test_reshape)
pd.DataFrame(result_val_model_7).to_csv("result_test_model_7.csv")
result_val_model_8 = model_8.predict(x_test_reshape)
pd.DataFrame(result_val_model_8).to_csv("result_test_model_8.csv")
result_val_model_9 = model_9.predict(x_test_reshape)
pd.DataFrame(result_val_model_9).to_csv("result_test_model_9.csv")
result_val_model_10 = model_10.predict(x_test_reshape)
pd.DataFrame(result_val_model_10).to_csv("result_test_model_10.csv")


# In[42]:


score_1 = model_1.evaluate(x_test_reshape, y_test_cat)
score_2 = model_2.evaluate(x_test_reshape, y_test_cat)
score_3 = model_3.evaluate(x_test_reshape, y_test_cat)
score_4 = model_4.evaluate(x_test_reshape, y_test_cat)
score_5 = model_5.evaluate(x_test_reshape, y_test_cat)
score_6 = model_6.evaluate(x_test_reshape, y_test_cat)
score_7 = model_7.evaluate(x_test_reshape, y_test_cat)
score_8 = model_8.evaluate(x_test_reshape, y_test_cat)
score_9 = model_9.evaluate(x_test_reshape, y_test_cat)
score_10 = model_10.evaluate(x_test_reshape, y_test_cat)

pd.DataFrame(np.array([score_1, score_2, score_3, score_4, score_5, score_6, score_7, score_8, score_9, score_10])).to_csv("val_score.csv")


# In[43]:


print(score_1)
print(score_2)
print(score_3)
print(score_4)
print(score_5)
print(score_6)
print(score_7)
print(score_8)
print(score_9)
print(score_10)


# ### save model

# In[44]:


#save weights

model_1.save_weights("model_1_weights.h5")
model_2.save_weights("model_2_weights.h5")
model_3.save_weights("model_3_weights.h5")
model_4.save_weights("model_4_weights.h5")
model_5.save_weights("model_5_weights.h5")
model_6.save_weights("model_6_weights.h5")
model_7.save_weights("model_7_weights.h5")
model_8.save_weights("model_8_weights.h5")
model_9.save_weights("model_9_weights.h5")
model_10.save_weights("model_10_weights.h5")


# In[45]:


#save model

model_1.save("model_1.h5")
model_2.save("model_2.h5")
model_3.save("model_3.h5")
model_4.save("model_4.h5")
model_5.save("model_5.h5")
model_6.save("model_6.h5")
model_7.save("model_7.h5")
model_8.save("model_8.h5")
model_9.save("model_9.h5")
model_10.save("model_10.h5")


