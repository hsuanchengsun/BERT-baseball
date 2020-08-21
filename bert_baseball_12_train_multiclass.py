#!/usr/bin/env python
# coding: utf-8

# ### Import Module

# In[26]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[27]:


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

print("There are " +  str(numTrainSize + numValidatSize) + " training data with " +  str(numVar) + " variables, which equals to 4 data information and 21 variables of 1 year for total 5 continuous years.")
print("There are " + str(numTestSize) + " data used for prediction.")


# In[28]:


x_train_load.head()


# In[29]:


column = ['FirstYear_TB', 'SecondYear_TB', 'ThirdYear_TB', 'FourthYear_TB', 'FifthYear_BB']


# In[30]:


x_train_load.drop(columns = column, inplace = True)
x_val_load.drop(columns = column, inplace = True)
x_test_load.drop(columns = column, inplace = True)


# ### Separating output data

# In[31]:


y_train = y_train_load[y_train_load.columns[3:]]
y_val = y_val_load[y_val_load.columns[3:]]
y_test = y_test_load


# ### Classify

# In[32]:


y_train_TY = y_train.values
for k in range(numTrainSize):
    y_train_TY[k] = np.floor(y_train_TY[k]/5)# each interval is 5 HR


# In[33]:


y_val_TY = y_val.values
for k in range(numValidatSize):
    y_val_TY[k] = np.floor(y_val_TY[k]/5)# each interval is 5 HR


# In[34]:


y_test_TY = y_test.astype('int').values
for k in range(numTestSize):
    y_test_TY[k] = np.floor(y_test_TY[k]/5)# each interval is 5 HR


# In[35]:


# max label number

numTotalLabel = np.max(np.array([y_train_TY.max(), y_val_TY.max(), y_test_TY.max()]))
numTotalLabel


# ### Combine

# In[36]:


K = x_train_load.astype('str')
L = x_val_load.astype('str')
M = x_test_load.astype('int').astype('str')


# In[37]:


k = ""
l = ""
m = ""


# In[38]:


for i in range(100):
    k = k + K.iloc[:, 4+i] + " "


# In[39]:


for i in range(100):
    l = l + L.iloc[:, 4+i] + " "


# In[40]:


for i in range(100):
    m = m + M.iloc[:, i] + " "


# In[41]:


A = k.to_frame('text')
A['label'] = y_train_TY


# In[42]:


B = l.to_frame('text')
B['label'] = y_val_TY


# In[43]:


C = m.to_frame('text')
C['label'] = y_test_TY


# In[68]:


E = pd.concat([A, B])


# In[69]:


E.head()


# In[70]:


E.reset_index().drop(columns = ['index'], inplace = True)


# In[71]:


len(E)


# In[47]:


c = 0
for i in range(len(C)):
    if len(C.text[i]) > c:
        c = len(C.text[i])
    else:
        pass


# In[48]:


b = 0
for i in range(len(B)):
    if len(B.text[i]) > b:
        b = len(B.text[i])
    else:
        pass


# In[49]:


a = 0
for i in range(len(A)):
    if len(A.text[i]) > a:
        a = len(A.text[i])
    else:
        pass


# In[50]:


print(a,b,c)


# ### Model

# In[ ]:


from simpletransformers.classification import ClassificationModel


# In[ ]:


model = ClassificationModel('bert', 'bert-base-multilingual-cased', 
                         num_labels = 12, 
                         args={"max_seq_length": 320, 
                               'learning_rate':1e-5, 
                               'num_train_epochs': 20, 
                               'train_batch_size': 14, 
                               'eval_batch_size': 14,
                               "evaluate_during_training": True,
                               'reprocess_input_data': True, 
                               'overwrite_output_dir': True})


# In[ ]:


from sklearn.metrics import f1_score, accuracy_score


def f1_multiclass(labels, preds):
    return f1_score(labels, preds, average='micro')

model.train_model(E, eval_df = C, f1 = f1_multiclass, acc = accuracy_score)    
result, model_outputs, wrong_predictions = model.eval_model(C, f1=f1_multiclass, acc=accuracy_score)


# In[ ]:


C_pre, C_raw = model.predict(C.text)


# In[ ]:


pd.DataFrame(C_pre).to_csv("test_result.csv")


# In[ ]:


E_pre, E_raw = model.predict(E.text)


# In[ ]:


pd.DataFrame(E_pre).to_csv("train_result.csv")

