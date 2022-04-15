#!/usr/bin/env python
# coding: utf-8

# In[4]:


#Import fastai and fastai.vision Library

from fastai import *
from fastai.vision import *
from fastai.widgets import *


# In[5]:


#Provide the path for parent folder of Men and Women directories containing respective images


# In[6]:


path = Path('data/humans')


# In[7]:


# classes = ['women', 'men']


# In[ ]:





# In[8]:


# for c in classes:
#     print(c)
#     verify_images(path/c, delete=True, max_workers=8)


# In[9]:


#Create a databunch object defined in fastai library.
#Pass the path to parent directory using from_folder(path)
#Labels for the dataset will begiven based on Directories in path. e.g.(Men/Women)
#Change the image size to 224 pixel for model training. transform(size=224 )
#The image dataset is divided in 80:20 ratio for Training and Validation using split_by_rand_pct(0.2)


# In[10]:


data = (ImageList.from_folder(path).split_by_rand_pct(0.2).label_from_folder().transform(size=224 ).databunch())


# In[11]:


#Get the classes in databunch object

data.classes


# In[12]:


#Show some sample images in the dataset

data.show_batch(rows=3,figsize=(3,3))


# In[142]:


print(data.classes)
len(data.classes),data.c


# In[14]:



#The cnn_learner factory method helps you to automatically get a pretrained model from a given architecture with a custom head that is suitable for your data.
#Used models.resnet34 for Transfer Learning


learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.model


# In[15]:


#fit_one_cycle() uses large, cyclical learning rates to train models significantly quicker and with higher accuracy.


# In[16]:


learn.fit_one_cycle(4)


# In[17]:


#Get Interpretation object from learn model
#Interpretation object will cntain information about losses ,indexes
interp = ClassificationInterpretation.from_learner(learn)


# In[18]:


losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# In[19]:


#Plot the losses
interp.plot_top_losses(4, figsize=(15,11))


# In[20]:


#Plot the confusion matrix

interp.plot_confusion_matrix(figsize=(12,12), dpi=60)


# In[21]:


#Unfreeze the rest of the layers of model
#Now all the layers of the model will be trained
learn.unfreeze()


# In[22]:


learn.fit_one_cycle(1)


# In[23]:


#save the current weights of the re-trained model

learn.save('stage-1');


# In[24]:


#Find the learning rate for the model
learn.lr_find()


# In[25]:


#plot the results for Learning rate finder
learn.recorder.plot()


# In[26]:


#Choose the learning rate which has low LOSS.
#Again fit the model with this learning rate
learn.unfreeze()
learn.fit_one_cycle(2, max_lr=slice(3e-04,1e-03))


# In[ ]:





# In[ ]:




