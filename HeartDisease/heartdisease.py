#!/usr/bin/env python
# coding: utf-8

# # Heart Disease Prediction

# ## 1. Import Libraries

# In[2]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


# ## 2. Load and Examine the Dataset

# In[10]:


data = pd.read_csv(r"C:\Users\iamke\OneDrive\Desktop\Heart Disease data\Heart Disease data.csv")


# In[13]:


data.head()


# In[14]:


data.info()


# ## 3. Handle Missing Values

# In[15]:


print(data.isnull().sum())


# In[23]:


features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 
            'exang', 'oldpeak', 'slope', 'ca', 'thal']
target = 'target'


# ## 4.Separate Features and Target

# In[24]:


x = data[features]
y = data[target]

print("Number of sample in x",x.shape[0])
print("Number of sample in y",y.shape[0])


# ## 5. Normalize/Standardize the Features

# In[25]:


scaler = StandardScaler()
x_scaled =  scaler.fit_transform(x)
  
    
print("Number of sample in x_scaled :",x_scaled.shape[0])


# ## 6. Split the Data into Training and Testing Sets

# In[26]:


x_train , x_test , y_train , y_test = train_test_split(x_scaled , y,test_size =0.2 ,random_state = 42)


print("Number of samples in X_train:", x_train.shape[0])
print("Number of samples in X_test:", x_test.shape[0])
print("Number of samples in y_train:", y_train.shape[0])
print("Number of samples in y_test:", y_test.shape[0])


# ##  7.Train a Logistic Regression Model

# In[27]:


lr_model = LogisticRegression()
lr_model.fit(x_train,y_train)
lr_pred = lr_model.predict(x_test)
lr_accuracy = accuracy_score(y_test,lr_pred)
lr_roc_auc = roc_auc_score(y_test,lr_model.predict_proba(x_test)[:,1])
print("logistic Regression - Accuracy:",lr_accuracy,"ROC-AUC :",lr_roc_auc)
print(classification_report(y_test,lr_pred))


# ## Evaluate Model Performance

# In[30]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc


# In[32]:


conf_matrix = confusion_matrix(y_test, lr_pred)
ConfusionMatrixDisplay(conf_matrix).plot()
plt.title('Confusion Matrix')
plt.show()


# In[35]:


fpr, tpr, _ = roc_curve(y_test, lr_model.predict_proba(x_test)[:, 1])
roc_auc = auc(fpr, tpr)


# In[36]:


plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()


# ## Interpret Model Coefficients

# In[37]:


feature_importance = pd.DataFrame({
    'Feature': features,
    'Coefficient': lr_model.coef_[0]
}).sort_values(by='Coefficient', ascending=False)

print(feature_importance)


# ## Save the Model

# In[38]:


import joblib

joblib.dump(lr_model, 'logistic_regression_model.pkl')

joblib.dump(scaler, 'scaler.pkl')


# ##  Load and Use the Model

# In[39]:


lr_model = joblib.load('logistic_regression_model.pkl')
scaler = joblib.load('scaler.pkl')

new_data = pd.DataFrame({
    'age': [60],
    'sex': [1],
    'cp': [2],
    'trestbps': [130],
    'chol': [250],
    'fbs': [0],
    'restecg': [1],
    'thalach': [160],
    'exang': [0],
    'oldpeak': [1.5],
    'slope': [2],
    'ca': [0],
    'thal': [2]
})
new_data_scaled = scaler.transform(new_data)

new_prediction = lr_model.predict(new_data_scaled)
new_prediction_proba = lr_model.predict_proba(new_data_scaled)[:, 1]

print(f"Prediction: {'Heart Disease' if new_prediction[0] == 1 else 'No Heart Disease'}, Probability: {new_prediction_proba[0]:.2f}")


# 

# In[ ]:




    


# In[ ]:



    


# In[ ]:




