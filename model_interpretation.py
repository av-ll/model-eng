import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import warnings

# ignore warnings for clean output
warnings.simplefilter('ignore')

# read in data
data = pd.read_csv('prepared_data.csv')

# import RobustScaler for scaling data
from sklearn.preprocessing import RobustScaler

# initialize scaler object
scaler = RobustScaler()

# separate features (X) from target variable (y)
X = data.iloc[:,:-1]
y = data.iloc[:,-1]

# get list of feature names
feature_names = data.columns[:-1].tolist()

# split data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(X,y,test_size=0.2,random_state=27)

# scale training and testing sets using RobustScaler
xtrain_scaled = scaler.fit_transform(xtrain)

xtest_scaled = scaler.transform(xtest)

# load trained logistic regression model using pickle
with open('Logistic_Regression_Model.pkl','rb') as f:
    
    lr = pickle.load(f)
    f.close()
    

# get absolute values of logistic regression coefficients and sort by magnitude 
coef_signed_lr = lr.coef_   
coef_lr = lr.coef_
feature_importance_lr = np.abs(coef_lr)
coef_lr = pd.DataFrame(feature_importance_lr,columns=feature_names).transpose()
coef_lr.columns = ['coefficients_absolute_value']
top_5_coef_lr = coef_lr.sort_values(by=['coefficients_absolute_value'],axis=0,ascending=False)[:5]
low_5_coef_lr = coef_lr.sort_values(by=['coefficients_absolute_value'],axis=0,ascending=False)[25:]


# use logistic regression model to predict on test set
lr_pred = lr.predict(xtest_scaled)

from sklearn.metrics import confusion_matrix

# create confusion matrix using predicted values and true labels from test set
cm_lr = confusion_matrix(ytest, lr_pred)

# import matplotlib for plotting confusion matrix as heatmap
import matplotlib.pyplot as plt

# Plot confusion matrix as a heatmap
fig, ax = plt.subplots()
im = ax.imshow(cm_lr, cmap='RdYlGn')

# Add colorbar
cbar = ax.figure.colorbar(im, ax=ax)

# Add labels, title, and ticks
ax.set(xticks=np.arange(cm_lr.shape[1]),
       yticks=np.arange(cm_lr.shape[0]),
       xticklabels=['Predicted Benign', 'Predicted Malign'],
       yticklabels=['True Benign', 'True Malign'],
       title='Confusion Matrix Logistic Regression')
plt.setp(ax.get_xticklabels(), rotation=45, ha='right',
         rotation_mode="anchor")

# Loop over data to add annotations
for i in range(cm_lr.shape[0]):
    for j in range(cm_lr.shape[1]):
        ax.text(j, i, format(cm_lr[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm_lr[i, j] > cm_lr.max() / 2. else "black")

# save figure and show plot
fig.tight_layout()
fig.savefig('Confusion_Matrix_LR.png',dpi=300)
plt.show()

# Error analysis for Logistic Regression

# combine test set features, true labels, and predicted labels into one DataFrame

data_compare = pd.concat([xtest,ytest],axis=1)

data_compare['prediction'] =lr_pred

errors = data_compare[data_compare['prediction'] != data_compare['diagnose']]

errors_important_features = errors[['texture_worst','area_se','smoothness_worst','area_worst','concave points_mean']]

predicted_cancer = data_compare[data_compare['prediction'] ==1]

predicted_cancer_important_features = predicted_cancer[['texture_worst','area_se','smoothness_worst','area_worst','concave points_mean']]

predicted_cancer_important_features_mean = predicted_cancer_important_features.mean()

predicted_cancer_important_features_std = predicted_cancer_important_features.std()


predicted_cancer[predicted_cancer['concave points_mean'] < 0.03]

# empty dataframe result which means that no values predicted as cancer had a concave points_mean less than 
# 0.03 so this could be one of the reasons the first data record was predicted wrong.

# analysis of the second error
predicted_cancer[(predicted_cancer['area_worst'] < 600)][['texture_worst','area_se','smoothness_worst','area_worst','concave points_mean']]


lr_pred_prob = lr.predict_proba(xtest_scaled)

prediction_prob = [1 if x <0.00001 else 0 for x in lr_pred_prob[:,1]]

cm_lr_lower_threshold = confusion_matrix(ytest, prediction_prob)





