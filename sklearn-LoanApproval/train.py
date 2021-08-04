
import argparse
import os
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

from azureml.core import Run
#from utils import load_data

# let user feed in 2 parameters, the location of the data files (from datastore), and the regularization rate of the logistic regression model
parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder', help='data folder mounting point')
parser.add_argument('--regularization', type=float, dest='reg', default=0.01, help='regularization rate')
args = parser.parse_args()

data_folder = args.data_folder
print('Data folder:', data_folder)

"""
# load train and test set into numpy arrays
# note we scale the pixel intensity values to 0-1 (by dividing it with 255.0) so the model can converge faster.
X_train = load_data(os.path.join(data_folder, 'train-images.gz'), False) / 255.0
X_test = load_data(os.path.join(data_folder, 'test-images.gz'), False) / 255.0
y_train = load_data(os.path.join(data_folder, 'train-labels.gz'), True).reshape(-1)
y_test = load_data(os.path.join(data_folder, 'test-labels.gz'), True).reshape(-1)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, sep = '\n')
"""

import os
import pandas as pd 
dataset = pd.read_csv(os.path.join(data_folder, 'Loan-Approval-Prediction.csv'))

dataset['Gender']=dataset['Gender'].fillna('Male')
dataset['Married']=dataset['Married'].fillna('No')
dataset['Self_Employed']=dataset['Self_Employed'].fillna('No')
dataset['Property_Area']=dataset['Property_Area'].fillna('Semiurban')

from sklearn.preprocessing import LabelEncoder
var_mod = ['Gender','Married','Education','Self_Employed','Property_Area','Loan_Status']
le = LabelEncoder()
for i in var_mod:
    dataset[i] = le.fit_transform(dataset[i])

    print(dataset.head(5))

dataset['LoanAmount']=dataset['LoanAmount'].fillna(0)
dataset['Loan_Amount_Term']=dataset['Loan_Amount_Term'].fillna(360)
dataset['Credit_History']=dataset['Credit_History'].fillna(0)

array = dataset.values
X = array[:,6:11]
Y = array[:,12]
Y = Y.astype('int')

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=66)

# get hold of the current run
run = Run.get_context()


print('Train a logistic regression model with regularization rate of', args.reg)
clf = LogisticRegression(C=1.0/args.reg, solver="liblinear", multi_class="auto", random_state=42)
clf.fit(x_train, y_train)

print('Predict the test set')
y_hat = clf.predict(x_test)

# calculate accuracy on the prediction
acc = np.average(y_hat == y_test)
print('Accuracy is', acc)

run.log('regularization rate', np.float(args.reg))
run.log('accuracy', np.float(acc))

os.makedirs('outputs', exist_ok=True)
# note file saved in the outputs folder is automatically uploaded into experiment record
joblib.dump(value=clf, filename='outputs/sklearn_loanapproval_model.pkl')