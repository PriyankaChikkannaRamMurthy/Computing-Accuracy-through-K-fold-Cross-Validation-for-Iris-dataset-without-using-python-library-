# Computing-Accuracy-through-K-fold-Cross-Validation-for-Iris-dataset-without-using-python-library-

File name:
Project1.py

iris.csv - to be present in the same folder as the python file is.

value of kfold = 10 [passed to the program]

Code Logic:

 Initially it was coded using the Jupyter Notebook and then downloaded as python file and been submitted.
 Pycharm is the IDE used.
 Python - 3.6 
 Numpy and sklearn.model_selection libraries used
 
 Iris data set is downloaded and converted to a csv file and read as the x and y matrices.
 Later using the beta formula the values are computed.
 Also, estimated y matrix is calculated using the formula from the slides "y= A(beta)"

Least sum squared values are computed and accuracy is computed.

for K-fold cross validation:
k fold is done using the sklearn kfold model where in it returns the training datasets and testing datasets of each fold.
Later in each fold we calculate the beta value and y estimate to compute the error of each fold.
Finally average error is computed and accuracy is calculated for the K-fold.

This is the summary of the project code.

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
1. Problem Sections:
• Loading Iris Dataset and reading it
The Iris dataset is taken from the http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data repository.
Later it is converted to a csv readable file format. It is read into 2 matrices with “x_features” of 150x4 matrix and “y_data” of 150x1 matrix. Next, we append the “x_features” matrix with column- 1 matrix to fill the position of x0 value to be 1’s for all 150 observations.
• Training the model using Linear Regression
Estimator for beta values calculated using the formula.
Sum of the squared error calculated using the formula.
• Classification of the trained model
• N-fold cross validation
Done using sklearn library for getting the indices of the training set and the testing set for X and Y matrices. 
For k-folds of 10 values the model is trained and tested using linear regression.


2. Data:
Iris data set where it is downloaded in the local machine : “iris.csv”

3. Method:
Loadtxt()
Concatenate()
Transpose()
Inv()
Using numpy library
from sklearn.model_selection import KFold -> KFold() is used


4. Results:
Below are the results for k-fold cross validation using the python language:

Kvalue Error_rate 

2      1.0368

3      1.6419

4      1.2145

5      1.3163

10     0.3305

Accuracy for 10 – folds : 99.26%

K value of 10 is better as error value is the least among all the folds cross validations when done on the trained model using Linear Regression.

References:

https://stackoverflow.com/questions/43156873/python-for-loop-how-to-save-every-iteration-under-different-variables

http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html#sklearn.model_selection.KFold

https://www.digitalocean.com/community/tutorials/how-to-construct-for-loops-in-python-3

https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6
