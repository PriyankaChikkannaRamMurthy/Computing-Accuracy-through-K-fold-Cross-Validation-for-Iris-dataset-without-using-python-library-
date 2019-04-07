
# coding: utf-8

# In[1]:


from numpy.linalg import inv
import numpy as np


# In[2]:


file = "iris.csv"


# In[3]:


x_features = np.loadtxt(file, delimiter=',', usecols=[0, 1, 2, 3])


# In[4]:


print(x_features)


# In[5]:


x_features.shape


# In[6]:


matrix_ones=np.ones((150,1))
print(matrix_ones)


# In[7]:


x_features_new=np.concatenate((matrix_ones, x_features), 1)
print(x_features_new)
x_features_new.shape


# In[8]:


data  = np.loadtxt(file, delimiter=',', usecols=[4], dtype=np.str)


# In[9]:


y_data = data.reshape((150,1))


# In[10]:


print(y_data)


# In[11]:


x_transpose =  x_features_new.transpose()


# In[12]:


print(x_transpose)


# In[13]:


x_transpose.shape


# In[14]:


x_multiply=np.matmul(x_transpose,x_features_new)


# In[15]:


print(x_multiply)


# In[16]:


x_multiply.shape


# In[17]:


x_inverse=inv(x_multiply)


# In[18]:


x_inverse.shape


# In[19]:


x_multiply2=np.matmul(x_inverse,x_transpose)


# In[20]:


print(x_multiply2)


# In[21]:


x_multiply2.shape


# In[22]:


for j in range(0,150):
    if y_data[j] == 'Iris-setosa':
        y_data[j]= 0
    elif y_data[j] == 'Iris-versicolor':
        y_data[j]= 1
    elif y_data[j] == 'Iris-virginica':
        y_data[j]= 2


# In[23]:


print(y_data)


# In[24]:


y_data.shape


# In[25]:


y_data = y_data.astype(float)


# In[26]:


beta = np.matmul(x_multiply2,y_data)


# In[27]:


print(beta.shape)


# In[28]:


y_estimated= np.matmul(x_features_new,beta)
print(y_estimated)
y_estimated.shape


# In[29]:


Sum_SquaredError = 0
for n in range(0,150):
    calculated_error= (y_estimated[n] - y_data[n])
    squared_error=np.square(calculated_error)
    Sum_SquaredError = squared_error + Sum_SquaredError
    
print(Sum_SquaredError)


# In[99]:


accuracy_rate=100-Sum_SquaredError
print("Accuracy of the predicted model using Linear Regression:",accuracy_rate,"%")


# In[31]:


new_matrix=np.concatenate((x_features_new, y_data), 1)
print(new_matrix)
new_matrix.shape


# In[32]:


np.random.shuffle(new_matrix)
print(new_matrix)
new_matrix.shape


# In[33]:


x_new = new_matrix[:, :-1].copy()
print(x_new)
x_new.shape


# In[34]:


y_new = new_matrix[:, -1].copy()
print(y_new)
y_new.shape


# In[111]:


from sklearn.model_selection import KFold
kf = KFold(n_splits=10)
print(kf)
split_value=(kf.get_n_splits(x_new))
print(split_value)


# In[112]:


count=1
error = []
for train_index, test_index in kf.split(x_new):
        print("k-Fold -", count)
        X_train, X_test = x_new[train_index], x_new[test_index]
        y_train, y_test = y_new[train_index], y_new[test_index]
        size = X_test.shape[0]
        #print("TRAin x size")
        #print(X_train.shape)
        #print("Test x size")
        #print(X_test.shape)
        #print("X-features Train Set:")
        #print(x_new[train_index])
        #print("X-features Test Set:")
        #print(x_new[test_index])
        #print("Train_y")
        #print(y_train.shape)
        #print("Test_y")
        #print(y_test.shape)
        #print("Y-features Train Set:")
        #print(y_new[train_index])
        #print("Y-features Test Set:")
        #print(y_new[test_index])
        s1= X_train.transpose()
        s1_multiply= np.matmul(s1,X_train)
        s1_inverse= inv(s1_multiply)
        s1_new= np.matmul(s1_inverse,s1)
        s1_beta= np.matmul(s1_new,y_train)
        y_estimate= np.matmul(X_test,s1_beta)
        
        Sum_SquaredError2 = 0
        for n in range(0,size):
            calculated_error2= (y_estimate[n] -  y_test[n])*(y_estimate[n] -  y_test[n])
            Sum_SquaredError2 = calculated_error2 + Sum_SquaredError2
            
        error.append(Sum_SquaredError2)
        print("Error for the fold",count,":")
        print(Sum_SquaredError2)
        count=count+1

print("List of Error valuesfor each folds :",error)
low_error=min(error)
print("Lowest error rate in the k-fold:",low_error)

err = 0
for e in error:
    err = err+e

total_error = err/int(split_value)
print("Average error for", split_value,"- folds : ",total_error)

accuracy_final=100-total_error
print("Accuracy for", split_value,"- folds : ",accuracy_final)

