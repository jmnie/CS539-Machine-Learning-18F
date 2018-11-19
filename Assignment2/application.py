import math
import numpy as np
from linear_regression import *
from sklearn.datasets import make_regression
# Note: please don't add any new package, you should solve this problem using only the packages above.
#-------------------------------------------------------------------------
'''
    Problem 2: Apply your Linear Regression
    In this problem, use your linear regression method implemented in problem 1 to do the prediction.
    Play with parameters alpha and number of epoch to make sure your test loss is smaller than 1e-2.
    Report your parameter, your train_loss and test_loss 
    Note: please don't use any existing package for linear regression problem, use your own version.
'''

#--------------------------

n_samples = 200
X,y = make_regression(n_samples= n_samples, n_features=4, random_state=1)
y = np.asmatrix(y).T
X = np.asmatrix(X)
Xtrain, Ytrain, Xtest, Ytest = X[::2], y[::2], X[1::2], y[1::2]

#########################################
## INSERT YOUR CODE HERE

def plot_result(x_content,train_loss,test_loss,label_content):
    import matplotlib.pyplot as plt
    from matplotlib.legend_handler import HandlerLine2D
    train_loss = np.array(train_loss)
    test_loss = np.array(test_loss)
    #length = [i for i in range(len(train_loss))]
    line1, = plt.plot(x_content, train_loss, label = 'Train Loss' )
    line2, = plt.plot(x_content, test_loss, label = 'Test Loss' )
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.legend(handler_map={line2: HandlerLine2D(numpoints=2)})
    plt.ylabel('MSE Loss')
    plt.xlabel(label_content[0])
    plt.title(label_content[1])
    plt.show()

def find_best_alpha(Xtrain,Ytrain,Xtest,Ytest):
    alpha = []
    epoch = 500
    train_loss = []
    interval = 1000
    test_loss = []
    for i in range(interval):
        temp_lr = (i+1)*(1/interval)
        alpha.append(temp_lr)
        w = train(Xtrain,Ytrain,temp_lr,epoch)
        train_yhat = np.array(Xtrain.dot(w))
        train_loss.append(compute_L(train_yhat,Ytrain))
        test_yhat = np.array(Xtest.dot(w))
        test_loss.append(compute_L(test_yhat,Ytest))
    
    best_alpha = alpha[np.argmin(np.array(test_loss))]
    print("Minimum Test Loss: ",test_loss[np.argmin(test_loss)])

    return alpha,train_loss,test_loss,best_alpha

def find_best_epoch(Xtrain,Ytrain,Xtest,Ytest,best_alpha):
    epoch = 200
    train_loss = []
    test_loss = []
    epoch_list = [i for i in range(epoch)]
    for i in range(epoch):
        w = train(Xtrain,Ytrain,best_alpha,i)
        train_yhat = compute_yhat(Xtrain,w)
        train_loss.append(compute_L(train_yhat,Ytrain))
        test_yhat = compute_yhat(Xtest,w)
        test_loss.append(compute_L(test_yhat,Ytest))
    
    best_epoch = epoch_list[np.argmin(np.array(test_loss))]
    print("Minimum Test Loss: ",test_loss[np.argmin(test_loss)])
    
    return epoch_list,train_loss,test_loss,best_epoch

def epoch_loss_plot(Xtrain,Ytrain,Xtest,Ytest):
    import matplotlib.pyplot as plt

    epoch = 80
    train_loss = []
    test_loss = []
    alpha = [(i+1)*(1/epoch) for i in range(epoch)]
    epoch_list = [i for i in range(epoch)]

    for i in range(epoch):

        temp_trainloss = []
        temp_testloss = []
        for j in range(epoch):
            w = train(Xtrain,Ytrain,alpha[j],i)
            train_yhat = compute_yhat(Xtrain,w)
            temp_trainloss.append(compute_L(train_yhat,Ytrain))
            test_yhat = compute_yhat(Xtest,w)
            temp_testloss.append(compute_L(test_yhat,Ytest))
        train_loss.append(temp_trainloss)
        test_loss.append(temp_testloss)

    test_loss = np.array(test_loss)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,6.15))

    X, Y = np.meshgrid(epoch_list, alpha)
    contours = ax[0].contour(X, Y, test_loss, 100)
    ax[0].clabel(contours)

    ax[0].set_title('MSE Loss')
    ax[0].set_xlabel(r'$Epoch$')
    ax[0].set_ylabel(r'$Alpha$')
    plt.show()




# First Part
alpha,train_loss,test_loss,best_alpha = find_best_alpha(Xtrain,Ytrain,Xtest,Ytest)
print("Best Alpha: ",best_alpha)
label_content = ['Alpha','Loss vs Alpha (Epoch = 500)']
plot_result(alpha,train_loss,test_loss,label_content)

### Second Part
epoch_list,train_loss,test_loss,best_epoch = find_best_epoch(Xtrain,Ytrain,Xtest,Ytest,best_alpha)
label_content = ['Epoch','Loss vs Epoch (Alpha = 0.689)']
print("Best Epoch: ",best_epoch)
plot_result(epoch_list,train_loss,test_loss,label_content)

epoch_loss_plot(Xtrain,Ytrain,Xtest,Ytest)






#########################################

