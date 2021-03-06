import os
import numpy
import scipy
import matplotlib
import mnist
import pickle
import scipy.special
matplotlib.use('agg')
from matplotlib import pyplot as plt
import time

mnist_data_directory = os.path.join(os.path.dirname(__file__), "data")

# TODO add any additional imports and global variables



def load_MNIST_dataset():
    PICKLE_FILE = os.path.join(mnist_data_directory, "MNIST.pickle")
    try:
        dataset = pickle.load(open(PICKLE_FILE, 'rb'))
    except:
        # load the MNIST dataset
        mnist_data = mnist.MNIST(mnist_data_directory, return_type="numpy", gz=True)
        Xs_tr, Lbls_tr = mnist_data.load_training();
        Xs_tr = Xs_tr.transpose() / 255.0
        Ys_tr = numpy.zeros((10, 60000))
        for i in range(60000):
            Ys_tr[Lbls_tr[i], i] = 1.0  # one-hot encode each label
        Xs_te, Lbls_te = mnist_data.load_testing();
        Xs_te = Xs_te.transpose() / 255.0
        Ys_te = numpy.zeros((10, 10000))
        for i in range(10000):
            Ys_te[Lbls_te[i], i] = 1.0  # one-hot encode each label

        dataset = (Xs_tr, Ys_tr, Xs_te, Ys_te)
        pickle.dump(dataset, open(PICKLE_FILE, 'wb'))
    return dataset

#calculate the prediction vector given W and X, namely softmax(WX)
#
# X        training examples (d * n)
# W         parameters       (c * d)
#
#returns preds, a (c * n) matrix of predictions
def mult_logreg_pred(W,X):
    WX= numpy.matmul(W,X) #apply the linear parameters
    preds=scipy.special.softmax(WX, axis=0) # apply softmax with respect to each observation
    
    return preds

# compute the gradient of the multinomial logistic regression objective, with regularization
#
# Xs        training examples (d * n)
# Ys        training labels   (c * n)
# gamma     L2 regularization constant
# W         parameters        (c * d)
#
# returns   the gradient of the model parameters
def multinomial_logreg_grad(Xs, Ys, gamma, W):
    # TODO students should implement this
    c, d = numpy.shape(W)
    dummy, n=numpy.shape(Xs)
    grad= numpy.zeros((c,d)) # gradient has the same size as W
    H=mult_logreg_pred(W,Xs)
    signed_error=(H-Ys)
    grad=(numpy.matmul(signed_error,Xs.T))/n+gamma*W
    return grad

# compute the error of the classifier
#
# Xs        examples          (d * n)
# Ys        labels            (c * n)
# W         parameters        (c * d)
#
# returns   the model error as a percentage of incorrect labels
def multinomial_logreg_error(Xs, Ys, W):
    # TODO students should implement this
    preds_real=mult_logreg_pred(W,Xs)
    c, n = numpy.shape(preds_real)
    pred_lables=numpy.zeros((c,n))
    label_position=numpy.argmax(preds_real,axis=0)
    for j,i in enumerate(label_position):
        pred_lables[i,j]=1
    #pred_lables=(numpy.zeros((c,n))+(1==preds_real/numpy.max(preds_real,axis=0)))
    #The above finds the max in each row and returns a matrix which has 1 at the greatet
    # probablility, and zero else where
    accuracy=numpy.trace(numpy.matmul(Ys,pred_lables.T))/n
    # we sum the vectors which only have one value equal to 1 in each column. thus the maximum
    # value per column is 2, which happens only if the lables agree. (sum(maxima)-n)/n gives accuracy

    error_percent=100-accuracy*100
    return error_percent

# run gradient descent on a multinomial logistic regression objective, with regularization
#
# Xs            training examples (d * n)
# Ys            training labels   (d * c)
# gamma         L2 regularization constant
# W0            the initial value of the parameters (c * d)
# alpha         step size/learning rate
# num_iters     number of iterations to run
# monitor_freq  how frequently to output the parameter vector
#
# returns       a list of models parameters, one every "monitor_freq" iterations
def gradient_descent(Xs, Ys, gamma, W0, alpha, num_iters, monitor_freq):
    # TODO students should implement this
    param_list=[]
    W_prev=W0;
    j=0;
    for i in range(num_iters):
        j+=1
        grad_f = multinomial_logreg_grad(Xs, Ys, gamma, W_prev);
        W_next = W_prev - alpha*grad_f
        if j == monitor_freq:
            param_list.append(W_next)
            j=0
            print("difference in W's",numpy.linalg.norm(W_prev-W_next))
        W_prev=W_next


    return param_list

# estimate the error of the classifier
#
# Xs        examples          (d * n)
# Ys        labels            (c * n)
# gamma     L2 regularization constant
# W         parameters        (c * d)
# nsamples  number of samples to use for the estimation
#
# returns   the error of the model
def estimate_multinomial_logreg_error(Xs, Ys, W, nsamples):
    # TODO students should implement this
    n = numpy.shape(Xs)[1]
    random_samples = numpy.random.randint(n, size=(1, nsamples)).flatten()
    X_subsample = Xs[:,random_samples]
    Y_subsample = Ys[:,random_samples]
    approx_error = multinomial_logreg_error(X_subsample, Y_subsample, W)
    return approx_error

if __name__ == "__main__":
    (Xs_tr, Ys_tr, Xs_te, Ys_te) = load_MNIST_dataset()
    d= Xs_tr.shape[0]
    c= Ys_tr.shape[0]
    
    #establish local variables
    monitor_freq = 10
    gamma = 0.0001
    alpha = 1
    num_iters = 1000
    W0 = numpy.random.rand(c,d) # we chose initialization with all zeros

    #first we run gradient descent
    Ws = gradient_descent(Xs_tr, Ys_tr, gamma, W0, alpha, num_iters, monitor_freq)

    #now having the list of weights we create plots of the errors
    iteration = range(monitor_freq,(len(Ws)+1)*monitor_freq,monitor_freq)
    errors_true_train = []
    errors_true_test = []
    errors_est_train_100 = []
    errors_est_test_100 = []
    errors_est_train_1000 = []
    errors_est_test_1000 = []
    for W in Ws:
        errors_true_train.append(multinomial_logreg_error(Xs_tr, Ys_tr, W))
        errors_true_test.append(multinomial_logreg_error(Xs_te, Ys_te, W))
        errors_est_train_100.append(estimate_multinomial_logreg_error(Xs_tr, Ys_tr, W, 100))
        errors_est_test_100.append(estimate_multinomial_logreg_error(Xs_te, Ys_te, W, 100))
        errors_est_train_1000.append(estimate_multinomial_logreg_error(Xs_tr, Ys_tr, W, 1000))
        errors_est_test_1000.append(estimate_multinomial_logreg_error(Xs_te, Ys_te, W, 1000))
    
    plt.figure()
    plt.title("error measurements on training set over each iteration")
    plt.xlabel("iteration")
    plt.ylabel("error")
    plt.plot(iteration,errors_true_train, 'r-')
    plt.plot(iteration,errors_est_train_100, 'g-')
    plt.plot(iteration,errors_est_train_1000, 'b-')
    plt.savefig("train_errs.pdf")


    plt.figure()
    plt.title("error measurements on testing set over each iteration")
    plt.xlabel("iteration")
    plt.ylabel("error")
    plt.plot(iteration,errors_true_test, 'r-')
    plt.plot(iteration,errors_est_test_100, 'g-')
    plt.plot(iteration,errors_est_test_1000, 'b-')
    plt.savefig("test_errs.pdf")

# finaly we compare the runtime of the error estimates
    start_time=time.time()
    multinomial_logreg_error(Xs_tr, Ys_tr, W)
    error_full_tr=start_time-time.time()

    start_time=time.time()
    multinomial_logreg_error(Xs_te, Ys_te, W)
    error_full_te=start_time-time.time()

    start_time=time.time()
    estimate_multinomial_logreg_error(Xs_tr, Ys_tr, W, 100)
    error_est_tr_100=start_time-time.time()

    start_time=time.time()
    estimate_multinomial_logreg_error(Xs_te, Ys_te, W, 100)
    error_est_te_100=start_time-time.time()

    start_time=time.time()
    estimate_multinomial_logreg_error(Xs_tr, Ys_tr, W, 1000)
    error_est_tr_1000=start_time-time.time()

    start_time=time.time()
    estimate_multinomial_logreg_error(Xs_te, Ys_te, W, 1000)
    error_est_te_1000=start_time-time.time()

    print("full evaluation of error on training set runtime: ",error_full_tr)
    print("estimate (100) evaluation of error on training set runtime: ",error_est_tr_100)
    print("estimate (1000) evaluation of error on training set runtime: ",error_est_tr_1000)
    print("\n")
    print("full evaluation of error on test set runtime: ",error_full_te)
    print("estimate (100) evaluation of error on test set runtime: ",error_est_te_100)
    print("estimate (1000) evaluation of error on test set runtime: ",error_est_te_1000)
