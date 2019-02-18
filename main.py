import os
import numpy
import scipy
import matplotlib
import mnist
import pickle
matplotlib.use('agg')
from matplotlib import pyplot

mnist_data_directory = os.path.join(os.path.dirname(__file__), "data")

# TODO add any additional imports and global variables
def run_and_plot(reg, step, iter, X_train, Y_train, X_val, Y_val, W0, num_iters, mon_freq):
    Ws = gradient_descent(X_train, Y_train, gamma, W0, alpha,num_iters, mon_freq )
    plot_given_W_vectors(X_train, X_val, Y_train, Y_val, Ws, 100)
    plot_given_W_vectors(X_train, X_val, Y_train, Y_val, Ws, 1000)


def plot_given_W_vectors(X_train, X_val, Y_train, Y_val, Ws, nsamples):
    y = np.zeros(2, len(Ws))
    x = np.zeros(1, len(Ws))
    for i in range(len(Ws)):
        y[0, i]= multinomial_logreg_error(X_train, Y_train, W)
        y[0, i] = estimate_multinomial_logreg_error(X_train, Y_train, W, nsamples)
        x[i] = i
        plot(x, y[0, :], x, y[1, :])
    y = np.zeros(2, len(Ws) / 10)
    x = np.zeros(1, len(Ws) / 10)
    for i in range(len(Ws)):
        y[0, i]= multinomial_logreg_error(X_val, Y_val, W)
        y[0, i] = estimate_multinomial_logreg_error(X_val Y_val, W, nsamples)
        x[i] = i
        plot(x, y[0, :], x, y[1, :])


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
    pred_lables=np.zeros((c,n))
    label_position=numpy.argmax(preds_real,axis=0)
    for j,i in enumerate(label_position):
        pred_lables[i,j]=1
    #pred_lables=(np.zeros((c,n))+(1==preds_real/np.max(preds_real,axis=0)))
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
    n = np.shape(Xs)[1]
    random_samples = numpy.random.randint(n, size=(1, nsamples)).flatten()
    X_subsample = Xs[:,random_samples]
    Y_subsample = Ys[:,random_samples]
    approx_error = multinomial_logreg_error(X_subsample, Y_subsample, W)
    return approx_error

if __name__ == "__main__":
    (Xs_tr, Ys_tr, Xs_te, Ys_te) = load_MNIST_dataset()
    d= Xs_tr.shape()[0]
    c= Ys_tr.shape()[0]
    W0 = np.zeros(c,d)
    run_and_plot(.0001, 1, 1000, Xs_tr, Ys_tr, Xs_te, Ys_te, W0)

