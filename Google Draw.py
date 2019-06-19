import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
from PIL import Image
from scipy import ndimage
import random
import cv2
np.set_printoptions(threshold=np.nan)


def load():
    """
         Used to load the dataset from the file
         Returns:
             datas -- dictionary where the key is the type and numpy array the training examples
             name -- the name of various types of examples or the keys of datas
    """

    Fn = 'C:\\Users\\Arsh Ahmed\\Desktop\\Final Google Draw\\data\\'
    name = ['airplane', 'apple', 'banana','basketball','bicycle','car','cat','face','fish','flower',\
             'horse', 'house', 'laptop', 'table', 'tree']
    datas = {}
    for i in name:
        datas[i] = np.load(Fn + str(i) + '.npy')
    return datas, name




def make_train_set(datas, cur):
    """
        Used to make the train set for the classification of the given name
        Returns:
            train_x: Set of x training examples
            train_y: Corresponding labels to train_x examples
    """
    m = len(datas)
    train_x = []
    train_y = []
    dn = name[:]
    dn.remove(cur)
    for i in range(2 * (m - 1) * 1000):  # Take 1000 examples of each example
        num = random.randint(10, m * 1000 * 10)
        t = num % m
        if (t == 0):
            train_x.append(datas[dn[t]][i])
            train_y.append(0)
        elif (t == 1):
            train_x.append(datas[dn[t]][i])
            train_y.append(0)
        elif (t == 2):
            train_x.append(datas[dn[t]][i])
            train_y.append(0)
        elif (t == 3):
            train_x.append(datas[dn[t]][i])
            train_y.append(0)
        elif (t == 4):
            train_x.append(datas[dn[t]][i])
            train_y.append(0)
        elif (t == 5):
            train_x.append(datas[dn[t]][i])
            train_y.append(0)
        elif (t == 6):
            train_x.append(datas[dn[t]][i])
            train_y.append(0)
        elif (t == 7):
            train_x.append(datas[dn[t]][i])
            train_y.append(0)
        elif (t == 8):
            train_x.append(datas[dn[t]][i])
            train_y.append(0)
        elif (t == 9):
            train_x.append(datas[dn[t]][i])
            train_y.append(0)
        elif (t == 10):
            train_x.append(datas[dn[t]][i])
            train_y.append(0)
        elif (t == 11):
            train_x.append(datas[dn[t]][i])
            train_y.append(0)
        elif (t == 12):
            train_x.append(datas[dn[t]][i])
            train_y.append(0)
        elif (t == 13):
            train_x.append(datas[dn[t]][i])
            train_y.append(0)
        else:
            train_x.append(datas[cur][i])
            train_y.append(1)
    train_y = np.array(train_y)
    train_y = train_y.reshape(1, train_y.size)
    train_x = np.array(train_x).T

    return train_x / 255, train_y


def make_test_set(datas , cur):
    m = len(datas)
    train_x = []
    train_y = []
    dn = name[:]
    dn.remove(cur)
    for i in range(1000):  # Take 1000 examples of each example
        num = random.randint(10, m * 1000 * 10)
        check = random.randint(1, 50000)
        t = num % m
        if (t == 0):
            train_x.append(datas[dn[t]][check])
            train_y.append(0)
        elif (t == 1):
            train_x.append(datas[dn[t]][check])
            train_y.append(0)
        elif (t == 2):
            train_x.append(datas[dn[t]][check])
            train_y.append(0)
        elif (t == 3):
            train_x.append(datas[dn[t]][check])
            train_y.append(0)
        elif (t == 4):
            train_x.append(datas[dn[t]][check])
            train_y.append(0)
        elif (t == 5):
            train_x.append(datas[dn[t]][check])
            train_y.append(0)
        elif (t == 6):
            train_x.append(datas[dn[t]][check])
            train_y.append(0)
        elif (t == 7):
            train_x.append(datas[dn[t]][check])
            train_y.append(0)
        elif (t == 8):
            train_x.append(datas[dn[t]][check])
            train_y.append(0)
        elif (t == 9):
            train_x.append(datas[dn[t]][check])
            train_y.append(0)
        elif (t == 10):
            train_x.append(datas[dn[t]][check])
            train_y.append(0)
        elif (t == 11):
            train_x.append(datas[dn[t]][check])
            train_y.append(0)
        elif (t == 12):
            train_x.append(datas[dn[t]][check])
            train_y.append(0)
        elif (t == 13):
            train_x.append(datas[dn[t]][check])
            train_y.append(0)
        else:
            train_x.append(datas[cur][check])
            train_y.append(1)
    train_y = np.array(train_y)
    test_y = np.array(train_y)
    test_y = test_y.reshape(1, train_y.size)
    test_x = np.array(train_x).T

    return test_x / 255, test_y



def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy

    Arguments:
    Z -- numpy array of any shape

    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """

    A = 1 / (1 + np.exp(-Z))
    cache = Z

    return A, cache


def relu(Z):
    """
    Implement the RELU function.
    Arguments:
    Z -- Output of the linear layer, of any shape
    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """

    A = np.maximum(0, Z)

    assert (A.shape == Z.shape)

    cache = Z
    return A, cache


def softmax(Z):
    temp = np.exp(Z)
    s = np.sum(temp)
    temp = temp / s
    temp.reshape(temp.size, 1)
    return temp, Z


def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.
    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently
    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    Z = cache
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.

    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)

    return dZ


def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.
    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently
    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    Z = cache

    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    assert (dZ.shape == Z.shape)

    return dZ


def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """

    parameters = {}
    L = len(layer_dims)  # number of layers in the network

    for l in range(1, L):

        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(2 / layer_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))


        assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters


def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """

    Z = np.dot(W, A) + b

    assert (Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value
    cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """

    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".

        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)


    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".

        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    elif activation == 'softmax':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = softmax(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, parameters, ):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()

    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network

    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A

        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)],
                                             activation='relu')
        caches.append(cache)


    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.

    #    AL, cache = linear_activation_forward(A, parameters['W'+str(L)],parameters['b'+str(L)] , activation='sigmoid')
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation='sigmoid')
    caches.append(cache)

    assert (AL.shape == (1, X.shape[1]))

    return AL, caches


def compute_cost(AL, Y):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """

    m = Y.shape[1]

    # Compute loss from aL and y.

    cost = np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL)) / (-m)
    #    cost=np.sum(np.sum(-Y*np.log(AL),axis=0))/m

    cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert (cost.shape == ())

    return cost


def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation='sigmoid'):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.

    Arguments:
    dA -- post-activation gradient for current layer l
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache

    if activation == "relu":

        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)


    elif activation == "sigmoid":

        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == 'softmax':
        dZ = dA - train_y
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ...
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...
    """
    grads = {}
    L = len(caches)  # the number of layers
    m = AL.shape[1]
    #    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL

    # Initializing the backpropagation

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    #    dAL=AL-Y

    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]

    current_cache = caches[-1]
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL,
                                                                                                      current_cache,
                                                                                                      activation='sigmoid')

    # Loop from l=L-2 to l=0
    for l in reversed(range(L - 1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads['dA' + str(l + 1)], current_cache,
                                                                    activation='relu')
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent

    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients, output of L_model_backward

    Returns:
    parameters -- python dictionary containing your updated parameters
                  parameters["W" + str(l)] = ...
                  parameters["b" + str(l)] = ...
    """

    L = len(parameters) // 2  # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.

    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters


def L_layer_model(X, Y, layers_dims, learning_rate=0.05, num_iterations=3000, print_cost=False):  # lr was 0.009
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []  # keep track of cost

    # Parameters initialization. (≈ 1 line of code)

    parameters = initialize_parameters_deep(layers_dims)


    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.

        AL, caches = L_model_forward(X, parameters)


        # Compute cost.

        cost = compute_cost(AL, Y)


        # Backward propagation.

        grads = L_model_backward(AL, Y, caches)


        # Update parameters.

        parameters = update_parameters(parameters, grads, learning_rate)


        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # plot the cost
    # plt.plot(np.squeeze(costs))
    # plt.ylabel('cost')
    # plt.xlabel('iterations (per tens)')
    # plt.title("Learning rate =" + str(learning_rate))
    # plt.show()

    return parameters


def Predict(X,test_y, parameters):
    """
    This function is used to predict the results of a  L-layer neural network.

    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model

    Returns:
    p -- predictions for the given dataset X
    """

    m = X.shape[1]
    n = len(parameters) // 2  # number of layers in the neural network
    p = np.zeros((1, m))

    # Forward propagation
    probas, caches = L_model_forward(X, parameters)

    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    # print results
    # print ("predictions: " + str(p))
    # print ("true labels: " + str(y))
    print("Accuracy: " + str(np.sum((p == test_y) / m)))
    probas = np.squeeze(probas)
    #    print('Accuracy: ', probas)
    #    print("I guess it's a",name," with", probas*100, "% probability")

    return p


def save_params(parameters, name):
    Fn='C:\\Users\\Arsh Ahmed\\Desktop\\Final Google Draw\\params\\'
    Fn = Fn + name + '\\'
    L=len(parameters)//2
    for l in range(L):
        np.save(Fn+"W"+str(l+1)+".npy",parameters["W" + str(l+1)])
        np.save(Fn+"b"+str(l+1)+".npy",parameters["b" + str(l+1)])


def model(name):
    for i in name:
        print('The current training is for', i)
        train_x,train_y = make_train_set(datas,i)
        parameters = L_layer_model(train_x, train_y,layers_dims, num_iterations = 800, print_cost = True)
        test_x,test_y=make_test_set(datas, i)
        save_params(parameters, i)
        Predict(test_x, test_y, parameters)


def predict_test(X, parameters, name):
    """
    This function is used to predict the results of a  L-layer neural network.

    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model

    Returns:
    p -- predictions for the given dataset X
    """

    m = X.shape[1]
    n = len(parameters) // 2  # number of layers in the neural network
    p = np.zeros((1, m))

    # Forward propagation
    probas, caches = L_model_forward(X, parameters)

    # convert probas to 0/1 predictions


    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    probas = np.squeeze(probas)

    print("I guess it's a", name, " with", probas * 100, "% probability")

    return p


def predict(name,X):
    fn='C:\\Users\\Arsh Ahmed\\Desktop\\Final Google Draw\\params\\'
    for i in name:
        parameters={}
        t=fn+str(i)+'\\'
        for l in range(len(layers_dims)-1):
            parameters['W'+str(l+1)]=np.load(t+'W'+str(l+1)+'.npy')
            parameters['b'+str(l+1)]=np.load(t+'b'+str(l+1)+'.npy')
        predict_test(X,parameters,i)


def test_image(image):
    image = cv2.imread(image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('gray_image.png', gray_image)
    num_px = 28
    image = np.array(ndimage.imread('gray_image.png', flatten=False))
    my_image = scipy.misc.imresize(image, size=(num_px, num_px)).reshape((1, num_px * num_px)).T

    plt.imshow(my_image.reshape(28, 28))
    plt.show()
    predict(name, my_image)




layers_dims = [784, 50,50, 25, 1]
datas, name=load()
##################model(name)
test_image('car.png')
#predict(name,datas['cat'][16022].reshape(784,1))
#plt.imshow(datas['cat'][16022].reshape(28,28))
#plt.show()

