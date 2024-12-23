import numpy as np
import _pickle as pkl 

'''
    Custom Multilayer Perceptron Neural Network Implementation
    Created by: Caio Azevedo
    Last Update: 10-06-2021
'''

#region Activation Functions

def linear(x, derivative=False):
    return np.ones_like(x) if derivative else x

def sigmoid(x, derivative=False):
    if derivative:
        y = sigmoid(x)
        return y*(1 - y)
    return 1.0/(1.0 + np.exp(-x))

def tanh(x, derivative=False):
    if derivative:
        y = tanh(x)
        return 1 - y**2
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

def relu(x, derivative=False):
    if derivative:
        return np.where(x <= 0, 0, 1)
    return np.maximum(0, x)

def leaky_relu(x, derivative=False):
    alpha = 0.1
    if derivative:
        return np.where(x <= 0, alpha, 1)
    return np.where(x <= 0, alpha*x, x)

def elu(x, derivative=False):
    alpha = 1.0
    if derivative:
        y = elu(x)
        return np.where(x <= 0, y + alpha, 1)
    return np.where(x <= 0, alpha*(np.exp(x) - 1), x)

#endregion 

#region Utility Functions 
def softmax(x, y_oh=None, derivative=False):
    if derivative:
        y_pred = softmax(x)
        k = np.nonzero(y_pred * y_oh)
        pk = y_pred[k]
        y_pred[k] = pk * (1.0 - pk)
        return y_pred
    exp = np.exp(x)
    return exp / np.sum(exp, axis=1, keepdims=True)
#endregion

#region Cost Function

#region 2 Regression issues

def mae(y,y_pred, derivative = False):
    if derivative:
        return np.where(y_pred > y, 1, -1) / y.shape[0]
    return np.mean(no.abs(y - y_pred))

def mse(y, y_pred, derivative = False):
    if derivative:
        return -(y - y_pred) / y.shape[0]
    return 0.5 * np.mean((y - y_pred) ** 2)    
#endregion

#region Binary Classification

def binary_cross_entropy(y, y_pred, derivative= False):
    if derivative:
        return -(y - y_pred) / (y_pred * (1 - y_pred) * y.shape[0])
    return -np.mean(y*np.log(y_pred) +  (1 - y)*np.log(1 - y_pred))

def sigmoid_cross_entropy(y, y_pred, derivative= False):
    y_sigmoid = sigmoid(y_pred)
    if derivative:
        return -(y - y_sigmoid) / y.shape[0]
    return -np.mean(y*np.log(y_sigmoid) +  (1 - y) * np.log(1 - y_sigmoid))  
#endregion

#region Multiclass classification
def neg_log_likelihood(y_oh, y_pred, derivative=False):
    k = np.nonzero(y_pred * y_oh)
    pk = y_pred[k]
    if derivative:
        y_pred[k] = (-1.0 / pk)
        return y_pred
    return np.mean(-np.log(pk))

def softmax_neg_log_likelihood(y_oh, y_pred, derivative=False):
    y_softmax = softmax(y_pred)
    if derivative:
        return -(y_oh - y_softmax) / y_oh.shape[0]    
    return neg_log_likelihood(y_oh, y_softmax)
#endregion

#endregion

#region Weight Initialization

def zeros(rows,cols):
    return np.zeros((rows,cols))

def ones(rows,cols):
    return np.ones((rows,cols))

def random_normal(rows,cols):
    return np.random.randn(rows,cols)

def random_uniform(rows,cols):
    return np.random.rand(rows,cols)

def glorot_normal(rows,cols):
    std_dev = np.sqrt(2.0 / (rows + cols))
    return std_dev * np.random.randn(rows,cols)

def glorot_uniform(rows,cols):
    limit = np.sqrt(6.0 / (rows + cols))
    return 2.0 * limit * np.random.rand(rows, cols) - limit
#endregion 

#region Regularization

def l1_regularization(weights, derivative=False):
    if derivative:
        weights = [np.where(w < 0, -1, w) for w in weights]
        return np.array([np.where(w > 0, 1, w) for w in weights])
    return np.sum([np.abs(w) for w in weights])
                   
def l2_regularization(weights, derivative=False):
    if derivative:
        return weights
    return 0.5 * np.sum(weights**2)

#endregion

#region Batch Generator
def batch_sequential(x, y, batch_size=None):
    batch_size = x.shape[0] if batch_size is None else batch_size
    n_batches = x.shape[0] // batch_size
    
    for batch in range(n_batches):
        offset = batch_size * batch
        x_batch, y_batch = x[offset:offset+batch_size], y[offset:offset+batch_size]
        yield (x_batch, y_batch)
        
def batch_shuffle(x, y, batch_size=None):
    shuffle_index = np.random.permutation(range(x.shape[0])) #permutation shuffles the indexes
    return batch_sequential(x[shuffle_index],y[shuffle_index])
#endregion

#region Learning Rate Decay

def none_decay(learning_rate, epoch, decay_rate, decay_steps=1):
    return learning_rate

def time_based_decay(learning_rate, epoch, decay_rate, decay_steps=1):
    return 1.0 / (1 + decay_rate * epoch)

def exponential_decay(learning_rate, epoch, decay_rate, decay_steps=1):
    return learning_rate * decay_rate ** epoch

def staircase_decay(learning_rate, epoch, decay_rate, decay_steps=1):
    return learning_rate * decay_rate ** (epoch // decay_steps)
#endregion

#region Batch Normalization 

def batchnorm_forward(layer, x, is_training = True):
    mu = np.mean(x, axis=0) if is_training else layer._pop_mean
    var = np.var(x,axis=0) if is_training else layer._pop_var
    x_norm = (x - mu) / np.sqrt(var+1e-8)
    out = layer.gamma * x_norm + layer.beta
    
    if is_training:
        layer._pop_mean = layer.bn_decay * layer._pop_mean + (1.0 - layer.bn_decay) * mu
        layer._pop_var = layer.bn_decay * layer._pop_var + (1.0 - layer.bn_decay) * var
        layer._bn_cache = (x, x_norm, mu, var)
    return out

def batchnorm_backward(layer, dactivation):
    x, x_norm, mu, var = layer._bn_cache
    
    m = layer._activ_inp.shape[0] #numero de amostras 
    x_mu = x - mu
    std_inv = 1.0 / np.sqrt(var + 1e-8) #inverso do desvio padrão
    
    dx_norm = dactivation * layer.gamma
    dvar = np.sum(dx_norm * x_mu, axis=0) * -0.5 * (std_inv ** 3)
    dmu = np.sum(dx_norm * -std_inv, axis=0) + dvar * np.mean(-2.0 * x_mu, axis=0)
    
    dx = (dx_norm * std_inv) + (dvar * 2.0 * x_mu / m) + (dmu / m)
    layer._dgamma = np.sum(dactivation * x_norm, axis= 0)
    layer._dbeta = np.sum(dactivation, axis=0)
    return dx
#endregion    

#region Gradient Checking

def __compute_approx_grads(nn, x, y, eps=1e-4): #"compute approximate gradient"
    approx_grads = []   #"Approximate gradients by any epsilon" (lim f(x+eps)-f(x)/eps, com eps->0)
    feed_forward = lambda inp: nn._NeuralNetwork__feedforward(inp, is_training=True) #"Simulate network training"
                                                                            # f. lambda 'feed_forward' with parameter 'inp' that 
                                                                            #is passed to the 'feedforward' method of the neural network 'nn'."
    #"For each weight of each layer, the approximate gradients are computed."
    for layer in nn.layers:
        assert(layer.dropout_prob == 0), "Gradient checking can not be applied in ANN with DROPOUT"
        
        w_ori = layer.weights.copy() #"Copy to avoid altering the original weights of the network."
        w_ravel = w_ori.ravel()      
        w_shape = w_ori.shape
        
        for i in range(w_ravel.size):
            w_plus = w_ravel.copy()
            w_plus[i] += eps
            layer.weights = w_plus.reshape(w_shape)
            J_plus = nn.cost_func(y, feed_forward(x)) + (1.0 / y.shape[0] * layer.reg_strength * layer.reg_func(layer.weights))
            
            w_minus = w_ravel.copy()
            w_minus[i] -= eps
            layer.weights = w_minus.reshape(w_shape)
            J_minus = nn.cost_func(y, feed_forward(x)) + (1.0 / y.shape[0] * layer.reg_strength * layer.reg_func(layer.weights))
            
            approx_grads.append((J_plus-J_minus) / (2.0 * eps))
        layer.weights = w_ori
    
    return approx_grads
    
#"Check how close the real and approximate gradients are."
def gradient_checking(nn, x, y, eps=1e-4, verbose=False, verbose_precision=5): #verifica se 
    from copy import deepcopy #"Used to make a copy of an object, in this case, of 'nn'."
    nn_copy = deepcopy(nn)
    
    nn.fit(x,y, epochs=0)
    grads = np.concatenate([layer._dweights.ravel() for layer in nn.layers]) #Original gradients
    
    approx_grads = __compute_approx_grads(nn_copy,x, y, eps) #compute the approximate gradients
    
    is_close = np.allclose(grads, approx_grads) #Check if the gradients are close with high precision, up to 8 decimal places.
    print("{}".format("\033[92mGRADIENTES OK" if is_close else "\033[91mGRADIENTS FAIL"))
    
    #Calculation of the relative errors between the gradients (any difference smaller than 1e-4 is acceptable).
    norm_num = np.linalg.norm(grads - approx_grads) #compute the norm of the differences.
    norm_den = np.linalg.norm(grads) + np.linalg.norm(approx_grads) #Sum of the norms.
    error = norm_num / norm_den  #Relative error (norm of the differences / sum of the norms).
    print("Relative error", error)
    
    if verbose:
        np.set_printoptions(precision=verbose_precision, linewidth=200, suppress=True)
        print("Gradients", grads)
        print("Approximated", np.array(approx_grads))

#endregion

#region Layer Implementation

class Layer():
    def __init__(self, input_dim, output_dim, weights_initializer = random_normal, biases_initializer = ones, activation=linear, dropout_prob = 0.0, reg_func=l2_regularization, reg_strength=0, batch_norm=False, bn_decay=0.9, is_trainable=True):
        self.input = None
        self.weights = weights_initializer(output_dim, input_dim) #self.weights = np.random.randn(output_dim, input_dim)
        self.biases  = biases_initializer(1, output_dim) #self.biases  = np.random.randn(1, output_dim)
        self.activation = activation
        self.dropout_prob = dropout_prob
        self.reg_func = reg_func #(l1 or l2)
        self.reg_strength = reg_strength #(hyperparameter - lambda)
        
        #Batch Normalization
        self.batch_norm = batch_norm
        self.bn_decay = bn_decay
        self.gamma, self.beta = ones(1,output_dim),zeros(1,output_dim) 
        #Freezing in backpropagation
        self.is_trainable = is_trainable
        
        
        #prefix '-', used for private attribute.
        self._activ_inp, self._activ_out = None, None
        self._dweights, self._dbiases, self._prev_dweights = None, None, 0.0
        #stores the attributes that underwent dropout.
        self._dropout_mask = None      
        self._dgamma, self._dbeta = None, None
        self._pop_mean, self._pop_var = zeros(1,output_dim), zeros(1,output_dim) #stores the estimated mean and variance of the population (batch norm).
        self._bn_cache = None 
#endregion
 
class MLPNeuralNetwork():
    def __init__(self,cost_func=mse, learning_rate = 1e-3, lr_decay_method = none_decay, lr_decay_rate= 0.0, lr_decay_steps = 1, momentum = 0.0, patience = np.inf):
        self.layers = []
        self.cost_func = cost_func
        #Stores the initial learning rate for the lr_decay methods.
        self.learning_rate = self.lr_initial = learning_rate 
        self.lr_decay_method = lr_decay_method
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_steps = lr_decay_steps
        #
        self.momentum = momentum
        #early stopping - Patience factor and waiting time (in epochs)
        self.patience, self.waiting = patience, 0   
        self._best_model, self._best_loss = self.layers, np.inf
    
    def fit(self,x_train, y_train, x_val=None, y_val=None, epochs=100,verbose=10,batch_gen=batch_sequential, batch_size=None):
        #early stopping - initialize a validation subset with training data if not provided.
        x_val, y_val = (x_train, y_train) if (x_val is None or y_val is None) else (x_val, y_val)
        
        for epoch in range(epochs + 1):
            #update the learning rate, considering possible decay.
            self.learning_rate = self.lr_decay_method(self.lr_initial, epoch, self.lr_decay_rate, self.lr_decay_steps)
            #mini-bacth
            for x_batch, y_batch in batch_gen(x_train, y_train, batch_size):
                y_pred = self.__feedforward(x_batch)
                self.__backprop(y_batch,y_pred)
            #early stopping 
            loss_val = self.cost_func(y_val, self.predict(x_val)) #compute loss at each epoch.
            if loss_val < self._best_loss:   #is it better?
                self._best_model, self._best_loss = self.layers, loss_val
                self.waiting = 0   #Reset the waiting time.
            else:
                self.waiting += 1
                #print("not improving! [{}] current: {} best: {}".format(self.waiting, loss_val, self._best_loss))
                if self.waiting >= self.patience:
                    self.layers = self._best_model
                    #print("early stopping at epoch ",epoch)
                    return
            
            if epoch % verbose == 0:
                loss_train = self.cost_func(y_train,self.predict(x_train)) #loss function
                loss_reg = (1.0 / y_train.shape[0]) * np.sum([layer.reg_strength * layer.reg_func(layer.weights) for layer in self.layers]) #loss regularitazion
                print("epoch: {0:=4}/{1} loss_train: {2:.8f} + {3:.8f} = {4:.8f} loss_val = {5:.8f}".format(epoch,epochs,loss_train, loss_reg, loss_train + loss_reg,loss_val))
    
    def predict(self,x):
        return self.__feedforward(x, is_training=  False)
    
    #wb: write bytes; -1 no compression and unlimited file size.
    def save(self, file_path):
        pkl.dump(self,open(file_path, 'wb'), -1)
    
    def load(file_path):
        return pkl.load(open(file_path,'rb')) #rb read bytes
    
    #'__' This prefix makes the 'method private'.
    def __feedforward(self,x,is_training = True):
        self.layers[0].input = x
        for current_layer, next_layer in zip(self.layers, self.layers[1:] + [Layer(0,0)]): #addition of an empty layer so both have the same size.
            y = np.dot(current_layer.input,current_layer.weights.T) + current_layer.biases #calculation of the activation function input.
            #batch normalization
            y = batchnorm_forward(current_layer, y, is_training) if current_layer.batch_norm else y
            #create an activation mask (on/off) for the neurons based on probability, and rescale the activations for each iteration.
            current_layer._dropout_mask = np.random.binomial(1,(1.0 - current_layer.dropout_prob),y.shape) / (1.0 - current_layer.dropout_prob)
            current_layer._activ_inp = y
            
            #compute the activation function (output of the current layer becomes the input of the next) & apply the dropout mask.
            current_layer._activ_out = current_layer.activation(y) * (current_layer._dropout_mask if is_training else 1)  
            next_layer.input = current_layer._activ_out
        return self.layers[-1]._activ_out
        
    #compute the derivatives and update the weights and biases.
    def __backprop(self, y, y_pred):
        last_delta = self.cost_func(y, y_pred, derivative=True)
        for layer in reversed(self.layers):
            #Derivative of the activation function, only for the active neurons (due to dropout).
            dactivation = layer.activation(layer._activ_inp,derivative=True) * last_delta * layer._dropout_mask
            #batch normalization - 
            dactivation = batchnorm_backward(layer, dactivation) if layer.batch_norm else dactivation
            
            last_delta = np.dot(dactivation, layer.weights) 
            layer._dweights = np.dot(dactivation.T, layer.input)
            layer._dbiases = 1.0 * dactivation.sum(axis=0, keepdims=True)
            
        for layer in reversed(self.layers):
            if layer.is_trainable:
                #increment backprop with the derivative of regularization (L1 or L2) before updating the weights.
                layer._dweights = layer._dweights + (1.0 / y.shape[0]) * layer.reg_strength * layer.reg_func(layer.weights,derivative=True)

                #compute the gradient of the previous iteration by applying the momentum factor (weight update).
                layer._prev_dweights = -self.learning_rate * layer._dweights + (self.momentum * layer._prev_dweights)
                layer.weights = layer.weights + layer._prev_dweights 
                #layer.weights = layer.weights - self.learning_rate * layer._dweights # atualização dos pesos sem momentum

                layer.biases = layer.biases - self.learning_rate * layer._dbiases
                #batch normalization - gamma and beta are new parameters of the network, just like weights and biases.
                if layer.batch_norm:
                    layer.gamma = layer.gamma - self.learning_rate * layer._dgamma
                    layer.beta = layer.beta - self.learning_rate * layer._dbeta
