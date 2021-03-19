import numpy as np
import pandas as pd
import queue
import math
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from random import *

class NeuralNetwork:
    """
    Class that creates a NN and includes methods to train and test it
    
    Parameters
    ---------
    setup
        the architecture of the network, where each element in the
        list is a separate layer. Index 0 of each layer is how many
        inputs go into the layer. Index 1 represents the number of
        output nodes for the layer. Index 2 represents the activation
        function for the layer.
    lr
        learning rate
    seed
        random seed
    bias
        value to initialize all of the biases to

    """
    def __init__(self, setup=[[68,25,"sigmoid"],[25,1,"sigmoid"]],lr=.05,seed=1,bias=0.5):
        self._lr = lr
        self._seed = seed
        self._bias = bias
        
        # stores weights
        weights = []
        # stores activations
        outputs = []
        # stores deltas for backprop
        change = []
        
        # initialize the given number of layers with weights
        for layer in setup:
            weights.append(self.make_weights(n_inputs=layer[0],n_nodes=layer[1]))
            outputs.append([0] * layer[1])
            change.append([0] * layer[1])
        
        self._weights = weights
        self._outputs = outputs
        self._change = change
        
    @property
    def lr(self):
        return self._lr

    @lr.setter
    def lr(self, lr):
        self._lr = lr

    @property
    def bias(self):
        return self._bias

    @bias.setter
    def bias(self, bias):
        self._bias = bias 
    
    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed 
    
    @property
    def outputs(self):
        return self._outputs

    @outputs.setter
    def outputs(self, outputs):
        self._outputs = outputs 
        
    @property
    def change(self):
        return self._change

    @change.setter
    def change(self, change):
        self._change = change 
    
    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = weights 
        
        
    def make_weights(self,n_inputs, n_nodes):
        """
        Generates random weights for the network initialization

        Parameters
        ---------
        n_inputs
            Number of input nodes to this layer
        n_nodes
            Number of nodes to generate weights for
            
        Returns
        ---------
        Layer with random weights initialized for each node
        """
        layer = []
        
        # Get n_inputs random float between -1 and 1 for each node
        for i in range(n_nodes):
            node_weights = [uniform(-1, 1) for j in range(n_inputs)]
            # add bias at end, whose starting value is defined as 0.5 during 
            # initialization. I tried also randomly initializing bias as well
            # but not sure if it was better or not
            node_weights.append(self.bias)
            layer.append(node_weights)
        
        return layer

    def feedforward(self, data):
        """
        Takes in data and passes it through the NN

        Parameters
        ---------
        data
            One datapoint
            
        Returns
        ---------
        The output(s) of the final layer in the network
        """
        inputs = data
        
        # pass data through all layers
        for layer in range(len(self.weights)):
            next_inputs = []
            for node in range(len(self.weights[layer])):
                sum = 0
                for i in range(len(inputs)): # multiply inputs by weights and add to sum
                    sum += inputs[i]*self.weights[layer][node][i]
                    
                sum += self.weights[layer][node][-1] # add bias
                output = sigmoid(sum) # Apply activation function
                self.outputs[layer][node] = output
                next_inputs.append(output)
            inputs = next_inputs
        # inputs should now be the final layer output
        return inputs
    
    def backprop(self, true_values, data):
        """
        Calculates the loss and gradient for each output node,
        starting at the last layer.
        Propagates the gradient through the network and records
        the error for each node. Updates the weights/biases based on
        the gradient
        
        Parameters
        ---------
        true_values
            true classification of example
        data
            training example

        Returns
        ---------
        None, weights and biases are updated
        """
        # start at last layer
        for layer in reversed(range(len(self.outputs))): 
            if layer == len(self.outputs) - 1: # for last layer, calculate loss using true values
                for node in range(len(self.outputs[layer])):
                    loss = (true_values[node] - self.outputs[layer][node])
                    # fill in change matrix
                    self.change[layer][node] = loss*sigmoid_derivative(self.outputs[layer][node])
            else: # for all other layers
                for node in range(len(self.outputs[layer])):
                    loss = 0
                    # sum weighted losses from previous layer
                    for prev_layer_node in range(len(self.weights[layer + 1])):
                        loss += self.weights[layer+1][prev_layer_node][node]*self.change[layer+1][prev_layer_node]
                    # fill in change matrix
                    self.change[layer][node] = loss*sigmoid_derivative(self.outputs[layer][node])
         
        
        # Update weights
        for layer in reversed(range(len(self.outputs))): 
            input = data[0] # the input to the first layer is the training example
            if layer != 0: # the input to rest of layers is output of prev layer
                input = [self.outputs[layer-1][node] for node in range(len(self.outputs[layer - 1]))]
            for node in range(len(self.outputs[layer])):
                for i in range(len(input)):
                    self.weights[layer][node][i] += self.lr*self.change[layer][node]*input[i]
                # update bias
                self.weights[layer][node][-1] += self.lr*self.change[layer][node]


    def fit(self, training):
        """
        Trains the neural network and computes training loss (MSE).
        Performs stochastic gradient descent using backpropogation.
        
        Parameters
        ---------
        training
            A list of training examples, where each training example
            is a list where index 0 is the example, and index 1 is
            the true class.

        Returns
        ---------
        Average training loss over the training examples
        """
        train_loss = 0
        for row in training:
            output = self.feedforward(row[0])
            expected = row[-1] # Expected class should be last element of training row
            # Sum loss of all output nodes
            train_loss += sum([(expected[i]-output[i])**2 for i in range(len(expected))])
            self.backprop(true_values=expected, data=row)
        return train_loss/len(training) # return MSE
        
    def predict(self, data):
        """
        Performs one forward pass through the network to predict the 
        result
        
        Parameters
        ---------
        data
            data to be passed through

        Returns
        ---------
        Prediction from the trained network
        """
        return self.feedforward(data)
    

def sigmoid(x):
    """
    Sigmoid activation function, which scales a value
    between 0 and 1

    Parameters
    ---------
    x
        value to apply sigmoid to

    Returns
    ---------
    Transformed value
    """
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    """
    Computes the derivative of the sigmoid function

    Parameters
    ---------
    x
        value to compute the sigmoid derivative for

    Returns
    ---------
    Sigmoid derivative of the value 
    """
    return sigmoid(x)*(1 - sigmoid(x))

def training(pos_batch_size, neg_batch_size, n_epochs, nn, pos_train, neg_train, validation):
    """
    Trains a neural network using given hyperparameters.
    Uses two queues of pos and neg examples to generate 
    class-balanced batches in each epoch. Evaluates validation
    loss after each epoch.

    Parameters
    ---------
    pos_batch_size
        How many positive examples to include in each batch
    neg_batch_size
        How many negative examples to include in each batch
    n_epochs
        How many epochs to train for
    nn
        Neural network to train
    pos_train
        list of positive training examples, where each example
        is a list of length 2, with index 0 being the one-hot
        encoded sequence and index 1 being the class (0 or 1)
    neg_train
        same as pos_train but for negative training examples
    validation
        list of validation examples
        
    Returns
    ---------
    A pandas dataframe containing training/validation loss per epoch
    """
    losses = pd.DataFrame(columns = ['Epoch', 'Train', 'Validation']) 
    
    # Add training data to queues
    pos_queue = queue.Queue(maxsize=0) 

    for i in range(len(pos_train)):
        pos_queue.put(pos_train[i])

    neg_queue = queue.Queue(maxsize=0) 

    for i in range(len(neg_train)):
        neg_queue.put(neg_train[i])

    for epoch in range(n_epochs):
        batch_losses = []
        # run until all neg examples are used,
        # while upsampling pos examples
        while neg_queue.empty() == False:
            # get a new batch of positive and neg examples
            neg_batch = [neg_queue.get() for i in range(neg_batch_size)]
            pos_batch = []
            for i in range(pos_batch_size):
                if pos_queue.empty(): # replenesh pos samples when necessary
                    for i in range(len(pos_train)):
                        pos_queue.put(pos_train[i])
                pos_batch.append(pos_queue.get())

            train = neg_batch + pos_batch
            shuffle(train)

            batch_losses.append(nn.fit(train))

        # Average over the batch losses    
        train_loss = sum(batch_losses)/len(batch_losses)
        
        # Compute validation loss
        val_loss = 0
        for val in validation:
            output = nn.predict(val[0])
            expected = val[-1]
            val_loss += sum([(expected[i]-output[i])**2 for i in range(len(expected))])
        losses = losses.append({'Epoch' : epoch, 'Train' : train_loss,
                                'Validation': val_loss/len(validation)}, ignore_index=True)
        if val_loss < 0.004: # stop if val loss gets low enough
            break
            
        # shuffle examples for the next epoch and repopulate queues
        shuffle(neg_train)
        for i in range(len(neg_train)):
            neg_queue.put(neg_train[i])

        shuffle(pos_train)
        pos_queue.queue.clear()
        for i in range(len(pos_train)):
            pos_queue.put(pos_train[i])

    return losses


def cross_validation(k, nn, pos_batch_size, neg_batch_size, n_epochs, pos_enc, neg_enc, return_model=False):
    """
    Performs k-fold cross validation on a set of training data.
    Splits the data into k different partitions and trains k 
    different networks where the validation set is the k-th fold,
    and the training set is everything minus that k-th fold. 

    Parameters
    ---------
    k
        Number of folds
    nn
        neural network object to use
    pos_batch_size
        How many positive examples to include in each batch
    neg_batch_size
        How many negative examples to include in each batch
    n_epochs
        How many epochs to train for
    pos_enc
        list of positive training examples, where each example
        is a list of length 2, with index 0 being the one-hot
        encoded sequence and index 1 being the class (0 or 1)
    neg_enc
        same as pos_train but for negative training examples
    return_model
        Return the trained nn for the last fold 
        
    Returns
    ---------
    A list of actual classes for each fold, and a list of predicted classes for each fold that can be used to calculate an average AUC across the folds to evaluate model performance
    """
    # n examples to include in each fold
    n_pos_folds = math.floor(len(pos_enc)/k)
    n_neg_folds = math.floor(len(neg_enc)/k)

    # dict to hold validation predicted and actual classes for plotting ROC later
    all_preds = []
    all_actuals = []
    for i in range(k):
        if i == k-1: # for the last fold, use all remaining data
            pos_val = pos_enc[i*n_pos_folds:]
            neg_val = neg_enc[i*n_neg_folds:]
            validation = pos_val + neg_val
            shuffle(validation)
        else:
            # get examples in current fold, use as validation set
            pos_val = pos_enc[i*n_pos_folds:i*n_pos_folds+n_pos_folds]
            neg_val = neg_enc[i*n_neg_folds:i*n_neg_folds+n_neg_folds]
            validation = pos_val + neg_val
            shuffle(validation)
        # train is just all examples minus the fold
        pos_train = pos_enc[:i*n_pos_folds] + pos_enc[i*n_pos_folds+n_pos_folds:]
        neg_train = neg_enc[:i*n_neg_folds] + neg_enc[i*n_neg_folds+n_neg_folds:]

        # train
        losses = training(pos_batch_size = pos_batch_size,neg_batch_size = neg_batch_size,
                          n_epochs = n_epochs, nn = nn, pos_train = pos_train, 
                          neg_train=neg_train, validation= validation)
        
        preds = []
        actuals = []
        for val in validation:
            preds.append(nn.predict(val[0])[0])
            actuals.append(val[-1][0])
        all_preds.append(preds)
        all_actuals.append(actuals)
    if return_model == False:
        return all_preds, all_actuals
    if return_model == True:
        return nn
    