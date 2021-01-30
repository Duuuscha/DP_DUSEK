import numpy as np
import nnfs
from nnfs.datasets import spiral_data

class Layer_Dense:
    
    def __init__(self, n_inputs, n_neurons):
        """initializes weights and biases"""
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.inputs = np.dot(dvalues, self.weights.T)
        

class Acti_Relu:
    
    def forward(self, inputs):
        self.output = np.maximum(0,inputs)
        
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class Acti_SoftMax:
    
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs,
                                           axis=1,
                                           keepdims=True))
        probabilities = exp_values / np.sum(exp_values,
                                           axis=1,
                                           keepdims=True)
        self.output = probabilities
        
    def backward(self, dvalues):
        
        self.dinputs = np.empty_like(dvalues)
        
        for i, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1,1)
            
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[i] = np.dot(jacobian_matrix, single_dvalues)


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


class Loss_CategoCrossentropy(Loss): # inheriting LOSS class
    
    def forward(self, y_pred, y_true):
        
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7) # to prevent overflow and dividing by 0
        
        if len(class_targets.shape)==1:
            correct_confidence = softmax_out[
                                    range(len(softmax_out)),
                                    class_targets]
        elif len(class_targets.shape) == 2:
            correct_confidence = np.sum(softmax_out * class_targets, axis=1)

        neg_log = -np.log(correct_confidence)

        return neg_log
    
    def backward(self, dvalues, y_true):
        
        samples=len(dvalues)
        
        labels = len(dvalues[0])
        
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
            
        self.dinputs = -y_true / dvalues
        
        self.dinputs = self.dinputs / samples
        

class Activation_Softmax_Loss_CategoricalCrossentropy():
    
    def __init__(self):
        self.activation = Acti_SoftMax()
        self.loss = Loss_CategoCrossentropy()
        
    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        
        self.output = self.activation.output
        
        return self.loss.calculate(self.output, y_true)
    
    def backward(self, dvalues, y_true):
        
        samples = len(dvalues)
        
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
            
        self.dinputs = dvalues.copy()
        
        self.dinputs[range(samples), y_true] -= 1
        
        self.dinputs = self.dinputs / samples

##############################################################################





X, y = spiral_data(samples=100, classes= 3)

dense1 = Layer_Dense(2, 3)
activation1 = Acti_Relu()

dense2 = Layer_Dense(3, 3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
loss = loss_activation.forward(dense2.output, y)

print(loss_activation.output[:5])

