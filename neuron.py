import numpy as np

class Neuron():
    def __init__(self,input_size) :
        self.weights = np.random.rand(input_size)
        self.bias = np.random.randn()
        self.erreurs = []
        
    
    

    def activation(self, x):
        """Fonction d'activation ReLu"""
        return x * ( x > 0)

    def outputs(self,inputs):
        '''Calcule la sortie du neurone'''
        x = np.dot(inputs, self.weights) + self.bias
        return self.activation(x)

    def entrainement(self,inputs,target,learning_rate=0.001):
        '''Mise à jour des poids avec descente de gradient'''
        sortie = self.outputs(inputs)
        
        erreur = sortie - target
        self.erreurs.append(erreur**2)
        #Descente de gradiant avec la fonction de ReLu
        gradiant = erreur*np.where(sortie > 0, 1, 0)

        #Mise à jour des poids et du bias
        self.weights -= learning_rate*gradiant*inputs
        self.bias -= learning_rate*gradiant
    

