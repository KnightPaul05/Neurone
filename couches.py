from neuron import Neuron

class Layer():
    def __init__(self,nbNeuron,nbinput):
        '''Initialise une couche, chacun recevant leur entrée'''
        self.nbNeuron = nbNeuron
        self.neurons = [Neuron(nbinput) for _ in range(nbNeuron)]

    def forward(self, inputs):
        '''Calcule la sortie de chaque neurone et renvoie toutes les sorties sous formes d'un tableau'''
        return [neuron.outputs(inputs) for neuron in self.neurons]