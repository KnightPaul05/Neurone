import numpy as np
from neuron import Neuron
import matplotlib.pyplot as plt
from couches import Layer

# neuron1 = Neuron(input_size=3)
# inputs = np.array([0.5,0.3,0.2])
# target = 1


# epochs = 1000

# for _ in range(epochs):
#     neuron1.entrainement(inputs,target)

# plt.plot(neuron1.sorties)
# plt.xlabel('Itération')
# plt.ylabel('Erreur quadratique')
# plt.title('Evolution de lerreur au fil des itérations')
# plt.show()

layer = Layer(nbNeuron=5,nbinput=3)
inputs = [0.5,0.3,0.2]
outputs = layer.forward(inputs)
print(outputs)
