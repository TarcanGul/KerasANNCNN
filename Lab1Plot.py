import matplotlib.pyplot as plt
import numpy as np

plt.rcdefaults()
plt.rcdefaults()

DATASETS = ["mnist_d", "mnist_f", "cifar_10", "cifar_100_f", "cifar_100_c"]
y_pos = np.arange(len(DATASETS))
#Fill with values.
accuracy_ANN = [0.9787, 0.8757,0.4418, 0.0971, 0.2806] 
accuracy_CNN = [0.9901,0.9256,0.6848,0.3715, 0.4981]

plt.bar(y_pos, accuracy_ANN, align='center', alpha=0.5)
plt.xticks(y_pos, DATASETS)
plt.ylabel('Usage')
plt.title('Accuracy of ANN')

plt.savefig("ANN_Accuracy_Plot.pdf")
plt.show()


plt.bar(y_pos, accuracy_CNN, align='center', alpha=0.5)
plt.xticks(y_pos, DATASETS)
plt.ylabel('Usage')
plt.title('Accuracy of CNN')

plt.savefig("CNN_Accuracy_Plot.pdf")
plt.show()
