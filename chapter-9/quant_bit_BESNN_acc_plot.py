import numpy as np
import matplotlib.pyplot as plt



######################################################################
## This script plots the accuracy of the BESNN for different bit precisions.
## The data is obtained from my experiment log
######################################################################

# Bit precision from 1 to 16 bits
bit_precision = np.arange(1, 17)

# Accuracy of the BESNN for different bit precisions
accuracy = np.array([5.56, 6, 6, 6.89, 68.15, 95.41, 97.04, 97.04, 97.19, 97.48, 97.48, 97.56, 97.56, 97.56, 97.48, 97.48])



# Plot the accuracy of the BESNN for different bit precisions
plt.figure()
plt.plot(bit_precision, accuracy, marker='o', markersize=10, linestyle='-', linewidth=2)
plt.xlabel('Bit precision', fontsize=12, weight='bold')
plt.ylabel('Accuracy (%)', fontsize=12, weight='bold')
plt.title('BESNN accuracy for different bit precisions', fontsize=14, weight='bold')
plt.grid()
plt.show()
