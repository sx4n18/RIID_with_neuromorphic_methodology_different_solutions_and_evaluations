import numpy as np
import matplotlib.pyplot as plt


######################################################################
# Plot the accuracy of the BESNN with full precision against time steps
# The data comes from my log
######################################################################


# Time steps
time_steps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Accuracy of the BESNN with full precision
ensemble_acc = [56.59, 96.07, 97.11, 97.48, 97.04, 97.19, 97.41, 97.19, 97.26, 97.04]




# Plot the accuracy of the BESNN with full precision against time steps
plt.figure()
plt.plot(time_steps, ensemble_acc, marker='o', markersize=10, color='b', linestyle='-', linewidth=2)
plt.xlabel('Time steps', fontsize=12, weight='bold')
plt.ylabel('Accuracy (%)', fontsize=12, weight='bold')
plt.xticks(time_steps, fontsize=12)
plt.title('BESNN accuracy with full precision', fontsize=14, weight='bold')
plt.grid()
plt.show()

