import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from seaborn import heatmap
######################################################################
# Confusion matrix for final results
# This script plots the confusion matrix for the final results of the BESNN in 8-bit.
# The data is obtained from my experiment log
# The corresponding class labels are:
# 0: Am241
# 1: Ba133
# 2: BGD
# 3: Co57
# 4: Co60
# 5: Cs137
# 6: DU
# 7: Eu152
# 8: Ga67
# 9: HEU
# 10: I131
# 11: Ir192
# 12: Np237
# 13: Ra226
# 14: Tc99m
# 15: Th232
# 16: Tl201
# 17: WGPu
######################################################################

# True labels
y_true = np.load("./results_collected/label.npy")

# corresponding class
label_class = ["Am241", "Ba133", "BGD", "Co57", "Co60", "Cs137", "DU", "Eu152", "Ga67", "HEU", "I131", "Ir192", "Np237", "Ra226", "Tc99m", "Th232", "Tl201", "WGPu"]

# Predicted labels
y_pred = np.load("./results_collected/hard_voting_prediction_8bit.npy")

# Confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Plot the confusion matrix using seaborn heatmap
plt.figure()
heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', xticklabels=label_class, yticklabels=label_class)
plt.xlabel('Predicted label', fontsize=12, weight='bold')
plt.ylabel('True label', fontsize=12, weight='bold')
plt.title('Confusion matrix for the final results of FPGA implementation', fontsize=14, weight='bold')
plt.show()

