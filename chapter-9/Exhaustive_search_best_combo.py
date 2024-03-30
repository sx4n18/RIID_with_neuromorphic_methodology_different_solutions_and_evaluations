import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import pickle

######################################################################
## This script tries to find the best combo of a variable ensemble size
## and plot out the accuracies in an error bar plot
######################################################################

def major_hardvote(final_class_of_each_ANN, total_testing_number):
    #final_class_of_each_ANN = np.argmax(all_diagonal_prediction, axis=2)
    final_class_of_each_ANN = np.asarray(final_class_of_each_ANN, 'int')
    final_prediction_after_voting = np.zeros(total_testing_number, 'int')
    for sample_index in range(total_testing_number):
        bin_count = np.bincount(final_class_of_each_ANN[:, sample_index])
        final_prediction_after_voting[sample_index] = np.argmax(bin_count)

    return final_prediction_after_voting

# Ensemble size
ensemble_size = [1, 2, 3, 4, 5]

# Collected predictions from the BESNN (20, 1350)
raw_predictions = np.load("./results_collected/SNN_ensemble_prediction_8bit.npy")

# True labels
y_true = np.load("./results_collected/label.npy")

# Accuracy of the BESNN for different ensemble sizes
accuracy = []
acc_mean = []
acc_std = []

# Loop through the ensemble size
for size in ensemble_size:
    acc_for_size = []
    all_combos = list(itertools.combinations(range(20), size))
    print(f"Analysing for {size} nets in the ensemble...\n")
    for combo in tqdm(all_combos):
        # Get the predictions for the combo
        combo_prediction = raw_predictions[combo, :]
        # Get the major voting prediction
        final_prediction = major_hardvote(combo_prediction, 1350)
        # Calculate the accuracy
        acc = accuracy_score(y_true, final_prediction)
        acc_for_size.append(acc)
    acc_mean.append(np.mean(acc_for_size))
    acc_std.append(np.std(acc_for_size))
    accuracy.append(acc_for_size)

# Save the accuracy list in pickle
with open('./results_collected/accuracy_list_'+ str(ensemble_size[0])+'_'+str(ensemble_size[-1])+'.pkl', 'wb') as f:
    pickle.dump(accuracy, f)



# Plot the accuracy of the BESNN for different ensemble sizes
plt.figure()
plt.errorbar(ensemble_size, acc_mean, yerr=acc_std, fmt='o', markersize=10, linestyle='-', linewidth=2)
plt.xlabel('Ensemble size', fontsize=12, weight='bold')
plt.ylabel('Accuracy (%)', fontsize=12, weight='bold')
plt.xticks(ensemble_size, fontsize=12)
plt.title('BESNN accuracy for different ensemble sizes', fontsize=14, weight='bold')
plt.grid()
plt.show()

