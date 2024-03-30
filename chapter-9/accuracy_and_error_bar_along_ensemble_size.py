import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load 4 lists of accuracy

with open('./results_collected/accuracy_list_1_5.pkl', 'rb') as f:
    accuracy_1 = pickle.load(f)

with open('./results_collected/accuracy_list_6_10.pkl', 'rb') as f:
    accuracy_2 = pickle.load(f)

with open('./results_collected/accuracy_list_11_15.pkl', 'rb') as f:
    accuracy_3 = pickle.load(f)

with open('./results_collected/accuracy_list_16_20.pkl', 'rb') as f:
    accuracy_4 = pickle.load(f)

# Plot the accuracy of the BESNN for different ensemble sizes
size = [i for i in range(1, 21)]

# merge the accuracy lists
accuracy = accuracy_1 + accuracy_2 + accuracy_3 + accuracy_4

# get the mean and std of the accuracy
acc_mean = [np.mean(acc) for acc in accuracy]
acc_std = [np.std(acc) for acc in accuracy]

# Plot the error bar plot
plt.figure(1)
plt.errorbar(size, acc_mean, yerr=acc_std, marker='o', markersize=10, linestyle='-', linewidth=2)
plt.xlabel('Ensemble size', fontsize=12, weight='bold')
plt.ylabel('Accuracy', fontsize=12, weight='bold')
plt.xticks(size, fontsize=12)
plt.title('BESNN accuracy for different ensemble sizes', fontsize=14, weight='bold')
plt.grid()


# plot the max and min accuracy for each ensemble size in a box plot
plt.figure(2)
plt.boxplot(accuracy, labels=size)
plt.xlabel('Ensemble size', fontsize=12, weight='bold')
plt.ylabel('Accuracy', fontsize=12, weight='bold')
plt.title('BESNN accuracy for different ensemble sizes', fontsize=14, weight='bold')
plt.grid()
plt.show()

