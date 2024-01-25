import matplotlib.pyplot as plt
import numpy as np

########################################################################################################################
## This script is used to generate the activation figures for the paper.
## The figure is Figure 5.17
########################################################################################################################


## Figure 1: plot the signoid function within the range of [-10, 10] with extra big font size and bold font weight using 5 ticks along each axis
def plot_sigmoid(fig_num=1):
    fig = plt.figure(fig_num)
    x = np.linspace(-10, 10, 100)
    y = 1 / (1 + np.exp(-x))
    plt.plot(x, y)
    plt.xticks(np.linspace(-10, 10, 5), fontsize=20, fontweight='bold')
    plt.yticks(np.linspace(0, 1, 2), fontsize=20, fontweight='bold')



## Figure 2: plot the tanh function within the range of [-10, 10]
def plot_tanh(fig_num=2):
    fig = plt.figure(fig_num)
    x = np.linspace(-10, 10, 100)
    y = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    plt.plot(x, y)
    plt.xticks(np.linspace(-10, 10, 5), fontsize=20, fontweight='bold')
    plt.yticks(np.linspace(-1, 1, 2), fontsize=20, fontweight='bold')


## Figure 3: plot the ReLU function within the range of [-10, 10]
def plot_relu(fig_num=3):
    fig = plt.figure(fig_num)
    x = np.linspace(-10, 10, 100)
    y = np.maximum(0, x)
    plt.plot(x, y)
    plt.xticks(np.linspace(-10, 10, 5), fontsize=20, fontweight='bold')
    plt.yticks(np.linspace(0, 10, 2), fontsize=20, fontweight='bold')


## Figure 4: plot the Leaky ReLU function within the range of [-10, 10]
def plot_leaky_relu(fig_num=4):
    fig = plt.figure(fig_num)
    x = np.linspace(-10, 10, 100)
    y = np.maximum(0.1 * x, x)
    plt.plot(x, y)
    plt.xticks(np.linspace(-10, 10, 5), fontsize=20, fontweight='bold')
    plt.yticks([-1, 10], fontsize=20, fontweight='bold')



## Figure 5: plot the ELU function within the range of [-10, 10]
def plot_elu(fig_num=5):
    fig = plt.figure(fig_num)
    x = np.linspace(-10, 10, 100)
    y = np.where(x >= 0, x, 0.1*(np.exp(x) - 1))
    plt.plot(x, y)
    plt.xticks(np.linspace(-10, 10, 5), fontsize=20, fontweight='bold')
    plt.yticks([0, 10], fontsize=20, fontweight='bold')



## main function
if __name__ == '__main__':
    plot_sigmoid()
    plot_tanh()
    plot_relu()
    plot_leaky_relu()
    plot_elu()
    plt.show()