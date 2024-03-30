import numpy as np
import matplotlib.pyplot as plt


######################################################################
# Poisson spike generator:
# This script generates a Poisson encoded spike train with a given firing rate.
# The spike train is generated using random numbers from a uniform distribution.
# This script also plots the spike train and plot rate text on the plot.
######################################################################


def poisson_generator(rate, T, dt):
    """
    Generates a Poisson spike train with a given firing rate.

    Args:
    rate : float
        Firing rate of the Poisson spike train (in Hz)
    T : float
        Length of the Poisson spike train (in seconds)
    dt : float
        Time step (in seconds)

    Returns:
    spike_train : 1D array
        Boolean array that indicates the spike times
    time : 1D array
        Time points
    """
    # Time points
    time = np.arange(0, T, dt)

    # Generate uniformly distributed random numbers
    r = np.random.rand(len(time))

    # Generate a Poisson spike train
    spike_train = r < rate * dt

    return spike_train, time


# Firing rate of the Poisson spike train (in Hz)
rate = 0.23
T = 1000  # Length of the Poisson spike train (in ms)
dt = 1  # Time step (in ms)

# Generate a Poisson spike train
spike_train, time = poisson_generator(rate, T, dt)
spike_train_cnt = np.where(spike_train == True)[0]
total_spikes = np.size(spike_train_cnt)

# Plot the spike train in a raster plot and put the text in the plot for the firing rate and actual number of spikes
plt.figure()
plt.eventplot(spike_train_cnt, lineoffsets=0.5, linelengths=0.5)
plt.text(0.5, 0.9, 'Firing rate: ' + str(rate) + ' Hz', ha='center', va='center', transform=plt.gca().transAxes, fontsize=12, weight='bold')
plt.text(0.5, 0.8, 'Number of spikes: ' + str(total_spikes), ha='center', va='center', transform=plt.gca().transAxes, fontsize=12, weight='bold')
plt.xlabel('Time (ms)', fontsize=12, weight='bold')
plt.ylabel('Neuron index', fontsize=12, weight='bold')
plt.title('Poisson spike train', fontsize=14, weight='bold')
plt.show()


