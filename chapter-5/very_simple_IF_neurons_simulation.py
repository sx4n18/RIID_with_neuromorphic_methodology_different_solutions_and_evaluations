import numpy as np
import matplotlib.pyplot as plt



###############################################
# Simulate the integrate and fire model defined as acuumulator
###############################################



## Define the parameters

t_max = 100.0  # ms
dt = 0.1  # ms
v_rest = 0.0  # mV
v_thresh = 40  # mV
v_reset = 0.0  # mV

weight = [5, 3, 2, 8, 1]
spike_times = [[i for i in np.sort(np.round(np.random.random(5)*100, 1))], [i for i in np.sort(np.round(np.random.random(5)*100, 1))], [i for i in np.sort(np.round(np.random.random(5)*100, 1))], [i for i in np.sort(np.round(np.random.random(5)*100, 1))], [i for i in np.sort(np.round(np.random.random(5)*100, 1))]]

# define the IF model states variables
v = v_rest
v_trace = [v]
t_trace = [0]
spikes = []

# Simulate the neuron behaviour
for t in np.arange(dt, t_max, dt):

    ## round the spike times to the keep the precision
    t = np.round(t, 1)

    # check if the timing is in the spike times
    for i in range(5):
        if t in spike_times[i]:
            print("spike detected and accumulated:", t, "ms")
            v += weight[i]

    # Check if the neuron crossed the threshold
    if v >= v_thresh:
        spikes += [t-dt]
        v = v_reset

    # Store the voltage trace and time
    v_trace += [v]
    t_trace += [t]

print(spikes)
print(spike_times)

# Plot the results of the voltage trace and spike times in two figures, spike times should be a raster plot
fig = plt.figure()
plt.plot(t_trace, v_trace, '#1f77b4')
plt.xlabel('Time (ms)', weight='bold', size=15)
plt.ylabel('Voltage (mV)', color='#1f77b4', weight='bold', size=15)
plt.tick_params('y', colors='#1f77b4', size=15)
plt.tight_layout()

fig = plt.figure()
# Create a raster plot with dots
plt.scatter(spike_times, [[i]*5 for i in range(5)], marker='.', color='black')
# Create a raster plot with dots
plt.scatter(spikes, [0]*len(spikes), marker='.', color='red')
plt.xlabel('Time (ms)', weight='bold', size=15)
plt.yticks([0, 1, 2, 3, 4], ['Neuron 1', 'Neuron 2', 'Neuron 3', 'Neuron 4', 'Neuron 5'], size=15)
plt.tight_layout()
plt.show()


