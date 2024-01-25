import numpy as np
import matplotlib.pyplot as plt
#############################################################################################################
## This script is used to do a simple simulation of the Izhikevich neuron model
## The figure is captioned at Figure 5.21
#############################################################################################################

# e = Excitatory neurons, i = inhibitory neurons
Ne = 800
Ni = 202
number_neurons = Ne + Ni
time_steps = 1000

re = np.random.rand(Ne, 1)
ri = np.random.rand(Ni, 1)

# First array of np.concatenate is the Excitatory neuron parameters, second part is the inhibitory neurons
a = np.concatenate([0.02 * np.ones((Ne, 1)), 0.02 + 0.08 * ri])
b = np.concatenate([0.2 * np.ones((Ne, 1)), 0.25 - 0.05 * ri])
c = np.concatenate([-65 + 15 * re ** 2, -65 * np.ones((Ni, 1))])
d = np.concatenate([8 - 6 * re ** 2, 2 * np.ones((Ni, 1))])
S = np.concatenate([0.5 * np.random.rand(Ne + Ni, Ne), -np.random.rand(Ne + Ni, Ni)], axis=1)

v = -65 * np.ones((Ne + Ni, 1))
u = b * -65
neurons_that_fired_across_time = []
voltage_across_time = []
current_injection = []
for t in range(1, time_steps + 1):
    # The Input Voltage
    I = np.concatenate([5 * np.random.rand(Ne, 1), 2 * np.random.rand(Ni, 1)])

    # When voltage goes above 30 mV, we find the index, and append it to fired,
    # then reset the membrane potnetial and membrane recovery variable
    neurons_that_fired = np.where(v > 30)
    voltage_across_time.append(float(v[50]))
    neurons_that_fired_across_time.append([t + 0 * neurons_that_fired[0], neurons_that_fired[0]])

    for i in neurons_that_fired[0]:
        v[i] = c[i]
        u[i] += d[i]

    I += np.expand_dims(np.sum(S[:, neurons_that_fired[0]], axis = 1), axis = 1)
    current_injection.append(float(I[50]))
    # We have to do 0.5ms increments for numerical stability
    v += 0.5 * (0.04 * v**2 + 5 * v + 140 - u + I)
    v += 0.5 * (0.04 * v**2 + 5 * v + 140 - u + I)
    u = u + a * (b * v - u)

voltage_across_time = np.array(voltage_across_time)
current_injection = np.array(current_injection)


# plot two figures in one row for the voltage and current injection
fig = plt.figure(1)
plt.plot(0.5*np.arange(time_steps), voltage_across_time)
plt.xlabel('Time (ms)', fontsize=20, fontweight='bold')
plt.ylabel('Voltage (mV)', fontsize=20, fontweight='bold')
plt.xticks(0.5*np.linspace(0, 1000, 5), fontsize=20, fontweight='bold')
plt.yticks(fontsize=20, fontweight='bold')
fig2 = plt.figure(2)
plt.plot(0.5*np.arange(time_steps), current_injection)
plt.xlabel('Time (ms)', fontsize=20, fontweight='bold')
plt.ylabel('Current Injection (pA)', fontsize=20, fontweight='bold')
plt.xticks(0.5*np.linspace(0, 1000, 5), fontsize=20, fontweight='bold')
plt.yticks(fontsize=20, fontweight='bold')
plt.show()