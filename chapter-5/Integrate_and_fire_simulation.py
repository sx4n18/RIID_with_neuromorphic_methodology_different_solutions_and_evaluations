import matplotlib.pyplot as plt

###############################################
# Simulate the integrate and fire model defined by the differential equation
# dV/dt = I/C i.e. t_tau (dV/dt) = IR
###############################################

# Define the parameters
t_max = 100.0  # ms
dt = 0.1  # ms
t_rest = 0.0  # initial refractory time

# Set up the current step
I_min = 0.0  # nA
I_max = 1.0  # nA
I_step = 0.1  # nA
I = I_min + 2*I_step  # nA

# define the IF model behaviour
def integrate_and_fire(I_inj, cap, t_max, dt, v_reset, v_rest, v_thresh):
    # Set up the simulation
    t = 0.0
    v = v_rest
    v_trace = [v]
    t_trace = [t]
    t_rest = 50.0
    current_trace = [0]

    # Simulate the neuron behaviour
    while t < t_max:

        if t > t_rest and t < t_rest + 40.0:
            v = v + (I_inj/cap)*dt
            current_trace.append(I_inj)
        else:
            v = v + (0/cap)*dt
            current_trace.append(0)


        if v > v_thresh:
            v = v_reset
        v_trace += [v]
        t += dt
        t_trace += [t]

    return t_trace, v_trace, current_trace

t_trace, v_trace, I_trace = integrate_and_fire(I, 0.1, t_max, dt, -70.0, -70.0, -40.0)

# Plot the results of the voltage trace and current trace in the same figure with twin axes
fig, ax1 = plt.subplots()
ax1.plot(t_trace, v_trace, '#1f77b4')
ax1.set_xlabel('Time (ms)', weight='bold', size=15)
ax1.set_ylabel('Voltage (mV)', color='#1f77b4', weight='bold', size=15)
ax1.tick_params('y', colors='#1f77b4', size=15)
ax2 = ax1.twinx()
ax2.plot(t_trace, I_trace, '#ff7f0e')
ax2.set_ylabel('Current (nA)', color='#ff7f0e', weight='bold', size=15)
ax2.tick_params('y', colors='#ff7f0e', size=15)
fig.tight_layout()
plt.show()

