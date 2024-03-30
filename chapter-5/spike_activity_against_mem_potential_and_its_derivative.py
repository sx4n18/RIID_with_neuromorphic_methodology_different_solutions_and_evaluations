import numpy as np
import matplotlib.pyplot as plt


###############################################
## This script is to plot the spike activity against the membrane potential (step function) and its derivative (dirac delta function)
###############################################


# Define the parameters
v_theta = 4

# define the dirac delta function approximation
def dirac_delta(t, t_spike):
    return np.where(np.abs(t-t_spike)<0.01, 1, 0)


# define the surrogate gradient function
def surrogate_gradient_exponential(v, v_theta, a_3 = 0.02):
    m = np.exp((v_theta - v)/a_3)
    n = (1+m)**2
    return m/(n*a_3)


def normal_dis_bell_curve(v, v_theta, a_4=0.02):
    m = np.exp(-(v-v_theta)**2/(2*a_4))
    n = np.sqrt(2*np.pi*a_4)
    return m/n


# Define the step function from -2 to 10 in the numpy array
t = np.linspace(-2, 10, 1000)
step_function = np.where(t>=4, 1, 0)
dirac_delta_function = dirac_delta(t, 4)
surrogate_gradient_exp = surrogate_gradient_exponential(t, v_theta, a_3=0.1)
surrogate_gradient_norm = normal_dis_bell_curve(t, v_theta)

# Plot step function and dirac delta function in figure 1
fig1 = plt.figure()
plt.plot(t, step_function, label='Spike Activity')
plt.plot(t, dirac_delta_function, label='derivative of Spike Activity', linestyle='--')
plt.xlabel('Voltage (mV)', weight='bold', size=15)


# Plot the surrogate gradient function in figure 2 with different colors
fig2 = plt.figure()
plt.plot(t, surrogate_gradient_exp, label='Exponential Surrogate Gradient', color='navy')
plt.plot(t, surrogate_gradient_norm, label='Normal Distribution Surrogate Gradient', linestyle='--', color='magenta')
plt.xlabel('Voltage (mV)', weight='bold', size=15)


# Show the plot
plt.show()


