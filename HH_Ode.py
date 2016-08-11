from __future__ import division
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import odeint

matplotlib.use('GTK')
plt.rcParams.update({'font.size':25})

# K channel
alpha_n = np.vectorize(lambda v: 0.01*(-v + 10)/(np.exp((-v+10)/10)-1) if v != 10 else 0.1)
beta_n = lambda v: 0.125*np.exp(-v/80)
n_inf = lambda v: alpha_n(v)/(alpha_n(v) + beta_n(v))

# Na channel (activating)
alpha_m = np.vectorize(lambda v: 0.1*(-v + 25)/(np.exp((-v + 25)/10) - 1) if v != 25 else 1)
beta_m = lambda v: 4*np.exp(-v/18)
m_inf = lambda v: alpha_m(v)/(alpha_m(v) + beta_m(v))

# Na channel (inactivating)
alpha_h = lambda v: 0.07*np.exp(-v/20)
beta_h = lambda v:  1/(np.exp((-v+30)/10)+1)
h_inf = lambda v: alpha_h(v)/(alpha_h(v) + beta_h(v))

# setup parameters and state variables
T = 100      #ms
dt = 0.025  #ms
time = np.arange(0, T+dt, dt)

# HH parameters
V_rest  = 0      #mV
Cm      = 1      #uF/cm2
gbar_Na = 120    #mS/cm2
gbar_K  = 36     #ms/cm2
gbar_l  = 0.3    #mS/cm2
E_Na    = 115    #mV
E_k     = -12    #mV
E_l     = 10.613 #mV
#I       = 30     #mV

def I(t):
	if (10 <= t <= 55):
	    return 30
	else:
		return 0


def Neuron(Y, time):
    Vm   = Y[0]
    m = Y[1]
    h  = Y[2]
    n  = Y[3]

    g_Na = gbar_Na*(m**3)*h
    g_K = gbar_K*(n**4)
    g_l = gbar_l

    m = (alpha_m(Vm)*(1-m) - beta_m(Vm)*m)
    h = (alpha_h(Vm)*(1-h) - beta_h(Vm)*h)
    n = (alpha_n(Vm)*(1-n) - beta_n(Vm)*n)

    Y =  (I(time) - g_Na*(Vm - E_Na) - g_K*(Vm -E_k) - g_l*(Vm -E_l))/Cm 
    return [Y, m, h, n]


m0 = m_inf(V_rest)
h0 = h_inf(V_rest)
n0 = n_inf(V_rest)


y_0 = [V_rest, m0, h0, n0]

Y = odeint(Neuron, y_0, time)


fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(time,Y[:,0], linewidth = 2, color='blue')

I = map(I,time)


ax2.plot(time, I, 'r-',linewidth = 2)
plt.grid(True)
plt.title("Hodgkin-Huxley model")
ax1.set_xlabel("Time [ms]")
ax1.set_ylabel("Membrane Potential [mV]", color='b')
ax2.set_ylabel(r"Current [$\frac{\mathtt{uA}}{\mathtt{cm^2}}$]", color='r')
ax2.set_ylim([-20, 120])
plt.text(56, 35, r"$I=30 \frac{\mathtt{uA}}{\mathtt{cm^2}}$ ", color = 'red')
plt.show()
