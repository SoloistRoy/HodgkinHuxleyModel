#-*-Encoding:utf-8-*-
from __future__ import division
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

matplotlib.use('GTK')
plt.rcParams.update({'font.size':25})

# HH parameters
V_rest  = 0      #mV
Cm      = 1      #uF/cm2
gbar_Na = 120    #mS/cm2
gbar_K  = 36     #ms/cm2
gbar_l  = 0.3    #mS/cm2
E_Na    = 115    #mV
E_k     = -12    #mV
E_l     = 10.613 #mV

#kanal K
def alpha_n(V):
    if V != 10:
        return 0.01*(-V + 10)/(np.exp((-V+10)/10)-1)
    else:
        return 0.1

def beta_n(V):
    return 0.125*np.exp(-V/80)

def n_inf(V):
    return alpha_n(V)/(alpha_n(V) + beta_n(V))

#kanal Na (aktywny)
def alpha_m(V):
     if V != 25:
         return 0.1*(-V + 25)/(np.exp((-V + 25)/10) - 1)
     else:
         return 1

def beta_m(V):
    return 4*np.exp(-V/18)

def m_inf(V):
    return alpha_m(V)/(alpha_m(V) + beta_m(V))

#knal Na (nieaktywny)
def alpha_h(V):
    return 0.07*np.exp(-V/20)

def beta_h(V):
    return 1/(np.exp((-V+30)/10)+1)

def h_inf(V):
    return alpha_h(V)/(alpha_h(V) + beta_h(V))


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
    

def I(t):
	if (10 <= t <= 55):
	    return 30
	else:
		return 0
  

def solve():
    T = 100      #ms
    dt = 0.025  #ms
    time = np.arange(0, T+dt, dt)
    m0 = m_inf(V_rest)
    h0 = h_inf(V_rest)
    n0 = n_inf(V_rest)
    y_0 = [V_rest, m0, h0, n0]
    Y = odeint(Neuron, y_0, time)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(time,Y[:,0], linewidth = 2, color='blue')
    I_s = map(I,time)
    ax2.plot(time, I_s, 'r-',linewidth = 2)
    plt.grid(True)
    plt.title("Hodgkin-Huxley model")
    ax1.set_xlabel("Time [ms]")
    ax1.set_ylabel("Membrane Potential [mV]", color='b')
    ax2.set_ylabel(r"Current [$\frac{\mathtt{uA}}{\mathtt{cm^2}}$]", color='r')
    ax2.set_ylim([-20, 120])
    plt.text(56, 35, r"$I=30 \frac{\mathtt{uA}}{\mathtt{cm^2}}$ ", color = 'red')
    plt.show()


def plot_chanel_parameters():
    V = np.linspace(-50,151,1000)
    Y_n_inf = map(n_inf, V)
    Y_m_inf = map(m_inf, V)
    Y_h_inf = map(h_inf, V)
    plt.plot(V, Y_n_inf, linewidth = 2)
    plt.plot(V, Y_m_inf, linewidth = 2)
    plt.plot(V, Y_h_inf, linewidth = 2)
    plt.legend(('n','m','h'))
    plt.xlabel(u"Napięcie [mV]")
    plt.grid(True)
    plt.show()

