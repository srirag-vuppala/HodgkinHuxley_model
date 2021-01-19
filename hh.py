import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint 

class HodgkinHuxley():
    C_m = 1
    """membrane capacitance, in uF/cm^2"""
    g_Na = 120
    """Sodium (Na) maximum conductances, in mS/cm^2"""
    g_K = 36
    """Postassium (K) maximum conductances, in mS/cm^2"""
    g_L = 0.3
    """Leak maximum conductances, in mS/cm^2"""
    V_Na = 115
    """Sodium (Na) Diffusion potentials, in mV"""
    V_K  = -12
    """Postassium (K) Diffusion potentials, in mV"""
    V_L  = -11
    """Leak current Diffusion potentials, in mV"""
    t = np.arange(0.0, 30.0, 0.01)
    """ The time to integrate over """

    def alpha_m(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 0.1*(25 - V)/(np.exp((25-V) / 10) - 1)
    def beta_m(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 4.0*np.exp(-(V / 18.0))

    def alpha_h(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 0.07*np.exp(-(V / 20.0))
    def beta_h(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 1.0/(1.0 + np.exp(-(30 - V) / 10.0))

    def alpha_n(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 0.01*(10 - V)/(np.exp((10 - V) / 10.0) - 1)
    def beta_n(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 0.125*np.exp(-(V / 80.0))
    
    def n_inf(self, Vm=0.0):
        """ Inflection point potassium conductance to easily write gK"""
        return self.alpha_n(Vm) / (self.alpha_n(Vm) + self.beta_n(Vm))
    def m_inf(self, Vm=0.0):
        """ Sodium activation variable """
        return self.alpha_m(Vm) / (self.alpha_m(Vm) + self.beta_m(Vm))
    def h_inf(self, Vm=0.0):
        """ Sodium inactivation variable """
        return self.alpha_h(Vm) / (self.alpha_h(Vm) + self.beta_h(Vm))
    
        # Input stimulus giver
    def Input_stimuli(self, t):
        """ Current applied to create stimulus which is dependent on time, in milli Ampere(A)/cm^2 """
        if 0.0 < t < 1.0:
            return 150.0
        elif 10.0 < t < 30.0:
            return 50.0
        return 0.0

    def derivatives(self, y, t0):
        dy = [0]*4
        V = y[0]
        n = y[1]
        m = y[2]
        h = y[3]
        
        #encapsulating the remaining terms within the equation
        GK = (self.g_K / self.C_m) * np.power(n, 4.0)
        GNa = (self.g_Na / self.C_m) * np.power(m, 3.0) * h
        GL = self.g_L / self.C_m
        
        dy[0] = (self.Input_stimuli(t0) / self.C_m) - (GK * (V - self.V_K)) - (GNa * (V - self.V_Na)) - (GL * (V - self.V_L))
        
        # dn/dt
        dy[1] = (self.alpha_n(V) * (1.0 - n)) - (self.beta_n(V) * n)
        
        # dm/dt
        dy[2] = (self.alpha_m(V) * (1.0 - m)) - (self.beta_m(V) * m)
        
        # dh/dt
        dy[3] = (self.alpha_h(V) * (1.0 - h)) - (self.beta_h(V) * h)
        
        return dy

    def main(self):
        Y = np.array([0.0, self.n_inf(), self.m_inf(), self.h_inf()])
        Vy = odeint(self.derivatives, Y, self.t)
        V = Vy[:,0]
        m = Vy[:,1]
        h = Vy[:,2]
        n = Vy[:,3]

        plt.figure()

        plt.subplot(4,1,1)
        plt.title('Hodgkin-Huxley for space-clamped axon')
        plt.plot(self.t, V, 'k')
        plt.ylabel('V (mV)')
        plt.xlabel('Time (ms)')

        plt.subplot(4,1,3)
        plt.plot(self.t, m, 'r', label='m')
        plt.plot(self.t, h, 'g', label='h')
        plt.plot(self.t, n, 'b', label='n')
        plt.ylabel('Gating Value')
        plt.xlabel('Time (ms)')
        plt.legend()

        plt.show()

if __name__ == '__main__':
    HodgkinHuxley().main()