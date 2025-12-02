#%%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from sampling import ThermodynamicHMCMC
from SINDy import DynamicSINDy

## HELPER FUNCTIONS

def tensorPlot(ax, A, t):

    _, ny, nx = np.shape(A)

    for ix in range(nx):
        for iy in range(ny):
            ax[iy, ix].plot(t, A[:, iy, ix])



## FUNCTIONS

def evolveFENEPnondim(Wi, Lm, L, tau, dt):
    
    dtau_dt = tau @ L + L.T @ tau - 1/Wi*(tau + np.trace(tau)/Lm**2*tau + np.trace(tau)/Lm**2*np.eye(3))
    tau += dtau_dt*dt

    return tau


def generateShearData(dt, Lt, eta, Wi, Lm, gamma_dot):
    
    nt = int(Lt/dt)
    L = np.array([[0, gamma_dot, 0], [0, 0, 0], [0, 0, 0]])
    DD = 0.5*(L + L.T)
    #W = 0.5*(L - L.T)

    tau_fill = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]])*1e-6
    tau = np.zeros((nt, 3, 3))
    tau[:] = tau_fill

    # Main loop
    for it in trange(nt - 1):

        tau[it + 1] = evolveFENEPnondim(Wi, Lm, L, tau[it], dt)
        tau[it + 1] += 2*eta*DD

    D = np.zeros((nt, 3, 3))
    D[:] = DD

    return tau, D


## MAIN

if __name__ == "__main__":

    # Time parameters
    dt = 1e-3
    Lt = 10
    nt = int(Lt/dt)
    t = np.linspace(0, Lt, nt)

    # Initialization
    eta = 0.01
    Wi = 3.0
    Lm = 5.0

    gamma_dot = 1.0
    
    tau, D = generateShearData(dt, Lt, eta, Wi, Lm, gamma_dot)

    fig, ax = plt.subplots(3, 3)
    tensorPlot(ax, tau, t)



# %%
