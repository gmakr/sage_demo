import numpy as np
import torch
def battery_simulator(A, C, E, At, Ct, t_end, dt):
    """
    A simple battery simulator based on the Shepherd model.

    Parameters:
    A, C: Parameters related to the anode and cathode materials.
    E: Electrolyte concentration.
    T: Electrode thickness.
    I: Discharge current.
    t_end: End time for the simulation.
    dt: Time step for the simulation.

    Returns:
    t_array: Array of time points.
    V_array: Array of voltages at each time point.
    E_array: Array of cumulative energy at each time point.
    Q_array: Array of remaining capacity at each time point.
    SOC_end: State of charge at the end of simulation.
    """
    X = At
    Y = Ct
    term1 = .75*np.exp(-((9*X - 2)**(2) + (9*Y - 2)**(2))/4)
    term2 = .75*np.exp(-((9*X + 1)**(2))/49 - (9*Y + 1)/10)
    term3 = .5*np.exp(-((9*X - 7)**(2) + (9*Y - 3)**(2))/4)
    term4 = .2*np.exp(-(9*X - 4)**(2) - (9*Y - 7)**(2))

    f = term1 + term2 + term3 - term4
    return f
