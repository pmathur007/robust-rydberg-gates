import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def clean_phases(phases, return_cleaned=False):
    steps = len(phases)
    for i in range(1, steps):
        while phases[i] - phases[i-1] > np.pi:
            phases[i] -= 2*np.pi
        while phases[i] - phases[i-1] < -np.pi:
            phases[i] += 2*np.pi
    phases -= phases[0]
    if return_cleaned:
        return phases
    else:
        return None

def plot_pulse(T, amplitude, phases):
    steps = len(phases)
    
    # clean phases
    clean_phases(phases)
            
    dt = T/steps
    times_mid = np.linspace(dt/2, T-dt/2, steps)
    amps = np.array([amplitude(t) for t in times_mid])
    
    time_points = []
    phase_points = []
    amp_points = []
    for i in range(steps):
        time_points.append(i*dt)
        time_points.append((i+1)*dt)
        for _ in range(2):
            phase_points.append(phases[i])
            amp_points.append(amps[i])
    
    fig, ax = plt.subplots(nrows=2, ncols = 1, sharex = True, figsize = (8,8))
    ax[0].plot(time_points, amp_points)
    ax[1].plot(time_points, phase_points)
    plt.show()


def amplitude_function_from_pulse_area(pulse_area, risetime_fraction, pulse_time=None):
    if pulse_time is None:
        T = pulse_area / (1 - risetime_fraction)
    else:
        T = pulse_time
    def amplitude(t):
        t_rel = t / T
        if t_rel < risetime_fraction:
            return np.sin(np.pi / 2 * t_rel / risetime_fraction) ** 2
        if t_rel > 1 - risetime_fraction:
            return np.sin(np.pi / 2 * (1 - t_rel) / risetime_fraction) ** 2
        return 1
    
    return T, amplitude

def stretch_pulse(phases, T, amplitude_function, new_steps=None):
    steps = len(phases)
    dt = T/steps
    
    pulse_areas = [0]
    for i in range(steps):
        t = (i+0.5)*dt
        pulse_areas.append(pulse_areas[-1] + dt*amplitude_function(t))
    pulse_areas = np.array(pulse_areas)
    total_area = pulse_areas[-1]
    pulse_areas = (pulse_areas[:-1]+pulse_areas[1:])/2
    
    phases_stretched = []
    for i in range(steps):
        idx = int(pulse_areas[i]/total_area*steps)
        phases_stretched.append(phases[idx])
    
    if new_steps is None:
        return phases_stretched
    else:
        phases_func = interp1d(np.linspace(0, T, steps, endpoint=False), phases_stretched, kind='linear', 
                               bounds_error=False, fill_value=(phases_stretched[0], phases_stretched[-1]))
        return phases_func(np.linspace(0, T, new_steps, endpoint=False))
    