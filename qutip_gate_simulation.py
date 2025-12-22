import qutip
import numpy as np
import matplotlib.pyplot as plt

# define basis states
psi_00 = qutip.tensor(qutip.basis(3, 0), qutip.basis(3, 0))
psi_01 = qutip.tensor(qutip.basis(3, 0), qutip.basis(3, 1))
psi_0r = qutip.tensor(qutip.basis(3, 0), qutip.basis(3, 2))
psi_10 = qutip.tensor(qutip.basis(3, 1), qutip.basis(3, 0))
psi_11 = qutip.tensor(qutip.basis(3, 1), qutip.basis(3, 1))
psi_1r = qutip.tensor(qutip.basis(3, 1), qutip.basis(3, 2))
psi_r0 = qutip.tensor(qutip.basis(3, 2), qutip.basis(3, 0))
psi_r1 = qutip.tensor(qutip.basis(3, 2), qutip.basis(3, 1))
psi_rr = qutip.tensor(qutip.basis(3, 2), qutip.basis(3, 2))

state_to_idx = {"00": 0, "01": 1, "0r": 2, "10": 3, "11": 4, "1r": 5, "r0": 6, "r1": 7, "rr": 8}

# general function for creating an Rydberg Hamiltonian with arbitrary drive strength and phase profile, but fixed detuning
def hamiltonian_from_pulse_profile(drive_strength, phase, blockade_strength, detuning=0):
    """drive strength should be real"""
    Omega = qutip.coefficient(lambda t: 0.5 * drive_strength(t) * np.exp(1j * phase(t)))
    Omega_conj = qutip.coefficient(lambda t: 0.5 * drive_strength(t) * np.exp(-1j * phase(t)))

    H01 = Omega * (psi_01 * psi_0r.dag()) + Omega_conj * (psi_0r * psi_01.dag()) - detuning * psi_0r * psi_0r.dag() 
    H10 = Omega * (psi_10 * psi_r0.dag()) + Omega_conj * (psi_r0 * psi_10.dag()) - detuning * psi_r0 * psi_r0.dag() 
    H11 = Omega * (psi_11 * psi_1r.dag() + psi_11 * psi_r1.dag() + psi_1r * psi_rr.dag() + psi_r1 * psi_rr.dag()) \
        + Omega_conj * (psi_1r * psi_11.dag() + psi_r1 * psi_11.dag() + psi_rr * psi_1r.dag() + psi_rr * psi_r1.dag()) \
        - detuning * (psi_1r * psi_1r.dag() + psi_r1 * psi_r1.dag()) \
        + (blockade_strength - 2*detuning) * psi_rr * psi_rr.dag()

    return H01 + H10 + H11

# utility function for getting pulse parameters at arbitrary times
def data_from_tstep(t, t_steps, data):
    i = 0
    while i < len(t_steps)-1 and t_steps[i+1] < t:
        i += 1
    return data[i]

# utility function for plotting
def plot_populations(results, psi0_strs, ts):
    for i in range(len(psi0_strs)):
        P0 = np.zeros(ts.size)
        for t in range(ts.size):
            P0[t] = np.abs(results[i].states[t].full()[state_to_idx[psi0_strs[i]], 0]) ** 2
        plt.plot(ts, P0, alpha=0.7, label=f"{psi0_strs[i]}")
    plt.ylim((-0.05, 1.05))
    plt.xlabel("$t\\Omega_{\\text{max}}$", fontsize=16)
    plt.ylabel("Population", fontsize=16)
    plt.legend(title="State")
    # plt.show()

def simulate_gate(drive_strength_func, phase_func, blockade_strength, detuning, t_min, t_max, nts_sim, plot=True):
    ts = np.linspace(t_min, t_max, nts_sim)

    psi0s = [psi_00, psi_01, psi_10, psi_11]
    psi0_strs = ["00", "01", "10", "11"]

    H = hamiltonian_from_pulse_profile(drive_strength_func,
                                          phase_func,
                                          blockade_strength,
                                          detuning=detuning)
    results = []
    for psi0 in psi0s:
        results.append(qutip.mesolve(H, psi0, ts, []))
    
    if plot:
        plot_populations(results, psi0_strs, ts)

    return results