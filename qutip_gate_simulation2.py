import qutip
import numpy as np
import matplotlib.pyplot as plt

# define basis states
psi_01 = qutip.basis(4, 0)
psi_11 = qutip.basis(4, 1)
psi_0r = qutip.basis(4, 2)
psi_W = qutip.basis(4, 3)

cz_target_states = [
    qutip.Qobj([1, 0, 0, 0]),
    qutip.Qobj([0, -1, 0, 0])
]

state_to_idx = {"01": 0, "11": 1, "0r": 2, "W": 3}


def data_from_tstep(t, t_steps, data):
    i = 0
    while i < len(t_steps)-1 and t_steps[i+1] < t:
        i += 1
    return data[i]

def plot_populations(results, psi0_strs, ts):
    for i in range(len(psi0_strs)):
        P0 = np.zeros(ts.size)
        for t in range(ts.size):
            P0[t] = np.abs(results[i].states[t].full()[state_to_idx[psi0_strs[i]], 0]) ** 2
        plt.plot(ts, P0, alpha=0.7, label=f"{psi0_strs[i]}")
    plt.ylim((-0.05, 1.05))
    plt.xlabel("$t\\Omega_{\\text{max}}$")
    plt.ylabel("Population")
    plt.legend(title="State")
    # plt.show()

def simulate_gate(drive_strength_func, phase_func, t_min, t_max, nts_sim, plot=True):
    ts = np.linspace(t_min, t_max, nts_sim)

    psi0s = [psi_01, psi_11]
    psi0_strs = ["01", "11"]
    
    H = hamiltonian_from_pulse_profile(drive_strength_func,
                                       phase_func)
    
    results = []
    for psi0 in psi0s:
        results.append(qutip.sesolve(H, psi0, ts, []))

    if plot:
        plot_populations(results, psi0_strs, ts)

    return results

def calculate_fidelity(results, single_qubit_phase=0, target_states=cz_target_states, metric="GOAT"):
    if metric == "TO":
        sq_gate = qutip.Qobj([[np.exp(1j * single_qubit_phase), 0, 0, 0],
                            [0, np.exp(2j * single_qubit_phase), 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0]])
        
        a01 = target_states[0].dag() * sq_gate * qutip.Qobj(results[0].states[-1].full())
        a11 = target_states[1].dag() * sq_gate * qutip.Qobj(results[1].states[-1].full())

        return np.abs((1/20) * (np.abs(1 + 2*a01 + a11)**2  + (1 + 2*a01*np.conj(a01) + a11*np.conj(a11)) ))
    else: # metric == "GOAT"
        basis01 = np.array([1, 0, 0, 0])
        basis11 = np.array([0, 1, 0, 0])
        U_T01 = results[0].states[-1].full().flatten()
        U_T11 = results[1].states[-1].full().flatten()

        N = 4
        U_T = np.array([[1, 0, 0, 0],
                        [0, basis01.conj().T @ U_T01, 0, basis01.conj().T @ U_T11],
                        [0, 0, basis01.conj().T @ U_T01, basis01.conj().T @ U_T11],
                        [0, 0, basis11.conj().T @ U_T01, basis11.conj().T @ U_T11]])
        sq_gate = np.array([[1, 0, 0, 0],
                            [0, np.exp(1j * single_qubit_phase), 0, 0],
                            [0, 0, np.exp(1j * single_qubit_phase), 0],
                            [0, 0, 0, np.exp(2j * single_qubit_phase)]])
        U_target = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, -1]])

        return np.abs(np.trace(U_target.conj().T @ sq_gate @ U_T)) / N

        # N = 2
        # U_T = np.array([[basis01.conj().T @ U_T01, basis01.conj().T @ U_T11],
        #                 [basis11.conj().T @ U_T01, basis11.conj().T @ U_T11]]) 
        # print(U_T)
        # sq_gate = np.array([[np.exp(1j * single_qubit_phase), 0],
        #                     [0, np.exp(2j * single_qubit_phase)]])
        # U_target = np.array([[1, 0],
        #                      [0, -1]])

        # return np.abs(np.trace(U_target.conj().T @ sq_gate @ U_T)) / N