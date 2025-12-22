import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import scipy.optimize
from IPython.display import clear_output
from utilities import clean_phases

info = {}

def run_grape(
    drift_hamiltonians,
    drive_hamiltonians,
    excitation_levels,
    single_qubit_phase_weights,
    T,
    steps,
    drive_amplitude=1.0,
    robustness=None,
    consider_global_phase=False,
    phases_guess=None,
    theta_guess=None,
    check_gradient=False,
    target_phase=np.pi,
    phase_variation_prefactor=0,
    minimize_options=None,
):
    global info
    info = {'F_var': 0, 'var': 0, 'iter': 0}

    dt = T / steps
    num_hams = len(drift_hamiltonians)
    if len(drive_hamiltonians) != num_hams:
        raise ValueError(
            "Number of drift Hamiltonians has to equal number of drive Hamiltonians"
        )
    dims = [h.shape[0] for h in drift_hamiltonians]

    target_phases = np.zeros(num_hams)
    target_phases[-1] = target_phase

    if robustness is None:
        robustness = []
    ham_rep = 1
    if 'DR' in robustness:
        ham_rep += 2
    if 'AR' in robustness:
        ham_rep += 1

    # Hamiltonian if laser phase = 0
    U0 = [np.zeros((steps, ham_rep * dim, ham_rep * dim), dtype=complex) for dim in dims]
    for i in range(steps):
        if callable(drive_amplitude):
            t = (i + 0.5) * dt
            amp = drive_amplitude(t)
        else:
            amp = drive_amplitude
        for j in range(num_hams):
            dim = dims[j]
            H0 = np.zeros((ham_rep * dim, ham_rep * dim), dtype=complex)
            for k in range(ham_rep):
                H0[k * dim : (k + 1) * dim, k * dim : (k + 1) * dim] = (
                    drift_hamiltonians[j] + amp * drive_hamiltonians[j]
                )
            if 'DR' in robustness:
                H0[dim : 2 * dim, :dim] = np.diag(excitation_levels[j][0])  # Delta_1
                H0[2 * dim : 3 * dim, :dim] = np.diag(excitation_levels[j][1])  # Delta_2
            if 'AR' in robustness:
                H0[-dim :, :dim] = amp * drive_hamiltonians[j]  # Omega
            U0[j][i] = scipy.linalg.expm(-1j * H0 * dt)

    excitation_levels_full = [np.zeros(ham_rep * dim) for dim in dims]
    for i in range(num_hams):
        dim = dims[i]
        for j in range(ham_rep):
            excitation_levels_full[i][j * dim : (j + 1) * dim] = np.sum(
                excitation_levels[i], axis=0
            )

    if phases_guess is None:
        phases_guess = 2 * np.pi * np.random.random(steps)
    if theta_guess is None:
        if consider_global_phase:
            if 'AR' in robustness:
                theta_guess = 2 * np.pi * np.random.random(3)
            else:
                theta_guess = 2 * np.pi * np.random.random(2)
        else:
            theta_guess = 2 * np.pi * np.random.random(1)
        
    params_guess = np.append(phases_guess, theta_guess)

    def infidelity(params):
        global info
        phases = params[:steps]
        theta = params[steps:]
        sq_phase = theta[-1]
        if consider_global_phase:
            global_phase = theta[0]
            if 'AR' in robustness:
                d_omega = theta[-2]
        else:
            global_phase, d_omega = 0, 0

        F = 0
        dF = np.zeros(len(params))

        for j in range(num_hams):
            dim = dims[j]
            psis = np.zeros((steps + 1, ham_rep * dim), dtype=complex)
            chis = np.zeros((steps + 1, ham_rep * dim), dtype=complex)
            psis[0, 0] = 1

            for i in range(steps):
                pre_mult = np.exp(1j * excitation_levels_full[j] * phases[i])
                post_mult = 1 / pre_mult
                psis[i + 1] = post_mult * (U0[j][i] @ (pre_mult * psis[i]))

            target_phase = single_qubit_phase_weights[j] * sq_phase + target_phases[j]
            if consider_global_phase:
                target_phase += global_phase
            chis[-1, 0] = -np.exp(1j * target_phase) * (
                1 + np.exp(-1j * target_phase) * psis[-1, 0]
            )
            if 'DR' in robustness:
                chis[-1, dim + 1 : 2 * dim] = psis[-1, dim + 1 : 2 * dim]
                chis[-1, 2 * dim + 1 : 3 * dim] = psis[-1, 2 * dim + 1 : 3 * dim]
            if 'AR' in robustness:
                chis[-1, : dim] += -1j * d_omega * psis[-1, -dim :] + d_omega ** 2 * psis[-1, : dim]  # The second term wasn't included by Sven.
                chis[-1, -dim :] += psis[-1, -dim :] + 1j * d_omega * psis[-1, : dim]

            for i in range(steps, 0, -1):
                pre_mult = np.exp(1j * excitation_levels_full[j] * phases[i - 1])
                post_mult = 1 / pre_mult
                chis[i - 1] = post_mult * (
                    np.conj(U0[j][i - 1]).T @ (pre_mult * chis[i])
                )

            F += 4 - np.abs((1 + np.exp(-1j * target_phase) * psis[-1, 0])) ** 2
            if 'DR' in robustness:
                F += np.linalg.norm(psis[-1, dim + 1 : 2 * dim]) ** 2
                F += np.linalg.norm(psis[-1, 2 * dim + 1 : 3 * dim]) ** 2
            if 'AR' in robustness:
                F += np.linalg.norm(psis[-1, -dim :] + 1j * d_omega * psis[-1, : dim]) ** 2

            for i in range(steps):
                dF[i] += 2 * np.real(
                    1j * np.vdot(chis[i], excitation_levels_full[j] * psis[i])
                )
                dF[i] -= 2 * np.real(
                    1j * np.vdot(chis[i + 1], excitation_levels_full[j] * psis[i + 1])
                )
            dF[-1] -= 2 * np.real(
                np.conj(1 + np.exp(-1j * target_phase) * psis[-1, 0])
                * (-1j)
                * single_qubit_phase_weights[j]
                * np.exp(-1j * target_phase)
                * psis[-1, 0]
            )
            if consider_global_phase:
                dF[steps] -= 2 * np.real(
                    np.conj(1 + np.exp(-1j * target_phase) * psis[-1, 0])
                    * (-1j)
                    * np.exp(-1j * target_phase)
                    * psis[-1, 0]
                )
                if 'AR' in robustness:
                    dF[steps + 1] += 2 * d_omega + 2 * np.real(
                        -1j * np.vdot(psis[-1, : dim], psis[-1, -dim :])
                    ) 

        if phase_variation_prefactor != 0:
            # Calculate variation of phase and add to cost function
            d_var = np.zeros(len(params))
            phase_diffs = np.diff(phases)
            var = np.sum(phase_diffs ** 2) / (T * dt)
            d_var[: len(phases) - 1] -= 2 * phase_diffs / (T * dt)
            d_var[1 : len(phases)] += 2 * phase_diffs / (T * dt)
            info['var'] = var
            F_var = phase_variation_prefactor * var
            dF_var = phase_variation_prefactor * d_var
            info['F_var'] = F_var
            F += F_var
            dF += dF_var

        info['F'] = F
        info['dF'] = dF
        return F, dF

    if check_gradient:
        F, dF = infidelity(params_guess)
        dF_num = np.zeros(len(params_guess))
        eps = 1e-6
        for i in range(len(params_guess)):
            params_guess[i] += eps
            F_new, _ = infidelity(params_guess)
            params_guess[i] -= eps
            dF_num[i] = (F_new - F) / eps

        # print("Analytic gradient: ", dF)
        # print("Numeric gradient: ", dF_num)
        return dF, dF_num

    def callback(xk):
        global info
        xk[:steps] = clean_phases(xk[:steps], return_cleaned=True)
        if info['iter'] % 30 == 0:
            clear_output(wait=True)
            plt.figure(figsize=(5, 5/1.618))
            plt.plot(np.linspace(0, T, steps, endpoint=False), xk[:steps], '.')
            plt.xlabel('Time')
            plt.ylabel('Phase')
            plt.show()
        print(
            f"{info['iter']} -- "
            f"cost = {info['F']:.5e} ({info['F'] - info['F_var']:.5e}), "
            f"gradient = {np.linalg.norm(info['dF']):.5e}, "
            f"smoothness = {info['var']:.5e}", end="\r", flush=True
        )
        # xk[:steps] = clean_phases(xk[:steps], return_cleaned=True)
        info['iter'] += 1
    
    kwargs = {"method": "BFGS", "options": {"gtol": 0, "maxiter": 100000000000}}
    if minimize_options is not None:
        kwargs.update(minimize_options)
    opt_res = scipy.optimize.minimize(
        infidelity,
        params_guess,
        jac=True,
        callback=callback,
        **kwargs
    )

    cost = opt_res.fun
    phases = opt_res.x[:steps]
    theta = opt_res.x[steps:]

    return cost, phases, theta


def calculate_infidelity_only(
    drift_hamiltonians,
    drive_hamiltonians,
    excitation_levels,
    single_qubit_phase_weights,
    T,
    phases,
    single_qubit_phase,
    robustness,
    drive_amplitude=1.0,
    drive_detuning=[0.0, 0.0],
    fidelity_weights=None,
    weight_sum=None,
    target_phase=np.pi,
):

    num_hams = len(drift_hamiltonians)
    if len(drive_hamiltonians) != num_hams:
        raise ValueError(
            "Number of drift Hamiltonians has to equal number of drive Hamiltonians"
        )
    dims = [h.shape[0] for h in drift_hamiltonians]

    target_phases = np.zeros(num_hams)
    target_phases[-1] = target_phase

    if fidelity_weights is None:
        fidelity_weights = [1, 1, 1, 1]
    
    if weight_sum is None:
        weight_sum = sum(fidelity_weights)

    if robustness is None:
        robustness = []
    if 'DR' in robustness:
        pulse_sgn = [1, -1]
        single_qubit_phase *= 2
    else:
        pulse_sgn = [1]

    psis = [np.zeros(dim, dtype=complex) for dim in dims]
    for i in range(num_hams):
        psis[i][0] = 1

    steps = len(phases)
    dt = T / steps
    for sgn_detuning in pulse_sgn:
        for i in range(steps):
            if callable(drive_amplitude):
                t = (i + 0.5) * dt
                amp = drive_amplitude(t)
            else:
                amp = drive_amplitude
            for j in range(num_hams):
                H = drift_hamiltonians[j] + amp * drive_hamiltonians[j] + np.sum([sgn_detuning * drive_detuning[k] * np.diag(excitation_levels[j][k]) for k in range(2)], axis=0)
                U = scipy.linalg.expm(-1j * H * dt)
                pre_mult = np.exp(1j * np.sum(excitation_levels[j], axis=0) * phases[i])
                post_mult = 1 / pre_mult
                psis[j] *= pre_mult
                psis[j] = U @ psis[j]
                psis[j] *= post_mult

    ol_sum = weight_sum - np.sum(fidelity_weights)
    for j in range(num_hams):
        target_phase = (
            single_qubit_phase * single_qubit_phase_weights[j] + target_phases[j]
        )
        ol_sum += fidelity_weights[j] * np.exp(-1j * target_phase) * psis[j][0]
    F = np.abs(ol_sum) ** 2 / weight_sum**2

    return 1 - F


def calculate_infidelity_and_derivative(
    drift_hamiltonians,
    drive_hamiltonians,
    excitation_levels,
    single_qubit_phase_weights,
    T,
    phases,
    single_qubit_phase,
    robustness,
    drive_amplitude=1.0,
    drive_detuning=[0.0, 0.0],
    fidelity_weights=None,
    weight_sum=None,
    target_phase=np.pi,
    eps=1e-3,
):

    def infid_func(eps_delta, eps_omega): 
        if callable(drive_amplitude):
            drive_amplitude_new = lambda t: (1 + eps_omega) * drive_amplitude(t)
        else:
            drive_amplitude_new = (1 + eps_omega) * drive_amplitude
        
        drive_detuning_new = [eps_delta, 0]

        infid = calculate_infidelity_only(
            drift_hamiltonians,
            drive_hamiltonians,
            excitation_levels,
            single_qubit_phase_weights,
            T,
            phases,
            single_qubit_phase,
            robustness,
            drive_amplitude_new,
            drive_detuning_new,
            fidelity_weights,
            weight_sum,
            target_phase,
        )

        return infid

    infid_0 = infid_func(0, 0)
    infid_plus_delta = infid_func(eps, 0)
    infid_minus_delta = infid_func(-eps, 0)
    infid_plus_omega = infid_func(0, eps)
    infid_minus_omega = infid_func(0, -eps)

    first_deriv_delta = (infid_plus_delta - infid_minus_delta) / (2 * eps)
    first_deriv_omega = (infid_plus_omega - infid_minus_omega) / (2 * eps)
    second_deriv_delta = (infid_plus_delta - 2 * infid_0 + infid_minus_delta) / eps**2
    second_deriv_omega = (infid_plus_omega - 2 * infid_0 + infid_minus_omega) / eps**2

    return infid_0, first_deriv_delta, first_deriv_omega, second_deriv_delta, second_deriv_omega
