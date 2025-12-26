import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from scipy.constants import h
hbar = h / 2*np.pi

def pack_complex_list(mats):
    if mats.ndim == 2:
        mats = mats[np.newaxis, ...]
    # n = mats.shape[0]
    real_vec = np.concatenate([mats.real.flatten(), mats.imag.flatten()])
    # flat = mats.reshape(n, -1)
    # real_vec = np.concatenate([flat.real, flat.imag], axis=1).ravel()
    return real_vec

def unpack_complex_vector(vec, n_blocks, N):
    """
    vec: real vector of length 2 * n_blocks * N*N
    returns complex array shape (n_blocks, N, N)
    """
    assert vec.size == 2 * n_blocks * N * N
    real_part = vec[:n_blocks*N*N].reshape(n_blocks, N, N)
    imag_part = vec[n_blocks*N*N:].reshape(n_blocks, N, N)
    return real_part + 1j * imag_part

def make_ode_rhs(H0, H_controls, control_funcs, control_param_derivs, control_extra_params=None):
    """
    Returns a function rhs(t, y, alpha) that computes the time derivative of the augmented state.
    - H0: (N,N) complex base Hamiltonian
    - H_controls: list of (N,N) complex control Hamiltonians H_k
    - control_funcs: list of functions c_k(t, alpha) -> scalar (real)
    - control_param_derivs: list of functions dc_k_dalpha_i(t, alpha) -> array(len(alpha))
       Alternatively, provide a single function that returns an (n_controls, n_params) array.
    The augmented state y contains:
       U (N,N) and for each parameter i: dU/dalpha_i (N,N)
    """
    N = H0.shape[0]
    n_controls = len(H_controls) # number of control hamiltonians

    def ode_rhs(t, y, alpha):
        C = alpha.size
        n_blocks = 1 + C
        mats = unpack_complex_vector(y, n_blocks, N)
        U = mats[0]
        dUs = mats[1:] # shape (C, N, N)

        # Built H(t)
        H = H0.copy()
        for k in range(n_controls):
            ck = control_funcs[k](t, alpha, control_extra_params=control_extra_params)
            H = H + ck * H_controls[k]

        # get derivatives of H w.r.t. alpha
        dc = control_param_derivs(t, alpha, control_extra_params=control_extra_params) # shape (n_controls, C)
        dH_dalpha = np.zeros((C, N, N), dtype=complex)
        for i in range(C):
            for k in range(n_controls):
                dH_dalpha[i] += dc[k,i] * H_controls[k]

        # using coefficient matrix, compute RHS of ODE 
        dU_dt = -1j * (H @ U)
        ddU_dt = np.zeros((C, N, N), dtype=complex)
        for i in range(C):
            ddU_dt[i] = -1j * (H @ dUs[i]) - 1j * (dH_dalpha[i] @ U)

        # pack back into real vector
        mats_out = np.vstack([dU_dt[np.newaxis, ...], ddU_dt])
        return pack_complex_list(mats_out)
    
    return ode_rhs

def goat_infidelity_and_grad(U_target, U, dU_dalpha_list, single_qubit_phase, single_qubit_phase_weights):
    N = U.shape[0]

    single_qubit_phase_weights = np.array(single_qubit_phase_weights)
    U_sqphase = np.diag(np.exp(1j * single_qubit_phase * single_qubit_phase_weights)) 
    dU_sqphase = np.diag(1j * single_qubit_phase_weights * np.exp(1j * single_qubit_phase * single_qubit_phase_weights))

    # print(f"U_sqphase: {U_sqphase}")
    # print(f"dU_sqphase: {dU_sqphase}")
    # print(f"U: {U}")
    O = np.trace(U_target.conj().T @ U_sqphase @ U)
    # print(f"O: {O}")
    # infidelity2 = 1 - (np.abs(O) / N) ** 2 # square so its differentiable
    infidelity = 1 - (np.abs(O)/N) 

    C = dU_dalpha_list.shape[0]
    grad = np.zeros(C + 1, dtype=float)

    for i in range(C):
        # print(f"dU_dalpha_{i}: {dU_dalpha_list[i]}")
        dO = np.trace(U_target.conj() @ U_sqphase @ dU_dalpha_list[i])
        # print(f"dO_{i}: {dO}")
        # grad[i] = -2 * np.real((np.conj(O) / (N**2)) * dO)
        grad[i] = -(1/N) * np.real( (np.conj(O) / np.abs(O)) * dO)
    dO = np.trace(U_target.conj() @ dU_sqphase @ U)
    # print(f"dO_sqphase: {dO}")
    # grad[-1] = -2 * np.real((np.conj(O) / (N**2)) * dO)
    grad[-1] = -(1/N) * np.real( (np.conj(O) / np.abs(O)) * dO)

    # return infidelity2, grad
    return infidelity, grad

def to_infidelity_and_grad(U_target, U, dU_dalpha_list, single_qubit_phase, single_qubit_phase_weights):
    """This function only works for time-optimal gates and using the truncated target unitary"""
    single_qubit_phase_weights = np.array(single_qubit_phase_weights)
    U_sqphase = np.diag(np.exp(1j * single_qubit_phase * single_qubit_phase_weights)) 
    dU_sqphase = np.diag(1j * single_qubit_phase_weights * np.exp(1j * single_qubit_phase * single_qubit_phase_weights))

    U_ops = U_target.conj().T @ U_sqphase @ U
    a01 = U_ops[0,0]
    a11 = U_ops[1,1]
    tr_sum = 1 + 2*a01 + a11
    fidelity = (1/20) * ( (np.abs(tr_sum) ** 2) + 1 + 2*(np.abs(a01) ** 2) + (np.abs(a11) ** 2) )

    C = dU_dalpha_list.shape[0]
    grad = np.zeros(C + 1, dtype=float)
    for i in range(C):
        dU_ops = U_target.conj().T @ U_sqphase @ dU_dalpha_list[i]
        da01 = dU_ops[0,0]
        da11 = dU_ops[1,1]
        grad[i] += (1/10) * np.real(np.conj(tr_sum) * (2*da01 + da11))
        grad[i] += (1/10) * (2*np.real(np.conj(a01) * da01) + np.real(np.conj(a11) * da11))
    dU_phase_ops = U_target.conj().T @ dU_sqphase @ U
    dphase_a01 = dU_phase_ops[0,0]
    dphase_a11 = dU_phase_ops[1,1]
    grad[-1] += (1/10) * (np.real(np.conj(tr_sum) * (1 + 2*dphase_a01 + dphase_a11)))
    grad[-1] += (1/10) * (1 + 2*np.real(np.conj(a01) * dphase_a01) + np.real(np.conj(a11) * dphase_a11))

    return 1 - fidelity, -grad


def run_goat_optimization(
        H0,
        H_controls,
        control_funcs,
        control_param_derivs,
        alpha0,
        U_target,
        t_span,
        U_truncator,
        single_qubit_phase_weights,
        fidelity_func_name="GOAT",
        alpha_bounds=None,
        constraints=None,
        control_extra_params=None,
        optimizer_opts=None,
        ode_rtol=1e-7,
        ode_atol=1e-9,
        callback=None
):
    N = H0.shape[0]
    C = alpha0.size - 1

    ode_rhs_func = make_ode_rhs(H0, H_controls, control_funcs, control_param_derivs, control_extra_params=control_extra_params)

    if fidelity_func_name == "TO":
        fidelity_func = to_infidelity_and_grad
    else:
        fidelity_func = goat_infidelity_and_grad 

    def eval_cost_and_grad(alpha, single_qubit_phase):
        n_blocks = 1 + C
        mats0 = np.zeros((n_blocks, N, N), dtype=complex)
        mats0[0] = np.eye(N, dtype=complex)
        y0 = pack_complex_list(mats0)

        sol = solve_ivp(
            fun=lambda t, y: ode_rhs_func(t, y, alpha),
            t_span=t_span,
            y0=y0,
            rtol=ode_rtol,
            atol=ode_atol,
            method="DOP853"
        )

        yf = sol.y[:, -1]
        mats_final = unpack_complex_vector(yf, n_blocks, N)
        # print(f"U_untruncated: {mats_final[0]}")
        # print(f"dU_untruncated: {mats_final[1:]}")
        if U_truncator is not None:
            U_final = U_truncator(mats_final[0])
            # print(U_final @ np.conj(U_final).T)
            dU_final = U_truncator(mats_final[1:])
        else:
            U_final = mats_final[0]
            dU_final = mats_final[1:]

        cost, grad = fidelity_func(U_target, U_final, dU_final, single_qubit_phase, single_qubit_phase_weights) 
        # print(f"Cost: {cost}, Grad: {grad}")
        # print("---")
        return cost, grad

    if optimizer_opts is None:
        optimizer_opts = {"maxiter": 200, "disp": True, "ftol": 1e-10, "gtol": 1e-10} # for L-BFGS-B
        # optimizer_opts = {"maxiter": 200, "disp": True, "gtol": 1e-10, "xtol": 1e-10} # for trust-constr

    res = minimize(
        fun=lambda a: eval_cost_and_grad(a[:-1], a[-1])[0],
        x0=alpha0,
        jac=lambda a: eval_cost_and_grad(a[:-1], a[-1])[1],
        bounds=alpha_bounds,
        constraints=constraints,
        method="L-BFGS-B",
        options=optimizer_opts,
        callback=callback
    )

    return res
