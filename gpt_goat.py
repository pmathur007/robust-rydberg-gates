"""
GOAT implementation (closed-system unitary version).
by: ChatGPT
References:
  Machnes et al., "Tunable, Flexible, and Efficient Optimization of Control Pulses for Practical Qubits"
  (arXiv / PRL) and its supplemental material. See:
    - arXiv:1507.04261 / Phys. Rev. Lett. 120, 150401 (2018).
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# -------------------------
# Utilities: Pauli matrices
# -------------------------
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
id2 = np.eye(2, dtype=complex)


# -------------------------
# Real/Imag packing helpers
# -------------------------
def pack_complex_list(mats):
    """
    Given list/array of complex matrices shape (n, N, N) or (N,N),
    return real vector stacking real and imag parts.
    """
    a = np.array(mats)
    if a.ndim == 2:
        a = a[np.newaxis, ...]
    n, N, _ = a.shape
    flat = a.reshape(n, -1)
    realvec = np.concatenate([flat.real, flat.imag], axis=1).ravel()
    return realvec


def unpack_complex_vector(vec, n_blocks, N):
    """
    vec: real vector of length 2 * n_blocks * N*N
    returns complex array shape (n_blocks, N, N)
    """
    total = 2 * n_blocks * N * N
    assert vec.size == total
    half = (n_blocks * N * N)
    real_part = vec[:half].reshape(n_blocks, N, N)
    imag_part = vec[half:].reshape(n_blocks, N, N)
    return real_part + 1j * imag_part


def pack_complex_vector(mat):
    """
    Given a single complex (N,N) matrix, pack to real vector [Re,Im].
    """
    N = mat.shape[0]
    flat = mat.ravel()
    return np.concatenate([flat.real, flat.imag])


# -------------------------
# GOAT core: augmented ODE
# -------------------------
def make_augmented_rhs(H0, H_controls, control_funcs, control_param_derivs):
    """
    Returns a function rhs(t, y, alpha) that computes the time derivative of the augmented state.
    - H0: (N,N) complex base Hamiltonian
    - H_controls: list of (N,N) complex control Hamiltonians H_k
    - control_funcs: list of functions c_k(t, alpha) -> scalar (real)
    - control_param_derivs: list of functions dc_k_dalpha_i(t, alpha) -> array(len(alpha))
       Alternatively, provide a single function that returns an (n_controls, n_params) array.
    The augmented state y contains:
       U (N,N) and for each parameter i: dU/dalpha_i (N,N)
    We'll pack them as a real vector for solve_ivp.
    """

    N = H0.shape[0]
    n_controls = len(H_controls)

    def rhs(t, y, alpha):
        # unpack
        n_params = alpha.size
        # total blocks = 1 + n_params
        blocks = 1 + n_params
        mats = unpack_complex_vector(y, blocks, N)  # shape (blocks, N, N)
        U = mats[0]
        dU_list = mats[1:]  # shape (n_params, N, N)

        # Build H(t)
        H = H0.copy()
        c_vals = np.zeros(n_controls, dtype=float)
        for k in range(n_controls):
            ck = control_funcs[k](t, alpha)
            c_vals[k] = ck
            H = H + ck * H_controls[k]

        # Build dH/dalpha_i = sum_k (∂c_k/∂alpha_i) H_k
        # We expect control_param_derivs(t, alpha) -> array (n_controls, n_params)
        dc = control_param_derivs(t, alpha)  # shape (n_controls, n_params)
        # dH_dalpha shape (n_params, N, N)
        dH_dalpha = np.zeros((n_params, N, N), dtype=complex)
        for i in range(n_params):
            for k in range(n_controls):
                if dc[k, i] != 0:
                    dH_dalpha[i] += dc[k, i] * H_controls[k]

        # Compute derivatives
        dU = -1j * (H @ U)
        ddU = np.zeros_like(dU_list)
        for i in range(n_params):
            ddU[i] = -1j * (H @ dU_list[i]) - 1j * (dH_dalpha[i] @ U)

        # pack back to real vector
        mats_out = np.vstack([dU[np.newaxis, ...], ddU])
        return pack_complex_list(mats_out)

    return rhs


# -------------------------
# Cost + gradient
# -------------------------
def gate_infidelity_and_grad(U_target, U, dU_dalpha_list):
    """
    Compute infidelity g = 1 - (|Tr(U_target^\dagger U)|/N)^2
    and its gradient w.r.t parameters alpha:
      d g / d alpha_i = -2 * Re( (O* / N^2) * dO/dalpha_i )
      where O = Tr(U_target^\dagger U), dO/dalpha_i = Tr(U_target^\dagger dU/dalpha_i)
    Inputs:
      U_target: (N,N) complex
      U: (N,N) complex (propagator at final time)
      dU_dalpha_list: list or array shape (n_params, N, N) of dU/dalpha_i
    Returns:
      g (scalar), grad (array length n_params)
    """
    N = U.shape[0]
    O = np.trace(U_target.conj().T @ U)
    denom = (N)
    g = 1.0 - (np.abs(O) / denom) ** 2

    n_params = dU_dalpha_list.shape[0]
    grad = np.zeros(n_params, dtype=float)
    for i in range(n_params):
        dO = np.trace(U_target.conj().T @ dU_dalpha_list[i])
        # derivative of |O|^2 = O* dO + O dO*
        # but we want derivative of (|O|/N)^2 = |O|^2 / N^2
        # dg/dalpha = - (1/N^2) * (O* dO + O dO*)
        # but it's simpler to use real part formula:
        grad[i] = -2.0 * np.real((np.conj(O) / (denom ** 2)) * dO)
    return g, grad


# -------------------------
# High-level GOAT optimize function
# -------------------------
def goat_optimize(
    H0,
    H_controls,
    control_funcs,
    control_param_derivs,
    alpha0,
    U_target,
    t_span,
    optimizer_opts=None,
    ode_rtol=1e-7,
    ode_atol=1e-9,
):
    """
    Run GOAT optimization.
    - H0, H_controls: Hamiltonian components.
    - control_funcs: list of c_k(t, alpha) functions
    - control_param_derivs: function(t, alpha) -> (n_controls, n_params) array of derivatives
    - alpha0: initial parameter vector (n_params,)
    - U_target: target unitary (N,N)
    - t_span: (t0, tf)
    Returns: result from scipy.optimize.minimize (fun, x, etc.)
    """

    N = H0.shape[0]
    n_params = alpha0.size

    rhs_fun = make_augmented_rhs(H0, H_controls, control_funcs, control_param_derivs)

    def eval_cost_and_grad(alpha):
        # initial augmented state: U(0)=I, dU/dalpha(0)=0
        blocks = 1 + n_params
        mats0 = np.zeros((blocks, N, N), dtype=complex)
        mats0[0] = np.eye(N, dtype=complex)
        # dU/dalpha initial zeros
        y0 = pack_complex_list(mats0)

        # integrate
        sol = solve_ivp(
            fun=lambda t, y: rhs_fun(t, y, alpha),
            t_span=t_span,
            y0=y0,
            rtol=ode_rtol,
            atol=ode_atol,
            method="RK45",
        )
        yf = sol.y[:, -1]
        mats_final = unpack_complex_vector(yf, blocks, N)
        U_final = mats_final[0]
        dU_final = mats_final[1:]

        g, grad = gate_infidelity_and_grad(U_target, U_final, dU_final)
        return g, grad

    # wrapper for optimizer: needs scalar return and gradient vector
    def fun_and_grad(alpha):
        g, grad = eval_cost_and_grad(alpha)
        return g, grad

    if optimizer_opts is None:
        optimizer_opts = {"maxiter": 200, "disp": True}

    res = minimize(
        fun=lambda a: fun_and_grad(a)[0],
        x0=alpha0,
        jac=lambda a: fun_and_grad(a)[1],
        method="L-BFGS-B",
        options=optimizer_opts,
    )
    return res


# -------------------------
# Example/demo: single qubit X gate
# -------------------------
if __name__ == "__main__":
    # system
    N = 2
    H0 = 0.5 * sz  # drift frequency (example)
    Hx = 0.5 * sx  # control Hamiltonian that generates X rotations

    # define an ansatz: sum of M sinusoids (amplitude and phase as params)
    M = 3
    n_params = 2 * M  # amplitudes & phases for each sinusoid
    freqs = np.linspace(1.0, 5.0, M)  # chosen basis frequencies

    def control_funcs_factory():
        def c_k_factory(k):
            def c_k(t, alpha):
                # alpha has shape (2*M,)
                amp = alpha[2 * k]
                ph = alpha[2 * k + 1]
                return amp * np.sin(2 * np.pi * freqs[k] * t + ph)

            return c_k

        return [c_k_factory(k) for k in range(M)]

    control_funcs = control_funcs_factory()
    H_controls = [Hx] * M  # each sinusoid multiplies same control operator

    def control_param_derivs(t, alpha):
        # returns (n_controls, n_params) array
        out = np.zeros((M, n_params), dtype=float)
        for k in range(M):
            amp = alpha[2 * k]
            ph = alpha[2 * k + 1]
            # derivative wrt amplitude
            out[k, 2 * k] = np.sin(2 * np.pi * freqs[k] * t + ph)
            # derivative wrt phase
            out[k, 2 * k + 1] = amp * np.cos(2 * np.pi * freqs[k] * t + ph)
        return out

    # target: X gate (up to global phase)
    U_target = np.array([[0, 1], [1, 0]], dtype=complex)  # X gate

    # initial parameters (small random)
    rng = np.random.default_rng(1)
    alpha0 = 0.1 * rng.normal(size=n_params)

    # time span (seconds, arbitrary units)
    t0, tf = 0.0, 1.0

    print("Starting GOAT optimization demo (single qubit X gate)...")
    res = goat_optimize(
        H0=H0,
        H_controls=H_controls,
        control_funcs=control_funcs,
        control_param_derivs=control_param_derivs,
        alpha0=alpha0,
        U_target=U_target,
        t_span=(t0, tf),
        optimizer_opts={"maxiter": 60, "disp": True},
    )

    print("Optimization finished. success:", res.success)
    print("Final parameters:", res.x)
    print("Final fun (infidelity):", res.fun)
