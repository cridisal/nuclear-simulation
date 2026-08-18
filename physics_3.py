import random
from collections import deque
import cupy as cp

def build_grid(grid_size, domain_half):
    """
    Build a 2D grid and its corresponding Fourier-space grid.
    First the physical-space grid is built with cp.meshgrid,
    then the Fourier-space grid is built with cp.fft.fftfreq and cp.meshgrid.

    Args:
        grid_size (int): Number of points on each size of the grid
        domain_half (float): Half amplitude of our physical domain

    Returns:
        tuple: A tuple which contains: X, Y, KX, KY, dx.
    """
    # The meshgrid is created using as dimentions half domains
    X, Y = cp.meshgrid(
        cp.linspace(-domain_half, domain_half, grid_size, dtype=cp.float32),
        cp.linspace(-domain_half, domain_half, grid_size, dtype=cp.float32),
    )
    dx = (2.0 * domain_half) / grid_size

    # This is the Fourier-space grid, which is used to calculate the gradients of the wavefunctions in Fourier space
    kx_1d = (cp.fft.fftfreq(grid_size, d=dx) * 2.0 * cp.pi).astype(cp.float32)
    KX, KY = cp.meshgrid(kx_1d, kx_1d)
    return X, Y, KX, KY, dx


def wave_packet(X, Y, x0, y0, kx, ky, dx, sigma=3.0):
    """
    Generate a normalized complex Gaussian wave packet, which is used as an initial condition
    for the Schrodinger equation.

    Args:
        X (cupy.ndarray): 2D array of x-coordinates
        Y (cupy.ndarray): 2D array of y-coordinates
        x0 (float): x-coordinate of the center of the wave packet
        y0 (float): y-coordinate of the center of the wave packet
        kx (float): x-component of the wave vector
        ky (float): y-component of the wave vector
        dx (float): grid spacing
        sigma (float, optional): The spatial width of the Gaussian envelope. Defaults to 3.0.

    Returns:
        cupy.ndarray: A 2D array representing the complex wave packet.
    """
    X = cp.asarray(X, dtype=cp.float32)
    Y = cp.asarray(Y, dtype=cp.float32)

    # The gaussian function is calculated around the given center (x0, y0)
    gaussian = cp.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma ** 2)).astype(cp.float32)
    phase = cp.exp(cp.complex64(1j) * (cp.float32(kx) * X + cp.float32(ky) * Y)).astype(cp.complex64)

    # The phase (the kick given to the particle) is multiplied for the wavefunction
    psi = (gaussian * phase).astype(cp.complex64)

    # The function is normalized, since the total probability must always add up to one
    norma = cp.sqrt(cp.sum(cp.abs(psi) ** 2) * (cp.float32(dx) ** 2))
    return (psi / norma).astype(cp.complex64)


def create_nucleus(center_x, center_y, num_protons, num_neutrons, kx_kick, ky_kick, X, Y, dx, sigma=1.2):
    """
    Build a compact shell arrangement of proton/neutron wave packets.

        Args:
        center_x (float): x-coordinate of the center of the nucleus
        center_y (float): y-coordinate of the center of the nucleus
        num_protons (int): number of protons in the nucleus
        num_neutrons (int): number of neutrons in the nucleus
        kx_kick (float): x-component of the kick for the wave packets
        ky_kick (float): y-component of the kick for the wave packets
        X (cupy.ndarray): 2D array of x-coordinates
        Y (cupy.ndarray): 2D array of y-coordinates
        dx (float): grid spacing
        sigma (float, optional): The spatial width of the Gaussian envelope. Defaults to 3.0.

    Returns:
        tuple: A tuple containing:
            - phi_array_gpu (cupy.ndarray): 3D array of complex wave packets for the nucleus
            - is_proton_gpu (cupy.ndarray): 1D boolean array indicating which wave packets are protons
            - spin_gpu (cupy.ndarray): array containing spin information
    """

    # The total number of nucleons is calculated and a list of booleans is created to indicate which wave packets are protons and which are neutrons
    total_nucleons = num_protons + num_neutrons
    is_proton_pool = [True] * num_protons + [False] * num_neutrons
    rng = random.Random(42)

    # This instruction generates a list of random spins, depending on whether the total nucleons are even or odd
    spin_pool = [1] * ((total_nucleons + 1) // 2) + [-1] * (total_nucleons // 2)
    rng_spin = random.Random(4242)
    rng_spin.shuffle(spin_pool)

    # This list is shuffled to randomize the order of protons and neutrons in the nucleus
    rng.shuffle(is_proton_pool)

    # These two variables are used to calculate the spacing between the shells and between the wave packets
    shell_spacing = sigma * 0.85
    arc_spacing = sigma * 0.85

    positions = []
    nucleons_left = total_nucleons

    # The first nucleon is placed at the center of the nucleus
    if nucleons_left > 0:
        positions.append((center_x, center_y))
        nucleons_left -= 1

    # Then the nucleons are placed in concentric shells around the center, with the two spacings defined before
    shell_index = 1
    while nucleons_left > 0:

        # The radius of the current shell is calculated and the circumference is used to determine how many nucleons can fit in that shell
        current_radius = shell_index * shell_spacing
        circumference = 2 * cp.pi * current_radius
        capacity = int(circumference / arc_spacing)
        if capacity == 0:
            capacity = 1

        # The angular distance at which two nucleons have to be positioned is calculated
        n_in_shell = min(capacity, nucleons_left)
        d_theta = (2 * cp.pi) / n_in_shell
        offset = (shell_index % 2) * (d_theta / 2.0)

        # The nucleons are actually placed in their position
        for i in range(n_in_shell):
            theta = i * d_theta + offset
            x_pos = center_x + current_radius * cp.cos(theta)
            y_pos = center_y + current_radius * cp.sin(theta)
            positions.append((float(x_pos), float(y_pos)))

        nucleons_left -= n_in_shell
        shell_index += 1

    # A list with the wave packets in their relative positions is generated
    local_phi_list = [
        wave_packet(X=X, Y=Y, x0=pos[0], y0=pos[1], kx=kx_kick, ky=ky_kick, dx=dx, sigma=sigma)
        for pos in positions[:total_nucleons]
    ]

    # A 3D tensor is created with all the wave functions and a 1D boolean array created to indicate which nucleon is a proton or a neutron
    phi_array_gpu = cp.stack(local_phi_list).astype(cp.complex64)
    is_proton_gpu = cp.array(is_proton_pool, dtype=cp.bool_)
    spin_gpu = cp.array(spin_pool[:total_nucleons], dtype=cp.float32)
    return phi_array_gpu, is_proton_gpu, spin_gpu


def create_free_particle(center_x, center_y, kx_kick, ky_kick, X, Y, dx, sigma=1.5, is_proton=False, spin = 1):
    """
    Lightweight single-particle constructor (e.g. an incoming neutron). Avoids the shell-packing machinery of `create_nucleus`, which is
    unnecessary overhead for a single free nucleon.

    Args:
        center_x (float): x-coordinate of the center of the particle
        center_y (float): y-coordinate of the center of the particle
        kx_kick (float): x-component of the kick for the wave packets
        ky_kick (float): y-component of the kick for the wave packets
        X (cupy.ndarray): 2D array of x-coordinates
        Y (cupy.ndarray): 2D array of y-coordinates
        dx (float): grid spacing
        sigma (float, optional): The spatial width of the Gaussian envelope. Defaults to 1.5.
        is_proton (bool, optional): Whether the particle is a proton or a neutron. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - phi_array_gpu (cupy.ndarray): 3D array of complex wave packets for the particle
            - is_proton_gpu (cupy.ndarray): 1D boolean array indicating whether the particle is a proton or a neutron
            - spin_gpu (cupy.ndarray): array containing the information on the spin
    """
    phi = wave_packet(X=X, Y=Y, x0=center_x, y0=center_y, kx=kx_kick, ky=ky_kick, dx=dx, sigma=sigma)
    phi_array_gpu = phi[cp.newaxis, :, :].astype(cp.complex64)
    is_proton_gpu = cp.array([is_proton], dtype=cp.bool_)
    spin_gpu = cp.array([spin], dtype=cp.float32)
    return phi_array_gpu, is_proton_gpu, spin_gpu


def _chebyshev_distance(X, Y):
    return cp.maximum(cp.abs(X), cp.abs(Y))


def build_boundary_mask(X, Y, domain_half, absorb_start_frac=0.80):
    """
    Large, smooth absorbing halo: the inner region stays effectively free, while only the outermost band is tapered down toward the boundary. This
    prevents reflections without disturbing the simulation until the packet reaches the true edge of the box.

        Args:
        X (cupy.ndarray): 2D array of x-coordinates
        Y (cupy.ndarray): 2D array of y-coordinates
        domain_half (float): Half amplitude of our physical domain
        absorb_start_frac (float, optional): Fraction of the domain where the absorbing halo starts. Defaults to 0.80.

        Returns:
            cupy.ndarray: A 2D array representing the absorbing mask.
    """
    absorb_start = domain_half * absorb_start_frac
    d = _chebyshev_distance(X, Y)
    span = max(domain_half - absorb_start, 1e-6)
    t = cp.clip((d - absorb_start) / span, 0.0, 1.0)

    # This is a smooth cubic taper (0->1) that is zero at the inner edge of the absorbing halo and 1 at the true boundary. 
    taper = 1.0 - (3.0 * t ** 2 - 2.0 * t ** 3)
    return taper.astype(cp.float32)


def build_absorbing_potential(X, Y, domain_half, absorb_start_frac=0.80, strength=6.0, power=2.0):
    """
    Real, non-negative potential ramping up from 0 at `absorb_start` to `strength` at the domain edge. Used as a damping and absorbing term.

    Args:
        X (cupy.ndarray): 2D array of x-coordinates
        Y (cupy.ndarray): 2D array of y-coordinates
        domain_half (float): Half amplitude of our physical domain
        absorb_start_frac (float, optional): Fraction of the domain where the absorbing halo starts. Defaults to 0.80.
        strength (float, optional): Maximum value of the absorbing potential. Defaults to 6.0.
        power (float, optional): Power law exponent for the potential profile. Defaults to 2.0.
    
    Returns:
        cupy.ndarray: A 2D array representing the absorbing potential.
    """
    absorb_start = domain_half * absorb_start_frac
    d = _chebyshev_distance(X, Y)
    excess = cp.clip(d - absorb_start, 0.0, None)
    span = max(domain_half - absorb_start, 1e-6)
    return (strength * (excess / span) ** power).astype(cp.float32)


def compute_densities(phi_array, is_proton_array, KX, KY):
    """
    This function computes the actual densities of the nucleons in our system, which will be crucial to calculate the
    mean field and the evolution of our system.

    Args:
        phi_array (cupy.ndarray): 3D array of complex wave packets for the system
        is_proton_array (cupy.ndarray): 1D boolean array indicating which wave packets are protons
        KX (cupy.ndarray): 2D array of x-components of the wave vectors
        KY (cupy.ndarray): 2D array of y-components of the wave vectors

    Returns:
        tuple: A tuple containing:
            - rho (cupy.ndarray): 2D array representing the total nucleon density
            - rho_p (cupy.ndarray): 2D array representing the proton density
            - tau (cupy.ndarray): 2D array representing the kinetic energy density
            - j_x (cupy.ndarray): 2D array representing the x-component of the current density
            - j_y (cupy.ndarray): 2D array representing the y-component of the current density
            - phi_fft_array (cupy.ndarray): 3D array of Fourier-transformed wave packets
    """
    # The wavefunctions are converted to the GPU and the Fourier transform is calculated
    phi_array = cp.asarray(phi_array, dtype=cp.complex64)
    is_proton_array = cp.asarray(is_proton_array, dtype=cp.bool_)
    phi_fft_array = cp.fft.fft2(phi_array, axes=(1, 2)).astype(cp.complex64)

    # The gradients of the wavefunctions are calculated in Fourier space using the Fourier transform properties
    grad_x = cp.fft.ifft2(1j * KX[cp.newaxis, :, :] * phi_fft_array, axes=(1, 2)).astype(cp.complex64)
    grad_y = cp.fft.ifft2(1j * KY[cp.newaxis, :, :] * phi_fft_array, axes=(1, 2)).astype(cp.complex64)

    # The first density is the total density of nucleons, which is calculated as the sum of the absolute value of the wavefunctions squared
    rho_nucleons = cp.abs(phi_array) ** 2
    rho = cp.sum(rho_nucleons, axis=0)

    # Then the proton density is calculated as the sum of the absolute value of the wavefunctions squared, but only for the wavefunctions that correspond to protons
    if is_proton_array.size > 0 and bool(cp.any(is_proton_array)):
        rho_p = cp.sum(rho_nucleons[is_proton_array], axis=0)
    else:
        rho_p = cp.zeros_like(rho)

    # The other three densities are the kinetic energy one and the current ones (both describe a movement)
    tau = cp.sum(cp.abs(grad_x) ** 2 + cp.abs(grad_y) ** 2, axis=0)
    j_x = cp.sum(cp.real(cp.imag(cp.conj(phi_array) * grad_x)), axis=0)
    j_y = cp.sum(cp.real(cp.imag(cp.conj(phi_array) * grad_y)), axis=0)

    return rho, rho_p, tau, j_x, j_y, phi_fft_array


def smooth_density(rho, KX, KY, corr_length):
    """
    This function is used to smooth the density of the nucleons in our system, which is crucial to avoid numerical
    instabilities and to have a more realistic description of the system.

    Args:
        rho (cupy.ndarray): 2D array representing the total nucleon density
        KX (cupy.ndarray): 2D array representing the x-component of the wave vector
        KY (cupy.ndarray): 2D array representing the y-component of the wave vector
        corr_length (float): Correlation length for smoothing

    Returns:
        cupy.ndarray: Smoothed density array
    """

    # The convolution is performed in the Fourier space, where we can apply it by simply multiplying the Fourier transform of the density with a Gaussian kernel.
    if corr_length <= 0:
        return rho
    k2 = KX ** 2 + KY ** 2
    kernel = cp.exp(-0.5 * k2 * (corr_length ** 2))
    rho_fft = cp.fft.fft2(rho)
    return cp.real(cp.fft.ifft2(rho_fft * kernel)).astype(cp.float32)


def compute_skyrme_fields(rho, rho_p, tau, j_x, j_y, KX, KY, params):
    """
    This function computes the mean-field potentials and effective mass for the Skyrme energy density functional,
    which is used to describe the nuclear interactions in our system.

    Args:
        rho (cupy.ndarray): 2D array representing the total nucleon density
        rho_p (cupy.ndarray): 2D array representing the proton density
        tau (cupy.ndarray): 2D array representing the kinetic energy density
        j_x (cupy.ndarray): 2D array representing the x-component of the current density
        j_y (cupy.ndarray): 2D array representing the y-component of the current density
        KX (cupy.ndarray): 2D array representing the x-component of the wave vector
        KY (cupy.ndarray): 2D array representing the y-component of the wave vector
        params (dict): Dictionary containing the Skyrme parameters

    Returns:
        tuple: A tuple containing:
            - B (cupy.ndarray): 2D array representing the effective mass
            - U (cupy.ndarray): 2D array representing the mean-field potential
            - A_x (cupy.ndarray): 2D array representing the x-component of the vector potential
            - A_y (cupy.ndarray): 2D array representing the y-component of the vector potential
            - U_coulomb (cupy.ndarray): 2D array representing the Coulomb potential for protons
            - SO_x
            - SO_y
    """
    hbar2_2m = params.get('hbar2_2m', 1.0)
    C_tau = params.get('C_tau', 0.1)
    alpha = params.get('alpha', -2.0)
    beta = params.get('beta', 3.0)
    kappa = params.get('kappa', -0.5)
    C_j = params.get('C_j', 0.1)
    e2 = params.get('e2_coulomb', 0.4)
    w_so = params.get('w_so', 0.0)

    # The effective mass is calculated as a function of the total density, which gets multiplied 
    B = hbar2_2m + C_tau * rho

    # These operations are used to calculate the mean-field potential (through Skyrme approximation) which is linked to strong-force
    rho_fft = cp.fft.fft2(rho)
    k_squared = KX ** 2 + KY ** 2
    laplacian_rho = cp.real(cp.fft.ifft2(-k_squared * rho_fft))
    U = (alpha * rho) + (beta * (rho ** 2)) + (C_tau * tau) + (kappa * laplacian_rho)

    # In this part, coulomb repulsion is calculated, linked to the electromagnetic force and affecting only protons
    rho_p_fft = cp.fft.fft2(rho_p)
    U_c_fft = cp.zeros_like(rho_p_fft)
    mask = k_squared > 0
    U_c_fft[mask] = (2.0 * cp.pi * e2 * rho_p_fft[mask]) / k_squared[mask]
    U_coulomb = cp.real(cp.fft.ifft2(U_c_fft))

    # These two potentials are directly linked to the currents
    A_x = C_j * j_x
    A_y = C_j * j_y

    # Spin-orbit terms
    grad_rho_x = cp.real(cp.fft.ifft2(1j * KX * rho_fft))
    grad_rho_y = cp.real(cp.fft.ifft2(1j * KY * rho_fft))
    SO_x = -w_so * grad_rho_y
    SO_y = w_so * grad_rho_x

    return (B.astype(cp.float32), U.astype(cp.float32), A_x.astype(cp.float32),
            A_y.astype(cp.float32), U_coulomb.astype(cp.float32),
            SO_x.astype(cp.float32), SO_y.astype(cp.float32))


def build_mean_field(rho, rho_p, tau, j_x, j_y, KX, KY, params, wall_potential=None):
    """
    This function is used to combine the mean-field potentials with the wall potential.
    """
    B, U, A_x, A_y, U_coulomb, SO_x, SO_y = compute_skyrme_fields(rho, rho_p, tau, j_x, j_y, KX, KY, params)
    if wall_potential is not None:
        U = U + wall_potential
    return B, U, A_x, A_y, U_coulomb, SO_x, SO_y

def _advective_term(phi_array, grad_x_phi, grad_y_phi, Ax, Ay, KX, KY):
    """
    Compute the symmetrized (Hermitian) advective term for quantum wave functions on GPU.
    This operation is shared by both the current density term (C_j) and the 2D Spin-Orbit 
    coupling term in the Skyrme energy density functional.
    """
    # First the current term is calculated, then the spin-orbit one
    A_dot_grad = Ax[cp.newaxis, :, :] * grad_x_phi + Ay[cp.newaxis, :, :] * grad_y_phi
    G_x = Ax[cp.newaxis, :, :] * phi_array
    G_y = Ay[cp.newaxis, :, :] * phi_array
    G_x_fft = cp.fft.fft2(G_x, axes=(1, 2)).astype(cp.complex64)
    G_y_fft = cp.fft.fft2(G_y, axes=(1, 2)).astype(cp.complex64)
    div_G = cp.fft.ifft2(1j * KX[cp.newaxis, :, :] * G_x_fft + 1j * KY[cp.newaxis, :, :] * G_y_fft,
                          axes=(1, 2)).astype(cp.complex64)
    return -0.5j * (A_dot_grad + div_G)

def apply_hamiltonian(phi_array, phi_fft_array, B, U_total, A_x, A_y, KX, KY, SO_x=None, SO_y=None, spin_mask=None):
    """
    This function applies the Hamiltonian operator (which includes all the densities, representing the total energy) to the
    wavefunctions, which is crucial to calculate the time evolution of the system.

    Args:
        phi_array (cupy.ndarray): 3D array of complex wave packets for the system
        phi_fft_array (cupy.ndarray): 3D array of Fourier-transformed wave packets
        B (cupy.ndarray): 2D array representing the effective mass
        U_total (cupy.ndarray): 2D array representing the total potential
        A_x (cupy.ndarray): 2D array representing the x-component of the vector potential
        A_y (cupy.ndarray): 2D array representing the y-component of the vector potential
        KX (cupy.ndarray): 2D array of x-components of the wave vector
        KY (cupy.ndarray): 2D array of y-components of the wave vector
        SO_x
        SO_y
        spin_mask
    
    Returns:
        cupy.ndarray: 3D array representing the result of applying the Hamiltonian to the wavefunctions.
    """
    # The gradients of the wavefunctions are calculated in Fourier space using the Fourier transform properties
    grad_x_phi = cp.fft.ifft2(1j * KX[cp.newaxis, :, :] * phi_fft_array, axes=(1, 2))
    grad_y_phi = cp.fft.ifft2(1j * KY[cp.newaxis, :, :] * phi_fft_array, axes=(1, 2))

    #With the following calculations, we obtain the kinetic term, we start by multiplying the gradients with B and then taking their divergence
    F_x = (B[cp.newaxis, :, :] * grad_x_phi).astype(cp.complex64)
    F_y = (B[cp.newaxis, :, :] * grad_y_phi).astype(cp.complex64)
    F_x_fft = cp.fft.fft2(F_x, axes=(1, 2))
    F_y_fft = cp.fft.fft2(F_y, axes=(1, 2))
    div_F = cp.fft.ifft2(1j * KX[cp.newaxis, :, :] * F_x_fft + 1j * KY[cp.newaxis, :, :] * F_y_fft,
                          axes=(1, 2)).astype(cp.complex64)
    kinetic_term = -div_F

    # The potential term is simply the multiplication of the total potential with the wavefunction
    potential_term = U_total * phi_array

    # The last term is the current one
    current_term = _advective_term(phi_array, grad_x_phi, grad_y_phi, A_x, A_y, KX, KY)

    # Finally the three components are added up
    total = kinetic_term + potential_term + current_term

    if SO_x is not None and spin_mask is not None:
        so_term = _advective_term(phi_array, grad_x_phi, grad_y_phi, SO_x, SO_y, KX, KY)
        total = total + spin_mask * so_term

    return total.astype(cp.complex64)


def hamiltonian_action(phi_array, phi_fft_array, is_proton_array, B, U, A_x, A_y, U_coulomb, KX, KY,
                        spin_array=None, SO_x=None, SO_y=None):
    """
    Apply the hamiltonian to the wavefunctions.
    """
    mask_p = is_proton_array.astype(cp.float32)[:, cp.newaxis, cp.newaxis]
    U_total = (U[cp.newaxis, :, :] + U_coulomb[cp.newaxis, :, :] * mask_p).astype(cp.float32)

    spin_mask = None
    if spin_array is not None:
        spin_mask = spin_array.astype(cp.float32)[:, cp.newaxis, cp.newaxis]

    return apply_hamiltonian(phi_array, phi_fft_array, B, U_total, A_x, A_y, KX, KY,
                              SO_x=SO_x, SO_y=SO_y, spin_mask=spin_mask)


def realtime_derivative(phi_array, is_proton_array, spin_array, KX, KY, params, wall_potential, absorb_coeff, hbar=1.0):
    """
    realtime_derivative computes the time derivative of the wavefunctions in real-time evolution, including the Hamiltonian action
    and the absorbing boundary conditions. This function directly employs Schrodinger's equation to calculate the derivative, which
    we will later use for the evolution of our system.

    Args:
        phi_array (cupy.ndarray): 3D array of complex wave packets for the system
        is_proton_array (cupy.ndarray): 1D boolean array indicating which wave packets are protons
        KX (cupy.ndarray): 2D array of x-components of the wave vector
        KY (cupy.ndarray): 2D array of y-components of the wave vector
        params (dict): Dictionary containing the Skyrme parameters
        wall_potential (cupy.ndarray): 2D array representing the wall potential
        absorb_coeff (cupy.ndarray): 2D array representing the absorbing coefficient
        hbar (float, optional): Reduced Planck's constant. Defaults to 1.0.

    Returns:
        cupy.ndarray: 3D array representing the time derivative of the wavefunctions.
    """
    rho, rho_p, tau, j_x, j_y, phi_fft = compute_densities(phi_array, is_proton_array, KX, KY)

    B, U, A_x, A_y, U_coulomb, SO_x, SO_y = build_mean_field(rho, rho_p, tau, j_x, j_y, KX, KY, params, wall_potential)

    h_phi = hamiltonian_action(phi_array, phi_fft, is_proton_array, B, U, A_x, A_y, U_coulomb, KX, KY,
                                spin_array=spin_array, SO_x=SO_x, SO_y=SO_y)

    damping = absorb_coeff[cp.newaxis, :, :] * phi_array
    return -(1j / hbar) * h_phi - damping


def rk4_realtime(phi_array, is_proton_array, spin_array, KX, KY, params, dt, wall_potential, absorb_coeff, hbar=1.0):
    """
    Perform a single Runge-Kutta 4th order (RK4) time step for the real-time evolution of the wavefunctions.

    Args:
        phi_array (cupy.ndarray): 3D array of complex wave packets for the system
        is_proton_array (cupy.ndarray): 1D boolean array indicating which wave packets are protons
        KX (cupy.ndarray): 2D array of x-components of the wave vector
        KY (cupy.ndarray): 2D array of y-components of the wave vector
        params (dict): Dictionary containing the Skyrme parameters
        dt (float): Time step for the RK4 integration
        wall_potential (cupy.ndarray): 2D array representing the wall potential
        absorb_coeff (cupy.ndarray): 2D array representing the absorbing coefficient
        hbar (float, optional): Reduced Planck's constant. Defaults to 1.0.

    Returns:
        cupy.ndarray: 3D array representing the wavefunctions after the RK4 time step.   
    """
    # The RK4 method is a numerical technique for solving ordinary differential equations. 
    k1 = realtime_derivative(phi_array, is_proton_array, spin_array, KX, KY, params, wall_potential, absorb_coeff, hbar)
    k2 = realtime_derivative(phi_array + 0.5 * dt * k1, is_proton_array, spin_array, KX, KY, params, wall_potential, absorb_coeff, hbar)
    k3 = realtime_derivative(phi_array + 0.5 * dt * k2, is_proton_array, spin_array, KX, KY, params, wall_potential, absorb_coeff, hbar)
    k4 = realtime_derivative(phi_array + dt * k3, is_proton_array, spin_array, KX, KY, params, wall_potential, absorb_coeff, hbar)
    phi_next = phi_array + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # If there are any non-valid values, the evolution is brought back to the previous state
    bad = ~cp.isfinite(phi_next)
    if bool(cp.any(bad)):
        phi_next = cp.where(bad, phi_array, phi_next)
    return phi_next.astype(cp.complex64)


def orthogonalize_wf(phi_array, is_proton_array, dx, custom_bodies=None):
    """
    Orthogonalizes wavefunctions using QR decomposition.
    If custom_bodies is provided, it orthogonalizes each nucleus independently 
    to avoid Gram-Schmidt asymmetry between higher and lower index bodies.
    """
    phi_out = phi_array.copy()

    # Block-wise orthogonalization for N-body configurations
    if custom_bodies and len(custom_bodies) > 0:
        for body in custom_bodies:
            start, end = body["start"], body["end"]
            
            # Extract wavefunctions and proton masks for the current nucleus
            phi_sub = phi_array[start:end]
            isp_sub = is_proton_array[start:end]

            # Orthogonalize protons and neutrons separately within this nucleus
            for is_p in [True, False]:
                idx = cp.where(isp_sub == is_p)[0]
                if len(idx) > 1:
                    # Reshape orbitals into matrix columns for QR decomposition
                    sub_matrix = phi_sub[idx].reshape(len(idx), -1).T
                    q, _ = cp.linalg.qr(sub_matrix)
                    
                    # Reconstruct spatial grids and re-normalize
                    for k, orbital_idx in enumerate(idx):
                        phi_reconstructed = q[:, k].reshape(phi_sub.shape[1:])
                        norm = cp.sqrt(cp.sum(cp.abs(phi_reconstructed) ** 2) * (dx ** 2))
                        if norm > 1e-12:
                            phi_sub[orbital_idx] = phi_reconstructed / norm
            
            phi_out[start:end] = phi_sub

    # Standard global orthogonalization for single nucleus / fused systems
    else:
        for is_p in [True, False]:
            idx = cp.where(is_proton_array == is_p)[0]
            if len(idx) > 1:
                sub_matrix = phi_array[idx].reshape(len(idx), -1).T
                q, _ = cp.linalg.qr(sub_matrix)
                
                for k, orbital_idx in enumerate(idx):
                    phi_reconstructed = q[:, k].reshape(phi_array.shape[1:])
                    norm = cp.sqrt(cp.sum(cp.abs(phi_reconstructed) ** 2) * (dx ** 2))
                    if norm > 1e-12:
                        phi_out[orbital_idx] = phi_reconstructed / norm

    return phi_out


def measure_orthogonality_deviation(phi_array, is_proton_array, dx):
    """
    Measures the orthogonality deviation of the wavefunctions by computing the maximum absolute value
    of the off-diagonal elements of the overlap matrix.

    Args:
        phi_array (cupy.ndarray): 3D array of complex wave packets for the system
        is_proton_array (cupy.ndarray): 1D boolean array indicating which wave packets are protons
        dx (float): grid spacing
    
    Returns:
        float: Maximum absolute value of the off-diagonal elements of the overlap matrix.
    """
    # The first step is to reshape the wavefunctions 3D matrix into a 2D one
    N, gy, gx = phi_array.shape
    flat = phi_array.reshape(N, gy * gx)
    max_dev = 0.0

    # Then the nucleons are separated by their type (proton or neutron) and the overlap matrix is calculated for each type
    for target_type in (True, False):
        idx = cp.where(is_proton_array == target_type)[0]
        if len(idx) < 2:
            continue
        # The operation is performed by calculating the scalar product of all the wavefunctions simultaneously
        block = flat[idx]
        overlap = (cp.conj(block) @ block.T) * (dx ** 2)
        off_diag = overlap - cp.diag(cp.diag(overlap))
        dev = float(cp.max(cp.abs(off_diag)))
        # The max deviation is updated if the current deviation is larger than the previous maximum
        max_dev = max(max_dev, dev)
    return max_dev


def compute_total_energy(phi_array, is_proton_array, KX, KY, params, dx):
    """
    This function calculates the total energy of the system, which is a crucial quantity to monitor during the simulation.
    The total energy is composed of several contributions: kinetic energy, bulk energy, surface energy, and Coulomb energy.

    Args:
            phi_array (cupy.ndarray): 3D array of complex wave packets for the system
            is_proton_array (cupy.ndarray): 1D boolean array indicating which wave packets are protons
            KX (cupy.ndarray): 2D array of x-components of the wave vector
            KY (cupy.ndarray): 2D array of y-components of the wave vector
            params (dict): Dictionary containing the Skyrme parameters
            dx (float): grid spacing
        
        Returns:
            float: Total energy of the system.
    """
    # The densities are computed first, which will be used to calculate the different energy contributions
    rho, rho_p, tau, j_x, j_y, phi_fft = compute_densities(phi_array, is_proton_array, KX, KY)

    hbar2_2m = params.get('hbar2_2m', 1.0)
    C_tau = params.get('C_tau', 0.1)
    alpha = params.get('alpha', -2.0)
    beta = params.get('beta', 3.0)
    kappa = params.get('kappa', -0.5)
    e2 = params.get('e2_coulomb', 0.4)

    area = dx ** 2
    B = hbar2_2m + C_tau * rho

    # The energies are calculated by integrating the corresponding energy densities over the spatial domain.
    E_kin = cp.sum(B * tau) * area
    E_bulk = cp.sum(0.5 * alpha * rho ** 2 + (beta / 3.0) * rho ** 3) * area

    k_squared = KX ** 2 + KY ** 2
    rho_fft = cp.fft.fft2(rho)
    grad_rho_x = cp.real(cp.fft.ifft2(1j * KX * rho_fft))
    grad_rho_y = cp.real(cp.fft.ifft2(1j * KY * rho_fft))
    E_surface = cp.sum(-0.5 * kappa * (grad_rho_x ** 2 + grad_rho_y ** 2)) * area

    rho_p_fft = cp.fft.fft2(rho_p)
    U_c_fft = cp.zeros_like(rho_p_fft)
    mask = k_squared > 0
    U_c_fft[mask] = (2.0 * cp.pi * e2 * rho_p_fft[mask]) / k_squared[mask]
    U_coulomb = cp.real(cp.fft.ifft2(U_c_fft))
    E_coulomb = 0.5 * cp.sum(U_coulomb * rho_p) * area

    total = E_kin + E_bulk + E_surface + E_coulomb
    return float(total.get()) if hasattr(total, "get") else float(total)


def compute_rms_radius(rho, X, Y, dx):
    """
    compute_rms_radius calculates the root-mean-square (RMS) radius of the nucleon density distribution,
    which is a measure of the spatial extent of the system.
    """
    # First the total density is calculated, and if it is zero or negative, the function returns 0.0 to avoid division by zero
    total = cp.sum(rho) * dx ** 2
    if float(total) <= 0:
        return 0.0

    # Then the RMS radius is calculated as the square root of the second moment of the density distribution, normalized by the total density.
    r2 = cp.sum((X ** 2 + Y ** 2) * rho) * dx ** 2
    rms = cp.sqrt(r2 / total)
    return float(rms.get()) if hasattr(rms, "get") else float(rms)


def estimate_fissility(n_protons, n_total, params):
    """
    estimate_fissility estimates the fissility parameter of a nucleus, which is a measure of its stability against fission. 
    The fissility parameter is defined as the ratio of the Coulomb energy to the surface energy, 
    and it depends on the number of protons, the total number of nucleons, and the Skyrme parameters.

    Args:
        n_protons (int): Number of protons in the nucleus
        n_total (int): Total number of nucleons in the nucleus
        params (dict): Dictionary containing the Skyrme parameters
    
    Returns:
        float: Estimated fissility parameter of the nucleus.
    """
    # If there are no or one nucleons, the fissility parameter is zero
    if n_total <= 1:
        return 0.0
    e2 = params.get('e2_coulomb', 0.0)
    kappa = params.get('kappa', -0.5)
    surface_cohesion = max(abs(kappa), 1e-3)

    # The coulomb term is proportional to the square of the number of protons, while the surface term is proportional to the square root of the total number of nucleons.
    coulomb_term = e2 * (n_protons ** 2)
    surface_term = surface_cohesion * (n_total ** 0.5)
    return coulomb_term / surface_term


class NuclearSystem:
    """
    NuclearSystem represents a collection of nucleons in a 2D simulation domain, which includes all their properties.
    """
    def __init__(self, phi_array, is_proton_array, spin_array, X, Y, KX, KY, dx, params, config):
        """
        __init__ initializes the NuclearSystem with the given wavefunctions, particle types, spatial grids and simulation parameters.
        """
        # The first operation is to store all the given parameters as attributes of the class.
        self.phi = cp.asarray(phi_array, dtype=cp.complex64)
        self.is_proton = cp.asarray(is_proton_array, dtype=cp.bool_)
        self.spin = cp.asarray(spin_array, dtype=cp.float32)
        self.X, self.Y = X, Y
        self.KX, self.KY = KX, KY
        self.dx = dx
        self.params = params
        self.cfg = config

        self.boundary_mask = build_boundary_mask(X, Y, config['domain_half'], config['absorb_start_frac'])
        self.absorb_potential = build_absorbing_potential(
            X, Y, config['domain_half'], config['absorb_start_frac'], config['absorb_strength']
        )

        # The cooling state is initialized to None, which will be used to store the mixed densities during the cooling process.
        self._rho_mix = None
        self._rho_p_mix = None
        self._prev_cool_energy = None
        self._cool_dt_scale = 1.0

        # Warm-up state machine
        self.warmup_remaining = 0
        self._was_cooling = True

        # Diagnostics history (for the live UI plot)
        self.energy_history = deque(maxlen=400)
        self.radius_history = deque(maxlen=400)
        self.last_energy = None
        self.last_rms_radius = None

    @property
    def n_particles(self):
        return int(self.phi.shape[0])

    def add_particles(self, phi_new, is_proton_new, spin_new):
        """
        Append new nucleons (such as an incoming neutron) to the live system.
        """
        self.phi = cp.concatenate([self.phi, cp.asarray(phi_new, dtype=cp.complex64)], axis=0)
        self.is_proton = cp.concatenate([self.is_proton, cp.asarray(is_proton_new, dtype=cp.bool_)], axis=0)
        self.spin = cp.concatenate([self.spin, cp.asarray(spin_new, dtype=cp.float32)], axis=0)

    def _mixed_densities(self, rho, rho_p):
        """
        This function computes the mixed densities for the cooling process, which is a weighted average 
        of the current densities and the previous mixed densities. This helps to stabilize the cooling process 
        and avoid numerical instabilities.
        """
        mix = self.cfg['density_mix']
        smoothing = self.cfg['field_smoothing']

        # The two densities are smoothed using a Gaussian kernel in Fourier space, which helps to reduce high-frequency noise.
        rho_s = smooth_density(rho, self.KX, self.KY, smoothing)
        rho_p_s = smooth_density(rho_p, self.KX, self.KY, smoothing)

        if self._rho_mix is None:
            self._rho_mix, self._rho_p_mix = rho_s, rho_p_s
        else:
            self._rho_mix = mix * rho_s + (1.0 - mix) * self._rho_mix
            self._rho_p_mix = mix * rho_p_s + (1.0 - mix) * self._rho_p_mix
        return self._rho_mix, self._rho_p_mix

    def _cooling_step(self, dt, hbar=1.0):
        """
        _cooling_step performs the cooling step in the imaginary time evolution, this allows only the ground state to be populated.
        """
        prev_phi = self.phi

        # The densities are computed and smoothed, which will be used to calculate the mean field and the evolution of our system.
        rho, rho_p, tau, j_x, j_y, phi_fft = compute_densities(self.phi, self.is_proton, self.KX, self.KY)
        rho_field, rho_p_field = self._mixed_densities(rho, rho_p)

        # Then the mean field is computed
        B, U, A_x, A_y, U_coulomb, SO_x, SO_y = build_mean_field(
            rho_field, rho_p_field, tau, j_x, j_y, self.KX, self.KY, self.params,
            wall_potential=self.absorb_potential,
        )
        h_phi = hamiltonian_action(self.phi, phi_fft, self.is_proton, B, U, A_x, A_y, U_coulomb, self.KX, self.KY,
                                    spin_array=self.spin, SO_x=SO_x, SO_y=SO_y)

        # Here the imaginary time evolution is performed, which is a steepest descent method to find the ground state of the system.
        k1 = -h_phi / hbar
        phi_next = prev_phi + dt * k1
        bodies = getattr(self, 'custom_bodies', None)
        phi_next = orthogonalize_wf(phi_next, self.is_proton, self.dx, custom_bodies=bodies)

        # At this point, the energy is computed and checked to see if it has increased.
        if self.cfg['adaptive_cooling']:
            energy_now = compute_total_energy(phi_next, self.is_proton, self.KX, self.KY, self.params, self.dx)
            if self._prev_cool_energy is not None and energy_now > self._prev_cool_energy + 1e-4:
                # If the energy has increased, the cooling time step is reduced to avoid overshooting the ground state.
                self._cool_dt_scale = max(self._cool_dt_scale * 0.5, 0.05)
                return
            else:
                self._cool_dt_scale = min(self._cool_dt_scale * 1.02, 1.0)
                self._prev_cool_energy = energy_now
        else:
            energy_now = None

        self.phi = phi_next
        self._record_diagnostics(energy_now)

    def _real_time_step(self, dt, hbar=1.0):
        # This is simply the real-time evolution of the system, which is performed using the RK4 method, defined in previous functions.
        self.phi = rk4_realtime(
            self.phi, self.is_proton, self.spin, self.KX, self.KY, self.params, dt,
            wall_potential=self.absorb_potential, absorb_coeff=self.absorb_potential, hbar=hbar,
        )
        self._record_diagnostics(None)

    def _record_diagnostics(self, energy_now):
        """
        This function records the diagnostics of the system, such as the total energy and the root-mean-square (RMS) radius.
        """
        rho, _, _, _, _, _ = compute_densities(self.phi, self.is_proton, self.KX, self.KY)
        if energy_now is None:
            energy_now = compute_total_energy(self.phi, self.is_proton, self.KX, self.KY, self.params, self.dx)
        rms = compute_rms_radius(rho, self.X, self.Y, self.dx)

        self.last_energy = energy_now
        self.last_rms_radius = rms
        self.energy_history.append(energy_now)
        self.radius_history.append(rms)

    def step(self, cooling, base_dt, hbar=1.0):
        """
        This method simply performs a single time step of the simulation, which can be in cooling mode or not
        """
        if self._was_cooling and not cooling:
            self.warmup_remaining = self.cfg['warmup_steps']
        self._was_cooling = cooling

        if self.phi.shape[0] == 0:
            return

        # Explicit boundary zeroing, applied every step in both modes, on top of the absorbing potential.
        self.phi = self.phi * self.boundary_mask[cp.newaxis, :, :]

        if cooling:
            dt = base_dt * self._cool_dt_scale
            self._cooling_step(dt, hbar=hbar)
        else:
            if self.warmup_remaining > 0:
                dt = self.cfg['warmup_dt']
                self.warmup_remaining -= 1
            else:
                dt = base_dt
            self._real_time_step(dt, hbar=hbar)

        self.phi = self.phi * self.boundary_mask[cp.newaxis, :, :]

    def densities(self):
        return compute_densities(self.phi, self.is_proton, self.KX, self.KY)