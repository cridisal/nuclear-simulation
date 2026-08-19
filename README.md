# Nuclear Simulation ⚛️

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)]()

<img width="800" height="800" alt="FUSION_ALPHA_SPEDUP-ezgif com-video-to-gif-converter(1)" src="https://github.com/user-attachments/assets/544b845e-0535-4429-a13b-c36a7c2def4b" />




## Short summary

Nuclear Simulation is a 2D experimental code which can explore several kinds of nuclear phenomena by employing a mean-field dynamics and a Skyrme potential approximation. This project allows to build nuclei as wave packets, to run the evolution of nuclei and to clearly visualize the real-time evolution for processes such as fusion and fission. Nuclear Simulation is optimized for GPU through Cupy and provides both an interactive UI and a headless script to generate videos.


## Main features

- Building nuclei such as a collection of wave functions in a 2D grid.
- Energy and fields calculated with a Skyrme-like energy density functional: kinetic terms, coulombian ones, spin-orbit interaction, density and current.
- Evolution in imaginary time (cooling) to find low-energy configurations.
- Real-time evolution, for the visualization of multiple phenomena.
- Instruments to launch neutrons, spawn nuclei and apply a kick to static objects.


## Using the Jupyter Notebook (run the code online, through Google Colab)

The easiest way to explore the simulation is through Google Colab, which allows you to run the notebook directly in your browser without installing Python or Jupyter locally.

1. Open https://colab.research.google.com
2. Select **Runtime → Change runtime type**
3. Set **Hardware accelerator** to **GPU** and click **Save**
4. Select **File → Upload notebook**
5. Upload the `.ipynb` file from this repository.
6. Run the three code sections in order.
7. Check the generated output files for the results.


## Physics background

### Theoretical framework
This project is based on the Time-Dependent Hartree-Fock (TDHF) approach combined with a Skyrme Energy Density Functional (EDF) description of nuclear interactions. This combination is clearly defined in the cited paper [1].

The TDHF is a mean-field theory used to describe the time evolution of nuclei and nuclear reactions. Instead of treating all nucleon-nucleon correlations explicitly, the many-body wavefunction is approximated as a single determinant built from single-particle states. The evolution of each nucleon is governed by a one-body Hamiltonian generated (and continuously updated) from the instantaneous density of the system. Below Schrödinger's Equation for our specific case:

$$i\hbar \frac{\partial \psi_\alpha}{\partial t} = \hat{h}[\rho] \psi_\alpha$$

where:

* $\psi_\alpha$ is a single-particle wavefunction,
* $\rho$ is the nuclear density,
* $\hat{h}[\rho]$ is the density-dependent mean-field Hamiltonian.

Because the Hamiltonian depends on the evolving density, all nucleons interact through a dynamically changing collective mean field.

### Skyrme Energy Density Functional

The effective nuclear interaction is represented through the Skyrme Energy Density Functional.

The total energy can be written as

$$E_{tot} = T + E_{Skyrme} + E_{Coulomb} + E_{pair}$$

where:

* **T** is the kinetic energy,
* **E**$_{Skyrme}$ contains the effective nuclear interaction,
* **E**$_{Coulomb}$ describes proton-proton electrostatic repulsion,
* **E**$_{pair}$ represents pairing correlations (when included).

The functional depends on local quantities such as:

* nucleon density $\rho$,
* kinetic density $\tau$,
* current density **j**,
* spin density **s**,
* spin-orbit density **J**.

These quantities are calculated directly from the occupied single-particle wavefunctions and determine the effective potentials acting on the nucleons.

### Mean-Field Approximation

The central approximation of TDHF is that nucleons evolve independently inside a common self-consistent field.

This approach captures several important collective phenomena:

* nuclear vibrations,
* giant resonances,
* fusion reactions,
* nucleon transfer,
* shape evolution,
* low-energy heavy-ion collisions.

However, because explicit two-body collisions are not included, TDHF is most reliable at relatively low excitation energies where mean-field dynamics dominate.

### Numerical Implementation

The simulation is performed on a three-dimensional Cartesian grid without symmetry assumptions.

Key numerical features include:

* 2D spatial discretization,
* self-consistent density evolution,
* Fourier-based derivative evaluation,
* time propagation through the TDHF evolution operator,
* unrestricted nuclear shapes and reaction geometries.

At each timestep the code:

1. Computes local densities from the wavefunctions.
2. Builds the Skyrme mean-field Hamiltonian.
3. Propagates all occupied orbitals.
4. Updates densities and observables.
5. Repeats until the final simulation time is reached.

This iterative procedure ensures full self-consistency during the evolution.


## References

[1] J. A. Maruhn, P.-G. Reinhard, P. D. Stevenson, A. S. Umar,
*The TDHF code Sky3D*,
Computer Physics Communications **185**, 2195-2216 (2014).

## Requirements

- NVIDIA GPU compatible with CUDA 12.x (CuPy: cupy-cuda12x)
- Python 3.10+
- Libraries: check requirements.txt
