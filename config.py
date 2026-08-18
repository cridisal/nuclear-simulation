# Skyrme parameters for the 2D simulation
ALPHA = -8.0
BETA = 2.6
KAPPA = -0.5
E2_COULOMB = 0.003
C_TAU = 0.05
C_J = 0.05

# Spin-orbit parameter
W_SO = 0.15 

SKYRME_PARAMS = dict(hbar2_2m=1.0, C_tau=C_TAU, alpha=ALPHA, beta=BETA,
                      kappa=KAPPA, C_j=C_J, e2_coulomb=E2_COULOMB, w_so=W_SO)

# Grid size and domain half-width (in fm)
GRID_SIZE = 256
DOMAIN_HALF = 80.0  # the box spans [-DOMAIN_HALF, +DOMAIN_HALF] in x and y

# Boundary absorption strength and start fraction (for absorbing halo)
ABSORB_START_FRAC = 0.80   # keep the absorbing halo near the true edge
ABSORB_STRENGTH = 6.0

# Time step sizes for cooling and real-time evolution, and number of sub-steps for real-time evolution
DT_COOLING = 0.05
DT_REALTIME = 0.01
SUB_STEPS = 2

# Warmup phase parameters for transitioning from cooling to real-time evolution
WARMUP_DT = 0.001
WARMUP_STEPS = 100

# Cooling parameters for density mixing, field smoothing, and adaptive cooling
DENSITY_MIX = 0.35
FIELD_SMOOTHING = 0.6
ADAPTIVE_COOLING = True

# Orthogonalization parameters for wavefunctions
ORTHO_AUTO_CORRECT_DEFAULT = False
ORTHO_TOLERANCE_DEFAULT = 0.05  # max allowed |<phi_i|phi_j>| dx^2 before correcting

# Neutron sigma parameter for the simulation
NEUTRON_SIGMA = 1.5

# Zoom parameters for the simulation visualization
ZOOM_MIN, ZOOM_MAX = 1.0, 50.0

# Window layout parameters for the simulation visualization
SCREEN_WIDTH, SCREEN_HEIGHT = 1200, 800
SCENE_W, SCENE_H = 600, 600

SYSTEM_CONFIG = dict(
    domain_half=DOMAIN_HALF, absorb_start_frac=ABSORB_START_FRAC, absorb_strength=ABSORB_STRENGTH,
    density_mix=DENSITY_MIX, field_smoothing=FIELD_SMOOTHING, adaptive_cooling=ADAPTIVE_COOLING,
    warmup_dt=WARMUP_DT, warmup_steps=WARMUP_STEPS,
)