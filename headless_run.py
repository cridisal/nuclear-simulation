import time
import cupy as cp
import numpy as np
from PIL import Image 
import physics_3 as physics
from config import GRID_SIZE, DOMAIN_HALF, DT_COOLING, DT_REALTIME, SUB_STEPS
import ui_and_control as ui_module

 
# The parameters are tuned for a light fusion scenario (e.g., two carbon-14 nuclei). Adjust them for different scenarios as needed.
SKYRME_PARAMS = dict(
    hbar2_2m=1.0, C_tau=0.05, alpha=-8.0, beta=2.6, kappa=-0.5,
    C_j=0.05, e2_coulomb=0.003, w_so=0.15,
)
SYSTEM_CONFIG = dict(
    domain_half=DOMAIN_HALF, absorb_start_frac=0.80, absorb_strength=6.0,
    density_mix=0.35, field_smoothing=0.6, adaptive_cooling=True,
    warmup_dt=0.001, warmup_steps=100,
)
 
# The time points for different phases of the simulation: cooling, launch, and expected capture.
T_COOL = 5.0   
T_CAPTURE = 40.0

# This is the main headless simulation script that sets up the nuclear system, can be adjusted for different scenarios.
SCENARIO = [
    {"t": 0.0, "action": "set_cooling", "on": True},
    {"t": T_COOL, "action": "set_cooling", "on": False},
    {"t": 11.0, "action": "launch", "body_index": 0, "speed": 0.2},
    {"t": 11.0, "action": "launch", "body_index": 1, "speed": -0.2},
    {"t": 11.0, "action": "log", "msg": "Lancio a bassa energia: cattura, non scattering."},
    {"t": 20.0, "action": "log"},
    {"t": 30.0, "action": "log"},
    {"t": 40.0, "action": "log"},
    {"t": 50.0, "action": "log"},
    {"t": 60.0, "action": "log"},
    {"t": 65.0, "action": "log"},
]
SIM_DURATION = 70.0

# Name of the output video file and its properties
OUTPUT_PATH = "fusion_light.mp4"

# Other video settings: frames per second, steps per saved frame, gamma correction, exposure rise/fall rates, and whether to show flow vectors.
OUTPUT_FPS = 30
STEPS_PER_SAVED_FRAME = 4
GAMMA = 0.55
EXPOSURE_RISE, EXPOSURE_FALL = 0.06, 0.02
SHOW_FLOW = True
 
# Zooming parameters for the simulation visualization, including output size, zoom limits, frame margin, and smoothing factor.
OUTPUT_SIZE = 720
ZOOM_MIN, ZOOM_MAX = 1.5, 8.0
FRAME_MARGIN = 3.2
ZOOM_SMOOTH = 0.08
 
_zoom_state = {'current': None}
 
 
_zoom_state = {'cx': None, 'cy': None, 'zoom': None}

def adaptive_zoom(rho, X, Y, grid_size, domain_half):
    """
    Calculates zoom and center of the required view.
    """
    total = cp.sum(rho)
    if total <= 0:
        return grid_size / 2.0, grid_size / 2.0, ZOOM_MIN

    # Center of mass in coordinate pixels
    ys, xs = cp.mgrid[0:grid_size, 0:grid_size]
    cx_px_target = float(cp.sum(xs * rho) / total)
    cy_px_target = float(cp.sum(ys * rho) / total)

    # RMS radius to calculate the zoom target
    cx_phys = float(cp.sum(X * rho) / total)
    cy_phys = float(cp.sum(Y * rho) / total)
    r2 = (X - cx_phys) ** 2 + (Y - cy_phys) ** 2
    rms_r = max(float(cp.sqrt(cp.sum(r2 * rho) / total)), 0.5)

    ideal_zoom = float(np.clip(domain_half / (FRAME_MARGIN * rms_r), ZOOM_MIN, ZOOM_MAX))

    # First frame
    if _zoom_state['zoom'] is None:
        _zoom_state['cx'] = cx_px_target
        _zoom_state['cy'] = cy_px_target
        _zoom_state['zoom'] = ideal_zoom
    else:
        # LERP interpolation for following frames
        _zoom_state['cx'] += (cx_px_target - _zoom_state['cx']) * ZOOM_SMOOTH
        _zoom_state['cy'] += (cy_px_target - _zoom_state['cy']) * ZOOM_SMOOTH
        _zoom_state['zoom'] += (ideal_zoom - _zoom_state['zoom']) * ZOOM_SMOOTH

    return _zoom_state['cx'], _zoom_state['cy'], _zoom_state['zoom']
 
 
def crop_and_zoom(frame_rgb_gpu, cx, cy, grid_size, zoom, output_size=OUTPUT_SIZE):
    """
    Crops a square centered on (cx, cy) and it zooms on it.
    """
    # It first identifies the square region to crop, ensuring it remains within the bounds of the grid
    half_px = grid_size / (2.0 * zoom)
    x0, x1 = int(cx - half_px), int(cx + half_px)
    y0, y1 = int(cy - half_px), int(cy + half_px)
 
    if x0 < 0:
        x1 -= x0; x0 = 0
    if y0 < 0:
        y1 -= y0; y0 = 0
    if x1 > grid_size:
        x0 -= (x1 - grid_size); x1 = grid_size
    if y1 > grid_size:
        y0 -= (y1 - grid_size); y1 = grid_size
    x0, y0 = max(0, x0), max(0, y0)

    # Crops the frame and resizes it to the desired output size using high-quality Lanczos resampling.
    frame_np = cp.asnumpy(frame_rgb_gpu)
    cropped = frame_np[y0:y1, x0:x1]
    img = Image.fromarray(cropped, mode='RGB').resize((output_size, output_size), Image.LANCZOS)
    return np.array(img)
 
 
def execute_event(ui, system, event):
    """
    Actually executes the event on the simulation, modifying the UI and system state as needed.
    """
    action = event["action"]
    if action == "set_cooling":
        ui.cooling_mode = bool(event.get("on", True))
    elif action == "fire_neutron":
        ui.neutron_speed = event.get("speed", ui.neutron_speed)
        ui.neutron_pos = event.get("y", ui.neutron_pos)
        ui.spawn_neutron(system)
    elif action == "spawn_nucleus":
        ui.spawn_x = event.get("x", 0.0)
        ui.spawn_y = event.get("y", 0.0)
        ui.spawn_num_protons = event.get("protons", 2)
        ui.spawn_num_neutrons = event.get("neutrons", 2)
        ui.spawn_custom_particle(system)
    elif action == "launch":
        ui.launch_body(system, event["body_index"], event.get("speed", 0.0), event.get("angle", 0.0))
    elif action == "set_param":
        system.params[event["name"]] = event["value"]
    elif action == "log":
        print(f"[t={event['t']:.2f}] {event.get('msg', '')}")
    else:
        raise ValueError(f"Unrecognized action: {action!r}")
 
X, Y, KX, KY, dx = physics.build_grid(GRID_SIZE, DOMAIN_HALF)


# ---------------------------------------------------------------------------------------------------------------------

# In this field you can choose how many nuclei the scenario will have
phi_nuc1, is_prot_nuc1, spin_nuc1 = physics.create_nucleus(
    center_x=-5.0, center_y=0.0, num_protons=2, num_neutrons=2,
    kx_kick=0, ky_kick=0, X=X, Y=Y, dx=dx,
)
phi_nuc2, is_prot_nuc2, spin_nuc2 = physics.create_nucleus(
    center_x=5.0, center_y=0.0, num_protons=2, num_neutrons=2,
    kx_kick=0, ky_kick=0, X=X, Y=Y, dx=dx,
)

phi_combined = cp.concatenate([phi_nuc1, phi_nuc2], axis=0)
is_prot_combined = cp.concatenate([is_prot_nuc1, is_prot_nuc2], axis=0)
spin_combined = cp.concatenate([spin_nuc1, spin_nuc2], axis=0)

system = physics.NuclearSystem(
    phi_combined.astype(cp.complex64), is_prot_combined.astype(cp.bool_), spin_combined.astype(cp.float32),
    X, Y, KX, KY, dx, dict(SKYRME_PARAMS), SYSTEM_CONFIG,
)
 
initial_bodies = [
    {"label": "NucA", "start": 0, "end": phi_nuc1.shape[0], "kick_speed": 0.0, "kick_angle": 0.0},
    {"label": "NucB", "start": phi_nuc1.shape[0], "end": phi_combined.shape[0], "kick_speed": 0.0, "kick_angle": 0.0},
]
ui = ui_module.SimulationUI(X, Y, dx, initial_bodies=initial_bodies)
system.custom_bodies = ui.custom_bodies

# ---------------------------------------------------------------------------------------------------------------------

events = sorted(SCENARIO, key=lambda e: e["t"])
event_ptr = 0
 
try:
    import imageio.v2 as imageio
except ImportError:
    raise SystemExit(
        "You need imageio (+ imageio-ffmpeg):\n    pip install imageio imageio-ffmpeg"
    )
 
writer = imageio.get_writer(OUTPUT_PATH, fps=OUTPUT_FPS)

# In this part, the simulation loop is executed, where the system evolves over time, events are processed, and frames are rendered.
 
exposure_state = {'ref': None}
sim_time = 0.0
step_count = 0
wall_start = time.time()
 
print(f"Running headless simulation: {SIM_DURATION} units' of simulated time -> {OUTPUT_PATH}")

# Here the simulation is actually run

while sim_time < SIM_DURATION:
    while event_ptr < len(events) and events[event_ptr]["t"] <= sim_time:
        execute_event(ui, system, events[event_ptr])
        event_ptr += 1

    # First the timestep is determined based on whether the system is in cooling mode or real-time mode.
    base_dt = DT_COOLING if ui.cooling_mode else DT_REALTIME
    # Then the steps are actually computed
    for _ in range(SUB_STEPS):
        system.step(cooling=ui.cooling_mode, base_dt=base_dt)
    sim_time += SUB_STEPS * base_dt

    # A check of orthogonality is also performed, which is important for the stability of the simulation.
    if not ui.cooling_mode:
        ui.update_orthogonality(system)
    else:
        ui.ortho_deviation = 0.0
 
    step_count += 1

    # The frame is rendered and saved at specified intervals, with adaptive zooming based on the density distribution of the system.
    if step_count % STEPS_PER_SAVED_FRAME == 0 and system.n_particles > 0:
        rho, rho_p, _, j_x, j_y, _ = system.densities()
        frame = ui_module.build_display_frame(
            rho, rho_p, j_x, j_y, exposure_state,
            gamma=GAMMA, rise_rate=EXPOSURE_RISE, fall_rate=EXPOSURE_FALL, show_flow=SHOW_FLOW,
        )

        # Calcola centro e zoom smussati in un'unica chiamata
        cx_px = GRID_SIZE / 2.0
        cy_px = GRID_SIZE / 2.0
        zoom = 6.0

        # Genera il frame finale
        final_frame = crop_and_zoom(frame, cx_px, cy_px, GRID_SIZE, zoom, OUTPUT_SIZE)
        writer.append_data(final_frame)

# Finally, the video writer is closed and a summary of the simulation is printed.
writer.close()
wall_elapsed = time.time() - wall_start
print(f"Done: {step_count} physical steps, {sim_time:.1f} units of simulated time, "
      f"video saved to {OUTPUT_PATH} ({wall_elapsed:.1f}s real computation time).")
