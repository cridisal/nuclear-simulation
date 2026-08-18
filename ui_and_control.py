import math
from platform import system
from turtle import speed
import cupy as cp
import numpy as np
import pygame
from imgui_bundle import imgui
import physics_3 as physics
from config import (
    DOMAIN_HALF, ZOOM_MIN, ZOOM_MAX,
    ORTHO_AUTO_CORRECT_DEFAULT, ORTHO_TOLERANCE_DEFAULT,
    NEUTRON_SIGMA,
)

def build_display_frame(rho, rho_p, j_x, j_y, exposure_state, gamma, rise_rate, fall_rate, show_flow):
    """
    build_display_frame constructs a visual representation of the simulation state based on the provided density and flow data. 
    It applies exposure control, color mapping, and optional flow visualization.
    """
    eps = 1e-8
    frame_max = float(cp.max(rho))

    if exposure_state['ref'] is None:
        exposure_state['ref'] = max(frame_max, eps)
    else:
        rate = rise_rate if frame_max > exposure_state['ref'] else fall_rate
        exposure_state['ref'] += (frame_max - exposure_state['ref']) * rate
    ref = max(exposure_state['ref'], eps)

    intensity = (cp.clip(rho / ref, 0.0, 1.0).astype(cp.float32)) ** cp.float32(gamma)

    proton_frac = cp.where(rho > eps, rho_p / cp.maximum(rho, eps), cp.float32(0.5)).astype(cp.float32)

    proton_color = cp.array([255.0, 170.0, 60.0], dtype=cp.float32)   # warm gold
    neutron_color = cp.array([60.0, 140.0, 255.0], dtype=cp.float32)  # cool blue
    base_color = proton_frac[..., cp.newaxis] * proton_color + (1.0 - proton_frac[..., cp.newaxis]) * neutron_color

    rgb = base_color * intensity[..., cp.newaxis]

    if show_flow:
        j_mag = cp.sqrt(j_x.astype(cp.float32) ** 2 + j_y.astype(cp.float32) ** 2)
        j_norm = cp.clip(j_mag / (cp.max(j_mag) + eps), 0.0, 1.0)
        flow_tint = cp.stack([cp.zeros_like(j_norm), j_norm * 200.0, j_norm * 40.0], axis=-1)
        rgb = rgb + flow_tint * intensity[..., cp.newaxis]

    return cp.clip(rgb, 0.0, 255.0).astype(cp.uint8)



class SimulationUI:
    """
    Owns every piece of UI-facing state that used to be a bare module-level
    global in main.py (cooling_mode, zoom/pan, spawner fields, ...), plus the
    body registry and the functions that act on a NuclearSystem.
    """

    def __init__(self, X, Y, dx, initial_bodies=None):
        self.X = X
        self.Y = Y
        self.dx = dx

        self.cooling_mode = False

        # Neutron gun (now exposed in the UI, see draw_panel)
        self.neutron_speed = 5.0
        self.neutron_pos = 0.0

        # Custom particle spawner
        self.spawn_x = 0.0
        self.spawn_y = 0.0
        self.spawn_num_protons = 2
        self.spawn_num_neutrons = 2

        # Launch controls
        self.selected_body_index = 0
        self.launch_speed = 0.0
        self.launch_angle = 0.0

        # Orthogonality check (see physics_2.measure_orthogonality_deviation)
        self.ortho_auto_correct = ORTHO_AUTO_CORRECT_DEFAULT
        self.ortho_tolerance = ORTHO_TOLERANCE_DEFAULT
        self.ortho_deviation = 0.0

        # View / zoom (display-only, never touches the physics grid)
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0

        self.gamma = 0.55
        self.exposure_rise = 0.06
        self.exposure_fall = 0.02
        self.show_flow = False

        # Single source of truth for the "Spawn & Launch" panel.
        self.custom_bodies = list(initial_bodies) if initial_bodies else []

    def spawn_neutron(self, system):
        """
        Spawns a neutron on the right edge at the chosen height, aimed left.
        """
        x0 = 25
        y0 = self.neutron_pos
        kx = -self.neutron_speed

        start_idx = system.n_particles
        phi_n, isp_n, spin_n = physics.create_free_particle(
            center_x=x0, center_y=y0, kx_kick=kx, ky_kick=0,
            X=self.X, Y=self.Y, dx=self.dx, sigma=NEUTRON_SIGMA, is_proton=False,
        )
        system.add_particles(phi_n, isp_n, spin_n)
        end_idx = system.n_particles

        n_previous = sum(1 for b in self.custom_bodies if "Neutron" in b["label"])
        label = f"Neutron {n_previous + 1} @ ({x0},{y0:.1f})"
        self.custom_bodies.append({
            "label": label, "start": start_idx, "end": end_idx,
            "kick_speed": self.neutron_speed, "kick_angle": 180.0,
        })

        self.cooling_mode = False
        print(f"Neutron fired: pos=({x0},{y0:.1f}) speed={self.neutron_speed:.1f}")

    def spawn_custom_particle(self, system):
        """
        Adds a new, arbitrary nucleus (any Z, any N) at rest at a chosen position.
        """
        if self.spawn_num_protons + self.spawn_num_neutrons <= 0:
            print("Nothing to spawn: pick at least 1 proton or neutron.")
            return

        start_idx = system.n_particles
        phi_new, isp_new, spin_new = physics.create_nucleus(
            center_x=self.spawn_x, center_y=self.spawn_y,
            num_protons=self.spawn_num_protons, num_neutrons=self.spawn_num_neutrons,
            kx_kick=0, ky_kick=0, X=self.X, Y=self.Y, dx=self.dx,
        )
        system.add_particles(phi_new, isp_new, spin_new)
        end_idx = system.n_particles

        label = (f"Body {len(self.custom_bodies) + 1} "
                 f"(Z={self.spawn_num_protons}, N={self.spawn_num_neutrons} "
                 f"@ {self.spawn_x:.1f},{self.spawn_y:.1f})")
        self.custom_bodies.append({
            "label": label, "start": start_idx, "end": end_idx,
            "kick_speed": 0.0, "kick_angle": 0.0,
        })
        print(f"Spawned {label} -> phi indices [{start_idx}:{end_idx}]")

    def launch_body(self, system, body_index, speed, angle_deg):
        """
        Applies a momentum phase kick to a selected nucleus using a Soft-Voronoi 
        spatial partition. Works for any arbitrary N-body configuration in 2D space.
        """
        if not self.custom_bodies or not (0 <= body_index < len(self.custom_bodies)):
            print("No valid body selected for launch.")
            return

        n_bodies = len(self.custom_bodies)
        theta = math.radians(angle_deg)
        kx = speed * math.cos(theta)
        ky = speed * math.sin(theta)

        # Compute the center of mass for EVERY registered body in the simulation domain
        centers = []
        distances_sq = []
        
        for b in self.custom_bodies:
            phi_b = system.phi[b["start"]:b["end"]]
            rho_b = cp.sum(cp.abs(phi_b) ** 2, axis=0)
            mass_b = cp.sum(rho_b)

            if mass_b > 0:
                cx = float(cp.sum(self.X * rho_b) / mass_b)
                cy = float(cp.sum(self.Y * rho_b) / mass_b)
            else:
                cx, cy = 0.0, 0.0

            centers.append((cx, cy))
            # Quadratic distance grid from this body's center of mass
            distances_sq.append((self.X - cx) ** 2 + (self.Y - cy) ** 2)

        target_cx, target_cy = centers[body_index]

        # Compute Soft-Voronoi spatial partition mask for the target body
        # sigma_sq controls the smoothness of the spatial boundary transition (e.g. 16.0 = 4 spatial units)
        sigma_sq = 16.0
        if n_bodies > 1:
            exp_terms = [cp.exp(-d / sigma_sq) for d in distances_sq]
            sum_exp = cp.sum(cp.stack(exp_terms), axis=0) + 1e-12
            # W_k is a 2D weight grid in [0, 1]: ~1 near target body, ~0 near all other N-1 bodies
            W_k = exp_terms[body_index] / sum_exp
        else:
            W_k = cp.ones_like(self.X)

        # Compute momentum phase factor relative to target body center of mass
        rel_X = self.X - target_cx
        rel_Y = self.Y - target_cy
        kick_phase = cp.exp(1j * (kx * rel_X + ky * rel_Y))

        # Apply phase kick weighted by spatial partition mask
        body = self.custom_bodies[body_index]
        start, end = body["start"], body["end"]

        for i in range(start, end):
            # Where W_k ~ 1 apply full phase kick; where W_k ~ 0 leave wavefunction untouched
            system.phi[i] = system.phi[i] * ((1.0 - W_k) + W_k * kick_phase)

        body["kick_speed"], body["kick_angle"] = speed, angle_deg
        self.cooling_mode = False
        print(f"Launched {body['label']} (Center: x={target_cx:.1f}, y={target_cy:.1f}) | speed={speed:.2f}, angle={angle_deg:.1f} deg")

    def update_orthogonality(self, system):
        """
        Call once per frame, real-time mode only, after system.step().
        Measures the maximum same-species overlap and optionally corrects it if it exceeds the tolerance.
        """
        # Updates the ortho_deviation and applies auto-correction if enabled and the deviation exceeds the tolerance.
        if system.n_particles == 0:
            self.ortho_deviation = 0.0
            return
        self.ortho_deviation = physics.measure_orthogonality_deviation(
            system.phi, system.is_proton, self.dx
        )
        if self.ortho_auto_correct and self.ortho_deviation > self.ortho_tolerance:
            system.phi = physics.orthogonalize_wf(system.phi, system.is_proton, self.dx)

    def handle_keydown(self, key, system):
        """ 
        Handles keydown events for toggling cooling mode and firing neutrons. 
        """

        if key == pygame.K_SPACE:
            self.cooling_mode = not self.cooling_mode
            print(f"Cooling mode toggled. Active: {self.cooling_mode}")
        elif key == pygame.K_n:
            self.spawn_neutron(system)

    def draw_panel(self, system, fps_val):
        """
        Draws the panel with all the controls and diagnostics using ImGui.
        """
        imgui.set_next_window_pos((700, 100), imgui.Cond_.first_use_ever)
        imgui.set_next_window_size((480, 900), imgui.Cond_.first_use_ever)
        imgui.begin("Control Panel")

        imgui.text(f"FPS: {fps_val:.1f}")
        n_total = system.n_particles
        n_protons = int(system.is_proton.sum()) if n_total else 0
        imgui.text(f"Total nucleons: {n_total}  (Z={n_protons}, N={n_total - n_protons})")
        imgui.separator()

        imgui.text("Simulation state:")
        if self.cooling_mode:
            imgui.text_colored((0.2, 0.8, 1.0, 1.0), "COOLING (imaginary time)")
        elif system.warmup_remaining > 0:
            imgui.text_colored((1.0, 0.8, 0.2, 1.0), f"WARM-UP ({system.warmup_remaining} steps left)")
        else:
            imgui.text_colored((1.0, 0.4, 0.4, 1.0), "REAL TIME (dynamic evolution)")

        if imgui.button("Toggle Mode (Space)", (200, 0)):
            self.cooling_mode = not self.cooling_mode

        imgui.spacing()
        imgui.separator()

        # --- Diagnostics ---
        imgui.text("Diagnostics:")
        if system.last_energy is not None:
            imgui.text(f"Total energy (monitor): {system.last_energy:.3f}")
        if system.last_rms_radius is not None:
            imgui.text(f"RMS radius: {system.last_rms_radius:.3f}")
            imgui.text_wrapped(
                "(Growing radius -> nucleus is expanding, try a more negative "
                "ALPHA. Shrinking radius -> collapsing, try a larger BETA.)"
            )
        if len(system.energy_history) > 1:
            history = np.asarray(system.energy_history, dtype=np.float32)
            imgui.plot_lines("Energy history", history, overlay_text="", scale_min=float(history.min()),
                              scale_max=float(history.max()), graph_size=imgui.ImVec2(400, 80))

        # --- Fissility indicator (NEW) ---
        imgui.spacing()
        fissility = physics.estimate_fissility(n_protons, n_total, system.params)
        imgui.text(f"Fissility indicator (rough, heuristic): {fissility:.3f}")
        if fissility < 0.3:
            imgui.text_colored((0.6, 0.6, 0.6, 1.0), "-> firmly bound, don't expect fission")
        elif fissility < 0.8:
            imgui.text_colored((1.0, 0.8, 0.2, 1.0), "-> borderline, induced fission may work")
        else:
            imgui.text_colored((1.0, 0.4, 0.4, 1.0), "-> likely fissile, try the neutron gun")
        imgui.text_wrapped(
            "Rises with e2_coulomb and with Z^2; falls with |kappa| and with A. "
            "Z=2 nuclei can basically never get this above ~0 -- spawn a "
            "bigger body to test fission."
        )

        imgui.spacing()
        imgui.separator()

        # --- Skyrme parameters: precise numeric input ---
        if imgui.collapsing_header("Skyrme parameters (precise values)"):
            _, system.params['alpha'] = imgui.input_float("Alpha (attraction)", system.params['alpha'], 0.1, 1.0, "%.3f")
            system.params['alpha'] = max(-15.0, min(5.0, system.params['alpha']))

            _, system.params['beta'] = imgui.input_float("Beta (repulsion)", system.params['beta'], 0.1, 1.0, "%.3f")
            system.params['beta'] = max(0.0, min(15.0, system.params['beta']))

            _, system.params['kappa'] = imgui.input_float("Kappa (surface)", system.params['kappa'], 0.05, 0.5, "%.3f")
            system.params['kappa'] = max(-5.0, min(5.0, system.params['kappa']))

            imgui.spacing()
            imgui.text_wrapped(
                "Coulomb + kinetic-density coefficients (previously not exposed "
                "here at all). e2_coulomb is the ONE term that drives fission -- "
                "alpha/beta/kappa alone can never produce it."
            )
            _, system.params['e2_coulomb'] = imgui.input_float(
                "e2_coulomb (Coulomb strength)", system.params['e2_coulomb'], 0.01, 0.1, "%.3f")
            system.params['e2_coulomb'] = max(0.0, min(5.0, system.params['e2_coulomb']))

            _, system.params['C_tau'] = imgui.input_float(
                "C_tau (kinetic density coupling)", system.params['C_tau'], 0.01, 0.1, "%.3f")
            system.params['C_tau'] = max(0.0, min(2.0, system.params['C_tau']))

            _, system.params['C_j'] = imgui.input_float(
                "C_j (current coupling)", system.params['C_j'], 0.01, 0.1, "%.3f")
            system.params['C_j'] = max(-2.0, min(2.0, system.params['C_j']))

        imgui.spacing()
        imgui.separator()

        # --- Orthogonalization check ---
        if imgui.collapsing_header("Orthogonalization check"):
            imgui.text_wrapped(
                "Real-time evolution should keep same-species nucleons orthogonal "
                "on its own; forcing an exact reset every frame can act like an "
                "artificial repulsion right when nucleons should be merging/fusing "
                "-- or, during a fission neck, splitting cleanly. If a nucleus "
                "seems to 'snap back together' right as it starts to neck, try "
                "unchecking Auto-correct and compare."
            )
            imgui.text(f"Max same-species overlap: {self.ortho_deviation:.5f}")
            _, self.ortho_auto_correct = imgui.checkbox("Auto-correct drift", self.ortho_auto_correct)
            _, self.ortho_tolerance = imgui.input_float("Tolerance", self.ortho_tolerance, 0.01, 0.05, "%.3f")
            self.ortho_tolerance = max(0.0, min(1.0, self.ortho_tolerance))
            if imgui.button("Force reorthogonalize now", (220, 0)):
                system.phi = physics.orthogonalize_wf(system.phi, system.is_proton, self.dx)
                self.ortho_deviation = 0.0

        imgui.spacing()
        imgui.separator()

        # --- Display / visualization (NEW) ---
        if imgui.collapsing_header("Display / Visualization", flags=imgui.TreeNodeFlags_.default_open):
            imgui.text_wrapped(
                "Brightness = total density, auto-exposed against a ceiling that "
                "only creeps toward a new max instead of snapping to it -- this "
                "is what stops a freshly spawned/fired particle from crushing the "
                "rest of the nucleus to black. Color = composition: proton-rich "
                "regions render warm gold, neutron-rich render cool blue."
            )
            _, self.gamma = imgui.input_float("Gamma (shadow lift)", self.gamma, 0.05, 0.1, "%.2f")
            self.gamma = max(0.15, min(1.5, self.gamma))
            _, self.exposure_rise = imgui.input_float(
                "Exposure rise rate", self.exposure_rise, 0.01, 0.05, "%.3f")
            self.exposure_rise = max(0.005, min(1.0, self.exposure_rise))
            _, self.exposure_fall = imgui.input_float(
                "Exposure fall rate", self.exposure_fall, 0.01, 0.05, "%.3f")
            self.exposure_fall = max(0.005, min(1.0, self.exposure_fall))
            _, self.show_flow = imgui.checkbox("Show probability-current flow (green tint)", self.show_flow)
            imgui.text_wrapped(
                "Flow tint uses j_x/j_y (already computed each step) to highlight "
                "where mass is actively moving -- handy for making a forming "
                "fission neck or a fusion merger visually obvious on camera."
            )

        imgui.spacing()
        imgui.separator()

        # --- View / zoom ---
        if imgui.collapsing_header("View / Zoom", flags=imgui.TreeNodeFlags_.default_open):
            imgui.text_wrapped("Zoom only changes how the fixed simulation grid is displayed -- DOMAIN_HALF is unchanged.")
            _, self.zoom = imgui.input_float("Zoom", self.zoom, 0.5, 2.0, "%.2f")
            self.zoom = max(ZOOM_MIN, min(ZOOM_MAX, self.zoom))
            _, self.pan_x = imgui.input_float("Pan X", self.pan_x, 1.0, 5.0, "%.1f")
            self.pan_x = max(-DOMAIN_HALF, min(DOMAIN_HALF, self.pan_x))
            _, self.pan_y = imgui.input_float("Pan Y", self.pan_y, 1.0, 5.0, "%.1f")
            self.pan_y = max(-DOMAIN_HALF, min(DOMAIN_HALF, self.pan_y))
            if imgui.button("Reset view", (150, 0)):
                self.zoom, self.pan_x, self.pan_y = 1.0, 0.0, 0.0

        imgui.spacing()
        imgui.separator()

        # --- Neutron gun (NEW: previously fixed values, only reachable via code) ---
        if imgui.collapsing_header("Neutron gun (or press N)"):
            _, self.neutron_speed = imgui.input_float("Neutron speed", self.neutron_speed, 0.5, 1.0, "%.2f")
            self.neutron_speed = max(0.0, min(30.0, self.neutron_speed))
            _, self.neutron_pos = imgui.input_float("Neutron height (Y)", self.neutron_pos, 0.5, 1.0, "%.1f")
            self.neutron_pos = max(-DOMAIN_HALF, min(DOMAIN_HALF, self.neutron_pos))
            imgui.text_wrapped("More speed = more kinetic/excitation energy delivered on capture.")
            if imgui.button("Fire neutron", (200, 30)):
                self.spawn_neutron(system)

        imgui.spacing()
        imgui.separator()

        # --- Particles: Spawn & Launch ---
        if imgui.collapsing_header("Particles: Spawn & Launch", flags=imgui.TreeNodeFlags_.default_open):
            imgui.text("Spawn a new nucleus at rest")
            margin = DOMAIN_HALF * 0.8

            _, self.spawn_x = imgui.input_float("Spawn X", self.spawn_x, 0.5, 1.0, "%.2f")
            self.spawn_x = max(-margin, min(margin, self.spawn_x))
            _, self.spawn_y = imgui.input_float("Spawn Y", self.spawn_y, 0.5, 1.0, "%.2f")
            self.spawn_y = max(-margin, min(margin, self.spawn_y))
            _, self.spawn_num_protons = imgui.input_int("Protons", self.spawn_num_protons, 1, 5)
            self.spawn_num_protons = max(0, min(30, self.spawn_num_protons))
            _, self.spawn_num_neutrons = imgui.input_int("Neutrons", self.spawn_num_neutrons, 1, 5)
            self.spawn_num_neutrons = max(0, min(30, self.spawn_num_neutrons))

            if imgui.button("Spawn Particle", (200, 30)):
                self.spawn_custom_particle(system)

            imgui.spacing()
            imgui.separator()
            imgui.text("Launch (select a body, set its kick, launch it)")

            labels = [b["label"] for b in self.custom_bodies]
            self.selected_body_index = max(0, min(self.selected_body_index, len(labels) - 1))
            combo_changed, self.selected_body_index = imgui.combo("Select body", self.selected_body_index, labels)
            if combo_changed:
                self.launch_speed = self.custom_bodies[self.selected_body_index]["kick_speed"]
                self.launch_angle = self.custom_bodies[self.selected_body_index]["kick_angle"]

            _, self.launch_speed = imgui.input_float("Kick speed", self.launch_speed, 0.1, 1.0, "%.2f")
            self.launch_speed = max(-20.0, min(20.0, self.launch_speed))
            _, self.launch_angle = imgui.input_float("Kick angle (deg)", self.launch_angle, 1.0, 10.0, "%.1f")
            self.launch_angle = max(-180.0, min(180.0, self.launch_angle))

            if imgui.button("Launch!", (200, 30)):
                self.launch_body(system, self.selected_body_index, self.launch_speed, self.launch_angle)

        imgui.end()