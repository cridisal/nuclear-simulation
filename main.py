import array
import cupy as cp
import moderngl
import pygame
from imgui_bundle import imgui
from imgui_bundle.python_backends.pygame_backend import PygameRenderer
import physics_3 as physics
from config import (
    GRID_SIZE, DOMAIN_HALF, DT_COOLING, DT_REALTIME, SUB_STEPS,
    SCREEN_WIDTH, SCREEN_HEIGHT, SCENE_W, SCENE_H,
    SKYRME_PARAMS, SYSTEM_CONFIG,
)
import ui_and_control as ui_module


#Initializing the PyGame interface and setting up the OpenGL context and ImGui renderer
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF)
clock = pygame.time.Clock()
pygame.display.set_caption("Fission Simulation")
ctx = moderngl.create_context()
imgui.create_context()
impl = PygameRenderer()
io = imgui.get_io()
io.display_size = (SCREEN_WIDTH, SCREEN_HEIGHT)

#Setting up the grid
X, Y, KX, KY, dx = physics.build_grid(GRID_SIZE, DOMAIN_HALF)

#Starting nucleus is defined here, the second one can be directly spawned through the interface
phi_nuc1, is_prot_nuc1, spin_nuc1 = physics.create_nucleus(
    center_x=0, center_y=0, num_protons=10, num_neutrons=15,
    kx_kick=0, ky_kick=0, X=X, Y=Y, dx=dx,
)
NUM_NUC1 = phi_nuc1.shape[0]
phi_array = phi_nuc1.astype(cp.complex64)
is_proton_array = is_prot_nuc1.astype(cp.bool_)
spin_array = spin_nuc1.astype(cp.float32)

system = physics.NuclearSystem(phi_array, is_proton_array, spin_array, X, Y, KX, KY, dx, SKYRME_PARAMS, SYSTEM_CONFIG)

#Defining the user interface and the initial bodies for the simulation, which can be manipulated through the UI
initial_bodies = [
    {"label": "Nucleus 1 (initial)", "start": 0, "end": NUM_NUC1, "kick_speed": 0.0, "kick_angle": 0.0},
]
ui = ui_module.SimulationUI(X, Y, dx, initial_bodies=initial_bodies)


#ModernGL section, we first select the texture coloring (RGB) and we activate the interpolation of colors
texture = ctx.texture((GRID_SIZE, GRID_SIZE), 3, dtype='f1')
texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
texture.repeat_x = False
texture.repeat_y = False

#Here there are two separate scripts that are also used for the image
prog = ctx.program(
    #This scripts defines the center of the rectangle and enables the zooming
    vertex_shader='''
        #version 330
        in vec2 in_vert;
        in vec2 in_texcoord;
        out vec2 v_texcoord;
        uniform float u_zoom;
        uniform vec2 u_center;
        void main() {
            gl_Position = vec4(in_vert, 0.0, 1.0);
            // Zoom/pan only changes how the fixed simulation texture is
            // sampled for display -- the physics grid/domain is untouched.
            v_texcoord = u_center + (in_texcoord - u_center) / u_zoom;
        }
    ''',
    #This one reads the color of every point
    fragment_shader='''
        #version 330
        uniform sampler2D Texture;
        in vec2 v_texcoord;
        out vec4 f_color;
        void main() {
            f_color = texture(Texture, v_texcoord);
        }
    '''
)
prog['u_zoom'].value = 1.0
prog['u_center'].value = (0.5, 0.5)

#Creating the coordinates of our rectangle
x_ndc_min = (50.0 / SCREEN_WIDTH) * 2.0 - 1.0
x_ndc_max = ((50.0 + SCENE_W) / SCREEN_WIDTH) * 2.0 - 1.0
y_ndc_min = 1.0 - ((100.0 + SCENE_H) / SCREEN_HEIGHT) * 2.0
y_ndc_max = 1.0 - (100.0 / SCREEN_HEIGHT) * 2.0

#Adding the four angles to an array and then mapping it
vertices = array.array('f', [
    x_ndc_min, y_ndc_max, 0.0, 0.0,
    x_ndc_min, y_ndc_min, 0.0, 1.0,
    x_ndc_max, y_ndc_max, 1.0, 0.0,
    x_ndc_max, y_ndc_min, 1.0, 1.0,
])
vbo = ctx.buffer(vertices)
vao = ctx.simple_vertex_array(prog, vbo, 'in_vert', 'in_texcoord')


#Main loop
running = True
exposure_state = {'ref': None}

#While the program is running, the main section is launched
while running:
    #No limits to the speed and measures the fps
    clock.tick(0)
    fps_val = clock.get_fps()

    #This loop collects all the events sent from the OS, in order to process them
    for event in pygame.event.get():
        #With this line, the events are processed by the ImGui interface, so that it can handle them
        impl.process_event(event)

        if event.type == pygame.QUIT:
            running = False

        #When keys are pressed (and the user is not writing text), they are dealt by the function
        if not io.want_capture_keyboard and event.type == pygame.KEYDOWN:
            ui.handle_keydown(event.key, system)

    #Physics section, which runs if there is at least one particle
    if system.n_particles > 0:

        #This cycle runs the number of substeps the user have set
        for _ in range(SUB_STEPS):
            system.step(cooling=ui.cooling_mode, base_dt=DT_COOLING if ui.cooling_mode else DT_REALTIME)

        #If the user is not in cooling mode, the orthogonality of the system is updated, otherwise it is reset to zero
        if not ui.cooling_mode:
            ui.update_orthogonality(system)
        else:
            ui.ortho_deviation = 0.0

        #The densities of the system are collected from the specific function
        rho, rho_p, _, j_x, j_y, _ = system.densities()

        #The frame is built with the render function, which takes the densities and other parameters to create a visual representation of the system
        img_array_gpu = ui_module.build_display_frame(
            rho, rho_p, j_x, j_y, exposure_state,
            gamma=ui.gamma, rise_rate=ui.exposure_rise, fall_rate=ui.exposure_fall,
            show_flow=ui.show_flow,
        )
        #The image is sent to the GPU for rendering
        img_bytes = cp.asnumpy(img_array_gpu).tobytes()
        texture.write(img_bytes)

    #The ImGui interface is processed and a new frame is created, where the user can interact with the simulation through the UI
    impl.process_inputs()
    imgui.new_frame()
    ui.draw_panel(system, fps_val)

    #The user interface and the image are actually displayed on the screen
    cx = 0.5 + ui.pan_x / (2.0 * DOMAIN_HALF)
    cy = 0.5 - ui.pan_y / (2.0 * DOMAIN_HALF)
    prog['u_zoom'].value = ui.zoom
    prog['u_center'].value = (cx, cy)

    #The background is set and then the texture is used to render the image, which is displayed on the screen
    ctx.clear(0.08, 0.08, 0.08)
    texture.use()
    vao.render(moderngl.TRIANGLE_STRIP)
    imgui.render()
    impl.render(imgui.get_draw_data())
    pygame.display.flip()

pygame.quit()