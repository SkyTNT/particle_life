import imgui


def _matrix_editor(label, mat, lo, hi, speed=0.01):
    n = mat.shape[0]
    dirty = False
    cell_w = 44.0
    for i in range(n):
        for j in range(n):
            imgui.push_id(f"{label}{i}{j}")
            imgui.set_next_item_width(cell_w)
            changed, val = imgui.drag_float("", float(mat[i, j]), speed, lo, hi, "%.2f")
            if changed:
                mat[i, j] = val
                dirty = True
            imgui.pop_id()
            if j < n - 1:
                imgui.same_line(spacing=2)
    return dirty


TOOL_NAMES = ["None", "Push", "Paint", "Erase"]


def draw_ui(sim, tool):
    imgui.set_next_window_size(300, 500, imgui.FIRST_USE_EVER)
    imgui.begin("Particle Life Settings")

    changed, val = imgui.slider_int("Particles", sim.num_particles, 100, 5000)
    if changed:
        sim.num_particles = val
        sim.reset_particles()

    changed, val = imgui.slider_int("Colors", sim.num_colors, 1, 8)
    if changed:
        sim.num_colors = val
        sim.reset_particles()
        sim.randomize_rules()

    _, sim.force_factor    = imgui.slider_float("Force Factor",   sim.force_factor,    0.01, 3.0)
    _, sim.friction_factor = imgui.slider_float("Friction",       sim.friction_factor, 0.0,  1.0)
    _, sim.repel           = imgui.slider_float("Repel",          sim.repel,           0.1,  5.0)
    _, sim.sim_speed = imgui.slider_float("Sim Speed", sim.sim_speed, 0.1, 10.0)
    _, sim.max_speed = imgui.slider_float("Max Speed (0=off)", sim.max_speed, 0.0, 20.0)
    _, sim.max_accel = imgui.slider_float("Max Accel (0=off)", sim.max_accel, 0.0, 50.0)
    _, sim.wrap            = imgui.checkbox("Wrap edges",         sim.wrap)

    changed, val = imgui.slider_float("World Width",  sim.world_w, 200.0, 5000.0)
    if changed:
        sim.world_w = val
        sim.reset_particles()
    changed, val = imgui.slider_float("World Height", sim.world_h, 200.0, 5000.0)
    if changed:
        sim.world_h = val
        sim.reset_particles()

    imgui.spacing()
    if imgui.button("Random Rules"):
        sim.randomize_rules()
    imgui.same_line()
    if imgui.button("Symmetric"):
        sim.symmetric_rules()
    imgui.same_line()
    if imgui.button("Reset Particles"):
        sim.reset_particles()

    imgui.spacing()
    imgui.separator()
    imgui.text("Tool (left click)")
    for i, name in enumerate(TOOL_NAMES):
        if imgui.radio_button(name, tool[0] == i):
            tool[0] = i
        if i < len(TOOL_NAMES) - 1:
            imgui.same_line()

    _, sim.brush_radius = imgui.slider_float("Brush Radius", sim.brush_radius, 10.0, 500.0)
    if tool[0] == 1:
        _, sim.brush_force = imgui.slider_float("Brush Force", sim.brush_force, 0.01, 1.0)
    if tool[0] == 2:
        changed, val = imgui.slider_int("Paint Color (-1=random)", sim.brush_color, -1, sim.num_colors - 1)
        if changed:
            sim.brush_color = val

    imgui.spacing()
    imgui.separator()

    rules_dirty = False
    expanded, _ = imgui.collapsing_header("Force Matrix")
    if expanded:
        rules_dirty |= _matrix_editor("force", sim.force_matrix, -1.0, 1.0)
    expanded, _ = imgui.collapsing_header("Min Radius Matrix")
    if expanded:
        rules_dirty |= _matrix_editor("minr", sim.min_r_matrix, 5.0, 100.0)
    expanded, _ = imgui.collapsing_header("Max Radius Matrix")
    if expanded:
        rules_dirty |= _matrix_editor("maxr", sim.max_r_matrix, 20.0, 300.0)

    if rules_dirty:
        sim._upload_rules()

    imgui.end()
