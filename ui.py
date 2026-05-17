import imgui
from presets import save_preset, load_preset, list_presets, delete_preset

TOOL_NAMES = ["None", "Push", "Paint", "Erase"]
_CELL_W = 44.0
_SWATCH = 16.0
_preset_name = [""]


def _color_swatch(renderer, idx):
    r, g, b = renderer.palette[idx, :3]
    imgui.color_button(f"##sw{idx}", r, g, b, 1.0, 0, _CELL_W, _SWATCH)


def _matrix_editor(label, mat, lo, hi, renderer, speed=0.01):
    n = mat.shape[0]
    dirty = False
    # column headers
    imgui.dummy(_CELL_W, _SWATCH)  # row-header placeholder
    for j in range(n):
        imgui.same_line(spacing=2)
        _color_swatch(renderer, j)
    # rows
    for i in range(n):
        _color_swatch(renderer, i)  # row header
        for j in range(n):
            imgui.same_line(spacing=2)
            imgui.push_id(f"{label}{i}{j}")
            imgui.set_next_item_width(_CELL_W)
            changed, val = imgui.drag_float("", float(mat[i, j]), speed, lo, hi, "%.2f")
            if changed:
                mat[i, j] = val
                dirty = True
            imgui.pop_id()
    return dirty


def draw_ui(sim, tool, renderer):
    imgui.set_next_window_size(300, 500, imgui.FIRST_USE_EVER)
    imgui.begin("Particle Life Settings")

    changed, val = imgui.slider_int("Particles", sim.num_particles, 100, 50000)
    if changed:
        sim.num_particles = val
        sim.reset_particles()

    changed, val = imgui.slider_int("Colors", sim.num_colors, 1, 20)
    if changed:
        sim.num_colors = val
        sim.reset_particles()
        sim.randomize_rules()

    _, sim.force_factor     = imgui.slider_float("Force Factor",      sim.force_factor,    0.01, 3.0)
    _, sim.friction_factor  = imgui.slider_float("Friction",          sim.friction_factor, 0.0,  1.0)
    _, sim.repel            = imgui.slider_float("Repel",             sim.repel,           0.1,  5.0)
    _, sim.sim_speed        = imgui.slider_float("Sim Speed",         sim.sim_speed,       0.1,  10.0)
    _, sim.substeps         = imgui.slider_int(  "Substeps",          sim.substeps,        1,    16)
    _, sim.max_speed        = imgui.slider_float("Max Speed (0=off)", sim.max_speed,       0.0,  20.0)
    _, sim.max_accel        = imgui.slider_float("Max Accel (0=off)", sim.max_accel,       0.0,  50.0)
    imgui.text("Mode:")
    imgui.same_line()
    if imgui.radio_button("2D", not sim.mode3d):
        if sim.mode3d:
            sim.mode3d = False
            sim.reset_particles()
    imgui.same_line()
    if imgui.radio_button("3D", sim.mode3d):
        if not sim.mode3d:
            sim.enter_3d()

    imgui.text("Boundary:")
    imgui.same_line()
    for mode, name in enumerate(["Bounce", "Wrap", "Infinite"]):
        if imgui.radio_button(name, sim.world_mode == mode):
            sim.world_mode = mode
        if mode < 2:
            imgui.same_line()

    _, renderer.show_grid = imgui.checkbox("Show Grid", renderer.show_grid)
    if sim.world_mode == 1:
        imgui.same_line()
        _, renderer.tile_wrap = imgui.checkbox("Tile Wrap", renderer.tile_wrap)
    if sim.mode3d:
        imgui.same_line()
        _, renderer.fog = imgui.checkbox("Fog", renderer.fog)

    changed, val = imgui.slider_float("World Width",  sim.world_w, 200.0, 5000.0)
    if changed:
        sim.world_w = val
        sim.reset_particles()
    changed, val = imgui.slider_float("World Height", sim.world_h, 200.0, 5000.0)
    if changed:
        sim.world_h = val
        sim.reset_particles()
    if sim.mode3d:
        changed, val = imgui.slider_float("World Depth", sim.world_d, 200.0, 5000.0)
        if changed:
            sim.world_d = val
            sim.reset_particles()

    imgui.spacing()
    if imgui.button("Random Rules"):   sim.randomize_rules()
    imgui.same_line()
    if imgui.button("Symmetric"):      sim.symmetric_rules()
    imgui.same_line()
    if imgui.button("Reset Particles"):sim.reset_particles()
    imgui.same_line()
    if imgui.button("Reset Params"):   sim.reset_params()
    imgui.same_line()
    if imgui.button("Reset View"):     sim.reset_view()

    _, sim.rand_force_range = imgui.drag_float2("Force Range", *sim.rand_force_range, 0.01, -1.0, 1.0, "%.2f")
    _, sim.rand_min_r_range = imgui.drag_float2("Min R Range", *sim.rand_min_r_range, 0.5,  1.0, 300.0, "%.0f")
    _, sim.rand_max_r_range = imgui.drag_float2("Max R Range", *sim.rand_max_r_range, 0.5,  1.0, 300.0, "%.0f")

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
    if tool[0] in (2, 3):
        label = "Paint Colors (empty=random):" if tool[0] == 2 else "Erase Colors (empty=all):"
        imgui.text(label)
        draw_list = imgui.get_window_draw_list()
        for ci in range(sim.num_colors):
            if ci > 0:
                imgui.same_line(spacing=2)
            r, g, b = renderer.palette[ci, :3]
            selected = ci in sim.brush_colors
            imgui.push_id(f"bc{ci}")
            if imgui.color_button("", float(r), float(g), float(b), 1.0, 0, _SWATCH, _SWATCH):
                if selected:
                    sim.brush_colors.discard(ci)
                else:
                    sim.brush_colors.add(ci)
            # draw overlay after button so we know its screen position
            x, y = imgui.get_item_rect_min()
            x2, y2 = imgui.get_item_rect_max()
            if selected:
                draw_list.add_rect(x, y, x2, y2, imgui.get_color_u32_rgba(1,1,1,1), 0, 0, 2.5)
            else:
                draw_list.add_rect_filled(x, y, x2, y2, imgui.get_color_u32_rgba(0,0,0,0.5))
            imgui.pop_id()
        imgui.same_line(spacing=6)
        if imgui.button("Clear"):
            sim.brush_colors.clear()

    imgui.spacing()
    imgui.separator()

    expanded, _ = imgui.collapsing_header("Palette")
    if expanded:
        if imgui.button("Randomize Colors"):
            import numpy as np
            renderer.palette[:sim.num_colors, :3] = np.random.rand(sim.num_colors, 3).astype(np.float32)
            renderer._upload_palette()
        palette_dirty = False
        for i in range(sim.num_colors):
            imgui.push_id(f"pal{i}")
            changed, col = imgui.color_edit3(f"Color {i}", *renderer.palette[i, :3])
            if changed:
                renderer.palette[i, :3] = col
                palette_dirty = True
            imgui.pop_id()
        if palette_dirty:
            renderer._upload_palette()

    imgui.spacing()
    imgui.separator()

    expanded, _ = imgui.collapsing_header("Presets")
    if expanded:
        imgui.set_next_item_width(160)
        _, _preset_name[0] = imgui.input_text("##pname", _preset_name[0], 64)
        imgui.same_line()
        if imgui.button("Save") and _preset_name[0].strip():
            save_preset(sim, _preset_name[0].strip())
        imgui.spacing()
        for name in list_presets():
            del_w = imgui.calc_text_size("x").x + imgui.get_style().frame_padding.x * 2
            btn_w = imgui.get_content_region_available_width() - del_w - imgui.get_style().item_spacing.x
            if imgui.button(f"{name}##load", btn_w):
                load_preset(sim, name)
                sim.reset_particles()
            imgui.same_line()
            imgui.push_id(f"del{name}")
            if imgui.button("x"):
                delete_preset(name)
            imgui.pop_id()

    imgui.spacing()
    imgui.separator()

    rules_dirty = False
    expanded, _ = imgui.collapsing_header("Force Matrix")
    if expanded:
        rules_dirty |= _matrix_editor("force", sim.force_matrix, -1.0, 1.0, renderer)
    expanded, _ = imgui.collapsing_header("Min Radius Matrix")
    if expanded:
        rules_dirty |= _matrix_editor("minr", sim.min_r_matrix, 5.0, 100.0, renderer, speed=0.5)
    expanded, _ = imgui.collapsing_header("Max Radius Matrix")
    if expanded:
        rules_dirty |= _matrix_editor("maxr", sim.max_r_matrix, 20.0, 300.0, renderer, speed=0.5)

    if rules_dirty:
        sim._upload_rules()

    imgui.end()
