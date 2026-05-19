import imgui
from presets import save_preset, load_preset, list_presets, delete_preset

TOOL_NAMES = ["None", "Push", "Paint", "Erase", "Attract", "Repel"]
_CELL_W = 44.0
_SWATCH = 16.0
_preset_name = [""]
_all_force = [0.0]
_all_min_r = [40.0]
_all_max_r = [120.0]


def _color_swatch(renderer, idx):
    r, g, b = renderer.palette[idx, :3]
    imgui.color_button(f"##sw{idx}", r, g, b, 1.0, 0, _CELL_W, _SWATCH)


def _matrix_editor(label, mat, lo, hi, renderer, speed=0.01):
    n = mat.shape[0]
    dirty = False
    imgui.dummy(_CELL_W, _SWATCH)  # row-header placeholder
    for j in range(n):
        imgui.same_line(spacing=2)
        _color_swatch(renderer, j)
    for i in range(n):
        _color_swatch(renderer, i)
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


def _section(label, opened=False):
    flags = imgui.TREE_NODE_DEFAULT_OPEN if opened else 0
    expanded, _ = imgui.collapsing_header(label, flags=flags)
    return expanded


def draw_ui(sim, tool, renderer):
    imgui.set_next_window_size(340, 720, imgui.FIRST_USE_EVER)
    imgui.begin("Particle Life Settings")

    # --- Presets --------------------------------------------------------------
    if _section("Presets"):
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

    # --- Matrix Settings ------------------------------------------------------
    rules_dirty = False
    if _section("Matrix Settings"):
        if imgui.button("Random Rules"):  sim.randomize_rules()
        imgui.same_line()
        if imgui.button("Symmetric"):     sim.symmetric_rules()

        if imgui.tree_node("Force Matrix"):
            rules_dirty |= _matrix_editor("force", sim.force_matrix, -1.0, 1.0, renderer)
            imgui.set_next_item_width(-1)
            changed, _all_force[0] = imgui.slider_float("##af", _all_force[0], -1.0, 1.0, "Set all: %.2f")
            if changed:
                sim.force_matrix[:] = _all_force[0]
                rules_dirty = True
            imgui.tree_pop()
        if imgui.tree_node("Min Radius Matrix"):
            rules_dirty |= _matrix_editor("minr", sim.min_r_matrix, 5.0, 100.0, renderer, speed=0.5)
            imgui.set_next_item_width(-1)
            changed, _all_min_r[0] = imgui.slider_float("##amr", _all_min_r[0], 1.0, 200.0, "Set all: %.0f")
            if changed:
                sim.min_r_matrix[:] = _all_min_r[0]
                rules_dirty = True
            imgui.tree_pop()
        if imgui.tree_node("Max Radius Matrix"):
            rules_dirty |= _matrix_editor("maxr", sim.max_r_matrix, 20.0, 300.0, renderer, speed=0.5)
            imgui.set_next_item_width(-1)
            changed, _all_max_r[0] = imgui.slider_float("##axr", _all_max_r[0], 5.0, 300.0, "Set all: %.0f")
            if changed:
                sim.max_r_matrix[:] = _all_max_r[0]
                rules_dirty = True
            imgui.tree_pop()

    # --- Randomizer Settings --------------------------------------------------
    if _section("Randomizer Settings"):
        _, sim.rand_force_range = imgui.drag_float2("Force Range",
            *sim.rand_force_range, 0.01, -1.0, 1.0, "%.2f")
        _, sim.rand_min_r_range = imgui.drag_float2("Min R Range",
            *sim.rand_min_r_range, 0.5,  1.0, 300.0, "%.0f")
        _, sim.rand_max_r_range = imgui.drag_float2("Max R Range",
            *sim.rand_max_r_range, 0.5,  1.0, 300.0, "%.0f")
        if imgui.button("Randomize Radii"):
            n = sim.num_colors
            import numpy as np
            lo, hi = sim.rand_min_r_range
            sim.min_r_matrix = (np.random.rand(n, n) * (hi - lo) + lo).astype(np.float32)
            lo, hi = sim.rand_max_r_range
            sim.max_r_matrix = (np.random.rand(n, n) * (hi - lo) + lo).astype(np.float32)
            rules_dirty = True
        imgui.same_line()
        if imgui.button("Randomize All"):
            sim.randomize_rules()
            sim.reset_particles()

    # --- World Settings -------------------------------------------------------
    if _section("World Settings", opened=True):
        changed, val = imgui.slider_int("Particles", sim.num_particles, 100, 1000000)
        if changed:
            sim.num_particles = val
            sim.reset_particles()
        changed, val = imgui.slider_int("Species (Colors)", sim.num_colors, 1, 20)
        if changed:
            sim.num_colors = val
            sim.reset_particles()
            sim.randomize_rules()

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
        if sim.world_mode == 1 and renderer.tile_wrap and sim.mode3d:
            _, renderer.tile_distance = imgui.slider_int("Tile Distance", renderer.tile_distance, 0, 8)

    # --- Physics Settings -----------------------------------------------------
    if _section("Physics Settings", opened=True):
        _, sim.force_factor    = imgui.slider_float("Force Multiplier", sim.force_factor,    0.01, 200.0)
        _, sim.friction_factor = imgui.slider_float("Friction",         sim.friction_factor, 0.0,  1.0)
        _, sim.repel           = imgui.slider_float("Repel Force",      sim.repel,           0.01, 5.0)
        _, sim.sim_speed       = imgui.slider_float("Sim Speed",        sim.sim_speed,       0.1,  10.0)
        _, sim.substeps        = imgui.slider_int(  "Substeps",         sim.substeps,        1,    16)
        _, sim.max_speed       = imgui.slider_float("Max Speed (0=off)", sim.max_speed,      0.0,  100.0)
        _, sim.max_accel       = imgui.slider_float("Max Accel (0=off)", sim.max_accel,      0.0,  100.0)

    # --- Graphics Settings ----------------------------------------------------
    if _section("Graphics Settings"):
        _, renderer.particle_size    = imgui.slider_float("Particle Size",    renderer.particle_size,    0.1, 6.0)
        _, renderer.particle_opacity = imgui.slider_float("Particle Opacity", renderer.particle_opacity, 0.0, 1.0)
        imgui.text("Particle Blending:")
        imgui.same_line()
        if imgui.radio_button("Normal", not renderer.additive_blending):
            renderer.additive_blending = False
        imgui.same_line()
        if imgui.radio_button("Additive", renderer.additive_blending):
            renderer.additive_blending = True

        imgui.separator()
        _, renderer.particle_glow = imgui.checkbox("Particle Glowing", renderer.particle_glow)
        _, renderer.glow_size      = imgui.slider_float("Glow Size",      renderer.glow_size,      0.0, 32.0)
        _, renderer.glow_intensity = imgui.slider_float("Glow Intensity", renderer.glow_intensity, 0.0, 0.5, "%.3f")
        _, renderer.glow_steepness = imgui.slider_float("Glow Steepness", renderer.glow_steepness, 0.0, 12.0)

        imgui.separator()
        _, renderer.tone_mapping = imgui.checkbox("Tone Mapping", renderer.tone_mapping)
        if renderer.tone_mapping:
            _, renderer.exposure = imgui.slider_float("Exposure", renderer.exposure, 0.1, 4.0, "%.2f")

        if sim.mode3d:
            imgui.separator()
            _, renderer.fog = imgui.checkbox("Fog", renderer.fog)
            if renderer.fog:
                _, renderer.fog_density = imgui.slider_float(
                    "Fog Density", renderer.fog_density, 0.0001, 0.005, "%.4f")

        imgui.separator()
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

    # --- Camera Settings ------------------------------------------------------
    if _section("Camera Settings"):
        _, sim.zoom_smoothing = imgui.slider_float("Zoom Smoothing", sim.zoom_smoothing, 0.01, 1.0)
        _, sim.pan_smoothing  = imgui.slider_float("Pan Smoothing",  sim.pan_smoothing,  0.01, 1.0)
        imgui.separator()
        imgui.text("Cinematic Mode:")
        _, sim.drift_cam_enabled = imgui.checkbox("Drift Cam", sim.drift_cam_enabled)
        if sim.drift_cam_enabled:
            _, sim.drift_cam_reset_on_pan = imgui.checkbox("Reset Path on Pan", sim.drift_cam_reset_on_pan)
            _, sim.drift_cam_speed     = imgui.slider_float("Drift Speed",   sim.drift_cam_speed,     0.01, 1.0)
            _, sim.drift_cam_amplitude = imgui.slider_float("Pan Amplitude", sim.drift_cam_amplitude, 0.05, 1.0)
            _, sim.drift_cam_zoom_range = imgui.drag_float2(
                "Zoom Range", *sim.drift_cam_zoom_range, 0.01, 0.1, 5.0, "%.2f")

    # --- Brush / Tools --------------------------------------------------------
    if _section("Brush / Tools", opened=True):
        imgui.text("Tool (left click):")
        for i, name in enumerate(TOOL_NAMES):
            if imgui.radio_button(name, tool[0] == i):
                tool[0] = i
            if i < len(TOOL_NAMES) - 1 and (i + 1) % 3 != 0:
                imgui.same_line()
        _, sim.brush_radius             = imgui.slider_float("Radius",       sim.brush_radius,             10.0, 1000.0)
        _, sim.brush_intensity          = imgui.slider_int(  "Intensity",    sim.brush_intensity,          1,    100)
        _, sim.brush_force              = imgui.slider_float("Push Force",   sim.brush_force,              0.01, 1.0)
        _, sim.attract_force            = imgui.slider_float("Attract Force", sim.attract_force,           0.0,  2.0)
        _, sim.repulse_force            = imgui.slider_float("Repulse Force", sim.repulse_force,           0.0,  2.0)
        _, sim.brush_directional_force  = imgui.slider_float("Velocity Mix", sim.brush_directional_force,  0.0,  2.0)
        _, sim.show_brush_circle        = imgui.checkbox("Show Brush Circle", sim.show_brush_circle)

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

    # --- Debug Tools ----------------------------------------------------------
    if _section("Debug Tools"):
        _, sim.use_spatial_hash = imgui.checkbox("Spatial Hash", sim.use_spatial_hash)
        if not sim.use_spatial_hash:
            imgui.text_colored("(Brute force - slow for many particles)", 1.0, 0.6, 0.3)
        if sim.use_spatial_hash:
            _, sim.cell_subdivisions = imgui.slider_int(
                "Cell Subdivisions", sim.cell_subdivisions, 1, 5)

    # --- Action Buttons -------------------------------------------------------
    imgui.spacing(); imgui.separator()
    if imgui.button("Reset Particles"): sim.reset_particles()
    imgui.same_line()
    if imgui.button("Reset Params"):    sim.reset_params()
    imgui.same_line()
    if imgui.button("Reset View"):      sim.reset_view()

    if rules_dirty:
        sim._upload_rules()

    imgui.end()
