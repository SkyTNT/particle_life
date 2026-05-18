import glfw
import imgui
import math
import random
import numpy as np
from imgui.integrations.glfw import GlfwRenderer
from OpenGL.GL import *

from simulation import Simulation
from renderer import Renderer
from ui import draw_ui


def screen_to_world(mx, my, fh, view_offset, view_scale):
    return mx / view_scale - view_offset[0], (fh - my) / view_scale - view_offset[1]


def _perspective(fovy, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fovy) / 2)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0,0] = f / aspect; m[1,1] = f
    m[2,2] = (far + near) / (near - far); m[2,3] = -1
    m[3,2] = (2 * far * near) / (near - far)
    return m


def _look_at(eye, yaw, pitch):
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    fwd   = np.array([cy*cp, sp, sy*cp], dtype=np.float32)
    right = np.array([math.cos(yaw - math.pi/2), 0, math.sin(yaw - math.pi/2)], dtype=np.float32)
    up    = np.cross(fwd, right)
    v = np.eye(4, dtype=np.float32)
    v[0,:3] = right; v[0,3] = -np.dot(right, eye)
    v[1,:3] = up;    v[1,3] = -np.dot(up, eye)
    v[2,:3] = -fwd;  v[2,3] =  np.dot(fwd, eye)
    return v


def _unproject(sx, sy, w, h, mvp):
    ndc = np.array([(sx/w)*2-1, 1-(sy/h)*2, -1.0, 1.0], dtype=np.float64)
    p = np.linalg.inv(mvp.astype(np.float64)) @ ndc
    return p[:3] / p[3]


# 8 corner offset directions reused by the tile-frustum cull.
_TILE_CORNERS = np.array([
    [0,0,0],[1,0,0],[0,1,0],[1,1,0],
    [0,0,1],[1,0,1],[0,1,1],[1,1,1],
], dtype=np.float32)


def _compute_tile_offsets(sim, mvp4x4, cam_pos, win_w, win_h, view_scale, tile_distance):
    """Build the list of tile offsets to instance. 2D: bounded by viewport.
    3D: candidate cube of `tile_distance` tiles around the camera, then per-tile
    frustum cull so we don't draw tiles behind the camera or off to the side."""
    if sim.world_mode != 1:
        return None
    if not sim.mode3d:
        vw, vh = win_w / view_scale, win_h / view_scale
        r = max(1, int(math.ceil(max(vw / sim.world_w, vh / sim.world_h))) + 1)
        ax = np.arange(-r, r + 1)
        IX, IY = np.meshgrid(ax, ax, indexing='ij')
        return np.stack([IX * sim.world_w, IY * sim.world_h,
                         np.zeros_like(IX, dtype=np.float32)], axis=-1
                        ).reshape(-1, 3).astype(np.float32)

    sizes = np.array([sim.world_w, sim.world_h, sim.world_d], dtype=np.float64)
    cam_tile = np.floor(np.asarray(cam_pos, dtype=np.float64) / sizes).astype(int)
    d = max(0, int(tile_distance))
    ax = np.arange(-d, d + 1)
    IX, IY, IZ = np.meshgrid(ax + cam_tile[0], ax + cam_tile[1], ax + cam_tile[2],
                             indexing='ij')
    centers = np.stack([IX * sim.world_w, IY * sim.world_h, IZ * sim.world_d],
                       axis=-1).reshape(-1, 3).astype(np.float32)
    tile_size = np.array([sim.world_w, sim.world_h, sim.world_d], dtype=np.float32)
    corners = centers[:, None, :] + _TILE_CORNERS[None, :, :] * tile_size
    h4 = np.concatenate(
        [corners, np.ones(corners.shape[:2] + (1,), dtype=np.float32)], axis=-1)
    # mvp4x4 @ v_col == v_row @ mvp4x4.T  (clip-space in column-vector convention)
    clip = h4 @ mvp4x4.astype(np.float32).T
    x, y, z, w = clip[..., 0], clip[..., 1], clip[..., 2], clip[..., 3]
    out = ((x >  w).all(axis=1) | (x < -w).all(axis=1) |
           (y >  w).all(axis=1) | (y < -w).all(axis=1) |
           (z >  w).all(axis=1) | (z < -w).all(axis=1))
    return centers[~out]


def main():
    if not glfw.init():
        raise RuntimeError("GLFW init failed")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    monitor = glfw.get_primary_monitor()
    mode = glfw.get_video_mode(monitor)
    win = glfw.create_window(mode.size.width, mode.size.height, "Particle Life", None, None)
    glfw.make_context_current(win)
    glfw.swap_interval(1)

    imgui.create_context()
    impl = GlfwRenderer(win)

    sim = Simulation()
    renderer = Renderer()
    renderer.init_gl()
    sim.world_w = float(mode.size.width)
    sim.world_h = float(mode.size.height)
    sim.init_gl()

    last_time = glfw.get_time()
    view_offset = [0.0, 0.0]
    view_scale  = 1.0
    # Smoothed targets for camera (zoom/pan smoothing settings).
    target_view_offset = [0.0, 0.0]
    drag_last = brush_last = None
    tool = [1]

    # Drift cam state (auto pan + zoom).
    drift_t  = [0.0]
    drift_seed = [random.random() * 1000.0]

    cam_pos   = np.array([sim.world_w/2, sim.world_h/2, sim.world_w*1.5], dtype=np.float32)
    cam_yaw   = -math.pi / 2
    cam_pitch = 0.0
    rotating  = False
    bp        = cam_pos.copy()
    locked_mx = locked_my = 0.0
    brush_dist = [sim.world_w * 0.3]

    def enter_3d():
        nonlocal cam_pos, cam_yaw, cam_pitch
        sim.world_w = sim.world_h = sim.world_d = 1600.0
        # Sit outside the cube on +z looking toward -z so the swarm is in front of us,
        # not surrounding us (which produces a wall of overlapping glow halos).
        cam_pos   = np.array([800.0, 800.0, 2800.0], dtype=np.float32)
        cam_yaw   = -math.pi / 2
        cam_pitch = 0.0
        sim.mode3d = True
        sim.reset_particles()

    sim.enter_3d = enter_3d

    # List-backed so closures can mutate without nonlocal gymnastics.
    target_view_scale = [1.0]

    def reset_view():
        nonlocal view_scale, cam_pos, cam_yaw, cam_pitch
        if sim.mode3d:
            cam_pos  = np.array([sim.world_w/2, sim.world_h/2, sim.world_d * 1.75],
                                dtype=np.float32)
            cam_yaw  = -math.pi / 2
            cam_pitch = 0.0
        else:
            view_offset[0] = view_offset[1] = 0.0
            view_scale = 1.0
            target_view_offset[0] = target_view_offset[1] = 0.0
            target_view_scale[0]  = 1.0

    sim.reset_view = reset_view

    def on_scroll(window, dx, dy):
        # Forward to imgui so its windows/sliders can scroll, then bail if imgui owns the cursor.
        impl.scroll_callback(window, dx, dy)
        if imgui.get_io().want_capture_mouse:
            return
        if sim.mode3d:
            brush_dist[0] = max(50.0, brush_dist[0] * (1.1 ** dy))
            return
        mx, my = glfw.get_cursor_pos(window)
        _, fh = glfw.get_framebuffer_size(window)
        # Use the *current* view to compute the world point under the cursor, then
        # set the smoothed *target* so that point stays fixed once the zoom catches up.
        wx, wy = screen_to_world(mx, my, fh, view_offset, view_scale)
        target_view_scale[0] *= 1.1 ** dy
        target_view_offset[0] = mx / target_view_scale[0] - wx
        target_view_offset[1] = (fh - my) / target_view_scale[0] - wy
        if sim.drift_cam_reset_on_pan and sim.drift_cam_enabled:
            drift_t[0] = 0.0
            drift_seed[0] = random.random() * 1000.0

    def on_key(window, key, scancode, action, mods):
        if action == glfw.PRESS and key == glfw.KEY_TAB:
            if sim.mode3d: sim.mode3d = False; sim.reset_particles()
            else: enter_3d()

    glfw.set_scroll_callback(win, on_scroll)
    glfw.set_key_callback(win, on_key)

    mouse_last = None

    while not glfw.window_should_close(win):
        glfw.poll_events()
        impl.process_inputs()

        io  = imgui.get_io()
        mx, my = glfw.get_cursor_pos(win)
        w,  h  = glfw.get_framebuffer_size(win)
        now = glfw.get_time()
        dt  = min(now - last_time, 0.1)
        last_time = now

        if sim.mode3d:
            # WASD movement (only need fwd/right, compute cheaply)
            if not io.want_capture_keyboard:
                cy, sy = math.cos(cam_yaw), math.sin(cam_yaw)
                cp     = math.cos(cam_pitch)
                fwd    = np.array([cy*cp, math.sin(cam_pitch), sy*cp], dtype=np.float32)
                right  = np.array([math.cos(cam_yaw - math.pi/2), 0, math.sin(cam_yaw - math.pi/2)], dtype=np.float32)
                speed  = sim.world_w * dt * 0.8
                if glfw.get_key(win, glfw.KEY_W)          == glfw.PRESS: cam_pos += fwd   * speed
                if glfw.get_key(win, glfw.KEY_S)          == glfw.PRESS: cam_pos -= fwd   * speed
                if glfw.get_key(win, glfw.KEY_A)          == glfw.PRESS: cam_pos -= right * speed
                if glfw.get_key(win, glfw.KEY_D)          == glfw.PRESS: cam_pos += right * speed
                if glfw.get_key(win, glfw.KEY_SPACE)      == glfw.PRESS: cam_pos[1] += speed
                if glfw.get_key(win, glfw.KEY_LEFT_SHIFT) == glfw.PRESS: cam_pos[1] -= speed

            # right-drag = rotate
            rotating = glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS and not io.want_capture_mouse
            if rotating:
                if mouse_last is None:
                    locked_mx, locked_my = mx, my
                    glfw.set_input_mode(win, glfw.CURSOR, glfw.CURSOR_DISABLED)
                else:
                    cam_yaw   -= (mx - mouse_last[0]) * 0.003
                    cam_pitch  = max(-math.pi/2+0.01, min(math.pi/2-0.01,
                                    cam_pitch - (my - mouse_last[1]) * 0.003))
                mouse_last = (mx, my)
            else:
                if mouse_last is not None:
                    glfw.set_input_mode(win, glfw.CURSOR, glfw.CURSOR_NORMAL)
                mouse_last = None

            # build MVP once after all camera updates
            view   = _look_at(cam_pos, cam_yaw, cam_pitch)
            proj   = _perspective(60.0, w / h, 1.0, sim.world_w * 10)
            mvp4x4 = proj @ view
            mvp    = mvp4x4.T.flatten().astype(np.float32)

            # brush position: screen center when rotating, else mouse
            sx = locked_mx if rotating else mx
            sy = locked_my if rotating else my
            near_w = _unproject(sx, sy, w, h, mvp4x4)
            ray = near_w - cam_pos.astype(np.float64)
            ray /= np.linalg.norm(ray)
            bp = (cam_pos + ray * brush_dist[0]).astype(np.float32)

            if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS and not io.want_capture_mouse:
                bvx = (bp[0] - brush_last[0]) if brush_last else 0.0
                bvy = (bp[1] - brush_last[1]) if brush_last else 0.0
                if   tool[0] == 1: sim.apply_brush(bp[0], bp[1], bvx, bvy, bp[2])
                elif tool[0] == 2: sim.paint_particles(bp[0], bp[1], bp[2])
                elif tool[0] == 3: sim.apply_eraser(bp[0], bp[1], bp[2])
                elif tool[0] == 4: sim.apply_attract(bp[0], bp[1], bp[2], attract=True)
                elif tool[0] == 5: sim.apply_attract(bp[0], bp[1], bp[2], attract=False)
                brush_last = (bp[0], bp[1])
            else:
                brush_last = None
        else:
            mvp = mvp4x4 = None
            wx, wy = screen_to_world(mx, my, h, view_offset, view_scale)

            if (glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS or
                    glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS) and not io.want_capture_mouse:
                if drag_last:
                    dx_v = (mx - drag_last[0]) / view_scale
                    dy_v = -(my - drag_last[1]) / view_scale
                    view_offset[0]        += dx_v
                    view_offset[1]        += dy_v
                    target_view_offset[0] += dx_v
                    target_view_offset[1] += dy_v
                    if sim.drift_cam_enabled and sim.drift_cam_reset_on_pan:
                        drift_t[0] = 0.0
                        drift_seed[0] = random.random() * 1000.0
                drag_last = (mx, my)
            else:
                drag_last = None

            if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS and not io.want_capture_mouse:
                bvx = (wx - brush_last[0]) if brush_last else 0.0
                bvy = (wy - brush_last[1]) if brush_last else 0.0
                if   tool[0] == 1: sim.apply_brush(wx, wy, bvx, bvy)
                elif tool[0] == 2: sim.paint_particles(wx, wy)
                elif tool[0] == 3: sim.apply_eraser(wx, wy)
                elif tool[0] == 4: sim.apply_attract(wx, wy, 0.0, attract=True)
                elif tool[0] == 5: sim.apply_attract(wx, wy, 0.0, attract=False)
                brush_last = (wx, wy)
            else:
                brush_last = None

        # Drift cam: smoothly steer the *target* using lissajous-like motion.
        if sim.drift_cam_enabled and not sim.mode3d:
            drift_t[0] += dt * sim.drift_cam_speed
            t = drift_t[0]
            s = drift_seed[0]
            ax = math.sin(t * 0.83 + s) * math.cos(t * 0.41 + s * 0.5)
            ay = math.cos(t * 0.67 + s * 1.7) * math.sin(t * 0.29 + s * 0.3)
            zmin, zmax = sim.drift_cam_zoom_range
            zmid = (zmin + zmax) * 0.5
            zhalf = (zmax - zmin) * 0.5
            target_view_scale[0] = zmid + math.sin(t * 0.37 + s * 2.1) * zhalf
            cx, cy = w * 0.5, h * 0.5
            target_view_offset[0] = cx / target_view_scale[0] - sim.world_w * 0.5 \
                                   + ax * sim.world_w * sim.drift_cam_amplitude * 0.5
            target_view_offset[1] = cy / target_view_scale[0] - sim.world_h * 0.5 \
                                   + ay * sim.world_h * sim.drift_cam_amplitude * 0.5

        # Apply zoom/pan smoothing (target → current). Clamp factor to [0,1].
        if not sim.mode3d:
            kz = max(0.001, min(1.0, sim.zoom_smoothing))
            kp = max(0.001, min(1.0, sim.pan_smoothing))
            view_scale     += (target_view_scale[0]     - view_scale)     * kz
            view_offset[0] += (target_view_offset[0]    - view_offset[0]) * kp
            view_offset[1] += (target_view_offset[1]    - view_offset[1]) * kp

        sub_dt = dt * sim.sim_speed / sim.substeps
        sim.step_multi(sub_dt, sim.substeps)

        glViewport(0, 0, w, h)
        glClearColor(0.05, 0.05, 0.08, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST) if sim.mode3d else glDisable(GL_DEPTH_TEST)

        tile_offsets = (_compute_tile_offsets(sim, mvp4x4, cam_pos, w, h, view_scale,
                                              renderer.tile_distance)
                        if renderer.tile_wrap else None)
        renderer.draw(sim, w, h, view_offset=view_offset, view_scale=view_scale,
                      mvp=mvp, tile_offsets=tile_offsets)
        renderer.draw_grid(sim, w, h, view_offset=view_offset, view_scale=view_scale, mvp=mvp,
                           cam_pos=cam_pos if sim.mode3d else None)

        if sim.show_brush_circle:
            if sim.mode3d:
                renderer.draw_cursor(0, 0, sim.brush_radius, w, h,
                                     mode3d=True, brush3d_pos=bp, mvp4x4=mvp4x4)
            else:
                renderer.draw_cursor(wx, wy, sim.brush_radius, w, h,
                                     view_offset=view_offset, view_scale=view_scale)

        imgui.new_frame()
        draw_ui(sim, tool, renderer)
        imgui.render()
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(win)

    impl.shutdown()
    glfw.terminate()


if __name__ == "__main__":
    main()
