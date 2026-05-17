import glfw
import imgui
from imgui.integrations.glfw import GlfwRenderer
from OpenGL.GL import *

from simulation import Simulation
from renderer import Renderer
from ui import draw_ui


def screen_to_world(mx, my, fh, view_offset, view_scale):
    wx = mx / view_scale - view_offset[0]
    wy = (fh - my) / view_scale - view_offset[1]
    return wx, wy


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
    view_scale = 1.0
    drag_last = None
    brush_last = None  # last world pos for brush velocity

    # tool: 0=none, 1=push, 2=paint, 3=erase
    tool = [1]

    def on_scroll(window, dx, dy):
        nonlocal view_scale
        if imgui.get_io().want_capture_mouse:
            return
        mx, my = glfw.get_cursor_pos(window)
        _, fh = glfw.get_framebuffer_size(window)
        factor = 1.1 ** dy
        wx, wy = screen_to_world(mx, my, fh, view_offset, view_scale)
        view_scale *= factor
        view_offset[0] = mx / view_scale - wx
        view_offset[1] = (fh - my) / view_scale - wy

    glfw.set_scroll_callback(win, on_scroll)

    while not glfw.window_should_close(win):
        glfw.poll_events()
        impl.process_inputs()

        io = imgui.get_io()
        mx, my = glfw.get_cursor_pos(win)
        w, h = glfw.get_framebuffer_size(win)
        wx, wy = screen_to_world(mx, my, h, view_offset, view_scale)

        # middle/right drag = pan
        panning = (glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS or
                   glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)
        if panning and not io.want_capture_mouse:
            if drag_last is not None:
                view_offset[0] += (mx - drag_last[0]) / view_scale
                view_offset[1] -= (my - drag_last[1]) / view_scale
            drag_last = (mx, my)
        else:
            drag_last = None

        # left click = active tool
        painting = (glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
                    and not io.want_capture_mouse)
        if painting:
            bvx = (wx - brush_last[0]) if brush_last else 0.0
            bvy = (wy - brush_last[1]) if brush_last else 0.0
            if tool[0] == 1:
                sim.apply_brush(wx, wy, bvx, bvy)
            elif tool[0] == 2:
                sim.paint_particles(wx, wy)
            elif tool[0] == 3:
                sim.apply_eraser(wx, wy)
            brush_last = (wx, wy)
        else:
            brush_last = None

        now = glfw.get_time()
        dt = min(now - last_time, 0.1)
        last_time = now
        frame_dt = dt * sim.sim_speed * 60.0
        sub_dt = frame_dt / sim.substeps
        for _ in range(sim.substeps):
            sim.step(dt_scale=sub_dt)

        glViewport(0, 0, w, h)
        glClearColor(0.05, 0.05, 0.08, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        renderer.draw(sim, w, h, view_offset=view_offset, view_scale=view_scale)
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
