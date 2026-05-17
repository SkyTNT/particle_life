import math
import numpy as np
from OpenGL.GL import *

VERT = """
#version 430 core
struct Particle { float x, y, vx, vy; int color; float z, vz, _p2; };
layout(std430, binding=0) buffer Particles { Particle p[]; };
layout(std430, binding=3) buffer Palette   { vec4 palette[]; };
uniform vec2 world_size; uniform vec2 view_offset; uniform float view_scale;
uniform int mode3d;
uniform mat4 mvp;
out vec3 vColor;
out float vDepth;
void main() {
    Particle pt = p[gl_VertexID];
    if (mode3d == 1) {
        gl_Position = mvp * vec4(pt.x, pt.y, pt.z, 1.0);
        gl_PointSize = max(1.0, 8000.0 / gl_Position.w);
        vDepth = gl_Position.w;
    } else {
        vec2 sp = (vec2(pt.x,pt.y) + view_offset) * view_scale;
        gl_Position = vec4(sp/world_size*2.0-1.0, 0.0, 1.0);
        gl_PointSize = max(1.0, 4.0 * view_scale);
        vDepth = 0.0;
    }
    vColor = palette[pt.color].rgb;
}
"""

FRAG = """
#version 430 core
in vec3 vColor; in float vDepth; out vec4 fragColor;
void main() {
    vec2 c = gl_PointCoord*2.0-1.0;
    if (dot(c,c)>1.0) discard;
    float brightness = (vDepth > 0.0) ? clamp(8000.0 / vDepth, 0.1, 1.0) : 1.0;
    fragColor = vec4(vColor * brightness, 1.0);
}
"""

GRID_VERT = """
#version 430 core
out vec2 uv;
void main() {
    vec2 pos = vec2((gl_VertexID & 1) * 2 - 1, (gl_VertexID >> 1) * 2 - 1);
    uv = pos;
    gl_Position = vec4(pos, 0.0, 1.0);
}
"""
GRID_FRAG = """
#version 430 core
in vec2 uv;
out vec4 fragColor;
uniform int mode3d;
uniform float step;
uniform vec2 view_offset; uniform float view_scale; uniform vec2 win_size;
uniform mat4 inv_mvp;
uniform vec3 cam_pos;
uniform float world_h;
float gridline(vec2 p) {
    vec2 g = abs(fract(p / step - 0.5) - 0.5) / fwidth(p / step);
    return 1.0 - min(min(g.x, g.y), 1.0);
}
void main() {
    float alpha;
    if (mode3d == 1) {
        vec4 near = inv_mvp * vec4(uv, -1.0, 1.0);
        vec4 far  = inv_mvp * vec4(uv,  1.0, 1.0);
        near /= near.w; far /= far.w;
        vec3 ray = normalize(near.xyz - far.xyz);
        // single ground plane at y=0
        float best_alpha = 0.0;
        if (abs(ray.y) > 1e-4) {
            float t = -cam_pos.y / ray.y;
            if (t > 0.0) {
                vec3 hit = cam_pos + ray * t;
                float fade = 1.0 - clamp(t / (step * 30.0), 0.0, 1.0);
                best_alpha = gridline(hit.xz) * fade;
            }
        }
        alpha = best_alpha;
    } else {
        vec2 screen = (uv + 1.0) * 0.5 * win_size;
        vec2 world = screen / view_scale - view_offset;
        alpha = gridline(world);
    }
    if (alpha < 0.01) discard;
    fragColor = vec4(1.0, 1.0, 1.0, alpha * 0.3);
}
"""

CURSOR_VERT = """
#version 430 core
layout(location=0) in vec2 pos;
void main() { gl_Position = vec4(pos, 0.0, 1.0); }
"""

CURSOR_FRAG = """
#version 430 core
out vec4 fragColor;
void main() { fragColor = vec4(1.0, 1.0, 1.0, 0.8); }
"""


def _compile_prog(vert_src, frag_src):
    def _s(src, kind):
        s = glCreateShader(kind)
        glShaderSource(s, src)
        glCompileShader(s)
        if not glGetShaderiv(s, GL_COMPILE_STATUS):
            raise RuntimeError(glGetShaderInfoLog(s).decode())
        return s
    vs, fs = _s(vert_src, GL_VERTEX_SHADER), _s(frag_src, GL_FRAGMENT_SHADER)
    prog = glCreateProgram()
    glAttachShader(prog, vs); glAttachShader(prog, fs)
    glLinkProgram(prog)
    glDeleteShader(vs); glDeleteShader(fs)
    return prog


DEFAULT_PALETTE = np.array([
    [1.0,0.2,0.2,1],[0.2,0.8,0.2,1],[0.2,0.4,1.0,1],[1.0,0.9,0.1,1],
    [0.9,0.3,0.9,1],[0.2,0.9,0.9,1],[1.0,0.6,0.1,1],[0.6,1.0,0.4,1],
    [1.0,0.5,0.5,1],[0.5,1.0,0.5,1],[0.5,0.5,1.0,1],[1.0,1.0,0.5,1],
    [1.0,0.5,1.0,1],[0.5,1.0,1.0,1],[0.8,0.4,0.2,1],[0.4,0.2,0.8,1],
    [0.2,0.6,0.4,1],[0.6,0.2,0.4,1],[0.9,0.7,0.3,1],[0.3,0.7,0.9,1],
], dtype=np.float32)


class Renderer:
    def __init__(self):
        self.prog = self.cursor_prog = self.grid_prog = None
        self.vao = self.cursor_vao = self.cursor_vbo = self.ssbo_palette = None
        self.grid_vao = self.grid_vbo = None
        self.palette = DEFAULT_PALETTE.copy()
        self.show_grid = False

    def init_gl(self):
        self.prog = _compile_prog(VERT, FRAG)
        self.cursor_prog = _compile_prog(CURSOR_VERT, CURSOR_FRAG)
        self.vao = glGenVertexArrays(1)
        self.cursor_vbo = glGenBuffers(1)
        self.ssbo_palette = glGenBuffers(1)
        glEnable(GL_PROGRAM_POINT_SIZE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.cursor_vao = glGenVertexArrays(1)
        glBindVertexArray(self.cursor_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.cursor_vbo)
        glBufferData(GL_ARRAY_BUFFER, 64 * 2 * 4, None, GL_DYNAMIC_DRAW)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)
        glBindVertexArray(0)

        self.grid_prog = _compile_prog(GRID_VERT, GRID_FRAG)
        self.grid_vao = glGenVertexArrays(1)  # empty VAO for attributeless draw

        self._upload_palette()

    def _upload_palette(self):
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo_palette)
        glBufferData(GL_SHADER_STORAGE_BUFFER, self.palette.nbytes, self.palette, GL_DYNAMIC_DRAW)

    def draw_grid(self, sim, win_w, win_h, view_offset=(0.0,0.0), view_scale=1.0, mvp=None,
                  cam_pos=None, **_):
        if not self.show_grid:
            return
        W, H = sim.world_w, sim.world_h
        D = sim.world_d if sim.mode3d else W
        step = float(10 ** math.ceil(math.log10(max(W, H, D) / 10)))
        glUseProgram(self.grid_prog)
        u = lambda name: glGetUniformLocation(self.grid_prog, name)
        glUniform1i(u("mode3d"), 1 if sim.mode3d else 0)
        glUniform1f(u("step"), step)
        glUniform2f(u("view_offset"), view_offset[0], view_offset[1])
        glUniform1f(u("view_scale"), view_scale)
        glUniform2f(u("win_size"), float(win_w), float(win_h))
        if sim.mode3d and mvp is not None and cam_pos is not None:
            inv = np.linalg.inv(mvp.reshape(4,4).T).T.flatten().astype(np.float32)
            glUniformMatrix4fv(u("inv_mvp"), 1, GL_FALSE, inv)
            glUniform3f(u("cam_pos"), *cam_pos)
            glUniform1f(u("world_h"), float(sim.world_h))
        glBindVertexArray(self.grid_vao)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)

    def draw(self, sim, win_w, win_h, view_offset=(0.0,0.0), view_scale=1.0, mvp=None):
        glUseProgram(self.prog)
        glUniform2f(glGetUniformLocation(self.prog,"world_size"), float(win_w), float(win_h))
        glUniform2f(glGetUniformLocation(self.prog,"view_offset"), view_offset[0], view_offset[1])
        glUniform1f(glGetUniformLocation(self.prog,"view_scale"), view_scale)
        glUniform1i(glGetUniformLocation(self.prog,"mode3d"), 1 if sim.mode3d else 0)
        if sim.mode3d and mvp is not None:
            glUniformMatrix4fv(glGetUniformLocation(self.prog,"mvp"), 1, GL_FALSE, mvp)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, sim.get_particle_ssbo())
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, self.ssbo_palette)
        glBindVertexArray(self.vao)
        glDrawArrays(GL_POINTS, 0, sim.num_particles)
        glBindVertexArray(0)

    def draw_cursor(self, wx, wy, radius, win_w, win_h, view_offset=(0.0,0.0), view_scale=1.0,
                    mode3d=False, brush3d_pos=None, mvp4x4=None):
        N = 64
        a = np.linspace(0, 2*math.pi, N, endpoint=False)
        cos_a, sin_a = np.cos(a), np.sin(a)
        if mode3d and brush3d_pos is not None and mvp4x4 is not None:
            bx, by, bz = brush3d_pos
            clip_c = mvp4x4 @ np.array([bx, by, bz, 1.0], dtype=np.float32)
            if clip_c[3] <= 0:
                return
            cx = clip_c[0] / clip_c[3]
            cy = clip_c[1] / clip_c[3]
            r_px = max(4.0, radius / math.tan(math.radians(30.0)) / clip_c[3] * win_h / 2)
            xs = cx + cos_a * r_px / win_w * 2.0
            ys = cy + sin_a * r_px / win_h * 2.0
        else:
            xs = ((wx + cos_a * radius + view_offset[0]) * view_scale) / win_w * 2 - 1
            ys = ((wy + sin_a * radius + view_offset[1]) * view_scale) / win_h * 2 - 1
        verts = np.column_stack([xs, ys]).astype(np.float32)

        glBindBuffer(GL_ARRAY_BUFFER, self.cursor_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, verts.nbytes, verts)

        glUseProgram(self.cursor_prog)
        glBindVertexArray(self.cursor_vao)
        glDrawArrays(GL_LINE_LOOP, 0, N)
        glBindVertexArray(0)
