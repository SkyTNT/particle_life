import math
from OpenGL.GL import *

VERT = """
#version 430 core
struct Particle { float x, y, vx, vy; int color; float _p0, _p1, _p2; };
layout(std430, binding = 0) buffer Particles { Particle p[]; };
const vec3 PALETTE[8] = vec3[8](
    vec3(1.0,0.2,0.2),vec3(0.2,0.8,0.2),vec3(0.2,0.4,1.0),vec3(1.0,0.9,0.1),
    vec3(0.9,0.3,0.9),vec3(0.2,0.9,0.9),vec3(1.0,0.6,0.1),vec3(0.6,1.0,0.4)
);
uniform vec2 world_size; uniform vec2 view_offset; uniform float view_scale;
out vec3 vColor;
void main() {
    Particle pt = p[gl_VertexID];
    vec2 sp = (vec2(pt.x,pt.y) + view_offset) * view_scale;
    gl_Position = vec4(sp/world_size*2.0-1.0, 0.0, 1.0);
    gl_PointSize = 4.0;
    vColor = PALETTE[pt.color % 8];
}
"""

FRAG = """
#version 430 core
in vec3 vColor; out vec4 fragColor;
void main() {
    vec2 c = gl_PointCoord*2.0-1.0;
    if (dot(c,c)>1.0) discard;
    fragColor = vec4(vColor, 1.0);
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


class Renderer:
    def __init__(self):
        self.prog = self.cursor_prog = None
        self.vao = self.cursor_vao = self.cursor_vbo = None

    def init_gl(self):
        self.prog = _compile_prog(VERT, FRAG)
        self.cursor_prog = _compile_prog(CURSOR_VERT, CURSOR_FRAG)
        self.vao = glGenVertexArrays(1)
        self.cursor_vao, = glGenVertexArrays(1),
        self.cursor_vbo = glGenBuffers(1)
        glEnable(GL_PROGRAM_POINT_SIZE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # build cursor circle (line loop, 64 segments)
        self.cursor_vao = glGenVertexArrays(1)
        glBindVertexArray(self.cursor_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.cursor_vbo)
        glBufferData(GL_ARRAY_BUFFER, 64 * 2 * 4, None, GL_DYNAMIC_DRAW)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)
        glBindVertexArray(0)

    def draw(self, sim, win_w, win_h, view_offset=(0.0,0.0), view_scale=1.0):
        glUseProgram(self.prog)
        glUniform2f(glGetUniformLocation(self.prog,"world_size"), float(win_w), float(win_h))
        glUniform2f(glGetUniformLocation(self.prog,"view_offset"), view_offset[0], view_offset[1])
        glUniform1f(glGetUniformLocation(self.prog,"view_scale"), view_scale)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, sim.get_particle_ssbo())
        glBindVertexArray(self.vao)
        glDrawArrays(GL_POINTS, 0, sim.num_particles)
        glBindVertexArray(0)

    def draw_cursor(self, wx, wy, radius, win_w, win_h, view_offset=(0.0,0.0), view_scale=1.0):
        import numpy as np
        N = 64
        angles = [2*math.pi*i/N for i in range(N)]
        # world-space circle -> screen -> NDC
        pts = []
        for a in angles:
            sx = (wx + math.cos(a)*radius + view_offset[0]) * view_scale
            sy = (wy + math.sin(a)*radius + view_offset[1]) * view_scale
            pts += [sx/win_w*2-1, sy/win_h*2-1]
        verts = np.array(pts, dtype=np.float32)

        glBindBuffer(GL_ARRAY_BUFFER, self.cursor_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, verts.nbytes, verts)

        glUseProgram(self.cursor_prog)
        glBindVertexArray(self.cursor_vao)
        glDrawArrays(GL_LINE_LOOP, 0, N)
        glBindVertexArray(0)
