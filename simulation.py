import numpy as np
from OpenGL.GL import *

_SIM_SRC = """
#version 430
layout(local_size_x = 64) in;
struct Particle { float x, y, vx, vy; int color; float _p0,_p1,_p2; };
struct Rule     { float force, min_r, max_r, _pad; };
layout(std430, binding=0) buffer Particles { Particle p[]; };
layout(std430, binding=1) buffer Rules     { Rule rules[]; };
uniform int num_particles, num_colors, wrap, world_mode;
uniform float force_factor, friction_factor, repel, world_w, world_h, dt_scale, max_speed, max_accel;

float get_force(float rule, float min_r, float max_r, float dist) {
    float softened = max(dist, min_r * 0.1); // prevent force explosion at dist~0
    if (softened < min_r) return (repel/min_r)*softened - repel;
    if (softened > max_r) return 0.0;
    float mid = (min_r+max_r)*0.5, slope = rule/(mid-min_r);
    return -(slope*abs(softened-mid)) + rule;
}
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(num_particles)) return;
    float ax=0,ay=0, px=p[i].x, py=p[i].y; int ci=p[i].color;
    for (int j=0; j<num_particles; j++) {
        if (j==int(i)) continue;
        float dx=p[j].x-px, dy=p[j].y-py;
        if (world_mode==1) {
            if (dx> world_w*.5) dx-=world_w; if (dx<-world_w*.5) dx+=world_w;
            if (dy> world_h*.5) dy-=world_h; if (dy<-world_h*.5) dy+=world_h;
        }
        float dist=sqrt(dx*dx+dy*dy);
        Rule r=rules[ci*num_colors+p[j].color];
        if (dist>r.max_r) continue;
        float safe_dist = max(dist, r.min_r * 0.1);
        float f=get_force(r.force,r.min_r,r.max_r,dist);
        ax+=dx/safe_dist*f; ay+=dy/safe_dist*f;
    }
    float a_len = length(vec2(ax, ay));
    if (max_accel > 0.0 && a_len > max_accel) { ax *= max_accel/a_len; ay *= max_accel/a_len; }
    p[i].vx=p[i].vx*(1-friction_factor)+ax*force_factor;
    p[i].vy=p[i].vy*(1-friction_factor)+ay*force_factor;
    if (max_speed > 0.0) {
        float spd = length(vec2(p[i].vx, p[i].vy));
        float gamma = 1.0 / sqrt(1.0 + (spd/max_speed)*(spd/max_speed));
        p[i].vx *= gamma; p[i].vy *= gamma;
    }
    p[i].x+=p[i].vx*dt_scale; p[i].y+=p[i].vy*dt_scale;
    if (world_mode==1) {
        if (p[i].x<0) p[i].x+=world_w; if (p[i].x>world_w) p[i].x-=world_w;
        if (p[i].y<0) p[i].y+=world_h; if (p[i].y>world_h) p[i].y-=world_h;
    } else if (world_mode==0) {
        if (p[i].x<0||p[i].x>world_w){p[i].x-=p[i].vx; p[i].vx*=-1.8;}
        if (p[i].y<0||p[i].y>world_h){p[i].y-=p[i].vy; p[i].vy*=-1.8;}
    }
    // world_mode==2: infinite, no boundary
}
"""

_BRUSH_SRC = """
#version 430
layout(local_size_x = 64) in;
struct Particle { float x, y, vx, vy; int color; float _p0,_p1,_p2; };
layout(std430, binding=0) buffer Particles { Particle p[]; };
uniform int num_particles, wrap;
uniform float brush_x, brush_y, brush_r, brush_vx, brush_vy, brush_force, world_w, world_h;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(num_particles)) return;
    float dx=p[i].x-brush_x, dy=p[i].y-brush_y;
    if (wrap==1) {
        if (dx> world_w*.5) dx-=world_w; if (dx<-world_w*.5) dx+=world_w;
        if (dy> world_h*.5) dy-=world_h; if (dy<-world_h*.5) dy+=world_h;
    }
    float distSq=dx*dx+dy*dy;
    if (distSq>=brush_r*brush_r||distSq<=0) return;
    float dist=sqrt(distSq), t=1.0-smoothstep(0.0,1.0,dist/brush_r);
    p[i].vx += dx/dist*brush_force*t*500.0 + brush_vx*t*40.0;
    p[i].vy += dy/dist*brush_force*t*500.0 + brush_vy*t*40.0;
}
"""

_ERASE_SRC = """
#version 430
layout(local_size_x = 64) in;
struct Particle { float x, y, vx, vy; int color; float _p0,_p1,_p2; };
layout(std430, binding=0) buffer Particles  { Particle p[]; };
layout(std430, binding=2) buffer KeepFlags  { uint keep[]; };
layout(std430, binding=4) buffer EraseTypes { int erase_types[]; };
uniform int num_particles, wrap, num_erase_types;
uniform float brush_x, brush_y, brush_r, world_w, world_h;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(num_particles)) return;
    float dx=p[i].x-brush_x, dy=p[i].y-brush_y;
    if (wrap==1) {
        if (dx> world_w*.5) dx-=world_w; if (dx<-world_w*.5) dx+=world_w;
        if (dy> world_h*.5) dy-=world_h; if (dy<-world_h*.5) dy+=world_h;
    }
    bool in_radius = dx*dx+dy*dy < brush_r*brush_r;
    bool color_match = (num_erase_types == 0);
    for (int k=0; k<num_erase_types; k++)
        if (p[i].color == erase_types[k]) { color_match = true; break; }
    keep[i] = (in_radius && color_match) ? 0u : 1u;
}
"""


def _compile(src):
    s = glCreateShader(GL_COMPUTE_SHADER)
    glShaderSource(s, src)
    glCompileShader(s)
    if not glGetShaderiv(s, GL_COMPILE_STATUS):
        raise RuntimeError(glGetShaderInfoLog(s).decode())
    prog = glCreateProgram()
    glAttachShader(prog, s)
    glLinkProgram(prog)
    glDeleteShader(s)
    return prog


class Simulation:
    def __init__(self):
        self.num_particles  = 1500
        self.num_colors     = 6
        self.force_factor   = 1.0
        self.friction_factor= 0.3
        self.repel          = 1.0
        self.wrap           = True
        self.world_mode     = 1  # 0=bounce, 1=wrap, 2=infinite
        self.world_w        = 800.0
        self.world_h        = 600.0
        self.sim_speed      = 1.0
        self.substeps       = 1
        self.max_speed      = 0.0   # 0 = disabled
        self.max_accel      = 0.0   # 0 = disabled
        self.brush_radius   = 80.0
        self.brush_force    = 0.1
        self.brush_colors   = set()  # empty = all colors (for paint: random; for erase: all)

        self.force_matrix = self.min_r_matrix = self.max_r_matrix = None
        self._ssbo_particles = self._ssbo_rules = self._ssbo_keep = None
        self._prog_sim = self._prog_brush = self._prog_erase = None
        self._dtype = np.dtype([
            ('x',np.float32),('y',np.float32),
            ('vx',np.float32),('vy',np.float32),
            ('color',np.int32),('_pad',np.float32,3),
        ])

    _DEFAULTS = dict(
        force_factor=1.0, friction_factor=0.3, repel=1.0,
        sim_speed=1.0, substeps=1, max_speed=0.0, max_accel=0.0,
        brush_radius=80.0, brush_force=0.1, brush_color=0,
    )

    def reset_params(self):
        for k, v in self._DEFAULTS.items():
            setattr(self, k, v)

    def init_gl(self):
        self._prog_sim   = _compile(_SIM_SRC)
        self._prog_brush = _compile(_BRUSH_SRC)
        self._prog_erase = _compile(_ERASE_SRC)
        self._ssbo_particles, self._ssbo_rules, self._ssbo_keep, self._ssbo_erase_types = glGenBuffers(4)
        self.randomize_rules()
        self.reset_particles()

    def reset_particles(self):
        n = self.num_particles
        data = np.zeros(n, dtype=self._dtype)
        data['x']     = np.random.rand(n).astype(np.float32) * self.world_w
        data['y']     = np.random.rand(n).astype(np.float32) * self.world_h
        data['color'] = np.random.randint(0, self.num_colors, n).astype(np.int32)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self._ssbo_particles)
        glBufferData(GL_SHADER_STORAGE_BUFFER, data.nbytes, data, GL_DYNAMIC_DRAW)

    def _upload_rules(self):
        n = self.num_colors
        buf = np.zeros((n,n,4), dtype=np.float32)
        buf[:,:,0] = self.force_matrix
        buf[:,:,1] = self.min_r_matrix
        buf[:,:,2] = self.max_r_matrix
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self._ssbo_rules)
        glBufferData(GL_SHADER_STORAGE_BUFFER, buf.nbytes, buf, GL_DYNAMIC_DRAW)

    def randomize_rules(self):
        n = self.num_colors
        self.force_matrix = (np.random.rand(n,n)*2-1).astype(np.float32)
        self.min_r_matrix = (np.random.rand(n,n)*30+30).astype(np.float32)
        self.max_r_matrix = (np.random.rand(n,n)*60+90).astype(np.float32)
        if self._ssbo_rules is not None:
            self._upload_rules()

    def symmetric_rules(self):
        self.randomize_rules()
        for i in range(self.num_colors):
            for j in range(i+1, self.num_colors):
                self.force_matrix[j,i] = self.force_matrix[i,j]
                self.min_r_matrix[j,i] = self.min_r_matrix[i,j]
                self.max_r_matrix[j,i] = self.max_r_matrix[i,j]
        self._upload_rules()

    def _u(self, prog, name):
        return glGetUniformLocation(prog, name)

    def step(self, dt_scale=1.0):
        glUseProgram(self._prog_sim)
        u = lambda n: self._u(self._prog_sim, n)
        glUniform1i(u("num_particles"),  self.num_particles)
        glUniform1i(u("num_colors"),     self.num_colors)
        glUniform1f(u("force_factor"),   self.force_factor)
        glUniform1f(u("friction_factor"),self.friction_factor)
        glUniform1f(u("repel"),          self.repel)
        glUniform1f(u("world_w"),        self.world_w)
        glUniform1f(u("world_h"),        self.world_h)
        glUniform1i(u("wrap"),           self.world_mode)
        glUniform1i(u("world_mode"),     self.world_mode)
        glUniform1f(u("dt_scale"),       dt_scale)
        glUniform1f(u("max_speed"),      self.max_speed)
        glUniform1f(u("max_accel"),      self.max_accel)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self._ssbo_particles)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self._ssbo_rules)
        glDispatchCompute((self.num_particles+63)//64, 1, 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

    def _set_brush_uniforms(self, prog, wx, wy, vx, vy):
        u = lambda n: self._u(prog, n)
        glUniform1i(u("num_particles"), self.num_particles)
        glUniform1i(u("wrap"),          self.world_mode)
        glUniform1f(u("brush_x"),       wx)
        glUniform1f(u("brush_y"),       wy)
        glUniform1f(u("brush_r"),       self.brush_radius)
        glUniform1f(u("world_w"),       self.world_w)
        glUniform1f(u("world_h"),       self.world_h)
        if glGetUniformLocation(prog, "brush_vx") >= 0:
            glUniform1f(u("brush_vx"),      vx)
            glUniform1f(u("brush_vy"),      vy)
            glUniform1f(u("brush_force"),   self.brush_force)

    def apply_brush(self, wx, wy, vx=0.0, vy=0.0):
        """Push particles away from (wx,wy)."""
        glUseProgram(self._prog_brush)
        self._set_brush_uniforms(self._prog_brush, wx, wy, vx, vy)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self._ssbo_particles)
        glDispatchCompute((self.num_particles+63)//64, 1, 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

    def apply_eraser(self, wx, wy):
        n = self.num_particles
        keep_init = np.ones(n, dtype=np.uint32)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self._ssbo_keep)
        glBufferData(GL_SHADER_STORAGE_BUFFER, keep_init.nbytes, keep_init, GL_DYNAMIC_DRAW)

        types = np.array(sorted(self.brush_colors), dtype=np.int32) if self.brush_colors else np.zeros(0, dtype=np.int32)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self._ssbo_erase_types)
        glBufferData(GL_SHADER_STORAGE_BUFFER, max(types.nbytes, 4), types if len(types) else np.zeros(1, dtype=np.int32), GL_DYNAMIC_DRAW)

        glUseProgram(self._prog_erase)
        self._set_brush_uniforms(self._prog_erase, wx, wy, 0, 0)
        glUniform1i(glGetUniformLocation(self._prog_erase, "num_erase_types"), len(types))
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self._ssbo_particles)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, self._ssbo_keep)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, self._ssbo_erase_types)
        glDispatchCompute((n+63)//64, 1, 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self._ssbo_keep)
        flags = np.frombuffer(glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, n*4), dtype=np.uint32).copy()
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self._ssbo_particles)
        particles = np.frombuffer(
            glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, n * self._dtype.itemsize),
            dtype=self._dtype).copy()
        kept = particles[flags == 1]
        self.num_particles = len(kept)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self._ssbo_particles)
        glBufferData(GL_SHADER_STORAGE_BUFFER, kept.nbytes, kept, GL_DYNAMIC_DRAW)

    def paint_particles(self, wx, wy, count=10):
        """Add particles near (wx,wy)."""
        angles = np.random.rand(count) * 2 * np.pi
        radii  = np.random.rand(count) * self.brush_radius
        data = np.zeros(count, dtype=self._dtype)
        data['x'] = wx + np.cos(angles) * radii
        data['y'] = wy + np.sin(angles) * radii
        color_pool = list(self.brush_colors) if self.brush_colors else list(range(self.num_colors))
        data['color'] = [color_pool[np.random.randint(len(color_pool))] for _ in range(count)]

        # append to existing buffer
        n_old = self.num_particles
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self._ssbo_particles)
        old = np.frombuffer(
            glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, n_old * self._dtype.itemsize),
            dtype=self._dtype).copy()
        combined = np.concatenate([old, data])
        glBufferData(GL_SHADER_STORAGE_BUFFER, combined.nbytes, combined, GL_DYNAMIC_DRAW)
        self.num_particles = len(combined)

    def get_particle_ssbo(self):
        return self._ssbo_particles
