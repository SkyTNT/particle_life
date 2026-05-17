import numpy as np
from OpenGL.GL import *

# --- Bin count: atomically count particles per bin ---
_BIN_COUNT_SRC = """
#version 430
layout(local_size_x = 64) in;
struct Particle { float x, y, vx, vy; int color; float z, vz, _p2; };
layout(std430, binding=0) buffer Particles  { Particle p[]; };
layout(std430, binding=5) buffer BinCount   { uint bin_count[]; };
uniform int num_particles, grid_w, grid_h, grid_d, mode3d, world_mode;
uniform float cell_size;
int bin_of(Particle pt) {
    int bx = int(floor(pt.x / cell_size));
    int by = int(floor(pt.y / cell_size));
    if (world_mode == 1) {
        bx = (bx%grid_w+grid_w)%grid_w;
        by = (by%grid_h+grid_h)%grid_h;
    } else {
        bx = clamp(bx, 0, grid_w-1);
        by = clamp(by, 0, grid_h-1);
    }
    if (mode3d == 1) {
        int bz = int(floor(pt.z / cell_size));
        bz = (world_mode==1) ? (bz%grid_d+grid_d)%grid_d : clamp(bz, 0, grid_d-1);
        return bz * grid_h * grid_w + by * grid_w + bx;
    }
    return by * grid_w + bx;
}
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(num_particles)) return;
    atomicAdd(bin_count[bin_of(p[i])], 1u);
}
"""

# --- Bin sort: scatter particles into sorted buffer ---
_BIN_SORT_SRC = """
#version 430
layout(local_size_x = 64) in;
struct Particle { float x, y, vx, vy; int color; float z, vz, _p2; };
layout(std430, binding=0) buffer Particles  { Particle p[]; };
layout(std430, binding=7) buffer Sorted     { Particle sorted[]; };
layout(std430, binding=5) buffer BinCount   { uint bin_count[]; };
layout(std430, binding=6) buffer BinOffset  { uint bin_offset[]; };
uniform int num_particles, grid_w, grid_h, grid_d, mode3d, world_mode;
uniform float cell_size;
int bin_of(Particle pt) {
    int bx = int(floor(pt.x / cell_size));
    int by = int(floor(pt.y / cell_size));
    if (world_mode == 1) {
        bx = (bx%grid_w+grid_w)%grid_w;
        by = (by%grid_h+grid_h)%grid_h;
    } else {
        bx = clamp(bx, 0, grid_w-1);
        by = clamp(by, 0, grid_h-1);
    }
    if (mode3d == 1) {
        int bz = int(floor(pt.z / cell_size));
        bz = (world_mode==1) ? (bz%grid_d+grid_d)%grid_d : clamp(bz, 0, grid_d-1);
        return bz * grid_h * grid_w + by * grid_w + bx;
    }
    return by * grid_w + bx;
}
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(num_particles)) return;
    int b = bin_of(p[i]);
    uint slot = atomicAdd(bin_count[b], 1u);
    sorted[bin_offset[b] + slot] = p[i];
}
"""

# --- Main sim: force + integrate, reads from sorted buffer ---
_SIM_SRC = """
#version 430
layout(local_size_x = 64) in;
struct Particle { float x, y, vx, vy; int color; float z, vz, _p2; };
struct Rule     { float force, min_r, max_r, _pad; };
layout(std430, binding=7) buffer Sorted     { Particle sorted[]; };
layout(std430, binding=0) buffer Particles  { Particle p[]; };
layout(std430, binding=1) buffer Rules      { Rule rules[]; };
layout(std430, binding=6) buffer BinOffset  { uint bin_offset[]; };
uniform int num_particles, num_colors, world_mode, mode3d;
uniform int grid_w, grid_h, grid_d, cell_subdivisions;
uniform float force_factor, friction_factor, repel, world_w, world_h, world_d, dt_scale, max_speed, max_accel, cell_size;

float get_force(float rule, float min_r, float max_r, float dist) {
    if (dist < min_r) return (dist/min_r - 1.0) * repel;
    float mid = (min_r+max_r)*0.5, slope = rule/(mid-min_r);
    return -(slope*abs(dist-mid)) + rule;
}

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(num_particles)) return;
    Particle me = sorted[i];
    float px=me.x, py=me.y, pz=me.z; int ci=me.color;
    float ax=0, ay=0, az=0;

    bool wrap = (world_mode==1);
    int cbx = wrap ? (int(floor(px/cell_size))%grid_w+grid_w)%grid_w : clamp(int(floor(px/cell_size)), 0, grid_w-1);
    int cby = wrap ? (int(floor(py/cell_size))%grid_h+grid_h)%grid_h : clamp(int(floor(py/cell_size)), 0, grid_h-1);
    int cbz = (mode3d==1) ? (wrap ? (int(floor(pz/cell_size))%grid_d+grid_d)%grid_d : clamp(int(floor(pz/cell_size)), 0, grid_d-1)) : 0;
    int cs = cell_subdivisions;

    for (int dbz=-(mode3d==1?cs:0); dbz<=(mode3d==1?cs:0); dbz++) {
    for (int dby=-cs; dby<=cs; dby++) {
    for (int dbx=-cs; dbx<=cs; dbx++) {
        int bx = cbx+dbx, by = cby+dby, bz = cbz+dbz;
        if (wrap) {
            if (bx < 0) bx += grid_w; else if (bx >= grid_w) bx -= grid_w;
            if (by < 0) by += grid_h; else if (by >= grid_h) by -= grid_h;
            if (mode3d==1) { if (bz < 0) bz += grid_d; else if (bz >= grid_d) bz -= grid_d; }
        } else {
            if (bx<0||bx>=grid_w||by<0||by>=grid_h) continue;
            if (mode3d==1&&(bz<0||bz>=grid_d)) continue;
        }
        int b = bz*grid_h*grid_w + by*grid_w + bx;
        uint jstart = bin_offset[b], jend = bin_offset[b+1];
        for (uint j=jstart; j<jend; j++) {
            if (j==i) continue;
            Particle o = sorted[j];
            float dx=o.x-px, dy=o.y-py, dz=(mode3d==1)?(o.z-pz):0.0;
            if (world_mode==1) {
                if (dx> world_w*.5) dx-=world_w; if (dx<-world_w*.5) dx+=world_w;
                if (dy> world_h*.5) dy-=world_h; if (dy<-world_h*.5) dy+=world_h;
                if (mode3d==1){if(dz>world_d*.5)dz-=world_d;if(dz<-world_d*.5)dz+=world_d;}
            }
            float dist2=dx*dx+dy*dy+dz*dz;
            if (dist2 < 0.0001) continue;
            Rule r=rules[ci*num_colors+o.color];
            if (dist2 > r.max_r*r.max_r) continue;
            float dist=sqrt(dist2);
            float f=get_force(r.force,r.min_r,r.max_r,dist);
            float inv=f/dist;
            ax+=dx*inv; ay+=dy*inv; az+=dz*inv;
        }
    }}}

    float a_len = length(vec3(ax,ay,az));
    if (max_accel>0.0 && a_len>max_accel){float s=max_accel/a_len;ax*=s;ay*=s;az*=s;}
    me.vx=me.vx*(1-friction_factor)+ax*force_factor;
    me.vy=me.vy*(1-friction_factor)+ay*force_factor;
    if (mode3d==1) me.vz=me.vz*(1-friction_factor)+az*force_factor;
    if (max_speed>0.0){
        float spd=length(vec3(me.vx,me.vy,me.vz));
        float g=1.0/sqrt(1.0+(spd/max_speed)*(spd/max_speed));
        me.vx*=g; me.vy*=g; me.vz*=g;
    }
    me.x+=me.vx*dt_scale; me.y+=me.vy*dt_scale;
    if (mode3d==1) me.z+=me.vz*dt_scale;
    if (world_mode==1) {
        if (me.x < 0.0) me.x += world_w; else if (me.x >= world_w) me.x -= world_w;
        if (me.y < 0.0) me.y += world_h; else if (me.y >= world_h) me.y -= world_h;
        if (mode3d==1) { if (me.z < 0.0) me.z += world_d; else if (me.z >= world_d) me.z -= world_d; }
    } else if (world_mode==0) {
        if (me.x < 0.0) { me.x = 0.0; me.vx = abs(me.vx); }
        else if (me.x > world_w) { me.x = world_w; me.vx = -abs(me.vx); }
        if (me.y < 0.0) { me.y = 0.0; me.vy = abs(me.vy); }
        else if (me.y > world_h) { me.y = world_h; me.vy = -abs(me.vy); }
        if (mode3d==1) {
            if (me.z < 0.0) { me.z = 0.0; me.vz = abs(me.vz); }
            else if (me.z > world_d) { me.z = world_d; me.vz = -abs(me.vz); }
        }
    }
    sorted[i] = me;
}
"""

_BRUSH_SRC = """
#version 430
layout(local_size_x = 64) in;
struct Particle { float x, y, vx, vy; int color; float z, vz, _p2; };
layout(std430, binding=0) buffer Particles { Particle p[]; };
uniform int num_particles, wrap, mode3d;
uniform float brush_x, brush_y, brush_z, brush_r, brush_vx, brush_vy, brush_force, world_w, world_h;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(num_particles)) return;
    float dx=p[i].x-brush_x, dy=p[i].y-brush_y, dz=(mode3d==1)?(p[i].z-brush_z):0.0;
    if (wrap==1) {
        if (dx> world_w*.5) dx-=world_w; if (dx<-world_w*.5) dx+=world_w;
        if (dy> world_h*.5) dy-=world_h; if (dy<-world_h*.5) dy+=world_h;
    }
    float distSq=dx*dx+dy*dy+dz*dz;
    if (distSq>=brush_r*brush_r||distSq<=0) return;
    float dist=sqrt(distSq), t=1.0-smoothstep(0.0,1.0,dist/brush_r);
    p[i].vx += dx/dist*brush_force*t*500.0 + brush_vx*t*40.0;
    p[i].vy += dy/dist*brush_force*t*500.0 + brush_vy*t*40.0;
    if (mode3d==1) p[i].vz += dz/dist*brush_force*t*500.0;
}
"""

_ERASE_SRC = """
#version 430
layout(local_size_x = 64) in;
struct Particle { float x, y, vx, vy; int color; float z, vz, _p2; };
layout(std430, binding=0) buffer Particles  { Particle p[]; };
layout(std430, binding=2) buffer KeepFlags  { uint keep[]; };
layout(std430, binding=4) buffer EraseTypes { int erase_types[]; };
uniform int num_particles, wrap, num_erase_types, mode3d;
uniform float brush_x, brush_y, brush_z, brush_r, world_w, world_h;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(num_particles)) return;
    float dx=p[i].x-brush_x, dy=p[i].y-brush_y, dz=(mode3d==1)?(p[i].z-brush_z):0.0;
    if (wrap==1) {
        if (dx> world_w*.5) dx-=world_w; if (dx<-world_w*.5) dx+=world_w;
        if (dy> world_h*.5) dy-=world_h; if (dy<-world_h*.5) dy+=world_h;
    }
    bool in_radius = dx*dx+dy*dy+dz*dz < brush_r*brush_r;
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
        self.num_particles  = 50000
        self.num_colors     = 6
        self.force_factor   = 1.0
        self.friction_factor= 0.3
        self.repel          = 1.0
        self.wrap           = True
        self.world_mode     = 1  # 0=bounce, 1=wrap, 2=infinite
        self.world_w        = 800.0
        self.world_h        = 600.0
        self.world_d        = 800.0
        self.sim_speed      = 1.0
        self.substeps       = 1
        self.cell_subdivisions = 1
        self.max_speed      = 0.0   # 0 = disabled
        self.max_accel      = 0.0   # 0 = disabled
        self.brush_radius   = 80.0
        self.brush_force    = 0.1
        self.brush_colors   = set()  # empty = all colors (for paint: random; for erase: all)

        self.rand_force_range = [-1.0, 1.0]
        self.rand_min_r_range = [12.0, 24.0]
        self.rand_max_r_range = [32.0, 64.0]
        self._ssbo_particles = self._ssbo_rules = self._ssbo_keep = None
        self._ssbo_bin_count = self._ssbo_bin_offset = self._ssbo_sorted = None
        self._prog_sim = self._prog_brush = self._prog_erase = None
        self._prog_bin_count = self._prog_bin_sort = None
        self._grid_w = self._grid_h = self._grid_d = 1
        self._cell_size = 150.0
        self.mode3d = False
        self._dtype = np.dtype([
            ('x',np.float32),('y',np.float32),
            ('vx',np.float32),('vy',np.float32),
            ('color',np.int32),('z',np.float32),('vz',np.float32),('_p2',np.float32),
        ])

    _DEFAULTS = dict(
        force_factor=1.0, friction_factor=0.3, repel=1.0,
        sim_speed=1.0, substeps=1, max_speed=0.0, max_accel=0.0,
        brush_radius=80.0, brush_force=0.1, brush_color=0,
    )

    def reset_params(self):
        for k, v in self._DEFAULTS.items():
            setattr(self, k, v)

    def _update_grid(self):
        max_r = float(np.max(self.max_r_matrix)) if hasattr(self, 'max_r_matrix') else self.rand_max_r_range[1]
        self._cell_size = max(max_r, 1.0)
        self._grid_w = max(1, int(np.ceil(self.world_w / self._cell_size)))
        self._grid_h = max(1, int(np.ceil(self.world_h / self._cell_size)))
        self._grid_d = max(1, int(np.ceil(self.world_d / self._cell_size))) if self.mode3d else 1
        num_bins = self._grid_w * self._grid_h * self._grid_d + 1
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self._ssbo_bin_count)
        glBufferData(GL_SHADER_STORAGE_BUFFER, num_bins * 4, None, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self._ssbo_bin_offset)
        glBufferData(GL_SHADER_STORAGE_BUFFER, num_bins * 4, None, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self._ssbo_sorted)
        glBufferData(GL_SHADER_STORAGE_BUFFER, self.num_particles * self._dtype.itemsize, None, GL_DYNAMIC_DRAW)

    def init_gl(self):
        self._prog_sim       = _compile(_SIM_SRC)
        self._prog_bin_count = _compile(_BIN_COUNT_SRC)
        self._prog_bin_sort  = _compile(_BIN_SORT_SRC)
        self._prog_brush     = _compile(_BRUSH_SRC)
        self._prog_erase     = _compile(_ERASE_SRC)
        (self._ssbo_particles, self._ssbo_rules, self._ssbo_keep,
         self._ssbo_erase_types, self._ssbo_bin_count,
         self._ssbo_bin_offset, self._ssbo_sorted) = glGenBuffers(7)
        def _locs(prog, names):
            return {n: glGetUniformLocation(prog, n) for n in names}
        _grid_unis = ["num_particles","grid_w","grid_h","grid_d","mode3d","world_mode","cell_size"]
        self._uloc_bin_count = _locs(self._prog_bin_count, _grid_unis)
        self._uloc_bin_sort  = _locs(self._prog_bin_sort,  _grid_unis)
        self._uloc_sim = _locs(self._prog_sim, [
            "num_particles","num_colors","force_factor","friction_factor","repel",
            "world_w","world_h","world_d","world_mode","mode3d","dt_scale",
            "max_speed","max_accel","cell_size","grid_w","grid_h","grid_d","cell_subdivisions"])
        self._uloc_brush = _locs(self._prog_brush, [
            "num_particles","wrap","mode3d","brush_x","brush_y","brush_z",
            "brush_r","world_w","world_h","brush_vx","brush_vy","brush_force"])
        self._uloc_erase = _locs(self._prog_erase, [
            "num_particles","wrap","mode3d","brush_x","brush_y","brush_z",
            "brush_r","world_w","world_h","num_erase_types"])
        self.randomize_rules()
        self.reset_particles()

    def reset_particles(self):
        n = self.num_particles
        data = np.zeros(n, dtype=self._dtype)
        data['x']     = np.random.rand(n).astype(np.float32) * self.world_w
        data['y']     = np.random.rand(n).astype(np.float32) * self.world_h
        data['z']     = np.random.rand(n).astype(np.float32) * self.world_d if self.mode3d else np.zeros(n, dtype=np.float32)
        data['color'] = np.random.randint(0, self.num_colors, n).astype(np.int32)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self._ssbo_particles)
        glBufferData(GL_SHADER_STORAGE_BUFFER, data.nbytes, data, GL_DYNAMIC_DRAW)
        self._update_grid()

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
        lo, hi = self.rand_force_range
        self.force_matrix = (np.random.rand(n,n)*(hi-lo)+lo).astype(np.float32)
        lo, hi = self.rand_min_r_range
        self.min_r_matrix = (np.random.rand(n,n)*(hi-lo)+lo).astype(np.float32)
        lo, hi = self.rand_max_r_range
        self.max_r_matrix = (np.random.rand(n,n)*(hi-lo)+lo).astype(np.float32)
        if self._ssbo_rules is not None:
            self._upload_rules()
            if self._ssbo_bin_count is not None:
                self._update_grid()

    def symmetric_rules(self):
        self.randomize_rules()
        for i in range(self.num_colors):
            for j in range(i+1, self.num_colors):
                self.force_matrix[j,i] = self.force_matrix[i,j]
                self.min_r_matrix[j,i] = self.min_r_matrix[i,j]
                self.max_r_matrix[j,i] = self.max_r_matrix[i,j]
        self._upload_rules()

    def _bin_sort_pass(self):
        n = self.num_particles
        gw, gh, gd = self._grid_w, self._grid_h, self._grid_d
        num_bins = gw * gh * gd
        cs = self._cell_size
        mode3d_i = 1 if self.mode3d else 0
        groups = (n + 63) // 64

        # 1. clear bin_count
        zeros = np.zeros(num_bins + 1, dtype=np.uint32)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self._ssbo_bin_count)
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, zeros.nbytes, zeros)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        # 2. count particles per bin
        glUseProgram(self._prog_bin_count)
        u = self._uloc_bin_count
        glUniform1i(u["num_particles"], n)
        glUniform1i(u["grid_w"], gw); glUniform1i(u["grid_h"], gh); glUniform1i(u["grid_d"], gd)
        glUniform1i(u["mode3d"], mode3d_i); glUniform1i(u["world_mode"], self.world_mode)
        glUniform1f(u["cell_size"], cs)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self._ssbo_particles)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, self._ssbo_bin_count)
        glDispatchCompute(groups, 1, 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        # 3. CPU prefix sum (exclusive scan)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self._ssbo_bin_count)
        counts = np.frombuffer(
            glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, (num_bins + 1) * 4),
            dtype=np.uint32).copy()
        offsets = np.zeros(num_bins + 1, dtype=np.uint32)
        offsets[1:] = np.cumsum(counts[:num_bins])
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self._ssbo_bin_offset)
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, offsets.nbytes, offsets)

        # reset bin_count to 0 for scatter
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self._ssbo_bin_count)
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, zeros.nbytes, zeros)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        # 4. scatter into sorted buffer
        glUseProgram(self._prog_bin_sort)
        u = self._uloc_bin_sort
        glUniform1i(u["num_particles"], n)
        glUniform1i(u["grid_w"], gw); glUniform1i(u["grid_h"], gh); glUniform1i(u["grid_d"], gd)
        glUniform1i(u["mode3d"], mode3d_i); glUniform1i(u["world_mode"], self.world_mode)
        glUniform1f(u["cell_size"], cs)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self._ssbo_particles)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, self._ssbo_sorted)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, self._ssbo_bin_count)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, self._ssbo_bin_offset)
        glDispatchCompute(groups, 1, 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

    def step_multi(self, dt_scale, substeps):
        n = self.num_particles
        groups = (n + 63) // 64
        mode3d_i = 1 if self.mode3d else 0
        for _ in range(substeps):
            self._bin_sort_pass()
            glUseProgram(self._prog_sim)
            u = self._uloc_sim
            glUniform1i(u["num_particles"],   n)
            glUniform1i(u["num_colors"],      self.num_colors)
            glUniform1f(u["force_factor"],    self.force_factor)
            glUniform1f(u["friction_factor"], self.friction_factor)
            glUniform1f(u["repel"],           self.repel)
            glUniform1f(u["world_w"],         self.world_w)
            glUniform1f(u["world_h"],         self.world_h)
            glUniform1f(u["world_d"],         self.world_d)
            glUniform1i(u["world_mode"],      self.world_mode)
            glUniform1i(u["mode3d"],          mode3d_i)
            glUniform1f(u["dt_scale"],        dt_scale)
            glUniform1f(u["max_speed"],       self.max_speed)
            glUniform1f(u["max_accel"],       self.max_accel)
            glUniform1f(u["cell_size"],       self._cell_size)
            glUniform1i(u["grid_w"],          self._grid_w)
            glUniform1i(u["grid_h"],          self._grid_h)
            glUniform1i(u["grid_d"],          self._grid_d)
            glUniform1i(u["cell_subdivisions"], self.cell_subdivisions)
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, self._ssbo_sorted)
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self._ssbo_particles)
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self._ssbo_rules)
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, self._ssbo_bin_offset)
            glDispatchCompute(groups, 1, 1)
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
        # copy sorted -> particles so brush/erase/paint see updated positions
        glBindBuffer(GL_COPY_READ_BUFFER, self._ssbo_sorted)
        glBindBuffer(GL_COPY_WRITE_BUFFER, self._ssbo_particles)
        glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0,
                            n * self._dtype.itemsize)

    def _set_brush_uniforms(self, prog, uloc, wx, wy, vx, vy, wz=0.0):
        u = uloc
        glUniform1i(u["num_particles"], self.num_particles)
        glUniform1i(u["wrap"],          self.world_mode)
        glUniform1i(u["mode3d"],        1 if self.mode3d else 0)
        glUniform1f(u["brush_x"],       wx)
        glUniform1f(u["brush_y"],       wy)
        glUniform1f(u["brush_z"],       wz)
        glUniform1f(u["brush_r"],       self.brush_radius)
        glUniform1f(u["world_w"],       self.world_w)
        glUniform1f(u["world_h"],       self.world_h)
        if u.get("brush_vx", -1) >= 0:
            glUniform1f(u["brush_vx"],    vx)
            glUniform1f(u["brush_vy"],    vy)
            glUniform1f(u["brush_force"], self.brush_force)

    def apply_brush(self, wx, wy, vx=0.0, vy=0.0, wz=0.0):
        glUseProgram(self._prog_brush)
        self._set_brush_uniforms(self._prog_brush, self._uloc_brush, wx, wy, vx, vy, wz)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self._ssbo_particles)
        glDispatchCompute((self.num_particles+63)//64, 1, 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

    def apply_eraser(self, wx, wy, wz=0.0):
        n = self.num_particles
        keep_init = np.ones(n, dtype=np.uint32)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self._ssbo_keep)
        glBufferData(GL_SHADER_STORAGE_BUFFER, keep_init.nbytes, keep_init, GL_DYNAMIC_DRAW)

        types = np.array(sorted(self.brush_colors), dtype=np.int32) if self.brush_colors else np.zeros(0, dtype=np.int32)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self._ssbo_erase_types)
        glBufferData(GL_SHADER_STORAGE_BUFFER, max(types.nbytes, 4), types if len(types) else np.zeros(1, dtype=np.int32), GL_DYNAMIC_DRAW)

        glUseProgram(self._prog_erase)
        self._set_brush_uniforms(self._prog_erase, self._uloc_erase, wx, wy, 0, 0, wz)
        glUniform1i(self._uloc_erase["num_erase_types"], len(types))
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
        self._update_grid()

    def paint_particles(self, wx, wy, wz=0.0, count=10):
        angles = np.random.rand(count) * 2 * np.pi
        radii  = np.random.rand(count) * self.brush_radius
        data = np.zeros(count, dtype=self._dtype)
        data['x'] = wx + np.cos(angles) * radii
        data['y'] = wy + np.sin(angles) * radii
        data['z'] = wz + (np.random.rand(count) - 0.5) * self.brush_radius * 2 if self.mode3d else 0.0
        color_pool = np.array(list(self.brush_colors) if self.brush_colors else list(range(self.num_colors)), dtype=np.int32)
        data['color'] = color_pool[np.random.randint(len(color_pool), size=count)]

        n_old = self.num_particles
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self._ssbo_particles)
        old = np.frombuffer(
            glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, n_old * self._dtype.itemsize),
            dtype=self._dtype).copy()
        combined = np.concatenate([old, data])
        glBufferData(GL_SHADER_STORAGE_BUFFER, combined.nbytes, combined, GL_DYNAMIC_DRAW)
        self.num_particles = len(combined)
        self._update_grid()

    def get_particle_ssbo(self):
        return self._ssbo_sorted
