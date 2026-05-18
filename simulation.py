import numpy as np
from OpenGL.GL import *


# Particle layout matches renderer (32 bytes, std430 friendly).
_COMMON_HEAD = """
#version 430
struct Particle { float x, y, vx, vy; int color; float z, vz, _p2; };
struct Rule     { float force, min_r, max_r, _pad; };
"""

# --- Spatial-binning pipeline shaders -----------------------------------------

_BIN_CLEAR_SRC = _COMMON_HEAD + """
layout(local_size_x = 64) in;
layout(std430, binding=6) buffer BinSize { uint binSize[]; };
uniform int num_bins;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(num_bins)) return;
    binSize[i] = 0u;
}
"""

_BIN_FILL_SRC = _COMMON_HEAD + """
layout(local_size_x = 64) in;
layout(std430, binding=0) buffer Particles { Particle p[]; };
layout(std430, binding=6) buffer BinSize   { uint binSize[]; };
uniform int num_particles, mode3d;
uniform int gridW, gridH, gridD;
uniform float invCellX, invCellY, invCellZ;
// Robust positive-result modulo: GLSL int `%` is implementation-defined for
// negative dividends (e.g. NVIDIA returns 3 for -1 % 12 instead of -1).
int wrapBin(int v, int n) {
    return v - int(floor(float(v) / float(n))) * n;
}
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(num_particles)) return;
    int bx = int(floor(p[i].x * invCellX));
    int by = int(floor(p[i].y * invCellY));
    int bz = (mode3d == 1) ? int(floor(p[i].z * invCellZ)) : 0;
    bx = wrapBin(bx, gridW);
    by = wrapBin(by, gridH);
    bz = (mode3d == 1) ? wrapBin(bz, gridD) : 0;
    int binIdx = (bz * gridH + by) * gridW + bx;
    atomicAdd(binSize[binIdx + 1], 1u);
}
"""

# Hillis-Steele prefix sum, ping-ponged via Src/Dst bound on bindings 6/7.
_PREFIX_SUM_SRC = """
#version 430
layout(local_size_x = 64) in;
layout(std430, binding=6) buffer Src { uint src[]; };
layout(std430, binding=7) buffer Dst { uint dst[]; };
uniform int step_size, count;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(count)) return;
    dst[i] = (i < uint(step_size)) ? src[i] : src[i - uint(step_size)] + src[i];
}
"""

_SORT_SRC = _COMMON_HEAD + """
layout(local_size_x = 64) in;
layout(std430, binding=0) buffer SrcP    { Particle src[]; };
layout(std430, binding=5) buffer DstP    { Particle dst[]; };
layout(std430, binding=6) buffer BinSize { uint binSize[]; };
layout(std430, binding=7) buffer BinOff  { uint binOffset[]; };
uniform int num_particles, mode3d;
uniform int gridW, gridH, gridD;
uniform float invCellX, invCellY, invCellZ;
int wrapBin(int v, int n) {
    return v - int(floor(float(v) / float(n))) * n;
}
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(num_particles)) return;
    Particle pt = src[i];
    int bx = int(floor(pt.x * invCellX));
    int by = int(floor(pt.y * invCellY));
    int bz = (mode3d == 1) ? int(floor(pt.z * invCellZ)) : 0;
    bx = wrapBin(bx, gridW);
    by = wrapBin(by, gridH);
    bz = (mode3d == 1) ? wrapBin(bz, gridD) : 0;
    int binIdx = (bz * gridH + by) * gridW + bx;
    uint write = binOffset[binIdx] + atomicAdd(binSize[binIdx], 1u);
    dst[write] = pt;
}
"""

# Force kernel: read sorted src, iterate 3x3x3 neighbor bins, write velocity-updated dst.
_FORCE_SRC = _COMMON_HEAD + """
layout(local_size_x = 64) in;
layout(std430, binding=0) buffer SrcP   { Particle src[]; };
layout(std430, binding=5) buffer DstP   { Particle dst[]; };
layout(std430, binding=1) buffer Rules  { Rule rules[]; };
layout(std430, binding=7) buffer BinOff { uint binOffset[]; };
uniform int num_particles, num_colors, world_mode, mode3d;
uniform int gridW, gridH, gridD;
uniform float invCellX, invCellY, invCellZ;
uniform float world_w, world_h, world_d;
uniform float force_factor, friction_factor, repel, max_accel;
int wrapBin(int v, int n) {
    return v - int(floor(float(v) / float(n))) * n;
}
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(num_particles)) return;
    Particle pt = src[i];
    float px = pt.x, py = pt.y, pz = pt.z;
    int   myOff = pt.color * num_colors;
    bool  wrap  = (world_mode == 1);
    float halfW = world_w * 0.5, halfH = world_h * 0.5, halfD = world_d * 0.5;

    int bx = wrapBin(int(floor(px * invCellX)), gridW);
    int by = wrapBin(int(floor(py * invCellY)), gridH);
    int bz = (mode3d == 1) ? wrapBin(int(floor(pz * invCellZ)), gridD) : 0;

    int dzMin = (mode3d == 1) ? -1 : 0;
    int dzMax = (mode3d == 1) ?  1 : 0;

    float ax = 0.0, ay = 0.0, az = 0.0;
    for (int dz = dzMin; dz <= dzMax; ++dz) {
        int rbz = bz + dz;
        if (rbz < 0) rbz += gridD; else if (rbz >= gridD) rbz -= gridD;
        for (int dy = -1; dy <= 1; ++dy) {
            int rby = by + dy;
            if (rby < 0) rby += gridH; else if (rby >= gridH) rby -= gridH;
            for (int dx = -1; dx <= 1; ++dx) {
                int rbx = bx + dx;
                if (rbx < 0) rbx += gridW; else if (rbx >= gridW) rbx -= gridW;
                int binIdx = (rbz * gridH + rby) * gridW + rbx;
                uint start = binOffset[binIdx];
                uint end   = binOffset[binIdx + 1];
                for (uint j = start; j < end; ++j) {
                    if (j == i) continue;
                    Particle q = src[j];
                    float rx = q.x - px;
                    float ry = q.y - py;
                    float rz = (mode3d == 1) ? (q.z - pz) : 0.0;
                    if (wrap) {
                        if (rx >  halfW) rx -= world_w; else if (rx < -halfW) rx += world_w;
                        if (ry >  halfH) ry -= world_h; else if (ry < -halfH) ry += world_h;
                        if (mode3d == 1) {
                            if (rz >  halfD) rz -= world_d; else if (rz < -halfD) rz += world_d;
                        }
                    }
                    float distSq = rx*rx + ry*ry + rz*rz;
                    Rule r = rules[myOff + q.color];
                    if (distSq >= r.max_r * r.max_r || distSq < 1e-12) continue;
                    float dist = sqrt(distSq);
                    float safe = max(dist, r.min_r * 0.1);
                    float f;
                    if (safe < r.min_r) {
                        f = (repel / r.min_r) * safe - repel;
                    } else {
                        float mid   = (r.min_r + r.max_r) * 0.5;
                        float slope = r.force / (mid - r.min_r);
                        f = -(slope * abs(safe - mid)) + r.force;
                    }
                    float invD = 1.0 / safe;
                    ax += rx * invD * f;
                    ay += ry * invD * f;
                    az += rz * invD * f;
                }
            }
        }
    }
    float aLen = length(vec3(ax, ay, az));
    if (max_accel > 0.0 && aLen > max_accel) {
        float s = max_accel / aLen; ax *= s; ay *= s; az *= s;
    }
    pt.vx = pt.vx * (1.0 - friction_factor) + ax * force_factor;
    pt.vy = pt.vy * (1.0 - friction_factor) + ay * force_factor;
    if (mode3d == 1) pt.vz = pt.vz * (1.0 - friction_factor) + az * force_factor;
    dst[i] = pt;
}
"""

_ADVANCE_SRC = _COMMON_HEAD + """
layout(local_size_x = 64) in;
layout(std430, binding=0) buffer Particles { Particle p[]; };
uniform int num_particles, world_mode, mode3d;
uniform float world_w, world_h, world_d, dt_scale, max_speed;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(num_particles)) return;
    if (max_speed > 0.0) {
        float spd = length(vec3(p[i].vx, p[i].vy, p[i].vz));
        float gamma = 1.0 / sqrt(1.0 + (spd/max_speed)*(spd/max_speed));
        p[i].vx *= gamma; p[i].vy *= gamma;
        if (mode3d == 1) p[i].vz *= gamma;
    }
    p[i].x += p[i].vx * dt_scale;
    p[i].y += p[i].vy * dt_scale;
    if (mode3d == 1) p[i].z += p[i].vz * dt_scale;
    if (world_mode == 1) {
        if (p[i].x < 0.0)         p[i].x += world_w;
        else if (p[i].x >= world_w) p[i].x -= world_w;
        if (p[i].y < 0.0)         p[i].y += world_h;
        else if (p[i].y >= world_h) p[i].y -= world_h;
        if (mode3d == 1) {
            if (p[i].z < 0.0)         p[i].z += world_d;
            else if (p[i].z >= world_d) p[i].z -= world_d;
        }
    } else if (world_mode == 0) {
        if (p[i].x < 0.0 || p[i].x > world_w) { p[i].x -= p[i].vx * dt_scale; p[i].vx *= -1.8; }
        if (p[i].y < 0.0 || p[i].y > world_h) { p[i].y -= p[i].vy * dt_scale; p[i].vy *= -1.8; }
        if (mode3d == 1 && (p[i].z < 0.0 || p[i].z > world_d)) {
            p[i].z -= p[i].vz * dt_scale; p[i].vz *= -1.8;
        }
    }
}
"""

# --- Brush / erase shaders (unchanged behavior) -------------------------------

_BRUSH_SRC = _COMMON_HEAD + """
layout(local_size_x = 64) in;
layout(std430, binding=0) buffer Particles { Particle p[]; };
uniform int num_particles, wrap, mode3d;
uniform float brush_x, brush_y, brush_z, brush_r, brush_vx, brush_vy, brush_force, world_w, world_h;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(num_particles)) return;
    float dx = p[i].x - brush_x, dy = p[i].y - brush_y, dz = (mode3d==1) ? (p[i].z - brush_z) : 0.0;
    if (wrap == 1) {
        if (dx >  world_w*0.5) dx -= world_w; if (dx < -world_w*0.5) dx += world_w;
        if (dy >  world_h*0.5) dy -= world_h; if (dy < -world_h*0.5) dy += world_h;
    }
    float distSq = dx*dx + dy*dy + dz*dz;
    if (distSq >= brush_r*brush_r || distSq <= 0.0) return;
    float dist = sqrt(distSq);
    float t = 1.0 - smoothstep(0.0, 1.0, dist/brush_r);
    p[i].vx += dx/dist*brush_force*t*500.0 + brush_vx*t*40.0;
    p[i].vy += dy/dist*brush_force*t*500.0 + brush_vy*t*40.0;
    if (mode3d == 1) p[i].vz += dz/dist*brush_force*t*500.0;
}
"""

_ERASE_SRC = _COMMON_HEAD + """
layout(local_size_x = 64) in;
layout(std430, binding=0) buffer Particles  { Particle p[]; };
layout(std430, binding=2) buffer KeepFlags  { uint keep[]; };
layout(std430, binding=4) buffer EraseTypes { int erase_types[]; };
uniform int num_particles, wrap, num_erase_types, mode3d;
uniform float brush_x, brush_y, brush_z, brush_r, world_w, world_h;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(num_particles)) return;
    float dx = p[i].x - brush_x, dy = p[i].y - brush_y, dz = (mode3d==1) ? (p[i].z - brush_z) : 0.0;
    if (wrap == 1) {
        if (dx >  world_w*0.5) dx -= world_w; if (dx < -world_w*0.5) dx += world_w;
        if (dy >  world_h*0.5) dy -= world_h; if (dy < -world_h*0.5) dy += world_h;
    }
    bool in_radius = dx*dx + dy*dy + dz*dz < brush_r*brush_r;
    bool color_match = (num_erase_types == 0);
    for (int k = 0; k < num_erase_types; ++k)
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
    if not glGetProgramiv(prog, GL_LINK_STATUS):
        raise RuntimeError(glGetProgramInfoLog(prog).decode())
    return prog


def _locs(prog, names):
    return {n: glGetUniformLocation(prog, n) for n in names}


class Simulation:
    def __init__(self):
        self.num_particles   = 50000
        self.num_colors      = 6
        self.force_factor    = 1.0
        self.friction_factor = 0.3
        self.repel           = 1.0
        self.wrap            = True
        self.world_mode      = 1   # 0=bounce, 1=wrap, 2=infinite
        self.world_w         = 800.0
        self.world_h         = 600.0
        self.world_d         = 1600.0
        self.sim_speed       = 1.0
        self.substeps        = 1
        self.max_speed       = 0.0
        self.max_accel       = 0.0
        self.brush_radius    = 80.0
        self.brush_force     = 0.1
        self.brush_colors    = set()

        self.rand_force_range = [-1.0, 1.0]
        self.rand_min_r_range = [12.0, 24.0]
        self.rand_max_r_range = [32.0, 64.0]

        self.mode3d = False
        self._dtype = np.dtype([
            ('x', np.float32),('y', np.float32),
            ('vx',np.float32),('vy',np.float32),
            ('color',np.int32),('z',np.float32),('vz',np.float32),('_p2',np.float32),
        ])

        # GL objects
        self._buf_a = self._buf_b = None
        self._ssbo_rules = self._ssbo_keep = self._ssbo_erase_types = None
        self._ssbo_bin_size = self._ssbo_bin_off = None
        self._a_current = True
        self._allocated = 0  # capacity of particle buffers

        # Bin geometry tracking (sentinel values force first-time allocation)
        self._cell_x = self._cell_y = self._cell_z = -1.0
        self._grid_w = self._grid_h = self._grid_d = -1
        self._bin_count = 0
        self._bin_capacity = 0   # currently allocated bin count

        # programs
        self._prog_bin_clear = self._prog_bin_fill = self._prog_prefix = None
        self._prog_sort = self._prog_force = self._prog_advance = None
        self._prog_brush = self._prog_erase = None

    _DEFAULTS = dict(
        force_factor=1.0, friction_factor=0.3, repel=1.0,
        sim_speed=1.0, substeps=1, max_speed=0.0, max_accel=0.0,
        brush_radius=80.0, brush_force=0.1, brush_color=0,
    )

    def reset_params(self):
        for k, v in self._DEFAULTS.items():
            setattr(self, k, v)

    # -------------------------------------------------------------------------
    # GL setup
    # -------------------------------------------------------------------------
    def init_gl(self):
        self._prog_bin_clear = _compile(_BIN_CLEAR_SRC)
        self._prog_bin_fill  = _compile(_BIN_FILL_SRC)
        self._prog_prefix    = _compile(_PREFIX_SUM_SRC)
        self._prog_sort      = _compile(_SORT_SRC)
        self._prog_force     = _compile(_FORCE_SRC)
        self._prog_advance   = _compile(_ADVANCE_SRC)
        self._prog_brush     = _compile(_BRUSH_SRC)
        self._prog_erase     = _compile(_ERASE_SRC)

        bufs = glGenBuffers(7)
        (self._buf_a, self._buf_b, self._ssbo_rules,
         self._ssbo_keep, self._ssbo_erase_types,
         self._ssbo_bin_size, self._ssbo_bin_off) = bufs

        self._uloc_bin_clear = _locs(self._prog_bin_clear, ["num_bins"])
        self._uloc_bin_fill  = _locs(self._prog_bin_fill, [
            "num_particles","mode3d","gridW","gridH","gridD",
            "invCellX","invCellY","invCellZ"])
        self._uloc_prefix    = _locs(self._prog_prefix, ["step_size","count"])
        self._uloc_sort      = _locs(self._prog_sort, [
            "num_particles","mode3d","gridW","gridH","gridD",
            "invCellX","invCellY","invCellZ"])
        self._uloc_force     = _locs(self._prog_force, [
            "num_particles","num_colors","world_mode","mode3d","gridW","gridH","gridD",
            "invCellX","invCellY","invCellZ","world_w","world_h","world_d",
            "force_factor","friction_factor","repel","max_accel"])
        self._uloc_advance   = _locs(self._prog_advance, [
            "num_particles","world_mode","mode3d","world_w","world_h","world_d",
            "dt_scale","max_speed"])
        self._uloc_brush = _locs(self._prog_brush, [
            "num_particles","wrap","mode3d","brush_x","brush_y","brush_z",
            "brush_r","world_w","world_h","brush_vx","brush_vy","brush_force"])
        self._uloc_erase = _locs(self._prog_erase, [
            "num_particles","wrap","mode3d","brush_x","brush_y","brush_z",
            "brush_r","world_w","world_h","num_erase_types"])

        self.randomize_rules()
        self.reset_particles()

    # -------------------------------------------------------------------------
    # Rules / cell-size
    # -------------------------------------------------------------------------
    def _upload_rules(self):
        n = self.num_colors
        buf = np.zeros((n, n, 4), dtype=np.float32)
        buf[:, :, 0] = self.force_matrix
        buf[:, :, 1] = self.min_r_matrix
        buf[:, :, 2] = self.max_r_matrix
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self._ssbo_rules)
        glBufferData(GL_SHADER_STORAGE_BUFFER, buf.nbytes, buf, GL_DYNAMIC_DRAW)

    def randomize_rules(self):
        n = self.num_colors
        lo, hi = self.rand_force_range
        self.force_matrix = (np.random.rand(n, n) * (hi - lo) + lo).astype(np.float32)
        lo, hi = self.rand_min_r_range
        self.min_r_matrix = (np.random.rand(n, n) * (hi - lo) + lo).astype(np.float32)
        lo, hi = self.rand_max_r_range
        self.max_r_matrix = (np.random.rand(n, n) * (hi - lo) + lo).astype(np.float32)
        if self._ssbo_rules is not None:
            self._upload_rules()

    def symmetric_rules(self):
        self.randomize_rules()
        for i in range(self.num_colors):
            for j in range(i + 1, self.num_colors):
                self.force_matrix[j, i] = self.force_matrix[i, j]
                self.min_r_matrix[j, i] = self.min_r_matrix[i, j]
                self.max_r_matrix[j, i] = self.max_r_matrix[i, j]
        self._upload_rules()

    # -------------------------------------------------------------------------
    # Particle buffers
    # -------------------------------------------------------------------------
    def _allocate_particle_buffers(self, n):
        nbytes = max(n, 1) * self._dtype.itemsize
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self._buf_a)
        glBufferData(GL_SHADER_STORAGE_BUFFER, nbytes, None, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self._buf_b)
        glBufferData(GL_SHADER_STORAGE_BUFFER, nbytes, None, GL_DYNAMIC_DRAW)
        self._allocated = n

    def reset_particles(self):
        n = self.num_particles
        data = np.zeros(n, dtype=self._dtype)
        data['x'] = np.random.rand(n).astype(np.float32) * self.world_w
        data['y'] = np.random.rand(n).astype(np.float32) * self.world_h
        if self.mode3d:
            data['z'] = np.random.rand(n).astype(np.float32) * self.world_d
        data['color'] = np.random.randint(0, self.num_colors, n).astype(np.int32)

        self._allocate_particle_buffers(n)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self._buf_a)
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, data.nbytes, data)
        self._a_current = True

    def _current_buf(self):
        return self._buf_a if self._a_current else self._buf_b

    def _other_buf(self):
        return self._buf_b if self._a_current else self._buf_a

    def get_particle_ssbo(self):
        return self._current_buf()

    # -------------------------------------------------------------------------
    # Bin grid management
    # -------------------------------------------------------------------------
    def _ensure_bins(self):
        # Per-axis cells exactly divide the world so wrap boundaries align with
        # bin boundaries — otherwise the 3-wide neighbor scan misses interactions
        # right at the wrap edge and the simulation feels like there's a wall.
        # Cell size per axis >= max(max_r) so 3-wide neighbor scan covers all
        # possible interactions. Floor max_r at 8 to keep the grid sane.
        cs_min = float(max(self.max_r_matrix.max(), 8.0))
        gw = max(1, int(self.world_w / cs_min))   # floor
        gh = max(1, int(self.world_h / cs_min))
        gd = max(1, int(self.world_d / cs_min)) if self.mode3d else 1
        # Cap total bins so 3D + tiny max_r doesn't blow memory.
        max_bins = 4_000_000
        while gw * gh * gd > max_bins:
            cs_min *= 1.25
            gw = max(1, int(self.world_w / cs_min))
            gh = max(1, int(self.world_h / cs_min))
            gd = max(1, int(self.world_d / cs_min)) if self.mode3d else 1
        cx = self.world_w / gw
        cy = self.world_h / gh
        cz = (self.world_d / gd) if self.mode3d else 1.0
        if (cx != self._cell_x or cy != self._cell_y or cz != self._cell_z or
                gw != self._grid_w or gh != self._grid_h or gd != self._grid_d):
            self._cell_x, self._cell_y, self._cell_z = cx, cy, cz
            self._grid_w, self._grid_h, self._grid_d = gw, gh, gd
            self._bin_count = gw * gh * gd
        if self._bin_count + 1 > self._bin_capacity:
            nbytes = (self._bin_count + 1) * 4
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self._ssbo_bin_size)
            glBufferData(GL_SHADER_STORAGE_BUFFER, nbytes, None, GL_DYNAMIC_DRAW)
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self._ssbo_bin_off)
            glBufferData(GL_SHADER_STORAGE_BUFFER, nbytes, None, GL_DYNAMIC_DRAW)
            self._bin_capacity = self._bin_count + 1

    # -------------------------------------------------------------------------
    # Step pipeline
    # -------------------------------------------------------------------------
    def _set_bin_uniforms(self, uloc):
        glUniform1i(uloc["mode3d"],   1 if self.mode3d else 0)
        glUniform1i(uloc["gridW"],    self._grid_w)
        glUniform1i(uloc["gridH"],    self._grid_h)
        glUniform1i(uloc["gridD"],    self._grid_d)
        glUniform1f(uloc["invCellX"], 1.0 / self._cell_x)
        glUniform1f(uloc["invCellY"], 1.0 / self._cell_y)
        glUniform1f(uloc["invCellZ"], 1.0 / self._cell_z if self._cell_z > 0 else 1.0)

    def _bin_clear(self):
        n = self._bin_count + 1
        glUseProgram(self._prog_bin_clear)
        glUniform1i(self._uloc_bin_clear["num_bins"], n)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, self._ssbo_bin_size)
        glDispatchCompute((n + 63) // 64, 1, 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

    def _bin_fill(self, src_buf):
        glUseProgram(self._prog_bin_fill)
        glUniform1i(self._uloc_bin_fill["num_particles"], self.num_particles)
        self._set_bin_uniforms(self._uloc_bin_fill)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, src_buf)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, self._ssbo_bin_size)
        glDispatchCompute((self.num_particles + 63) // 64, 1, 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

    def _prefix_sum(self):
        # Hillis-Steele: produces inclusive prefix sum of binSize → binOff.
        # After log2(N) ping-pong passes, binOff[i] = sum(binSize[0..i-1]) of original sizes
        # (since binSize stores counts at index+1, inclusive scan gives starting offset of bin i).
        count = self._bin_count + 1
        glUseProgram(self._prog_prefix)
        glUniform1i(self._uloc_prefix["count"], count)
        src = self._ssbo_bin_size
        dst = self._ssbo_bin_off
        groups = (count + 63) // 64
        step = 1
        passes = 0
        while step < count:
            glUniform1i(self._uloc_prefix["step_size"], step)
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, src)
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, dst)
            glDispatchCompute(groups, 1, 1)
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
            src, dst = dst, src
            step *= 2
            passes += 1
        # After loop, `src` holds the final result.
        # Ensure final result lives in _ssbo_bin_off.
        if src is not self._ssbo_bin_off:
            # odd number of passes → copy from bin_size to bin_off
            glBindBuffer(GL_COPY_READ_BUFFER, src)
            glBindBuffer(GL_COPY_WRITE_BUFFER, self._ssbo_bin_off)
            glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, count * 4)
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

    def _sort(self, src_buf, dst_buf):
        glUseProgram(self._prog_sort)
        glUniform1i(self._uloc_sort["num_particles"], self.num_particles)
        self._set_bin_uniforms(self._uloc_sort)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, src_buf)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, dst_buf)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, self._ssbo_bin_size)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, self._ssbo_bin_off)
        glDispatchCompute((self.num_particles + 63) // 64, 1, 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

    def _force(self, src_buf, dst_buf):
        u = self._uloc_force
        glUseProgram(self._prog_force)
        glUniform1i(u["num_particles"],   self.num_particles)
        glUniform1i(u["num_colors"],      self.num_colors)
        glUniform1i(u["world_mode"],      self.world_mode)
        self._set_bin_uniforms(u)
        glUniform1f(u["world_w"],         self.world_w)
        glUniform1f(u["world_h"],         self.world_h)
        glUniform1f(u["world_d"],         self.world_d)
        glUniform1f(u["force_factor"],    self.force_factor)
        glUniform1f(u["friction_factor"], self.friction_factor)
        glUniform1f(u["repel"],           self.repel)
        glUniform1f(u["max_accel"],       self.max_accel)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, src_buf)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, dst_buf)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self._ssbo_rules)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, self._ssbo_bin_off)
        glDispatchCompute((self.num_particles + 63) // 64, 1, 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

    def _advance(self, buf, dt_scale):
        u = self._uloc_advance
        glUseProgram(self._prog_advance)
        glUniform1i(u["num_particles"], self.num_particles)
        glUniform1i(u["world_mode"],    self.world_mode)
        glUniform1i(u["mode3d"],        1 if self.mode3d else 0)
        glUniform1f(u["world_w"],       self.world_w)
        glUniform1f(u["world_h"],       self.world_h)
        glUniform1f(u["world_d"],       self.world_d)
        glUniform1f(u["dt_scale"],      dt_scale)
        glUniform1f(u["max_speed"],     self.max_speed)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buf)
        glDispatchCompute((self.num_particles + 63) // 64, 1, 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

    def step(self, dt_scale=1.0):
        self.step_multi(dt_scale, 1)

    def step_multi(self, dt_scale, substeps):
        if self.num_particles <= 0:
            return
        self._ensure_bins()
        for _ in range(substeps):
            src = self._current_buf()
            dst = self._other_buf()
            self._bin_clear()
            self._bin_fill(src)
            self._prefix_sum()
            self._bin_clear()
            self._sort(src, dst)
            self._a_current = not self._a_current  # sorted result is now current

            src = self._current_buf()
            dst = self._other_buf()
            self._force(src, dst)
            self._a_current = not self._a_current  # force-updated now current

            self._advance(self._current_buf(), dt_scale)

    # -------------------------------------------------------------------------
    # Brush / erase / paint (operate on current buffer)
    # -------------------------------------------------------------------------
    def _set_brush_uniforms(self, uloc, wx, wy, vx, vy, wz=0.0):
        glUniform1i(uloc["num_particles"], self.num_particles)
        glUniform1i(uloc["wrap"],          1 if self.world_mode == 1 else 0)
        glUniform1i(uloc["mode3d"],        1 if self.mode3d else 0)
        glUniform1f(uloc["brush_x"],       wx)
        glUniform1f(uloc["brush_y"],       wy)
        glUniform1f(uloc["brush_z"],       wz)
        glUniform1f(uloc["brush_r"],       self.brush_radius)
        glUniform1f(uloc["world_w"],       self.world_w)
        glUniform1f(uloc["world_h"],       self.world_h)
        if uloc.get("brush_vx", -1) >= 0:
            glUniform1f(uloc["brush_vx"],    vx)
            glUniform1f(uloc["brush_vy"],    vy)
            glUniform1f(uloc["brush_force"], self.brush_force)

    def apply_brush(self, wx, wy, vx=0.0, vy=0.0, wz=0.0):
        if self.num_particles <= 0:
            return
        glUseProgram(self._prog_brush)
        self._set_brush_uniforms(self._uloc_brush, wx, wy, vx, vy, wz)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self._current_buf())
        glDispatchCompute((self.num_particles + 63) // 64, 1, 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

    def apply_eraser(self, wx, wy, wz=0.0):
        n = self.num_particles
        if n <= 0:
            return
        keep_init = np.ones(n, dtype=np.uint32)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self._ssbo_keep)
        glBufferData(GL_SHADER_STORAGE_BUFFER, keep_init.nbytes, keep_init, GL_DYNAMIC_DRAW)

        types = np.array(sorted(self.brush_colors), dtype=np.int32) if self.brush_colors else np.zeros(0, dtype=np.int32)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self._ssbo_erase_types)
        glBufferData(GL_SHADER_STORAGE_BUFFER,
                     max(types.nbytes, 4),
                     types if len(types) else np.zeros(1, dtype=np.int32),
                     GL_DYNAMIC_DRAW)

        cur = self._current_buf()
        glUseProgram(self._prog_erase)
        self._set_brush_uniforms(self._uloc_erase, wx, wy, 0, 0, wz)
        glUniform1i(self._uloc_erase["num_erase_types"], len(types))
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, cur)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, self._ssbo_keep)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, self._ssbo_erase_types)
        glDispatchCompute((n + 63) // 64, 1, 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self._ssbo_keep)
        flags = np.frombuffer(glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, n * 4),
                              dtype=np.uint32).copy()
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, cur)
        particles = np.frombuffer(
            glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, n * self._dtype.itemsize),
            dtype=self._dtype).copy()
        kept = particles[flags == 1]
        self.num_particles = len(kept)
        self._allocate_particle_buffers(self.num_particles)
        if self.num_particles > 0:
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self._buf_a)
            glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, kept.nbytes, kept)
        self._a_current = True

    def paint_particles(self, wx, wy, wz=0.0, count=10):
        angles = np.random.rand(count) * 2 * np.pi
        radii  = np.random.rand(count) * self.brush_radius
        data = np.zeros(count, dtype=self._dtype)
        data['x'] = wx + np.cos(angles) * radii
        data['y'] = wy + np.sin(angles) * radii
        if self.mode3d:
            data['z'] = wz + (np.random.rand(count) - 0.5) * self.brush_radius * 2
        color_pool = np.array(list(self.brush_colors) if self.brush_colors
                              else list(range(self.num_colors)), dtype=np.int32)
        data['color'] = color_pool[np.random.randint(len(color_pool), size=count)]

        n_old = self.num_particles
        cur = self._current_buf()
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, cur)
        old = np.frombuffer(
            glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, n_old * self._dtype.itemsize),
            dtype=self._dtype).copy()
        combined = np.concatenate([old, data])
        self.num_particles = len(combined)
        self._allocate_particle_buffers(self.num_particles)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self._buf_a)
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, combined.nbytes, combined)
        self._a_current = True
