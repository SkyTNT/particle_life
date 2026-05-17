import json
import numpy as np
from pathlib import Path

PRESETS_DIR = Path(__file__).parent / "presets"


def save_preset(sim, name):
    PRESETS_DIR.mkdir(exist_ok=True)
    data = {
        "num_colors":   sim.num_colors,
        "force_matrix": sim.force_matrix.tolist(),
        "min_r_matrix": sim.min_r_matrix.tolist(),
        "max_r_matrix": sim.max_r_matrix.tolist(),
    }
    (PRESETS_DIR / f"{name}.json").write_text(json.dumps(data, indent=2))


def load_preset(sim, name):
    data = json.loads((PRESETS_DIR / f"{name}.json").read_text())
    sim.num_colors    = data["num_colors"]
    sim.force_matrix  = np.array(data["force_matrix"], dtype=np.float32)
    sim.min_r_matrix  = np.array(data["min_r_matrix"], dtype=np.float32)
    sim.max_r_matrix  = np.array(data["max_r_matrix"], dtype=np.float32)
    sim._upload_rules()


def delete_preset(name):
    p = PRESETS_DIR / f"{name}.json"
    if p.exists():
        p.unlink()


def list_presets():
    if not PRESETS_DIR.exists():
        return []
    return sorted(p.stem for p in PRESETS_DIR.glob("*.json"))
