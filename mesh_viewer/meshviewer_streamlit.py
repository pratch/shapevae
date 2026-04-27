from __future__ import annotations

from pathlib import Path
from typing import Optional

import pyvista as pv
import streamlit as st

try:
    from stpyvista import stpyvista
except Exception:  # pragma: no cover
    stpyvista = None

DEFAULT_ROOT = Path("/ist/ist-share/scads/ploy/scene2/big_file/shapenet/shapenet")
DEFAULT_CLASS_ID = "02818832"
DEFAULT_MESH_ID = "1f6eed9be4aa249de76bc197b3a3ffc0"

st.set_page_config(page_title="ShapeNet Viewer", layout="wide")


def build_model_paths(root_path: Path, class_id: str, mesh_id: str) -> tuple[Path, Path, Path]:
    model_dir = root_path / class_id / mesh_id / "models"
    obj_path = model_dir / "model_normalized.obj"
    mtl_path = model_dir / "model_normalized.mtl"
    return model_dir, obj_path, mtl_path


def parse_texture_from_mtl(mtl_path: Path) -> Optional[Path]:
    if not mtl_path.exists():
        return None

    for raw_line in mtl_path.read_text(errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.lower().startswith("map_kd"):
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                texture_path = (mtl_path.parent / parts[1].strip()).resolve()
                return texture_path if texture_path.exists() else None
    return None


def load_mesh_with_optional_texture(obj_path: Path, mtl_path: Path):
    mesh = pv.read(str(obj_path))
    texture = None

    texture_path = parse_texture_from_mtl(mtl_path)
    if texture_path is not None:
        try:
            texture = pv.read_texture(str(texture_path))
        except Exception:
            texture = None

    return mesh, texture, texture_path


def render_plotter(mesh, texture, model_label: str):
    pv.global_theme.smooth_shading = True

    plotter = pv.Plotter(window_size=(1200, 800))
    plotter.set_background("#0f111a")
    plotter.add_axes(interactive=True)
    plotter.add_text(model_label, position="upper_left", font_size=10, color="white")

    if texture is not None:
        plotter.add_mesh(mesh, texture=texture, smooth_shading=True)
    else:
        plotter.add_mesh(mesh, color="#c7d7ff", smooth_shading=True)

    plotter.enable_anti_aliasing("msaa")
    plotter.reset_camera()

    if stpyvista is None:
        st.warning("stpyvista is not installed. Showing a static preview image.")
        image = plotter.screenshot(return_img=True)
        st.image(image, caption=model_label)
        plotter.close()
        return

    stpyvista(plotter, key=f"viewer_{model_label}")


def mesh_counts(mesh) -> tuple[int, int]:
    return mesh.n_points, mesh.n_cells


st.title("ShapeNet Mesh Viewer (PyVista + Streamlit)")
st.caption("Load ShapeNet model_normalized.obj with texture from model_normalized.mtl")

with st.sidebar:
    st.header("Model Selection")
    root_path_str = st.text_input("Root Path", value=str(DEFAULT_ROOT))
    class_id = st.text_input("Class ID", value=DEFAULT_CLASS_ID)
    mesh_id = st.text_input("Mesh ID", value=DEFAULT_MESH_ID)
    load_clicked = st.button("Load Mesh", type="primary", use_container_width=True)

if load_clicked or "last_model" not in st.session_state:
    st.session_state.last_model = {
        "root": root_path_str.strip(),
        "class_id": class_id.strip(),
        "mesh_id": mesh_id.strip(),
    }

selected = st.session_state.last_model
root_path = Path(selected["root"])
class_id = selected["class_id"]
mesh_id = selected["mesh_id"]

model_dir, obj_path, mtl_path = build_model_paths(root_path, class_id, mesh_id)

st.write(f"Model directory: {model_dir}")

if not obj_path.exists():
    st.error(f"OBJ not found: {obj_path}")
    st.stop()

mesh, texture, texture_path = load_mesh_with_optional_texture(obj_path, mtl_path)
vertex_count, face_count = mesh_counts(mesh)

with st.expander("Loaded Files", expanded=True):
    st.write(f"OBJ: {obj_path}")
    st.write(f"MTL: {mtl_path if mtl_path.exists() else 'missing'}")
    st.write(f"Texture: {texture_path if texture_path is not None else 'not found in MTL or missing'}")

if texture is None:
    st.info("Texture could not be loaded from MTL. Rendering mesh with fallback color.")

render_plotter(mesh, texture, f"{class_id}/{mesh_id}")
st.markdown(
    f"<div style='font-size: 1.1rem; font-weight: 600; margin-top: 0.5rem;'>"
    f"Vertices: {vertex_count} | Faces: {face_count}"
    f"</div>",
    unsafe_allow_html=True,
)

st.markdown(
    """
Mouse controls (inside viewer):
- Left drag: rotate
- Right drag: pan
- Scroll: zoom
"""
)
