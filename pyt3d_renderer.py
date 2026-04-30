from __future__ import annotations

from typing import List
import os


def _load_mtl_diffuse(obj_path: str) -> tuple[float, float, float] | None:
    try:
        with open(obj_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("mtllib "):
                    mtl_name = line.split(None, 1)[1].strip()
                    mtl_path = os.path.join(os.path.dirname(obj_path), mtl_name)
                    if not os.path.isfile(mtl_path):
                        return None
                    return _parse_mtl_diffuse(mtl_path)
    except OSError:
        return None
    return None


def _parse_mtl_diffuse(mtl_path: str) -> tuple[float, float, float] | None:
    try:
        with open(mtl_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("Kd "):
                    parts = line.split()
                    if len(parts) >= 4:
                        r, g, b = (float(parts[1]), float(parts[2]), float(parts[3]))
                        return (r, g, b)
    except OSError:
        return None
    return None


def render_mesh_views(
    mesh_path: str,
    n_views: int,
    image_size: int = 224,
    device: str = "cpu",
    dist: float = 1.3,
    elev: float = 20.0,
) -> List:
    if n_views <= 0:
        return []

    try:
        import torch
        from pytorch3d.io import load_obj, load_objs_as_meshes
        from pytorch3d.renderer import (
            FoVPerspectiveCameras,
            RasterizationSettings,
            MeshRenderer,
            MeshRasterizer,
            HardPhongShader,
            PointLights,
            TexturesVertex,
            look_at_view_transform,
        )
        from pytorch3d.structures import Meshes
        from PIL import Image
    except Exception as exc:  # pragma: no cover - informative error
        raise RuntimeError(
            "PyTorch3D is required for mesh rendering. Install it before using --render-mode mesh.\n"
            f"Original error: {exc}"
        )

    device_t = torch.device(device)
    try:
        mesh = load_objs_as_meshes([mesh_path], device=device_t)
    except Exception:
        verts, faces, _ = load_obj(mesh_path, load_textures=False)
        verts = verts.to(device_t)
        faces_idx = faces.verts_idx.to(device_t)
        diffuse = _load_mtl_diffuse(mesh_path)
        if diffuse is None:
            verts_rgb = torch.ones_like(verts)[None]
        else:
            rgb = torch.tensor(diffuse, device=device_t, dtype=verts.dtype)
            verts_rgb = rgb[None, None, :].expand(1, verts.shape[0], 3)
        textures = TexturesVertex(verts_features=verts_rgb)
        mesh = Meshes(verts=[verts], faces=[faces_idx], textures=textures)

    if mesh.textures is None:
        diffuse = _load_mtl_diffuse(mesh_path)
        if diffuse is None:
            verts_rgb = torch.ones_like(mesh.verts_padded())
        else:
            rgb = torch.tensor(diffuse, device=device_t, dtype=mesh.verts_padded().dtype)
            verts_rgb = rgb[None, None, :].expand_as(mesh.verts_padded())
        mesh.textures = TexturesVertex(verts_features=verts_rgb)

    raster_settings = RasterizationSettings(image_size=image_size)
    lights = PointLights(device=device_t, location=[[0.0, 1.0, -2.0]])

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=HardPhongShader(device=device_t, lights=lights),
    )

    if n_views == 5:
        azims = [90, 135, 180, 225, 270]
    elif n_views == 1:
        azims = [180]
    else:
        step = (270.0 - 90.0) / (n_views - 1)
        azims = [90.0 + step * i for i in range(n_views)]

    images: List = []
    for azim in azims:
        R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
        cameras = FoVPerspectiveCameras(device=device_t, R=R, T=T)

        with torch.no_grad():
            rendered = renderer(mesh, cameras=cameras)

        rgb = rendered[0, ..., :3].cpu().numpy()
        rgb = (rgb * 255).astype("uint8")
        images.append(Image.fromarray(rgb))

    return images
