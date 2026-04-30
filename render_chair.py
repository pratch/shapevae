import torch
import os
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    PointLights,
    look_at_view_transform,
)
from PIL import Image

# -------------------------
# Config
# -------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

obj_path = "/ist/ist-share/scads/ploy/scene2/big_file/shapenet/shapenet/03001627/1b5e876f3559c231532a8e162f399205/models/model_normalized.obj"

output_dir = "./renders_chair"
os.makedirs(output_dir, exist_ok=True)

dist = 1.3
elev = 20
azims = [90, 135, 180, 225, 270]

mesh = load_objs_as_meshes([obj_path], device=device)

raster_settings = RasterizationSettings(image_size=512)

# shading
lights = PointLights(
    device=device,
    # location=[[2.0, 2.0, 2.0]],
    location=[[0.0, 1.0, -2.0]],
)


renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        raster_settings=raster_settings
    ),
    shader=HardPhongShader(
        device=device,
        lights=lights,
    ),
)

for azim in azims:
    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    images = renderer(mesh, cameras=cameras)

    rgb = images[0, ..., :3].cpu().numpy()
    alpha = images[0, ..., 3].cpu().numpy()

    rgb = (rgb * 255).astype("uint8")


    Image.fromarray(rgb).save(
        os.path.join(output_dir, f"chair_azim_{azim}.png")
    )

    print(f"Saved azim {azim}")