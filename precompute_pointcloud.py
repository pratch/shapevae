import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import random

import numpy as np
import torch
import trimesh
import io
from typing import List

from pyt3d_renderer import render_mesh_views


def preprocess_points(points_np: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    points = torch.from_numpy(points_np).float()
    centroid = points.mean(dim=0, keepdim=True)
    points = points - centroid
    scale = points.norm(dim=1).max()
    points = points / scale
    return points.numpy().astype(np.float32), centroid.squeeze(0).numpy().astype(np.float32), float(scale.item())


def render_pointcloud_views(points_np: np.ndarray, n_views: int, image_size: int = 224) -> List:
    """Render `points_np` into `n_views` square RGB PIL images using matplotlib (Agg backend).

    Returns a list of PIL.Image images sized `image_size` x `image_size`.
    """
    if n_views <= 0:
        return []

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from PIL import Image

    pts = points_np.copy()
    images: List = []

    for i in range(n_views):
        az = 360.0 * i / max(1, n_views)
        el = 30.0

        fig = plt.figure(figsize=(image_size / 100.0, image_size / 100.0), dpi=100)
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, c='black', depthshade=False)
        ax.view_init(elev=el, azim=az)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_box_aspect([1, 1, 1])
        ax.set_axis_off()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        img = img.resize((image_size, image_size), Image.BICUBIC)
        images.append(img)

    return images


def compute_clip_embeddings(images: List, model_name: str = "ViT-B/32", device: str = "cpu") -> np.ndarray:
    """Compute CLIP image embeddings for a list of PIL images.

    Returns an (n_images, dim) numpy array. Requires the `clip` package (openai/clip).
    """
    if len(images) == 0:
        return np.zeros((0, 0), dtype=np.float32)

    try:
        import clip
    except Exception as exc:  # pragma: no cover - informative error
        raise RuntimeError("CLIP library not available. Install with `pip install git+https://github.com/openai/CLIP.git`\n"
                           f"Original error: {exc}")

    import torch

    model, preprocess = clip.load(model_name, device=device, jit=False)
    model.eval()
    model.to(device)

    tensors = torch.stack([preprocess(img) for img in images], dim=0).to(device)
    with torch.no_grad():
        emb = model.encode_image(tensors)
    emb = emb.cpu().numpy()
    return emb


def save_clip_embeddings(path: str, embeddings: np.ndarray) -> None:
    """Save per-view embeddings and their mean into a compressed npz file."""
    if embeddings.size == 0:
        np.savez_compressed(path, embeddings=embeddings)
    else:
        mean = embeddings.mean(axis=0)
        np.savez_compressed(path, embeddings=embeddings, mean=mean)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute ShapeNet surface-sampled point clouds into .npy files")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to raw ShapeNet root (contains class folders like 03001627)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./sampled_poincloud",
        help="Output root folder for sampled point clouds and normalization metadata",
    )
    parser.add_argument(
        "--object-class",
        type=str,
        default="03001627",
        help="Single ShapeNet class id (backward compatible; can also be comma-separated)",
    )
    parser.add_argument(
        "--object-classes",
        type=str,
        nargs="+",
        default=None,
        help="One or more ShapeNet class ids, e.g. --object-classes 03001627 02691156",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=1024,
        help="Number of surface points to sample per mesh",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .npy files",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for number of objects to process",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) - 1),
        help="Number of parallel worker processes",
    )
    parser.add_argument(
        "--render-views",
        type=int,
        default=0,
        help="Number of rendered views to produce per point cloud (0 to disable)",
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        choices=["pointcloud", "mesh"],
        default="pointcloud",
        help="Render mode for views: pointcloud (matplotlib) or mesh (PyTorch3D)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Square size (pixels) for rendered views",
    )
    parser.add_argument(
        "--compute-clip",
        action="store_true",
        help="Compute CLIP embeddings for rendered views (requires CLIP package)",
    )
    parser.add_argument(
        "--clip-model",
        type=str,
        default="ViT-B/32",
        help="CLIP model name to load (passed to clip.load)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for CLIP computations: 'cpu' or 'cuda'",
    )
    return parser.parse_args()


def process_one(
    mesh_path: str,
    out_path: str,
    norm_path: str,
    obj_id: str,
    num_points: int,
    render_views: int = 0,
    render_mode: str = "pointcloud",
    compute_clip: bool = False,
    clip_model: str = "ViT-B/32",
    image_size: int = 224,
    device: str = "cpu",
) -> tuple[str, str, str]:
    try:
        mesh = trimesh.load(mesh_path, force="mesh")
        points, _ = trimesh.sample.sample_surface(mesh, num_points)
        points, centroid, scale = preprocess_points(points)
        np.save(out_path, points)
        np.savez_compressed(norm_path, centroid=centroid, scale=scale)

        # Optional: render views and compute CLIP embeddings
        if render_views and render_views > 0:
            try:
                if render_mode == "mesh":
                    images = render_mesh_views(
                        mesh_path,
                        render_views,
                        image_size=image_size,
                        device=device,
                    )
                else:
                    images = render_pointcloud_views(points, render_views, image_size)
            except Exception as exc:
                return "failed", obj_id, f"render_failed:{exc}"

            if compute_clip:
                try:
                    emb = compute_clip_embeddings(images, model_name=clip_model, device=device)
                    emb_path = os.path.splitext(out_path)[0] + ".clip.npz"
                    print(f"Saving CLIP embeddings to {emb_path} for {obj_id} shape {emb.shape}")
                    # save embeddings
                    save_clip_embeddings(emb_path, emb)

                    # save for 1%: store rendered view images in a `views` subdirectory
                    if random.random() < 1:
                        views_dir = os.path.join(os.path.dirname(out_path), "views")
                        os.makedirs(views_dir, exist_ok=True)
                        base_name = os.path.splitext(os.path.basename(out_path))[0]
                        try:
                            for idx, img in enumerate(images):
                                img_path = os.path.join(views_dir, f"{base_name}.view{idx:02d}.jpg")
                                img.save(img_path, format="JPEG", quality=90)
                        except Exception as exc:
                            # non-fatal: warn and continue
                            print(f"warning: failed saving view images for {obj_id}: {exc}")
                except Exception as exc:
                    return "failed", obj_id, f"clip_failed:{exc}"

        return "saved", obj_id, ""
    except Exception as exc:
        return "failed", obj_id, str(exc)


def _resolve_object_classes(args: argparse.Namespace) -> list[str]:
    if args.object_classes:
        classes = args.object_classes
    else:
        # Keep backward compatibility for --object-class and allow comma-separated values.
        classes = [c.strip() for c in args.object_class.split(",") if c.strip()]

    unique_classes: list[str] = []
    for class_id in classes:
        if class_id not in unique_classes:
            unique_classes.append(class_id)
    return unique_classes


def main() -> None:
    args = parse_args()
    object_classes = _resolve_object_classes(args)

    total = 0
    saved = 0
    skipped = 0
    failed = 0

    tasks = []
    for class_id in object_classes:
        class_dir = os.path.join(args.data_dir, class_id)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"Class directory not found: {class_dir}")

        output_class_dir = os.path.join(args.output_dir, class_id)
        os.makedirs(output_class_dir, exist_ok=True)

        obj_ids = sorted(os.listdir(class_dir))
        if args.limit is not None:
            obj_ids = obj_ids[: args.limit]

        total += len(obj_ids)

        for obj_id in obj_ids:
            mesh_path = os.path.join(class_dir, obj_id, "models", "model_normalized.obj")
            out_path = os.path.join(output_class_dir, f"{obj_id}.npy")
            norm_path = os.path.join(output_class_dir, f"{obj_id}.norm.npz")
            clip_path = os.path.join(output_class_dir, f"{obj_id}.clip.npz")

            if not os.path.isfile(mesh_path):
                skipped += 1
                continue

            if os.path.exists(out_path) and os.path.exists(norm_path) and os.path.exists(clip_path) and not args.overwrite:
                skipped += 1
                print(f"skipping existing files for {class_id}/{obj_id}")
                continue

            # Keep class_id in obj label for clearer failure logs.
            tasks.append(
                (
                    mesh_path,
                    out_path,
                    norm_path,
                    f"{class_id}/{obj_id}",
                    args.num_points,
                    args.render_views,
                    args.render_mode,
                    args.compute_clip,
                    args.clip_model,
                    args.image_size,
                    args.device,
                )
            )

    if len(tasks) == 0:
        print("nothing to do")
        print(f"classes: {', '.join(object_classes)}")
        print(f"total:   {total}")
        print(f"saved:   {saved}")
        print(f"skipped: {skipped}")
        print(f"failed:  {failed}")
        print(f"output:  {args.output_dir}")
        return

    workers = max(1, args.workers)
    # If computing CLIP on CUDA, avoid multiple processes each loading the GPU model
    # which often causes OOMs and abrupt worker termination. Force single-worker in that case.
    # if args.compute_clip and args.device and args.device.startswith("cuda") and workers > 1:
    #     print(
    #         "warning: compute-clip with CUDA and multiple workers may crash workers (GPU OOM)."
    #         " For safety, reducing workers to 1."
    #     )
    #     workers = 1
    completed = 0
    print(f"classes: {', '.join(object_classes)}")
    print(f"launching {workers} workers for {len(tasks)} files")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                process_one,
                mesh_path,
                out_path,
                norm_path,
                obj_id,
                num_points,
                render_views,
                render_mode,
                compute_clip,
                clip_model,
                image_size,
                device,
            )
            for mesh_path, out_path, norm_path, obj_id, num_points, render_views, render_mode, compute_clip, clip_model, image_size, device in tasks
        ]

        for future in as_completed(futures):
            try:
                status, obj_id, message = future.result()
            except Exception as exc:
                # Worker crashed or raised an unpicklable exception
                failed += 1
                completed += 1
                print(f"[worker-crash] a worker crashed: {exc}")
                continue

            completed += 1

            if status == "saved":
                saved += 1
            else:
                failed += 1
                print(f"[failed] {obj_id}: {message}")

            if completed % 200 == 0 or completed == len(tasks):
                print(
                    f"progress: {completed}/{len(tasks)} done | saved={saved} skipped={skipped} failed={failed}"
                )

    print("done")
    print(f"classes: {', '.join(object_classes)}")
    print(f"total:   {total}")
    print(f"saved:   {saved}")
    print(f"skipped: {skipped}")
    print(f"failed:  {failed}")
    print(f"output:  {args.output_dir}")


if __name__ == "__main__":
    main()
