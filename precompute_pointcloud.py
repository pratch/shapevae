import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
import trimesh


def preprocess_points(points_np: np.ndarray) -> np.ndarray:
    points = torch.from_numpy(points_np).float()
    points = points - points.mean(dim=0, keepdim=True)
    scale = points.norm(dim=1).max()
    points = points / scale
    return points.numpy().astype(np.float32)


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
        help="Output root folder for sampled point clouds",
    )
    parser.add_argument(
        "--object-class",
        type=str,
        default="03001627",
        help="ShapeNet class id (default: chair)",
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
    return parser.parse_args()


def process_one(mesh_path: str, out_path: str, obj_id: str, num_points: int) -> tuple[str, str, str]:
    try:
        mesh = trimesh.load(mesh_path, force="mesh")
        points, _ = trimesh.sample.sample_surface(mesh, num_points)
        points = preprocess_points(points)
        np.save(out_path, points)
        return "saved", obj_id, ""
    except Exception as exc:
        return "failed", obj_id, str(exc)


def main() -> None:
    args = parse_args()

    class_dir = os.path.join(args.data_dir, args.object_class)
    if not os.path.isdir(class_dir):
        raise FileNotFoundError(f"Class directory not found: {class_dir}")

    output_class_dir = os.path.join(args.output_dir, args.object_class)
    os.makedirs(output_class_dir, exist_ok=True)

    obj_ids = sorted(os.listdir(class_dir))
    if args.limit is not None:
        obj_ids = obj_ids[: args.limit]

    total = len(obj_ids)
    saved = 0
    skipped = 0
    failed = 0

    tasks = []
    for obj_id in obj_ids:
        mesh_path = os.path.join(class_dir, obj_id, "models", "model_normalized.obj")
        out_path = os.path.join(output_class_dir, f"{obj_id}.npy")

        if not os.path.isfile(mesh_path):
            skipped += 1
            continue

        if os.path.exists(out_path) and not args.overwrite:
            skipped += 1
            continue

        tasks.append((mesh_path, out_path, obj_id, args.num_points))

    if len(tasks) == 0:
        print("nothing to do")
        print(f"class:   {args.object_class}")
        print(f"total:   {total}")
        print(f"saved:   {saved}")
        print(f"skipped: {skipped}")
        print(f"failed:  {failed}")
        print(f"output:  {output_class_dir}")
        return

    workers = max(1, args.workers)
    completed = 0
    print(f"launching {workers} workers for {len(tasks)} files")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(process_one, mesh_path, out_path, obj_id, num_points)
            for mesh_path, out_path, obj_id, num_points in tasks
        ]

        for future in as_completed(futures):
            status, obj_id, message = future.result()
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
    print(f"class:   {args.object_class}")
    print(f"total:   {total}")
    print(f"saved:   {saved}")
    print(f"skipped: {skipped}")
    print(f"failed:  {failed}")
    print(f"output:  {output_class_dir}")


if __name__ == "__main__":
    main()
