import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# precomputed point clouds are organized as:
# sampled_poincloud/
#   ├── object_class/ (e.g. 03001627)
#   │   ├── object_id.npy

shapenet_id_to_category = {
    "02691156": "airplane",
    "02933112": "cabinet",
    "03001627": "chair",
    "03636649": "lamp",
    "04090263": "rifle",
    "04379243": "table",
    "04530566": "watercraft",
    "02828884": "bench",
    "02958343": "car",
    "03211117": "display",
    "03691459": "speaker",
    "04256520": "sofa",
    "04401088": "telephone",
}

class ShapeNetDataset(Dataset):
    def __init__(self, data_dir, object_class='03001627'):
        super().__init__()
        self.data_dir = data_dir
        self.object_class = object_class

        class_dir = os.path.join(data_dir, object_class)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"Class directory not found: {class_dir}")

        self.file_paths = []
        self.norm_paths = []
        self.object_ids = []  # for debugging

        for file_name in sorted(os.listdir(class_dir)):
            if file_name.endswith('.npy'):
                obj_id = file_name[:-4]
                path = os.path.join(class_dir, file_name)
                norm_path = os.path.join(class_dir, f"{obj_id}.norm.npz")
                self.file_paths.append(path)
                self.norm_paths.append(norm_path)
                self.object_ids.append(obj_id)

        if len(self.file_paths) == 0:
            raise RuntimeError(f"No .npy point clouds found under: {class_dir}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        points = np.load(self.file_paths[idx]).astype(np.float32)
        points = torch.from_numpy(points)  # shape (N, 3)

        return {
            'points': points,  # shape (N, 3)
            'norm_path': self.norm_paths[idx],  # loaded lazily during unnormalized plotting
            'object_id': self.object_ids[idx],  # for debugging
            'category': shapenet_id_to_category[self.object_class],  # for debugging
        }

# unit test
if __name__ == '__main__':
    dataset = ShapeNetDataset(data_dir='./sampled_poincloud', object_class='03001627')
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    for batch in dataloader:
        print(batch['points'].shape)  # should be (2, 1024, 3)
        print(batch['object_id'])  # for debugging
        print(batch['category'])  # for debugging
        break
