from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import trimesh
import torch

# shapenet is organized as:
# data_dir/
#   ├── object_class/ (e.g. chair) 
#   │   ├── object_id/
#   │   │   ├── models/
#   │   │   │   ├── model_normalized.obj

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
    def __init__(self, data_dir, object_class='03001627', num_points=1024):
        super().__init__()
        self.data_dir = data_dir
        self.object_class = object_class
        self.num_points = num_points

        class_dir = os.path.join(data_dir, object_class)
        self.file_paths = []
        self.object_ids = [] # for debugging
        
        for obj_id in os.listdir(class_dir):
            path = os.path.join(class_dir, obj_id, 'models', 'model_normalized.obj')
            if os.path.exists(path):
                self.file_paths.append(path)
                self.object_ids.append(obj_id)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # load a mesh and sample points from its surface
        mesh = trimesh.load(self.file_paths[idx], force='mesh') # return Mesh instead of Scene
        points, _ = trimesh.sample.sample_surface(mesh, self.num_points)

        points = torch.from_numpy(points).float() # shape (N, 3)
        points = points - points.mean(dim=0, keepdim=True)  # center the point cloud
        scale = points.norm(dim=1).max()
        points = points / scale  # normalize to unit sphere

        return {
            'points': points,  # shape (N, 3)
            'object_id': self.object_ids[idx],  # for debugging
            'category': shapenet_id_to_category[self.object_class],  # for debugging
        }

# unit test
if __name__ == '__main__':
    dataset = ShapeNetDataset(data_dir='/ist/ist-share/scads/ploy/scene2/big_file/shapenet/shapenet/', object_class='03001627', num_points=1024)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    for batch in dataloader:
        print(batch['points'].shape)  # should be (2, 1024, 3)
        print(batch['object_id'])  # for debugging
        print(batch['category'])  # for debugging
        break
