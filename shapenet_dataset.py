from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import trimesh

# shapenet is organized as:
# data_dir/
#   ├── object_class/ (e.g. chair) 
#   │   ├── object_id/
#   │   │   ├── models/
#   │   │   │   ├── model_normalized.obj

class ShapeNetDataset(Dataset):
    def __init__(self, data_dir, object_class='03001627', num_points=1024):
        super().__init__()
        self.data_dir = data_dir
        self.object_class = object_class
        self.num_points = num_points

        class_dir = os.path.join(data_dir, object_class)
        self.file_paths = [
            os.path.join(class_dir, obj_id, 'models', 'model_normalized.obj')
            for obj_id in os.listdir(class_dir)
        ]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # load pointcloud from file, sample num_points, and return as tensor
        mesh = trimesh.load(self.file_paths[idx])
        points, _ = trimesh.sample.sample_surface(mesh, self.num_points)
        return torch.from_numpy(points).float()

# unit test
if __name__ == '__main__':
    dataset = ShapeNetDataset(data_dir='/ist/ist-share/scads/ploy/scene2/big_file/shapenet/shapenet/', object_class='03001627', num_points=1024)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    for batch in dataloader:
        print(batch.shape)  # should be (8, 1024, 3)
        break
