import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# precomputed point clouds are organized as:
# sampled_poincloud/
#   ├── object_class/ (e.g. 03001627)
#   │   ├── object_id.npy

shapenet_id_to_category = {#https://gist.githubusercontent.com/tejaskhot/15ae62827d6e43b91a4b0c5c850c168e/raw/5064af3603d509b79229f6931998d4e197575ad3/shapenet_synset_list
    "04379243": "table",
    "03593526": "jar",
    "04225987": "skateboard",
    "02958343": "car",
    "02876657": "bottle",
    "04460130": "tower",
    "03001627": "chair",
    "02871439": "bookshelf",
    "02942699": "camera",
    "02691156": "airplane",
    "03642806": "laptop",
    "02801938": "basket",
    "04256520": "sofa",
    "03624134": "knife",
    "02946921": "can",
    "04090263": "rifle",
    "04468005": "train",
    "03938244": "pillow",
    "03636649": "lamp",
    "02747177": "trash bin",
    "03710193": "mailbox",
    "04530566": "watercraft",
    "03790512": "motorbike",
    "03207941": "dishwasher",
    "02828884": "bench",
    "03948459": "pistol",
    "04099429": "rocket",
    "03691459": "loudspeaker",
    "03337140": "file cabinet",
    "02773838": "bag",
    "02933112": "cabinet",
    "02818832": "bed",
    "02843684": "birdhouse",
    "03211117": "display",
    "03928116": "piano",
    "03261776": "earphone",
    "04401088": "telephone",
    "04330267": "stove",
    "03759954": "microphone",
    "02924116": "bus",
    "03797390": "mug",
    "04074963": "remote",
    "02808440": "bathtub",
    "02880940": "bowl",
    "03085013": "keyboard",
    "03467517": "guitar",
    "04554684": "washer",
    "02834778": "bicycle",
    "03325088": "faucet",
    "04004475": "printer",
    "02954340": "cap"
}


class ShapeNetDataset(Dataset):
    def __init__(self, data_dir, object_class="03001627"):
        super().__init__()
        self.data_dir = data_dir
        self.object_class = object_class

        class_dir = os.path.join(data_dir, object_class)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"Class directory not found: {class_dir}")

        self.file_paths = []
        self.norm_paths = []
        self.clip_paths = []
        self.object_ids = []  # for debugging

        for file_name in sorted(os.listdir(class_dir)):
            if file_name.endswith(".npy"):
                obj_id = file_name[:-4]
                path = os.path.join(class_dir, file_name)
                norm_path = os.path.join(class_dir, f"{obj_id}.norm.npz")
                clip_path = os.path.join(class_dir, f"{obj_id}.clip.npz")
                self.file_paths.append(path)
                self.norm_paths.append(norm_path)
                self.clip_paths.append(clip_path)
                self.object_ids.append(obj_id)

        if len(self.file_paths) == 0:
            raise RuntimeError(f"No .npy point clouds found under: {class_dir}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        points = np.load(self.file_paths[idx]).astype(np.float32)
        points = torch.from_numpy(points)  # shape (N, 3)
        clip_path = self.clip_paths[idx]

        output = {
            "points": points,  # shape (N, 3)
            "norm_path": self.norm_paths[
                idx
            ],  # loaded lazily during unnormalized plotting
            "object_id": self.object_ids[idx],  # for debugging
            "category": shapenet_id_to_category[self.object_class] ,  # for debugging
        }
        if os.path.exists(clip_path):
            clip_latent = np.load(self.clip_paths[idx])['mean'].astype(np.float32)
            output.update(
                {
                    "clip_path": self.clip_paths[
                        idx
                    ],  # for loading CLIP embeddings during training
                    "clip_latent": clip_latent
                }
            )
        return output


# unit test
if __name__ == "__main__":
    dataset = ShapeNetDataset(data_dir="./sampled_poincloud", object_class="03001627")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    for batch in dataloader:
        print(batch["points"].shape)  # should be (2, 1024, 3)
        print(batch["object_id"])  # for debugging
        print(batch["category"])  # for debugging
        break
