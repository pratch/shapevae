# shapevae

## TODO

### Encoder
- [x] Point AE (PTv3 as encoder, MLP as decoder, loss: CD/repulsion, flag for AE or VAE) ```model/ptv3_based_model.py```

### Decoder
- [ ] Predict Occupancy (Pratch)

### Loss
- [ ] VAE, β-VAE 

### Experiments
- [x] Latent interpolation
  - CLIP-guided axes?
- [ ] Number of pointcloud samples
- [ ] Latent sizes
- [ ] Number of class

### Demo
- [ ] Final Web Demo
  - [ ] Text-based Shape Query
  - [ ] Pointcloud/Mesh-based Shape Query
  - [ ] Interpolation
  - [ ] Shape Attribute Extrapolation (e.g. increase spikiness) 
- [ ] CLIP-aligned latent adapter
   - [x] Update `precompute_pointcloud` and `shapenet_dataset` to precompute and present the [5 rendered views, CLIP embeddings of rendered views, point cloud] for each shape in the dataset (add falgs render-views, image-size, compute-clip, clip-model, device)
     - [x] output to `*.clip.npz` files as tenor of shape (render_views, 512) 
     - [x] render with textured/mesh? pt3d? / Boss
     - [x] render with upright orientation? (currently random orientation, which may be suboptimal for CLIP alignment)
   - [x] Train a small projection head to align the latent space with CLIP embeddings
   - [ ] AI-generated shopping web-app
```
point cloud -> PTv3 encoder -> latent z -> small projection head -> CLIP-aligned embedding
text prompt -> frozen CLIP text encoder -> text embedding
dir = embed("chair with armrests") - embed("chair without armrests")
```

### Shared Resources
- wandb project: https://wandb.ai/alephnir-vistec/shapevae
- dataset: /ist/ist-share/scads/ploy/scene2/big_file/shapenet/shapenet
- preprocessed data: /ist-nas/ist-share/vision/pratchp/shapevae_preprocessed
- trained weights: /ist-nas/ist-share/vision/pratchp/shapevae_weights