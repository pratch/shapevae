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

### Demo
- [ ] CLIP-aligned latent adapter
   - [ ] Train a small projection head to align the latent space with CLIP embeddings
   - [ ] Update `precompute_pointcloud` and `shapenet_dataset` to precompute and present the [5 rendered views, CLIP embeddings of rendered views, point cloud] for each shape in the dataset
   - [ ] AI-generated shopping web-app
```
point cloud -> PTv3 encoder -> latent z -> small projection head -> CLIP-aligned embedding
text prompt -> frozen CLIP text encoder -> text embedding
dir = embed("chair with armrests") - embed("chair without armrests")
```
