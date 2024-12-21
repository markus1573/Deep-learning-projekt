This repo contains the code for four implementations of DDPM's as discussed in our exam project "02456 - Deep Learning\\ Denoising Diffusion Probabilistic Models" by s224384, s224217 and s203295.

-----
Model 1: Linear noise schedule and concatenated time embeddings before the first convolution in the U-net.

Model 2: Cosine noise schedule and sinosoidal time embeddings in every layer of the U-net.

Model 3: Model 1 with shallower U-net (three down- and three up-convs) but with a 4-convolution wide bottleneck with residual connections.

Model 4: Model 3 but with multihead attention after concatenation in the second up-conv layer.


