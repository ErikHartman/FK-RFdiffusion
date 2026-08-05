# Model-agnostic FK

This directory contains a standalone Feynman--Kac (FK) particle-filter loop.
It does not import RFdiffusion or make assumptions about tensors, coordinate
formats, or a model's self-conditioning implementation.

The adapter contract has five operations:

1. create an initial particle;
2. make a reverse-diffusion step, returning both the next state and the model's
   denoised/x0 prediction;
3. clone a particle after resampling;
4. expose the first and last timesteps; and
5. serialize a particle to the artifact accepted by the reward function.

Particles should include all per-particle model cache. This is essential for
self-conditioning or recurrent models: resampling must copy that state with the
particle lineage instead of leaving it on a shared model object.
