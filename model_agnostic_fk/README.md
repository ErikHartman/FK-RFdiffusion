# Model-agnostic FK

The Feynman-Kac (FK) method is model agnostic, however, the main implementation was done with RFdiffusion. However, this repository contains a standalone adapter module that can be used to implement FK for other models. You can therefore write your own adapter and run FK using your model.

The adapter contract has five operations:

1. create an initial particle;
2. make a reverse-diffusion step, returning both the next state and the model's
   denoised/x0 prediction;
3. clone a particle after resampling;
4. expose the first and last timesteps; and
5. serialize a particle to the artifact accepted by the reward function.

This was tested for [Genie 2](https://github.com/aqlaboratory/genie2) and its adapter is present in /adapters.
