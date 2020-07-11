Dependencies
----

1. *carnatic_DL* : Custom PyTorch dataloader for our Carnatic Violin Dataset
2. *vae_krishna* : (C)VAE, code for our network is inspired from <a href="https://github.com/timbmg/VAE-CVAE-MNIST" target="_blank">timbmg's</a> implementation of the same
3. *HpRNet* : Joint Modeling (JNet) of the harmonic and residual sum and differences
4. *sampling_synth* : Sample points from (C)VAE latent space using random walk, and synthesize audio corresponding to the sampled points
5. *HpR_modified* : Modified HpR model incorporating additional modifications (TAE, pYIN over TwM)
6. *func_envs* : TAE algorithm and other miscellaneous envelope extractors
7. *morphing* : Morphing between different audio representations (linear interpolation only)