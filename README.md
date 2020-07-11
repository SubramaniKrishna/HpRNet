# HpRNet : Incorporating Residual Noise Modeling for Violin in a Variational Parametric Synthesizer
### Krishna Subramani, Preeti Rao : IIT Bombay

<!-- <a href="https://www.ee.iitb.ac.in/student/~krishnasubramani/data/icassp_paper.pdf" target="_blank">Paper</a> 	/	 <a href="https://www.ee.iitb.ac.in/student/~krishnasubramani/icassp2020.html" target="_blank">Accompanying Webpage</a> 	/	<a href="https://www.ee.iitb.ac.in/student/~krishnasubramani/data/vapar.bib" target="_blank">BibTeX</a> -->

----

This repository contains the code for **HpRNet**, an extension of our previous work <a href="https://github.com/SubramaniKrishna/VaPar-Synth" target="_blank">VaPar Synth</a>. It is a  Conditional Variational Autoencoder trained on a source-filter inspired parametric representation. However we focus on the generative modeling of the residual bow noise to make for more natural tone quality.

We also introduce a new dataset, the **Carnatic Violin Dataset** [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3940330.svg)](https://doi.org/10.5281/zenodo.3940330)


#### Setting up an Anaconda Environment
For the necessary libraries/prerequisites, please use conda/anaconda to create an environment (from the environment.yml file in this repository) with the command   
~~~
conda env create -f environment.yml
~~~
Also install <a href="https://github.com/MTG/sms-tools" target="_blank">SMS-Tools</a> in the same environment. With these, all the code in the repository can be run inside this environment by activating it.

#### Code
A lot of our code is recycled and modified from our previous project <a href="https://github.com/SubramaniKrishna/VaPar-Synth" target="_blank">VaPar Synth</a>. 

1. [Dependencies](./Data_Loading/README.md): Functions for TAE extraction, PyTorch Dataloading, Sampling from the network etc.
2. [Parametric](./Parametric/README.md): Obtaining the parametric representation of the audio.
3. [Network](./Network/README.md): PyTorch code for the various networks. 
4. [Analysis](./Analysis/README.md): Code to analyze the network outputs (compute MSE/visualize Latent Space with t-SNE)
5. [GUI](./GUI/README.md): We also present a simple GUI (inspired from SMS-Tools!) for researchers to play around with. They can load pre-trained network weights and reconstruct/generate user input audio files.

----

If you use the dataset or the code, please refer to our work as:
~~~
@dataset{krishna_subramani_2020_3940330,
  author       = {Krishna Subramani and
                  Preeti Rao},
  title        = {Carnatic Violin Dataset},
  month        = jul,
  year         = 2020,
  publisher    = {Zenodo},
  version      = {1.0},
  doi          = {10.5281/zenodo.3940330},
  url          = {https://doi.org/10.5281/zenodo.3940330}
}
~~~


<!-- Accompanying repository for HpRNet : Incorporating Residual Noise Modeling for Violin in a Variational Parametric Synthesizer -->
