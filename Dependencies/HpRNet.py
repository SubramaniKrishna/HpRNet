"""
Script containing defining code for ``HpRNet''
This network jointly models the harmonic and residual in the following way:
let the cepstral representations of the harmonic component be x, and the residual be y (the network reconstructions being x' and y').
HpRNet is two independent CVAEs (N1,N2), however they both draw points from the same latent space i.e. the epsilon drawn from the N(0,1) is the same for both
N1 autoencodes 0.5(x + y) -> 0.5(x + y) -> N1 -> 0.5(x' + y') = n1
N2 autoencodes 0.5(x - y) -> 0.5(x - y) -> N2 -> 0.5(x' - y') = n2

The harmonic reconstruction is (n1 + n2), and the residual reconstruction is (n1 - n2).
"""

# Dependencies
import torch
import torch.nn as nn
import copy


# Defining the Classes corresponding to the Conditional Encoder and Decoder first

class c_Enc(nn.Module):
	"""
	Class specifying the conditional encoder architecture(data conditioned on Y{X|Y} -> latent space(z|Y))
	
	Constructor Parameters
	----------------------
	layer_dims : list of integers
		List containing the dimensions of consecutive layers(Including input layer{dim(X)} and excluding latent layer)
	latent_dims : integer
		Size of latent dimenstion(default = 2)
	flag_cond: boolean
		Flag to implement conditional VAE, if True will implement Conditional VAE, else not(default = True)
	num_cond : integer
		Number of conditional variables(dimension of the conditional vector Y in X|Y)
	"""

	# Defining the constructor to initialize the network layers and activations
	def __init__(self, layer_dims, num_cond, latent_dims = 2, flag_cond = True):
		super().__init__()
		self.layer_dims = layer_dims
		self.latent_dims = latent_dims
		self.flag_cond = flag_cond
		self.num_cond = num_cond

		# Check if flag_cond is deployed, if yes, size of the input vector is dim(X) + num_cond, else it is just dim(X)
		if(self.flag_cond):
			layer_dims[0] = layer_dims[0] + num_cond

		# Initializing the Model as a torch sequential model
		self.cENC_NN = nn.Sequential()

		# Currently using Linear layers with ReLU activations(potential hyperparams)
		# This Loop defines the layers just before the latent space (input(X) -> layer[0] -> layer[1] .... -> layer[n])
		for i, (in_size, out_size) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
			self.cENC_NN.add_module(name = "Encoder_Layer_{:d}".format(i), module = nn.Linear(in_size, out_size))
			self.cENC_NN.add_module(name = "Activation_{:d}".format(i), module = nn.ReLU())

		# Defining the final layer from layer[n] -> latent space gaussian parameters(mean and variance)
		self.mu = nn.Linear(layer_dims[-1], latent_dims)
		self.sigma = nn.Linear(layer_dims[-1], latent_dims)

	def forward(self, x, c = None):
		"""
		Forward pass of the encoder to obtain the latent space parameters

		Inputs
		------
		x : torch.tensor
			Input tensor
		c : torch.tensor
			Conditioning variable tensor
		"""
		if(self.flag_cond):
			x = torch.cat((x,c),dim = -1)

		# Forward pass till the n-1'th layer
		x = self.cENC_NN(x)

		# Forward through the final layer to obtain Parameters(mean and variances)
		mu = self.mu(x)
		sigma_covar = self.sigma(x)

		return mu, sigma_covar


class c_Dec(nn.Module):
	"""
	Class specifying the conditional decoder architecture(latent space(z|Y) -> Reconstructed Data X'|Y)
	
	Constructor Parameters
	----------------------
	layer_dims : list of integers
		List containing the dimensions of consecutive layers(Including latent dimension{dim(X)} and including final layer, which should be same size as input)
	flag_cond: boolean
		Flag to implement conditional VAE, if True will implement Conditional VAE, else not(default = True)
	num_cond : integer
		Number of conditional variables(dimension of the conditional vector Y in X|Y)
	"""

	# Defining the constructor to initialize the network layers and activations
	def __init__(self, layer_dims, num_cond, flag_cond = True):
		super().__init__()
		self.layer_dims = layer_dims
		self.flag_cond = flag_cond
		self.num_cond = num_cond

		# Check if flag_cond is deployed, if yes, size of the input vector to decoder is dim(latent space) + num_cond, else it is just dim(latent space)
		if(self.flag_cond):
			layer_dims[0] = layer_dims[0] + num_cond

		# Initializing the Model as a torch sequential model
		self.cDEC_NN = nn.Sequential()

		# Currently using Linear layers with ReLU activations(potential hyperparams)
		# This Loop defines the layers after the latent space(latent space -> layer[0] -> layer[1] .... -> layer[n] -> Reconstructed output)
		# <Point to note>, the final layer should allow negative values as well. Also, the inputs should be scaled before feeding to the network
		# to restrict them to (-1,1), and then use an appropriate activation(like tanh) to get the output in the range(-1,1) abd rescale it back.
		for i, (in_size, out_size) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
			self.cDEC_NN.add_module(name="Decoder_Layer{:d}".format(i), module=nn.Linear(in_size, out_size))
			if i + 1 < len(layer_dims) - 1:
				self.cDEC_NN.add_module(name="Activation{:d}".format(i), module=nn.LeakyReLU(negative_slope = 0.1))
			else:
				self.cDEC_NN.add_module(name="Reconstruct_LReLU", module=nn.LeakyReLU(negative_slope = 1))


	def forward(self, z, c = None):
		"""
		Forward pass of the decoder to obtain the reconstructed input
		
		Inputs
		------
		z : torch.tensor
			Latent variable
		c : torch.tensor
			Conditioning variable tensor
		"""

		if(self.flag_cond):
			z = torch.cat((z, c), dim=-1)

		x = self.cDEC_NN(z)

		return x


# Overall cVAE class combining the Encoder and Decoder classes

class HPRNet(nn.Module):
	"""
	Class defining the {conditional} autoencoder for the HpRNet by combining the Encoder and Decoder
	X|Y -> Encoder -> Latent{z|Y} -> Decoder -> X'|Y
	
	Constructor Parameters
	----------------------
	flag_cond: boolean
		Flag to implement conditional VAE, if True will implement Conditional VAE, else not(default = True)
	layer_dims_enc : list of integers
		List containing the dimensions of consecutive layers for the encoder(Including input layer{dim(X)} and excluding latent layer)
	layer_dims_dec : list of integers
		List containing the dimensions of consecutive layers for the decoder(Including latent dimension{dim(X)} and including final layer, which should be same size as input)
	latent_dims : integer
		Size of latent dimenstion(default = 2)
	num_cond : integer
		Number of conditional variables(dimension of the conditional vector Y in X|Y)
	device : 'cuda' or 'cpu'
		Where to run the optimization
	"""

	# Constructor defining the architecture
	def __init__(self, num_cond, device, layer_dims_enc, layer_dims_dec, latent_dims = 2, flag_cond = True):
		super().__init__() #Understand more about how super works!

		self.flag_cond = flag_cond
		self.layer_dims_enc = layer_dims_enc
		self.layer_dims_dec = layer_dims_dec
		self.latent_dims = latent_dims
		self.num_cond = num_cond
		self.device = device

		l_enc_N1 = copy.deepcopy(layer_dims_enc)
		l_enc_N2 = copy.deepcopy(layer_dims_enc)
		l_dec_N1 = copy.deepcopy(layer_dims_dec)
		l_dec_N2 = copy.deepcopy(layer_dims_dec) 

		# Defining the Encoder and Decoder architecture by calling the previously defined classes
		# Encoder, Decoder for N1
		self.main_cENC_N1 = c_Enc(layer_dims = l_enc_N1, latent_dims = latent_dims, flag_cond = flag_cond,num_cond = num_cond)
		self.main_cDEC_N1 = c_Dec(layer_dims = l_dec_N1,flag_cond = flag_cond,num_cond = num_cond)
		# Encoder, Decoder for N2
		self.main_cENC_N2 = c_Enc(layer_dims = l_enc_N2, latent_dims = latent_dims, flag_cond = flag_cond,num_cond = num_cond)
		self.main_cDEC_N2 = c_Dec(layer_dims = l_dec_N2,flag_cond = flag_cond,num_cond = num_cond)


	# Forward pass
	def forward(self, x, y , c = None):
		"""
		Forward pass of the Autoencoder
		
		Inputs
		------
		x : torch.tensor
			Input tensor
		c : torch.tensor
			Conditioning variable tensor
		"""

		batch_size = x.size(0)

		# Encode data x to obtain parameters of the distribution
		mu_n1, sigma_covar_n1 = self.main_cENC_N1(0.5*(x + y), c)
		mu_n2, sigma_covar_n2 = self.main_cENC_N2(0.5*(x - y), c)
		
		# Reparametrization
		# z = mu + sigma_covar * eps, where eps is sampled from N(0,I)
		sigma_covar_n1 = torch.exp(sigma_covar_n1)
		sigma_covar_n2 = torch.exp(sigma_covar_n2)
		eps = torch.randn([batch_size, self.latent_dims]).to(self.device)

		z_n1 = mu_n1 + (sigma_covar_n1 * eps)
		z_n2 = mu_n2 + (sigma_covar_n2 * eps)

		# Run this through decoder
		n1_recon = self.main_cDEC_N1(z_n1,c)
		n2_recon = self.main_cDEC_N2(z_n2,c)

		return n1_recon, mu_n1, sigma_covar_n1, z_n1, n2_recon, mu_n2, sigma_covar_n2, z_n2


	# Set of functions to perform inference and generate sequences of corresponding outputs

	# Sampling from Latent space
	def sample_latent_space(self, z, c = None):
		"""
		Sampling z from the latent space and obtaining the corresponding reconstructed point
		
		Inputs
		------
		z - torch.tensor
			Tensor containing the latent variables
		c - torch.tensor
			Tensor containing the corresponding conditional variables
		"""

		sampled_X = self.main_cDEC(z,c)

		return sampled_X

	# Return reconstructed H',R' for input H,R
	def return_HR(self, x, y , c = None):

		batch_size = x.size(0)

		# Encode data x to obtain parameters of the distribution
		mu_n1, sigma_covar_n1 = self.main_cENC_N1(0.5*(x + y), c)
		mu_n2, sigma_covar_n2 = self.main_cENC_N2(0.5*(x - y), c)
		
		# Reparametrization
		# z = mu + sigma_covar * eps, where eps is sampled from N(0,I)
		sigma_covar_n1 = torch.exp(sigma_covar_n1)
		sigma_covar_n2 = torch.exp(sigma_covar_n2)
		eps = torch.randn([batch_size, self.latent_dims]).to(self.device)

		z_n1 = mu_n1 + (sigma_covar_n1 * eps)
		z_n2 = mu_n2 + (sigma_covar_n2 * eps)

		# Run this through decoder
		n1_recon = self.main_cDEC_N1(z_n1,c)
		n2_recon = self.main_cDEC_N2(z_n2,c)

		x_recon = (n1_recon + n2_recon)
		y_recon = (n1_recon - n2_recon)

		return x_recon,y_recon

	

