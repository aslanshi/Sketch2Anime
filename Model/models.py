import torch
import torch.nn as nn
from . import network

class Component_AE(nn.Module):

	def __init__(self, params):
	"""params = {'component': str, 'CAE_norm': str}"""  
        super(Component_AE, self).__init__()

        self.encoder = define_part_encoder(model=params['component'], norm=params['CAE_norm'])
        self.decoder = define_part_decoder(model=params['component'], norm=params['CAE_norm'])

    def load_model(self, network, weights):
    	# load weights for inference and set model to eval mode.
        assert network in ['encoder', 'decoder'], 'Can only load encoder and decoder!'
        if network == 'encoder':
        	self.encoder.load_state_dict(torch.load(weights))
        	self.encoder.eval()
        else:
            self.decoder.load_state_dict(torch.load(weights))
            self.decoder.eval()

    def get_latent(self, input_component):

        latent = self.encoder(input_component)
        return latent

    def get_projection(self, input_component):
    	# get manifold projection
        latent = self.get_latent(input_component)
    	pass

class Feature_Decoder(nn.Module):

	def __init__(self, params):
	"""params = {'component': str, 'FD_norm': str, 'FD_output_channels': int}"""  
        super(Feature_Decoder, self).__init__()
        self.decoder = define_feature_decoder(model=params['component'], norm=params['FD_norm'], output_nc = params['FD_output_channels'])

    def load_model(self, weights):
        # load weights for inference and set model to eval mode.
    	self.decoder.load_state_dict(torch.load(weights))
    	self.decoder.eval()

class netG(nn.Module):

	def __init__(self, params):
	"""params = {'G_input_nc': int, 'ngf': int, 'G_norm': str}"""
	#input_nc, output_nc, ngf, n_downsample_global=3, n_blocks_global=9, norm='instance'
		super(netG, self).__init__()
		self.G = define_G(input_nc=params['G_input_nc'], ngf=params['ngf'], norm=['G_norm'])

	def load_model(self, weights):

		self.G.load_state_dict(torch.load(weights))
		self.G.eval()


