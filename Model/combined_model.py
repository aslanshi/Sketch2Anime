import torch
import torch.nn as nn
from models import *

####################
# work in progress #
####################

class Combined_Model(nn.Module):

    """
    combine all networks and use for inference
    """

    def __init__(self, params, inference=False):
        super(Combined_Model, self).__init__()

        self.params = params
        self.parts = self.params['parts']

        # define networks
        self.part_encoder = {}
        self.part_feature_decoder = {}
        for key in self.parts.keys():
            self.part_encoder[key] = Component_AE(self.params, key, encoder_only=True)
            self.part_feature_decoder[key] = Feature_Decoder(self.params, key)

        self.G = netG(self.params)

        for key in self.parts.keys():
            self.part_encoder[key].load_model('encoder', self.parts[key]['cae_weights'])

        # load weights for FD and G when inferencing
        if inference:
            for key in self.parts.keys():
                self.part_feature_decoder[key].load_model('encoder', self.parts[key]['fd_weights'])
            self.G.load_model(self.params['g_weights'])

    def inference(self, sketch, hint):

        ###############################################################
        
        # data processing

        # return dict components = {'eye1': torch.Tensor([x, y]), etc.}
        
        ##############################################################

        part_projections = {}
        part_decoded = {}
        for key in self.parts.keys():
            part_projections[key] = self.part_encoder[key].get_projection(components[key])
            part_decoded[key] = self.feature_decoder[key](part_projections[key])

        ##############################################################################

        # part_decoded['face'][:, :, 301:301+192, 169:169+192] = part_decoded['mouth']
        # etc

        ##############################################################################

        input_concat = torch.concat((part_decoded['face'], hint), 1)

        generated_image = self.G(input_concat)

        #######################

        # transpose for showing

        #######################

        return generated_image

