import torch
import torch.nn as nn
from PIL import Image
from models import *

####################
# work in progress #
####################

use_gpu = True

class Combined_Model(nn.Module):

    """
    combine all networks
    """

    def __init__(self, params, inference=False):
        super(Combined_Model, self).__init__()

        self.params = params
        self.parts = self.params['parts']

        # define networks
        self.part_encoder = {}
        self.part_feature_decoder = {}
        for key in self.parts.keys():
          if use_gpu:
            self.part_encoder[key] = Component_AE(self.params, key).cuda()
            self.part_feature_decoder[key] = Feature_Decoder(self.params, key).cuda()
          else:
            self.part_encoder[key] = Component_AE(self.params, key)
            self.part_feature_decoder[key] = Feature_Decoder(self.params, key)

        self.G = netG(self.params)
        # self.D = netD(self.params)

        for key in self.parts.keys():
            # self.part_encoder[key].load_model('encoder', self.parts[key]['cae_weights'])
            self.part_encoder[key].load_state_dict(torch.load(self.parts[key]['cae_weights']))
            self.part_encoder[key].eval()
            for p in self.part_encoder[key].parameters(): p.requires_grad = False

        # load weights for FD and G when inferencing
        if inference:
            for key in self.parts.keys():
                self.part_feature_decoder[key].load_model(self.parts[key]['fd_weights'])
            self.G.load_model(self.params['g_weights'])


    def forward(self, sketch, target, user_hints=None):

        #####################################################################################
        # part_projections = {}

        # for key in self.parts.keys():
        #     part_projections[key] = self.part_encoder[key].get_projection(components[key])
        #     part_decoded[key] = self.feature_decoder[key](part_projections[key])
        #####################################################################################

        latent = self.part_encoder['face'].get_latent(sketch)
        decoded_latent = self.part_feature_decoder['face'](latent)
        
        generated_image, hints = self.G(decoded_latent, target, user_hints)

        return generated_image, hints

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

        pass

