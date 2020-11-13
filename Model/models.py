import torch
import torch.nn as nn
from networks import *
from hint import *
import torchvision.transforms as transforms

use_gpu = True
if use_gpu:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = "cpu"

class Component_AE(nn.Module):

    def __init__(self, params, part, encoder_only=False):
    # params = {'CAE_norm': str}
        super(Component_AE, self).__init__()

        self.encoder_only = encoder_only

        if not self.encoder_only:
            self.encoder = define_part_encoder(model=part, norm=params['CAE_norm'])
            self.decoder = define_part_decoder(model=part, norm=params['CAE_norm'])
        else:
            self.encoder = define_part_encoder(model=part, norm=params['CAE_norm'])

    def forward(self, input):
        return self.decoder(self.encoder(input))

    def load_model(self, network, weights):
        # load weights for inference and set model to eval mode.
        # assert network in ['encoder', 'decoder'], 'Can only load encoder and decoder!'
        # if network == 'encoder':
        #     self.encoder.load_state_dict(torch.load(weights))
        #     self.encoder.eval()
        # else:
        #     self.decoder.load_state_dict(torch.load(weights))
        #     self.decoder.eval()
        pass

    def get_latent(self, input_component):
        # call after load weights
        latent = self.encoder(input_component)
        return latent

    def get_projection(self, input_component):
        # get manifold projection
        latent = self.get_latent(input_component)
        pass

class Feature_Decoder(nn.Module):

    def __init__(self, params, part):
    # params = {FD_norm': str, 'FD_output_channels': int}
        super(Feature_Decoder, self).__init__()
        self.decoder = define_feature_decoder(model=part, norm=params['FD_norm'], output_nc = params['FD_output_channels'])

    def forward(self, input_component):
        return self.decoder(input_component)

    def load_model(self, weights):
        # load weights for inference and set model to eval mode.
        self.decoder.load_state_dict(torch.load(weights))
        self.decoder.eval()

class netG(nn.Module):

    def __init__(self, params):
    # params = {'G_input_nc': int, 'ngf': int, 'G_norm': str, 'G_n_downsampling': int, 'G_n_blocks': int}
        super(netG, self).__init__()
        self.G = define_G(input_nc=params['G_input_nc'], ngf=params['ngf'], norm=params['G_norm'], n_downsample_global=params['G_n_downsampling'], n_blocks_global=params['G_n_blocks'])

    def forward(self, decoded_sketch, target):
        
        x = self.G.model1(decoded_sketch)
        # print(x.shape)
        hints = get_hints(target)
        # print(hints.shape)
        x = torch.cat([x, hints], 1)
        g_image = self.G.model2(x)

        return g_image, hints

    def load_model(self, weights):

        self.G.load_state_dict(torch.load(weights))
        self.G.eval()

class netD(nn.Module):

    def __init__(self, params):
        super(netD, self).__init__()
        self.D = define_D(n_downsampling=params['D_n_downsampling'], n_blocks=params['D_n_blocks'])

    def forward(self, input, hints=None):
        
        x = self.D.model1(input)
        if hints == None:
            hints = get_hints(input)
        x = torch.cat([x, hints], 1)
        score = self.D.model2(x)

        return score

    def load_model(self, weights):

        self.D.load_state_dict(torch.load(weights))
        self.D.eval()

def get_hints(target):

    """ Note the target has been normalized """

    if not torch.is_tensor(target):
        target = transforms.ToTensor()(target)

    mean, std = torch.tensor([0.5, 0.5, 0.5], device=device), torch.tensor([0.5, 0.5, 0.5], device=device)
    # target = target * std + mean

    img_size = (512, 512)
    mask_gen = Hint((img_size[0] // 4, img_size[1] // 4), 120, (1, 5), 5, (10, 10))
    htransform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    hints = []

    for i in range(target.shape[0]):
        if use_gpu:
            hint = mask_gen((target[i].permute(2, 1, 0) * std + mean).detach().cpu().numpy() * 255., htransform).cuda()
        else:
            hint = mask_gen((target[i].permute(2, 1, 0) * std + mean).detach().cpu().numpy() * 255., htransform)
        hints.append(hint)

    return torch.stack(hints)



