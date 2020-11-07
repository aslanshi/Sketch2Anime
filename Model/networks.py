# coding: utf-8


import torch
import torch.nn as nn
import numpy as np
import torchvision.models as M

VGG16_PATH = './Model/vgg16-397923af.pth'

#########################################
#               Functions               #   
#########################################

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
        
def get_norm_layer(norm_type='instance'):
    if (norm_type == 'batch'):
        norm_layer = nn.BatchNorm2d
    elif (norm_type == 'instance'):
        norm_layer = nn.InstanceNorm2d
    else:
        raise NotImplementedError(('normalization layer [%s] is not found' % norm_type))
    return norm_layer

# class MSELoss:
#     def __init__(self):
#         pass

#     def __call__(self, output, target):
#         return nn.functional.mse_loss(output, target)

# class BCELoss:
#     def __init__(self):
#         pass

#     def __call__(self, output, target):
#         return nn.functional.binary_cross_entropy(output, target)


#########################################
#               Networks                #   
#########################################

class EncoderBlock(nn.Module):
    
    def __init__(self, channel_in, channel_out, kernel_size=7, padding=3, stride=4):
        super(EncoderBlock, self).__init__()
        # convolution to halve the dimensions
        self.conv = nn.Conv2d(channel_in, channel_out, kernel_size, padding=padding, stride=stride)
        self.bn = nn.BatchNorm2d(channel_out, momentum=0.9)
        self.relu = nn.LeakyReLU()

    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DecoderBlock(nn.Module):

    def __init__(self, channel_in, channel_out, kernel_size=4, padding=1, stride=2, output_padding=0, norelu=False):
        super(DecoderBlock, self).__init__()
        layers_list = []
        layers_list.append(nn.ConvTranspose2d(channel_in, channel_out, kernel_size, padding=padding, stride=stride, output_padding=output_padding))
        layers_list.append(nn.BatchNorm2d(channel_out, momentum=0.9))
        if (norelu == False):
            layers_list.append(nn.LeakyReLU())
        self.conv = nn.Sequential(*layers_list)

    def forward(self, x):
        x = self.conv(x)
        return x


class ResnetBlock(nn.Module):
    
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if (padding_type == 'reflect'):
            conv_block += [nn.ReflectionPad2d(1)]
        elif (padding_type == 'replicate'):
            conv_block += [nn.ReplicationPad2d(1)]
        elif (padding_type == 'zero'):
            p = 1
        else:
            raise NotImplementedError(('padding [%s] is not implemented' % padding_type))
        conv_block += [nn.Conv2d(dim, dim, 3, padding=p), norm_layer(dim), activation]
        
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if (padding_type == 'reflect'):
            conv_block += [nn.ReflectionPad2d(1)]
        elif (padding_type == 'replicate'):
            conv_block += [nn.ReplicationPad2d(1)]
        elif (padding_type == 'zero'):
            p = 1
        else:
            raise NotImplementedError(('padding [%s] is not implemented' % padding_type))
        conv_block += [nn.Conv2d(dim, dim, 3, padding=p), norm_layer(dim)]
        
        return nn.Sequential(*conv_block)

    def forward(self, x):

        x = (x + self.conv_block(x))
        return x

class Encoder_Res(nn.Module):
    """docstring for  EncoderGenerator"""
    
    def __init__(self, norm_layer, image_size, input_nc, latent_dim=512):
        super(Encoder_Res, self).__init__()
        layers_list = []
        
        latent_size = int(image_size/32)
        longsize = 512*latent_size*latent_size
        self.longsize = longsize
        # print(image_size,latent_size, longsize)

        activation = nn.ReLU()
        padding_type='reflect'
        norm_layer=nn.BatchNorm2d

        # encode
        layers_list.append(EncoderBlock(channel_in=input_nc, channel_out=32, kernel_size=4, padding=1, stride=2))  # 176 176 

        dim_size = 32
        for i in range(4):
            layers_list.append(ResnetBlock(dim_size, padding_type=padding_type, activation=activation, norm_layer=norm_layer)) 
            layers_list.append(EncoderBlock(channel_in=dim_size, channel_out=dim_size*2, kernel_size=4, padding=1, stride=2)) 
            dim_size *= 2

        layers_list.append(ResnetBlock(512, padding_type=padding_type, activation=activation, norm_layer=norm_layer))  

        # final shape Bx256*7*6
        self.conv = nn.Sequential(*layers_list)
        self.fc_mu = nn.Sequential(nn.Linear(in_features=longsize, out_features=latent_dim))

        # self.fc_var = nn.Sequential(nn.Linear(in_features=longsize, out_features=latent_dim))

        for m in self.modules():
            weights_init_normal(m)

    def forward(self, x):

        x = self.conv(x)
        x = torch.reshape(x,(x.size()[0],-1))
        mu = self.fc_mu(x)
        return mu

class Decoder_Res(nn.Module):

    def __init__(self, norm_layer, image_size, output_nc, latent_dim=512):  
        super(Decoder_Res, self).__init__()
        # start from B*1024
        latent_size = int(image_size/32)
        self.latent_size = latent_size
        longsize = 512*latent_size*latent_size

        activation = nn.ReLU()
        padding_type='reflect'
        norm_layer=nn.BatchNorm2d

        self.fc = nn.Sequential(nn.Linear(in_features=latent_dim, out_features=longsize))
        layers_list = []

        layers_list.append(ResnetBlock(512, padding_type=padding_type, activation=activation, norm_layer=norm_layer))  # 176 176 
        
        dim_size = 256
        for i in range(4):
            layers_list.append(DecoderBlock(channel_in=dim_size*2, channel_out=dim_size, kernel_size=4, padding=1, stride=2, output_padding=0)) #latent*2
            layers_list.append(ResnetBlock(dim_size, padding_type=padding_type, activation=activation, norm_layer=norm_layer))  
            dim_size = int(dim_size/2)

        layers_list.append(DecoderBlock(channel_in=32, channel_out=32, kernel_size=4, padding=1, stride=2, output_padding=0)) #352 352
        layers_list.append(ResnetBlock(32, padding_type=padding_type, activation=activation, norm_layer=norm_layer))  # 176 176 

        # layers_list.append(DecoderBlock(channel_in=64, channel_out=64, kernel_size=4, padding=1, stride=2, output_padding=0)) #96*160
        layers_list.append(nn.ReflectionPad2d(2))
        layers_list.append(nn.Conv2d(32, output_nc, kernel_size=5, padding=0))

        self.conv = nn.Sequential(*layers_list)

        for m in self.modules():
            weights_init_normal(m)

    def forward(self, x):

        x = self.fc(x)
        x = torch.reshape(x,(x.size()[0],512, self.latent_size, self.latent_size))
        x = self.conv(x)

        return x

class Decoder_feature_Res(nn.Module):

    def __init__(self, norm_layer, image_size, output_nc, latent_dim=512):  
        super(Decoder_feature_Res, self).__init__()
        # start from B*1024
        latent_size = int(image_size/32)
        self.latent_size = latent_size
        longsize = 512*latent_size*latent_size

        activation = nn.ReLU()
        padding_type='reflect'
        norm_layer=nn.BatchNorm2d

        self.fc = nn.Sequential(nn.Linear(in_features=latent_dim, out_features=longsize))
        layers_list = []

        layers_list.append(ResnetBlock(512, padding_type=padding_type, activation=activation, norm_layer=norm_layer))  # 176 176 

        layers_list.append(DecoderBlock(channel_in=512, channel_out=256, kernel_size=4, padding=1, stride=2, output_padding=0)) #22 22
        layers_list.append(ResnetBlock(256, padding_type=padding_type, activation=activation, norm_layer=norm_layer))  # 176 176 

        layers_list.append(DecoderBlock(channel_in=256, channel_out=256, kernel_size=4, padding=1, stride=2, output_padding=0)) #44 44
        layers_list.append(ResnetBlock(256, padding_type=padding_type, activation=activation, norm_layer=norm_layer))  # 176 176 

        layers_list.append(DecoderBlock(channel_in=256, channel_out=128, kernel_size=4, padding=1, stride=2, output_padding=0)) #88 88 
        layers_list.append(ResnetBlock(128, padding_type=padding_type, activation=activation, norm_layer=norm_layer))  # 176 176 

        layers_list.append(DecoderBlock(channel_in=128, channel_out=64, kernel_size=4, padding=1, stride=2, output_padding=0)) #176 176
        layers_list.append(ResnetBlock(64, padding_type=padding_type, activation=activation, norm_layer=norm_layer))  # 176 176 

        layers_list.append(DecoderBlock(channel_in=64, channel_out=64, kernel_size=4, padding=1, stride=2, output_padding=0)) #352 352
        layers_list.append(ResnetBlock(64, padding_type=padding_type, activation=activation, norm_layer=norm_layer))  # 176 176 

        # layers_list.append(DecoderBlock(channel_in=64, channel_out=64, kernel_size=4, padding=1, stride=2, output_padding=0)) #96*160
        layers_list.append(nn.ReflectionPad2d(2))
        layers_list.append(nn.Conv2d(64, output_nc, kernel_size=5, padding=0))

        self.conv = nn.Sequential(*layers_list)

        for m in self.modules():
            weights_init_normal(m)

    def forward(self, x):

        x = self.fc(x)
        x = torch.reshape(x,(x.size()[0],512, self.latent_size, self.latent_size))
        x = self.conv(x)

        return x

class GlobalGenerator(nn.Module):

    def __init__(self, input_nc, output_nc=3, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        assert (n_blocks >= 0)
        super(GlobalGenerator, self).__init__()
        activation = nn.ReLU()

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, 7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = (2 ** i)
            model += [nn.Conv2d((ngf * mult), ((ngf * mult) * 2), 3, stride=2, padding=1), norm_layer(((ngf * mult) * 2)), activation]
        
        ### resnet blocks
        mult = (2 ** n_downsampling)
        for i in range(n_blocks):
            model += [ResnetBlock((ngf * mult), padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        
        ### upsample 
        for i in range(n_downsampling):
            mult = (2 ** (n_downsampling - i))
            model += [nn.ConvTranspose2d((ngf * mult), int(((ngf * mult) / 2)), 3, stride=2, padding=1, output_padding=1), norm_layer(int(((ngf * mult) / 2))), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, 7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)

        for m in self.modules():
            weights_init_normal(m)

    def forward(self, input):
        return self.model(input)

class Discriminator(nn.Module):

    def __init__(self, input_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d):
        super(Discriminator, self).__init__()

        activation = nn.ReLU()
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, 7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = (2 ** i)
            model += [nn.Conv2d((ngf * mult), ((ngf * mult) * 2), 3, stride=2, padding=1), norm_layer(((ngf * mult) * 2)), activation]
        
        ### resnet blocks
        mult = (2 ** n_downsampling)
        for i in range(n_blocks):
            model += [ResnetBlock((ngf * mult), padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        model += [nn.Linear(ngf * mult, 1)]
        self.model = nn.Sequential(*model)

        for m in self.modules():
            weights_init_normal(m)

    def forward(self, input):
        return self.model(input)
        
class VGGFeature(nn.Module):

    def __init__(self):
        super(VGGFeature, self).__init__()
        vgg = M.vgg16()
        vgg.load_state_dict(torch.load(VGG16_PATH))
        vgg.features = nn.Sequential(*list(vgg.features.children())[:9])
        self.model   = vgg.features

        self.register_buffer('mean', torch.FloatTensor([0.485 - 0.5, 0.456 - 0.5, 0.406 - 0.5]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, images):
        return self.model((images.mul(0.5) - self.mean) / self.std)

#########################################
#           Model Functions             #   
#########################################

def define_part_encoder(model='mouth', norm='instance', input_nc=1, latent_dim=512):
    """set encoder for each facial component during Component Embedding phase"""
    
    norm_layer = get_norm_layer(norm_type=norm)
    image_size = 512
    if 'eye' in model:
        image_size = 128
    elif 'mouth' in model:
        image_size = 192
    elif 'nose' in model:
        image_size = 160
    elif 'face' in model:
        image_size = 512
    else:
        print("Whole Image !!")
        
    # input longsize 256 to 512*4*4
    net_encoder = Encoder_Res(norm_layer, image_size, input_nc, latent_dim)     
    # print("net_encoder of part " + model + " is:", image_size)

    return net_encoder

def define_part_decoder(model='mouth', norm='instance', output_nc=1, latent_dim=512):
    """set decoder for each facial component during Component Embedding phase"""
    
    norm_layer = get_norm_layer(norm_type=norm)
    image_size = 512
    if 'eye' in model:
        image_size = 128
    elif 'mouth' in model:
        image_size = 192
    elif 'nose' in model:
        image_size = 160
    elif 'face' in model:
        image_size = 512
    else:
        print("Whole Image !!")
    
     # input longsize 256 to 512*4*4
    net_decoder = Decoder_Res(norm_layer, image_size, output_nc, latent_dim) 
    # print("net_decoder to image of part "+model+" is:",image_size)

    return net_decoder

def define_feature_decoder(model='mouth', norm='instance', output_nc=1, latent_dim=512):
    """set decoder for each facial component during Feature Mapping phase"""
    
    norm_layer = get_norm_layer(norm_type=norm)
    image_size = 512
    if 'eye' in model:
        image_size = 128
    elif 'mouth' in model:
        image_size = 192
    elif 'nose' in model:
        image_size = 160
    elif 'face' in model:
        image_size = 512
    else:
        print("Whole Image !!")

    # input longsize 256 to 512*4*4
    net_decoder = Decoder_feature_Res(norm_layer,image_size,output_nc, latent_dim) 
    # print("net_decoder to image of part "+model+" is:",image_size)
    
    return net_decoder

def define_G(input_nc, output_nc, ngf, n_downsample_global=3, n_blocks_global=9, norm='instance'):
    """define generator during Image Synthesis phase"""
    
    norm_layer = get_norm_layer(norm_type=norm)     
    netG = GlobalGenerator(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)
    return netG


