import torch.nn as nn 
import torch.nn.functional as F 
import torch 

import src.clip as clip 
import src.fusion as fusion
from src.blocks import Up, ConvBlock, IdentityBlock

class LEDModel(nn.Module):
    """ CLIP RN50 with U-Net skip connections """
    def __init__(self, config):
        super(LEDModel, self).__init__()
        self.config = config 
        self.up_factor = 2 if self.config['bilinear'] else 1
        self.clip_rn50, self.preprocess = clip.load("RN50")

        # Freezing the CLIP model
        for param in self.clip_rn50.parameters():
            param.requires_grad = False

        self._build_decoder()


    def _build_decoder(self):
        # language
        self.lang_fuser1 = fusion.names[self.config['lang_fusion_type']](input_dim=self.config['num_post_clip_channels'] // 2)
        self.lang_fuser2 = fusion.names[self.config['lang_fusion_type']](input_dim=self.config['num_post_clip_channels'] // 4)
        self.lang_fuser3 = fusion.names[self.config['lang_fusion_type']](input_dim=self.config['num_post_clip_channels'] // 8)

        # CLIP encoder output -> 1024
        self.proj_input_dim = 512 if 'word' in self.config['lang_fusion_type'] else 1024
        self.lang_proj1 = nn.Linear(self.proj_input_dim, 1024)
        self.lang_proj2 = nn.Linear(self.proj_input_dim, 512)
        self.lang_proj3 = nn.Linear(self.proj_input_dim, 256)

        # vision
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.config['num_post_clip_channels'], 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True)
        )
        self.up1 = Up(2048, 1024 // self.up_factor, self.config['bilinear'])

        self.up2 = Up(1024, 512 // self.up_factor, self.config['bilinear'])

        self.up3 = Up(512, 256 // self.up_factor, self.config['bilinear'])

        self.layer1 = nn.Sequential(
            ConvBlock(128, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.config['batch_norm']),
            IdentityBlock(64, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.config['batch_norm']),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        self.layer2 = nn.Sequential(
            ConvBlock(64, [32, 32, 32], kernel_size=3, stride=1, batchnorm=self.config['batch_norm']),
            IdentityBlock(32, [32, 32, 32], kernel_size=3, stride=1, batchnorm=self.config['batch_norm']),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        self.layer3 = nn.Sequential(
            ConvBlock(32, [16, 16, 16], kernel_size=3, stride=1, batchnorm=self.config['batch_norm']),
            IdentityBlock(16, [16, 16, 16], kernel_size=3, stride=1, batchnorm=self.config['batch_norm']),
            nn.UpsamplingBilinear2d(scale_factor=1),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, self.config['num_output_channels'], kernel_size=1)
        )

    def encode_image(self, img):
        with torch.no_grad():
            # The default CLIP function has been updated to be able to get intermediate prepools 
            img_encoding, img_im = self.clip_rn50.visual.prepool_im(img)
        return img_encoding, img_im

    def encode_text(self, x):
        x = x.type(torch.LongTensor)
        with torch.no_grad():
            text_feat = self.clip_rn50.encode_text(x)
            text_feat = torch.repeat_interleave(text_feat, self.config['max_floors'], 0)

        text_mask = torch.where(x==0, x, 1)  # [1, max_token_len]
        return text_feat, text_mask



    def forward(self, x, l):
        B, num_maps, C, H, W = x.size()
        x = x.view(B*num_maps, C, H, W)
        in_type = x.dtype
        in_shape = x.shape
        x = x[:,:3]  # select RGB
        x, im = self.encode_image(x)
        x = x.to(in_type)

        # encode text
        l_enc, l_mask = self.encode_text(l)
        l_input = l_enc
        l_input = l_input.to(dtype=x.dtype)

        # # encode image
        assert x.shape[1] == self.config['num_post_clip_channels']
        # print('after CLIP encoding: ', x.size())
        x = self.conv1(x)

        # print('after convolution after CLIP encoding: ', x.size())


        x = self.lang_fuser1(x, l_input, x2_mask=l_mask, x2_proj=self.lang_proj1)
        # print('after lang_fuser 1: ', x.size())
        x = self.up1(x, im[-2])
        # print('after up after lang_fuser 1: ', x.size())

        x = self.lang_fuser2(x, l_input, x2_mask=l_mask, x2_proj=self.lang_proj2)
        # print('after lang_fuser 2: ', x.size())
        x = self.up2(x, im[-3])
        # print('after up after lang_fuser 2: ', x.size())

        x = self.lang_fuser3(x, l_input, x2_mask=l_mask, x2_proj=self.lang_proj3)
        # print('after lang_fuser 3: ', x.size())
        x = self.up3(x, im[-4])
        # print('after up after lang_fuser 3: ', x.size())

        for enum, layer in enumerate([self.layer1, self.layer2, self.layer3, self.conv2]):
            x = layer(x)
            # print(f'after layer {enum} after all lang_fusions', x.size())
        
        h, w = x.size()[-2], x.size()[-1]
        x = x.squeeze(1)
        x = x.view(B, num_maps, x.size()[-2], x.size()[-1])
        x = F.log_softmax(x.view(B, -1), 1).view(B, num_maps, h, w)
        return x
