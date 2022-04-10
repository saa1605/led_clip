import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import copy
import clip
import src.clip_lingunet.fusion as fusion
import sys

def clones(module, N):
    """Produce N identical layers"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LinearProjectionLayers(nn.Module):
    def __init__(
        self, image_channels, linear_hidden_size, rnn_hidden_size, num_hidden_layers
    ):
        super(LinearProjectionLayers, self).__init__()

        if num_hidden_layers == 0:
            # map pixel feature vector directly to score without activation
            self.out_layers = nn.Linear(image_channels + rnn_hidden_size, 1, bias=False)
        else:
            self.out_layers = nn.Sequential(
                nn.Conv2d(
                    image_channels + rnn_hidden_size,
                    linear_hidden_size,
                    kernel_size=1,
                    padding=0,
                    stride=1,
                ),
                nn.ReLU(),
                nn.Conv2d(linear_hidden_size, 1, kernel_size=1, padding=0, stride=1),
            )
            self.out_layers.apply(self.init_weights)

    def forward(self, x):
        return self.out_layers(x)

    def init_weights(self, m):
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

class IdentityBlock(nn.Module):
    def __init__(self, in_planes, filters, kernel_size, stride=1, final_relu=True, batchnorm=True):
        super(IdentityBlock, self).__init__()
        self.final_relu = final_relu
        self.batchnorm = batchnorm

        filters1, filters2, filters3 = filters
        self.conv1 = nn.Conv2d(in_planes, filters1, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters1) if self.batchnorm else nn.Identity()
        self.conv2 = nn.Conv2d(filters1, filters2, kernel_size=kernel_size, dilation=1,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters2) if self.batchnorm else nn.Identity()
        self.conv3 = nn.Conv2d(filters2, filters3, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(filters3) if self.batchnorm else nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += x
        if self.final_relu:
            out = F.relu(out)
        return out

class ConvBlock(nn.Module):
    def __init__(self, in_planes, filters, kernel_size, stride=1, final_relu=True, batchnorm=True):
        super(ConvBlock, self).__init__()
        self.final_relu = final_relu
        self.batchnorm = batchnorm

        filters1, filters2, filters3 = filters
        self.conv1 = nn.Conv2d(in_planes, filters1, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters1) if self.batchnorm else nn.Identity()
        self.conv2 = nn.Conv2d(filters1, filters2, kernel_size=kernel_size, dilation=1,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters2) if self.batchnorm else nn.Identity()
        self.conv3 = nn.Conv2d(filters2, filters3, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(filters3) if self.batchnorm else nn.Identity()

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_planes, filters3,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(filters3) if self.batchnorm else nn.Identity()
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        if self.final_relu:
            out = F.relu(out)
        return out

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),                                     # (Mohit): argh... forgot to remove this batchnorm
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),                                     # (Mohit): argh... forgot to remove this batchnorm
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# class LingUNet(nn.Module):
#     def __init__(self, rnn_args, args):
#         super(LingUNet, self).__init__()
#         self.rnn_args = rnn_args

#         self.m = args.num_lingunet_layers
#         self.image_channels = args.linear_hidden_size
#         self.freeze_resnet = args.freeze_resnet
#         self.res_connect = args.res_connect
#         self.device = args.device

#         resnet = models.resnet18(pretrained=True)
#         modules = list(resnet.children())[:-4]
#         self.resnet = nn.Sequential(*modules)
#         if self.freeze_resnet:
#             for p in self.resnet.parameters():
#                 p.requires_grad = False

#         if not args.bidirectional:
#             self.rnn_hidden_size = args.rnn_hidden_size
#         else:
#             self.rnn_hidden_size = args.rnn_hidden_size * 2
#         assert self.rnn_hidden_size % self.m == 0

#         self.rnn = RNN(
#             rnn_args["input_size"],
#             args.embed_size,
#             args.rnn_hidden_size,
#             args.num_rnn_layers,
#             args.embed_dropout,
#             args.bidirectional,
#             args.embedding_dir,
#         ).to(args.device)

#         sliced_text_vector_size = self.rnn_hidden_size // self.m
#         flattened_conv_filter_size = 1 * 1 * self.image_channels * self.image_channels
#         self.text2convs = clones(
#             nn.Linear(sliced_text_vector_size, flattened_conv_filter_size), self.m
#         )

#         self.conv_layers = nn.ModuleList([])
#         for i in range(self.m):
#             self.conv_layers.append(
#                 nn.Sequential(
#                     nn.Conv2d(
#                         in_channels=self.image_channels
#                         if i == 0
#                         else self.image_channels,
#                         out_channels=self.image_channels,
#                         kernel_size=5,
#                         padding=2,
#                         stride=1,
#                     ),
#                     nn.BatchNorm2d(self.image_channels),
#                     nn.ReLU(True),
#                 )
#             )

#         # create deconv layers with appropriate paddings
#         self.deconv_layers = nn.ModuleList([])
#         for i in range(self.m):
#             in_channels = self.image_channels if i == 0 else self.image_channels * 2
#             out_channels = self.image_channels
#             self.deconv_layers.append(
#                 nn.Sequential(
#                     nn.ConvTranspose2d(
#                         in_channels=in_channels,
#                         out_channels=out_channels,
#                         kernel_size=5,
#                         padding=2,
#                         stride=1,
#                     ),
#                     nn.BatchNorm2d(out_channels),
#                     nn.ReLU(True),
#                 )
#             )

#         self.conv_dropout = nn.Dropout(p=0.25)
#         self.deconv_dropout = nn.Dropout(p=0.25)

#         self.out_layers = LinearProjectionLayers(
#             image_channels=self.image_channels,
#             linear_hidden_size=args.linear_hidden_size,
#             rnn_hidden_size=0,
#             num_hidden_layers=args.num_linear_hidden_layers,
#         )
#         self.sliced_size = self.rnn_hidden_size // self.m

#         # initialize weights
#         self.text2convs.apply(self.init_weights)
#         self.conv_layers.apply(self.init_weights)
#         self.deconv_layers.apply(self.init_weights)
#         self.out_layers.apply(self.init_weights)

#     def init_weights(self, m):
#         if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
#             torch.nn.init.xavier_uniform_(m.weight)
#             m.bias.data.fill_(0.01)

#     def forward(self, images, texts, seq_lengths):
#         B, num_maps, C, H, W = images.size()
#         images = images.view(B * num_maps, C, H, W)
#         images = self.resnet(images)

#         batch_size, image_channels, height, width = images.size()

#         text_embed = self.rnn(texts, seq_lengths)
#         text_embed = torch.repeat_interleave(text_embed, num_maps, dim=0)

#         Gs = []
#         image_embeds = images

#         for i in range(self.m):
#             image_embeds = self.conv_dropout(image_embeds)
#             image_embeds = self.conv_layers[i](image_embeds)
#             text_slice = text_embed[
#                 :, i * self.sliced_size : (i + 1) * self.sliced_size
#             ]

#             conv_kernel_shape = (
#                 batch_size,
#                 self.image_channels,
#                 self.image_channels,
#                 1,
#                 1,
#             )
#             text_conv_filters = self.text2convs[i](text_slice).view(conv_kernel_shape)

#             orig_size = image_embeds.size()
#             image_embeds = image_embeds.view(1, -1, *image_embeds.size()[2:])
#             text_conv_filters = text_conv_filters.view(
#                 -1, *text_conv_filters.size()[2:]
#             )
#             G = F.conv2d(image_embeds, text_conv_filters, groups=orig_size[0]).view(
#                 orig_size
#             )
#             image_embeds = image_embeds.view(orig_size)
#             if self.res_connect:
#                 G = G + image_embeds
#                 G = F.relu(G)
#             Gs.append(G)

#         # deconvolution operations, from the bottom up
#         H = Gs.pop()
#         for i in range(self.m):
#             if i == 0:
#                 H = self.deconv_dropout(H)
#                 H = self.deconv_layers[i](H)
#             else:
#                 G = Gs.pop()
#                 concated = torch.cat((H, G), 1)
#                 H = self.deconv_layers[i](concated)
#         out = self.out_layers(H).squeeze(-1)
#         out = out.view(B, num_maps, out.size()[-2], out.size()[-1])
#         out = F.log_softmax(out.view(B, -1), 1).view(B, num_maps, height, width)
#         return out

class LingUNet(nn.Module):
    """ CLIP RN50 with U-Net skip connections """
    def __init__(self, args):
        super(LingUNet, self).__init__()
        self.args = args 
        # self.output_dim = self.args.output_dim
        self.num_maps = self.args.num_maps
        self.output_dim = self.args.output_dim
        self.input_dim = self.args.input_dim  # penultimate layer channel-size of CLIP-RN50
        self.device = self.args.device 
        self.batchnorm = self.args.batchnorm
        self.lang_fusion_type = self.args.lang_fusion_type
        self.bilinear = self.args.bilinear
        self.batch_size = self.args.batch_size
        self.up_factor = 2 if self.bilinear else 1
        self.clip_rn50, self.preprocess = clip.load("RN50", device=self.args.device)

        self._build_decoder()


    def _build_decoder(self):
        # language
        self.lang_fuser1 = fusion.names[self.lang_fusion_type](input_dim=self.input_dim // 2)
        self.lang_fuser2 = fusion.names[self.lang_fusion_type](input_dim=self.input_dim // 4)
        self.lang_fuser3 = fusion.names[self.lang_fusion_type](input_dim=self.input_dim // 8)

        # CLIP encoder output -> 1024
        self.proj_input_dim = 512 if 'word' in self.lang_fusion_type else 1024
        self.lang_proj1 = nn.Linear(self.proj_input_dim, 1024)
        self.lang_proj2 = nn.Linear(self.proj_input_dim, 512)
        self.lang_proj3 = nn.Linear(self.proj_input_dim, 256)

        # vision
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_dim, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True)
        )
        self.up1 = Up(2048, 1024 // self.up_factor, self.bilinear)

        self.up2 = Up(1024, 512 // self.up_factor, self.bilinear)

        self.up3 = Up(512, 256 // self.up_factor, self.bilinear)

        self.layer1 = nn.Sequential(
            ConvBlock(128, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(64, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        self.layer2 = nn.Sequential(
            ConvBlock(64, [32, 32, 32], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(32, [32, 32, 32], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        self.layer3 = nn.Sequential(
            ConvBlock(32, [16, 16, 16], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(16, [16, 16, 16], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, self.output_dim, kernel_size=1)
        )

    def encode_image(self, img):
        with torch.no_grad():
            # The default CLIP function has been updated to be able to get intermediate prepools 
            img_encoding, img_im = self.clip_rn50.visual.prepool_im(img)
        return img_encoding, img_im

    def encode_text(self, x):
        with torch.no_grad():
            tokens = clip.tokenize(x).to(self.device)

            # The same dialog is repeated for all 5 maps 
            tokens = torch.repeat_interleave(tokens, self.num_maps, dim=0)
            text_feat = self.clip_rn50.encode_text(tokens)

        text_mask = torch.where(tokens==0, tokens, 1)  # [1, max_token_len]
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
        assert x.shape[1] == self.input_dim
        x = self.conv1(x)

        x = self.lang_fuser1(x, l_input, x2_mask=l_mask, x2_proj=self.lang_proj1)
        x = self.up1(x, im[-2])

        x = self.lang_fuser2(x, l_input, x2_mask=l_mask, x2_proj=self.lang_proj2)
        x = self.up2(x, im[-3])

        x = self.lang_fuser3(x, l_input, x2_mask=l_mask, x2_proj=self.lang_proj3)
        x = self.up3(x, im[-4])

        for layer in [self.layer1, self.layer2, self.layer3, self.conv2]:
            x = layer(x)

        x = F.interpolate(x, size=(780, 455), mode='bilinear')
        return x

def load_oldArgs(args, oldArgs):
    args.m = oldArgs["num_lingunet_layers"]
    args.image_channels = oldArgs["linear_hidden_size"]
    args.freeze_resnet = oldArgs["freeze_resnet"]
    args.res_connect = oldArgs["res_connect"]
    args.embed_size = oldArgs["embed_size"]
    args.rnn_hidden_size = oldArgs["rnn_hidden_size"]
    args.num_rnn_layers = oldArgs["num_rnn_layers"]
    args.embed_dropout = oldArgs["embed_dropout"]
    args.bidirectional = oldArgs["bidirectional"]
    args.linear_hidden_size = oldArgs["linear_hidden_size"]
    args.num_linear_hidden_layers = oldArgs["num_linear_hidden_layers"]
    args.ds_percent = oldArgs["ds_percent"]
    args.ds_height = oldArgs["ds_height"]
    args.ds_width = oldArgs["ds_width"]

    return args


def convert_model_to_state(model, optimizer, args):
    
    state = {
        "args": args,
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }

    return state 
