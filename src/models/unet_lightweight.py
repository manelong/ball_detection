from torch import nn
import functools
import torch


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            # model = down + up
            if use_dropout:
                model = down + up + [nn.Dropout(0.1)]
            else:
                model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.1)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class UnetLightweight(nn.Module):
    def __init__(self, input_nc, output_nc, use_dropout=False):
        super(UnetLightweight, self).__init__()

        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)

        # construct unet structure 3 layers
        unet_block = UnetSkipConnectionBlock(128, 256, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(64, 128, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)  # add the innermost layer
        unet_block = UnetSkipConnectionBlock(32, 64, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        self.unet = UnetSkipConnectionBlock(output_nc, 32, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, x):
        return {0: self.unet(x)}


if __name__ == '__main__':
    model = UnetLightweight(9, 3, use_dropout=True)
    x = torch.randn(1, 9, 288, 512)
    y = model(x)
    print(y.shape)