import torch
from torch import nn

class Unet2D(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # self.contract_blocks = []
        # for c_block in cfg.UNETSTRUCTURE.CONTRACTBLOCK:
        #     self.contract_blocks = self.contract_block(*c_block)
        self.conv1 = self.contract_block(cfg.MODEL.IN_CHANNELS, 32, 7, 3)
        self.conv2 = self.contract_block(32, 64, 3, 1)
        self.conv3 = self.contract_block(64, 128, 3, 1)
        self.conv4 = self.contract_block(128,256,3,1)
        self.conv5 = self.contract_block(256,512,3,1)


        # self.upconv_blocks = []
        # for e_block in cfg.UNETSTRUCTURE.EXPANDBLOCK:
        #     self.upconv_blocks = self.expand_block(*e_block)

        self.upconv5 = self.expand_block(512,256,3,1)
        self.upconv4 = self.expand_block(256*2,128,3,1)
        self.upconv3 = self.expand_block(128*2, 64, 3, 1)
        self.upconv2 = self.expand_block(64*2, 32, 3, 1)
        self.upconv1 = self.expand_block(32*2, cfg.MODEL.OUT_CHANNELS, 3, 1)

    def __call__(self, x):
        #downsampling
        # outputs = []
        # for c_block in self.contract_blocks:
        #     print(c_block)
        #     if outputs:
        #         outputs.append(c_block(outputs[-1]))
        #     else:
        #         outputs.append(c_block(x))
            
        
        # #upsampling
        # upconv_output = []
        # for op, up_block in zip(outputs.reverse(), self.upconv_blocks):
        #     if upconv_output:
        #         print(upconv_output[-1])
        #         upconv_output.append(up_block(torch.cat([upconv_output[-1], op], 1)))
        #     else:
        #         upconv_output.append(up_block(op))


        #downsampling part
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        #print(conv3.size())
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        
        #upsample
        upconv5 = self.upconv5(conv5)
        upconv4 = self.upconv4(torch.cat([upconv5, conv4], 1))
        #print(upconv4.size())
        upconv3 = self.upconv3(torch.cat([upconv4, conv3], 1))
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        return upconv1
        #return upconv_output[-1]

    def contract_block(self, in_channels, out_channels, kernel_size, padding):

        contract = nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),

            torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),

            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                                 )

        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):

        expand = nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
                            torch.nn.BatchNorm2d(out_channels),
                            torch.nn.ReLU(),

                            torch.nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
                            torch.nn.BatchNorm2d(out_channels),
                            torch.nn.ReLU(),

                            torch.nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1) 
                            )
        return expand
