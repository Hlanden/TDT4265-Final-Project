import torch
from torch import nn

class Unet2D(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.contract_blocks = []
        for c_block in cfg.UNETSTRUCTURE.CONTRACTBLOCK:
            self.contract_blocks.append(self.contract_block(*c_block))


        self.upconv_blocks = []
        for e_block in cfg.UNETSTRUCTURE.EXPANDBLOCK:
            self.upconv_blocks.append(self.expand_block(*e_block))
        
        #dette er fordi han trenger en liste når han skal hente ut model
        self.contract_blocks = nn.ModuleList(self.contract_blocks)
        self.upconv_blocks = nn.ModuleList(self.upconv_blocks)

    def __call__(self, x):
        #downsampling
        outputs = []
        for c_block in self.contract_blocks:
            if outputs:
                outputs.append(c_block(outputs[-1]))
            else:
                outputs.append(c_block(x))
            
        
        #upsampling
        upconv_output = []
        
        for op, up_block in zip(reversed(outputs), self.upconv_blocks):
            
            if upconv_output:
                upconv_output.append(up_block(torch.cat([upconv_output[-1], op], 1)))
            else:
                upconv_output.append(up_block(op))

        return upconv_output[-1]

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
