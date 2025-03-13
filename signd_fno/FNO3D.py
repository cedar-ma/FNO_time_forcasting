import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np

torch.manual_seed(0)
np.random.seed(0)


################################################################
# fourier layer
################################################################
class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes3 = modes3  # Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        #self.weights5 = nn.Parameter(
        #    self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
        #                            dtype=torch.cfloat))
        #self.weights6 = nn.Parameter(
        #    self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
        #                            dtype=torch.cfloat))
        #self.weights7 = nn.Parameter(
        #    self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
        #                            dtype=torch.cfloat))
        #self.weights8 = nn.Parameter(
        #    self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
        #                            dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,z), (in_channel, out_channel, x,y,z) -> (batch, out_channel, x,y,z)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)
       # out_ft[:, :, :self.modes1, :self.modes2, -self.modes3:] = \
       #     self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, -self.modes3:], self.weights5)
       # out_ft[:, :, -self.modes1:, :self.modes2, -self.modes3:] = \
       #     self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, -self.modes3:], self.weights6)
       # out_ft[:, :, :self.modes1, -self.modes2:, -self.modes3:] = \
       #     self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, -self.modes3:], self.weights7)
       # out_ft[:, :, -self.modes1:, -self.modes2:, -self.modes3:] = \
       #     self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, -self.modes3:], self.weights8)


        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv3d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv3d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

class LP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(LP, self).__init__()
        self.lp1 = nn.Linear(in_channels, mid_channels)
        self.lp2 = nn.Linear(mid_channels, out_channels)

    def forward(self, x):
        x = self.lp1(x)
        x = F.gelu(x)
        x = self.lp2(x)
        return x

class GetFNO3DModel(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3, width):
        super(GetFNO3DModel, self).__init__()
        self.input_channels = in_channels
        self.output_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.padding = 6

        # self.encoder = OneHotEncoder(sparse_output=False).fit_transform()
        # self.decoder = OneHotEncoder(sparse_output=False).inverse_transform()
        self.p = nn.Linear(self.input_channels + 3, self.width)  # input channel is 3: (sigma(x, y, z), x, y, z)
        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)
        self.q = LP(self.width, self.output_channels, self.width * 4)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        # f = x.clone()
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = F.pad(x, [0, self.padding])  # XM: delete two paddding

        x1 = self.conv0(x)
        # x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        # x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        # x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        # x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding]  # XM: delete two padding
        x = x.permute(0, 2, 3, 4, 1)  # XM: dimension coversion
        x = self.q(x)
        #x = x.softmax(dim=-1)

        # TODO: Add solid phase masking
        # x[f == 2] = 2
        return x.float()

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[:-1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)
