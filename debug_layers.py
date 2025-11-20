
import torch
import torch.nn as nn
import MinkowskiEngine as ME
import sys
import logging

# 配置简单的 logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DebugSparseGenerator(nn.Module):
    def __init__(
        self,
        latent_dim=256,
        channels=[256, 128, 64, 32, 16],
        output_channels=1,
        initial_tensor_stride=16,
        resolution=96,
        kernel_size=3,
    ):
        super().__init__()
        self.initial_tensor_stride = initial_tensor_stride
        self.resolution = resolution
        self.initial_size = resolution // initial_tensor_stride
        self.channels = channels
        
        print(f"DebugGenerator Init:")
        print(f"  Resolution: {resolution}")
        print(f"  Initial Stride: {initial_tensor_stride}")
        print(f"  Initial Size: {self.initial_size}")
        
        # FC
        self.fc = nn.Linear(latent_dim, channels[0] * (self.initial_size ** 3))
        
        self.blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            
            # 拆分 Block 以便逐层打印
            layers = nn.ModuleList([
                ME.MinkowskiGenerativeConvolutionTranspose(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=2,
                    stride=2,
                    dimension=3
                ),
                ME.MinkowskiConvolution(
                    in_channels=out_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    dimension=3
                )
            ])
            self.blocks.append(layers)
            
    def forward(self, z):
        batch_size = z.size(0)
        device = z.device
        
        # FC
        h = self.fc(z)
        h = h.view(batch_size, self.channels[0], self.initial_size, self.initial_size, self.initial_size)
        
        # Coords
        coords = []
        for b in range(batch_size):
            for x in range(self.initial_size):
                for y in range(self.initial_size):
                    for z in range(self.initial_size):
                        coords.append([b, x, y, z])
        
        coords = torch.tensor(coords, dtype=torch.int32, device=device)
        feats = h.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.channels[0])
        
        st = ME.SparseTensor(
            features=feats,
            coordinates=coords,
            # tensor_stride=self.initial_tensor_stride, # Remove this
            device=device
        )
        
        print(f"\nInput: Stride={st.tensor_stride[0]}, MaxCoord={st.C[:, 1:].max().item()}, Count={len(st)}")
        
        for i, block in enumerate(self.blocks):
            print(f"\n--- Block {i} ---")
            # ConvTranspose
            st = block[0](st)
            print(f"After ConvTranspose: Stride={st.tensor_stride[0]}, MaxCoord={st.C[:, 1:].max().item()}, Count={len(st)}")
            
            # Conv
            st = block[1](st)
            print(f"After Conv3x3:       Stride={st.tensor_stride[0]}, MaxCoord={st.C[:, 1:].max().item()}, Count={len(st)}")
            
        return st

if __name__ == "__main__":
    print("Running Layer-wise Debug...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test Stride 16
    model = DebugSparseGenerator(initial_tensor_stride=16).to(device)
    z = torch.randn(1, 256).to(device)
    model(z)

