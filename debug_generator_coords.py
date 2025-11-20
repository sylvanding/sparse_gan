
import torch
import MinkowskiEngine as ME
from sparse_gan_models import SparseGenerator

def test_generator_output(initial_stride):
    print(f"\n=== Testing with initial_tensor_stride = {initial_stride} ===")
    resolution = 96
    channels = [256, 128, 64, 32, 16]
    latent_dim = 256
    
    try:
        gen = SparseGenerator(
            latent_dim=latent_dim,
            channels=channels,
            output_channels=1,
            initial_tensor_stride=initial_stride,
            resolution=resolution,
            kernel_size=3
        )
        
        print(f"Initial Size: {gen.initial_size}")
        
        # Dummy input
        batch_size = 1
        z = torch.randn(batch_size, latent_dim)
        
        if torch.cuda.is_available():
            gen = gen.cuda()
            z = z.cuda()
            
        out = gen(z)
        
        coords = out.C
        if len(coords) > 0:
            max_coords = coords[:, 1:].max(dim=0)[0]
            min_coords = coords[:, 1:].min(dim=0)[0]
            print(f"Output Coordinate Range: Min {min_coords.tolist()}, Max {max_coords.tolist()}")
            print(f"Output Tensor Stride: {out.tensor_stride}")
        else:
            print("Output is empty!")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Test original config
    test_generator_output(32)
    
    # Test my fix
    test_generator_output(16)
    
    # Test potential fix (if needed)
    test_generator_output(8)

