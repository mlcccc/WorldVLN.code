import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class DepthwiseSeparableResBlock(nn.Module):
    """Lightweight residual block: DWConv + PWConv."""
    def __init__(self, channels, dropout=0.1):
        super().__init__()
        groups = 8 if channels % 8 == 0 else 4 if channels % 4 == 0 else 1
        self.dw = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(groups, channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout2d(dropout)

    def forward(self, x):
        residual = x
        x = self.dw(x)
        x = self.pw(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        return x + residual

class LatentToPatchEmbed(nn.Module):
    """
    Enhanced Patch Embedding layer for processing latent codes.
    It takes summed_codes (latent representations) as input and projects them 
    to the Transformer's embedding dimension, specifically focusing on 
    extracting features relevant to the last frame in the chunk.
    
    Input: (B, T_latent, C_latent, H_latent, W_latent) or (B, C_latent, H_latent, W_latent) if T is merged
    Output: (B*T_out, Num_Patches, Embed_Dim), T_out, W_out
    """
    def __init__(self, 
                 latent_dim=16,          # Input latent dimension (from VAE)
                 embed_dim=384,          # Output embedding dimension (TSformer dim)
                 img_size=(192, 640),    # Original image size (for reference)
                 patch_size=16,          # TSformer patch size
                 latent_size=(24, 80),   # Latent spatial size (H_latent, W_latent) - estimated from downsample factor
                 hidden_dim=96,          # Bottleneck width for lightweight adapter
                 num_layers=2,           # Number of lightweight residual blocks
                 dropout=0.1):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # Calculate number of patches in TSformer
        # H_patches = img_size[0] // patch_size
        # W_patches = img_size[1] // patch_size
        # The latent spatial size might not match the patch grid size directly.
        # We need to upsample/downsample or project to match (Num_Patches).
        
        # Assuming the VAE latent spatial resolution needs to be adapted to the Transformer's patch grid.
        # If TSformer expects (B*T, N, D), where N = (H//P)*(W//P)
        # We need to produce spatial tokens that correspond to this grid.
        
        # Lightweight latent adapter before tokenization.
        stem_groups = 8 if hidden_dim % 8 == 0 else 4 if hidden_dim % 4 == 0 else 1
        in_groups = 8 if latent_dim % 8 == 0 else 4 if latent_dim % 4 == 0 else 1
        self.input_norm = nn.GroupNorm(in_groups, latent_dim)
        self.stem = nn.Sequential(
            nn.Conv2d(latent_dim, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(stem_groups, hidden_dim),
            nn.GELU()
        )
        
        self.res_blocks = nn.ModuleList([
            DepthwiseSeparableResBlock(hidden_dim, dropout=dropout) for _ in range(num_layers)
        ])
        
        # Learnable patchifier: stride-2 maps 24x80 -> 12x40 (for default settings).
        self.patchifier = nn.Conv2d(hidden_dim, embed_dim, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Target size from TSformer patch grid.
        # Target size: (img_size[0] // patch_size, img_size[1] // patch_size)
        self.target_h = img_size[0] // patch_size
        self.target_w = img_size[1] // patch_size
        self.token_norm = nn.LayerNorm(embed_dim)
        self.token_drop = nn.Dropout(dropout)
        
        print(f"LatentToPatchEmbed initialized: {latent_dim} -> {hidden_dim} -> {embed_dim}")
        print(f"Target token grid: {self.target_h}x{self.target_w}")

    def forward(self, x):
        # Expected Input: (B, T_latent, C_latent, H, W)
        # If T_latent dim is missing, assume (B, C, H, W)
        
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = rearrange(x, 'b t c h w -> (b t) c h w')
        else:
            B, C, H, W = x.shape
            T = 1
            
        # 1. Lightweight latent adapter
        x = self.input_norm(x)
        x = self.stem(x)
        
        # 2. Residual Blocks
        for block in self.res_blocks:
            x = block(x)
            
        # 3. Learnable patchification / downsampling
        x = self.patchifier(x) # (B*T, embed_dim, H/2, W/2)
        
        # 4. Safety path for unexpected latent resolutions
        if x.shape[-2:] != (self.target_h, self.target_w):
            x = F.adaptive_avg_pool2d(x, output_size=(self.target_h, self.target_w))
            
        # 5. Flatten and normalize tokens
        # N = target_h * target_w
        x = x.flatten(2).transpose(1, 2) # (B*T, N, embed_dim)
        x = self.token_norm(x)
        x = self.token_drop(x)
        
        return x, T, self.target_w

if __name__ == "__main__":
    # Test the module
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Mock latent input: 16 channels, say 24x80 spatial (from VAE usually downsampled by 8 or 16)
    dummy_latent = torch.randn(2, 4, 16, 24, 80).to(device) # B=2, T=4 (latents), C=16, H=24, W=80
    
    model = LatentToPatchEmbed(latent_dim=16, embed_dim=384, img_size=(192, 640), patch_size=16).to(device)
    
    output, T_out, W_out = model(dummy_latent)
    print(f"Input shape: {dummy_latent.shape}")
    print(f"Output shape: {output.shape}")
    print(f"T_out: {T_out}, W_out: {W_out}")
