import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import relu
from torch.nn import LayerNorm,Linear,Dropout,BatchNorm2d
import torch.nn.functional as F
from bimamba import Mamba, MambaConfig 
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from sassm import StructureRouter,build_indices,unfold,fold

icc=64+2
ic=72
oc=128
ws=13
fs=(ws+1)//2

num_class=11

d_model=ws*ws

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_convs=nn.Sequential(
            nn.Conv2d(in_channels,mid_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        return self.double_convs(x)

class PositionalEncoding(nn.Module):
    def __init__(self, channels, height, width):
        super().__init__()
        self.register_buffer('position_ids', torch.arange(height * width))
        self.embedding = nn.Embedding(height * width, channels)
        self.height = height
        self.width = width

    def forward(self, x):
        B, C, H, W = x.shape
        #assert H == self.height and W == self.width

        # [H*W, C]
        pos_embed = self.embedding(self.position_ids)
        # [1, C, H, W]
        pos_embed = pos_embed.view(1, C, H, W)
        # expand to batch
        pos_embed = pos_embed.expand(B, -1, -1, -1)
        return x + pos_embed    
    
class Pooling(nn.Module):
    """
    @article{ref-vit,
    title={An image is worth 16x16 words: Transformers for image recognition at scale},
    author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, 
            Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and others},
    journal={arXiv preprint arXiv:2010.11929},
    year={2020}
    }
    """
    def __init__(self, pool: str = "mean"):
        super().__init__()
        if pool not in ["mean", "cls"]:
            raise ValueError("pool must be one of {mean, cls}")

        self.pool_fn = self.mean_pool if pool == "mean" else self.cls_pool

    def mean_pool(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)

    def cls_pool(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_fn(x)

class Classifier(nn.Module):

    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(in_features=dim, out_features=num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up,self).__init__()

        if bilinear:

            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:

            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

        self.up_match_channels=nn.Conv2d(in_channels,out_channels,kernel_size=1)
 
    def forward(self,x1,x2):
        x1=self.up(x1)
        x1=self.up_match_channels(x1)

        diffY=x2.size()[2]-x1.size()[2]
        diffX= x2.size()[3]-x1.size()[3]

        x1=F.pad(x1,[diffX//2,diffX-diffX//2,diffY//2,diffY-diffY//2])

        x=torch.cat([x2,x1],dim=1)
        return self.conv(x)

def BasicConv(in_channels, out_channels, kernel_size=3, stride=1, padding=None):
    if not padding:
        padding = (kernel_size - 1) // 2 if kernel_size else 0
    else:
        padding = padding
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),)

class EncoderLayers(nn.Module):
    def __init__(self,encoder_in=ic,num_encoder_layers=3,dim_feedforward=384,nhead=8,reverse=False,dropout=0.1):
        super(EncoderLayers, self).__init__()
        encoder_layer = TransformerEncoderLayer(encoder_in, nhead, dim_feedforward, dropout,norm_first=False)
        encoder_norm =LayerNorm(encoder_in)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.reverse=reverse

    def forward(self, x):
        bs, c, h, w = x.shape
        x = x.reshape(bs, c, -1)
        x = x.permute(0, 2, 1)

        if self.reverse:
            x=torch.flip(x, dims=[1])

        x = self.encoder(x)

        if self.reverse:
            x=torch.flip(x, dims=[1])

        x = x.permute(0, 2, 1)
        x = x.reshape(bs, c, h, w)
        return x   

class MambaLayers(nn.Module):
    def __init__(self, d_model, n_layers):
        super(MambaLayers, self).__init__()
        self.mamba = Mamba(MambaConfig(d_model=d_model, n_layers=n_layers,use_cuda=True))

    def forward(self, x):
        bs, c, h, w = x.shape
        x = x.reshape(bs, c, -1)
        x = x.permute(0, 2, 1)
        x = self.mamba(x)
        x = x.permute(0, 2, 1)
        x = x.reshape(bs, c, h, w)
        return x

class TransBlock(nn.Module):
    def __init__(self, ic: int, ws: int,num_encoder_layers=1):
        """
        Args:
            ic (int): input channels
            ws (int): width = height
        """
        super(TransBlock, self).__init__()
        self.ic = ic
        self.ws = ws  # window size or feature map size

        self.BN1 = nn.BatchNorm2d(ic)
        self.pos1 = PositionalEncoding(ic, ws, ws)
        self.attn1 = EncoderLayers(
            encoder_in=ic,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=ic*2,
            nhead=8
        )
    def forward(self, x):
        x = self.BN1(x)        
        x = self.pos1(x)        
        x = self.attn1(x)      
        return x

class PNC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PNC, self).__init__()
        self.pnc = nn.Sequential(
            BasicConv(in_channels,out_channels),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.pnc(x)

class StandardCA(nn.Module):
    def __init__(self, channels, reduction=8): 
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c) 
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class UpWOconv(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True):
        super(UpWOconv,self).__init__()

        if bilinear:

            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
            self.oneconv = nn.Conv2d(in_channels,out_channels,kernel_size=1)
        else:

            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
            

        self.up_match_channels=nn.Conv2d(in_channels,out_channels,kernel_size=1)
 
    def forward(self,x1,x2):
        x1=self.up(x1)
        x1=self.up_match_channels(x1)

        diffY=x2.size()[2]-x1.size()[2]
        diffX= x2.size()[3]-x1.size()[3]

        x1=F.pad(x1,[diffX//2,diffX-diffX//2,diffY//2,diffY-diffY//2])

        x=torch.cat([x2,x1],dim=1)
        
        return self.oneconv(x)  
    

class LinearAttention(nn.Module):
    def __init__(self, dim, num_heads=2, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 kernel_function=nn.ReLU, kernel_size=5):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # depth-wise convolution 
        self.dwc = nn.Conv2d(
            in_channels=head_dim,
            out_channels=head_dim,
            kernel_size=kernel_size,
            groups=head_dim,
            padding=kernel_size // 2
        )
        self.kernel_function = kernel_function()

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        assert H * W == N, "The sequence length N must be a perfect square (e.g., 13×13=169 )."

        q = self.q(x)  # (B, N, C)
        kv = self.kv(x).reshape(B, -1, 2, C).permute(2, 0, 1, 3)  # (2, B, N, C)
        k, v = kv[0], kv[1]

        q = self.kernel_function(q) + 1e-6
        k = self.kernel_function(k) + 1e-6

        q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)  # (B, H, N, C/H)
        k = k.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k.transpose(-2, -1) * (N ** -0.5)) @ (v * (N ** -0.5))
        x = q @ kv * z

        x = x.transpose(1, 2).reshape(B, N, C)  # (B, N, C)

        # DWC 
        v = v.reshape(B * self.num_heads, H, W, -1).permute(0, 3, 1, 2)  # (B*H, C/H, H, W)
        x = x + self.dwc(v).reshape(B, C, N).permute(0, 2, 1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# ---------------- DiT Block ----------------
class DiTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)

        self.attn = LinearAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=True,
            attn_drop=dropout,
            proj_drop=dropout,
            kernel_function=nn.ReLU,
            kernel_size=5
        )

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(approximate='tanh'),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # 
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------- DiT Backbone ----------------
class DiTBackbone(nn.Module):
    def __init__(self, in_chans, dim, depth, num_heads,
                 input_size, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_chans, dim, kernel_size=1)

        num_patches = input_size * input_size
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim))
        self.pos_drop  = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            DiTBlock(dim, num_heads, mlp_ratio, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim, eps=1e-6)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)                     # (B, dim, H, W)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)            # (B, N, C)
        x = x + self.pos_embed                      # 
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)                            # (B, N, dim)
        return x


class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        self.conv0 = BasicConv(in_channels=icc, out_channels=ic,
                               kernel_size=1, stride=1, padding=0)

        self.mamba0 = MambaLayers(d_model=ic, n_layers=2)

        self.tb     = TransBlock(ic, ws, num_encoder_layers=1)
        self.tb02   = TransBlock(ic, ws, num_encoder_layers=1)
        self.ca     = StandardCA(ic)
        self.lmd    = nn.Parameter(torch.tensor(0.))

        self.dit = DiTBackbone(
            in_chans   = ic,          # 72
            dim        = oc,          # 128 
            depth      = 6,
            num_heads  = 2,           # 
            input_size = ws,          # 
            mlp_ratio  = 4.0,
            dropout    = 0.1,
        )

        self.pool       = Pooling(pool="mean")
        self.classifier = Classifier(dim=oc, num_classes=num_class)

        self.router=StructureRouter(ic)


    def forward(self, x):
        x, x1 = x
        x = torch.cat((x, x1), 1)
        x = self.conv0(x)                                     # (B, 72, 13, 13)

        ## SASSM
        labels, _= self.router(x)
        sort_index, sort_index_reverse = build_indices(labels)
        x = unfold(x, sort_index)

        x = self.mamba0(x)

        x = fold(x, sort_index_reverse)

        x1=x.permute(0, 1, 3, 2)
        x1=self.tb02(x1).permute(0, 1, 3, 2)
        x=self.tb(x)

        lmd=torch.sigmoid(self.lmd)
        x=lmd*x1+(1-lmd)*x

        x       = self.ca(x)                                  # (B, 72, 13, 13)


        ## GL-DiT
        x = self.dit(x)                                       # (B, 169, 128)
        x = self.pool(x)                                      # (B, 128)
        x = self.classifier(x)                                # (B, 11)
        return x

if __name__=='__main__':
    print('start')
    batch_size=32
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = MyModel().to(device)

    # inputs (batch_size, ic, height, width)
    dummy_input = torch.randn(batch_size, 64, ws, ws).to(device)
    dummy_input2 = torch.randn(batch_size, 2, ws, ws).to(device)
    dummy=(dummy_input, dummy_input2)

    output = model(dummy)

    print("Output shape:", output.shape)