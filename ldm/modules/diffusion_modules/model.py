import math
import numpy as np
import torch
import torch.nn.functional as F

from torch import nn
from einops import rearrange
from typing import Optional, Any

from ldm.modules.attention import MemoryEfficientCrossAttention

try: 
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILABLE = True
except:
    XFORMERS_IS_AVAILABLE = False
    print("No module 'xformers'. Proceeding without it.")


def get_timestep_embedding(timesteps, embedding_dim):
    # Build sinsoidal embeddings
    assert len(timesteps.shape) == 1
    
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def Normalize(in_channels, num_groups=32):
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

def swish(x):
    return x * torch.sigmoid(x)
    

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()

        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()

        self.with_conv= with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = F.pad(x, pad, mode='constant', value=0)
            x= self.conv(x)
        else: 
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x
    

class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout, temb_channels=512):
        super().__init__()

        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)
        
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.use_conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(swish(temb))[:, :, None, None]
        
        h = self.norm2(h)
        h = swish(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.use_conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h
    

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.in_channels = in_channels
        
        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)      # b, hw, c
        k = k.reshape(b, c, h*w)    # b, c, hw
        w_ = torch.bmm(q, k)       # b, hw, hw  
        w_ = w_ * (int(c) ** (-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)    # b, hw, hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)       # b,    c, hw (hw of q)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_
    

class MemoryEfficientAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.attention_op = Optional[Any] = None

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q, k, v = map(lambda x: rearrange(x, 'b c h w -> b (h w) c'), (q, k, v))

        q, k, v = map(
            lambda t: 
            t.unsqueeze(3)
            .reshape(b, t.shape[1], 1, c)
            .permute(0, 2, 1, 3)
            .reshape(b * 1, t.shape[1], c)
            .contiguous(),
            (q, k, v),
        )
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        out = (
            out.unsqueeze(0)
            .reshape(b, 1, out.shape[1], c)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], c)
        )
        out = rearrange(out, 'b (h w) c -> b c h w', b=b, h=h, w=w, c=c)
        out = self.proj_out(out)

        return x + out


class MemoryEfficientCrossAttentionWrapper(MemoryEfficientCrossAttention):
    def forward(self, x, context=None, mask=None):
        b, c, h, w, = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        out = super().forward(x, context=context, mask=mask)
        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w, c=c)
        return x + out
    

def make_attn(in_channels, attn_type='vanilla', attn_kwargs=None):
    assert attn_type in ['vanilla', 'vanilla-xformers', 'memory-efficient-cross-attn', 'linear', 'none'], f'attn_type {attn_type} unknown'
    if XFORMERS_IS_AVAILABLE and attn_type == 'vanilla':
        attn_type = 'vanilla-xformers'
    print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == 'vanilla':
        assert attn_kwargs is None
        return AttentionBlock(in_channels)
    elif attn_type == 'vanilla-xformers':
        print(f"building MemoryEfficientAttentionBlock with {in_channels} in_channels...")
        return MemoryEfficientAttentionBlock(in_channels)
    elif type == 'memory-efficient-cross-attn':
        attn_kwargs['query_dim'] = in_channels
        return MemoryEfficientCrossAttention(**attn_kwargs)
    elif attn_type == 'none':
        return nn.Identity(in_channels) 
    else:
        raise NotImplementedError()


class Model(nn.Module):
    def __init__(
            self,
            *,
            channels,
            out_channels,
            in_channels,
            channel_multiplier=(1, 2, 4, 8),
            num_res_blocks,
            attn_resolutions,
            dropout=0.0,
            resample_with_conv=True,
            resolution,
            use_timestep=True,
            use_linear_attn=False,
            attn_type='vanilla',
    ):
        super().__init__()
        
        if use_linear_attn: 
            attn_type = 'linear'
        self.channels = channels
        self.in_channels = in_channels
        self.temb_channels = self.channels * 4
        self,num_resolutions = len(channel_multiplier)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        
        self.use_timestep = use_timestep
        if self.use_timestep:
            # timestep embedding
            self.temb = nn.Module()
            self.temb.dense = nn.ModuleList([
                nn.Linear(self.channels, self.temb_channels),
                nn.Linear(self.temb_channels, self. temb_channels)
            ])
        
        # Downsampling Layers
        self.conv_in = nn.Conv2d(in_channels, self.channels, kernel_size=3, stride=1, padding=1)

        current_resolution = resolution
        in_channel_multiplier = (1, ) + tuple(channel_multiplier)
        self.down = nn.ModuleList()
        for i in range(self.num_resolutions):
            res = nn.ModuleList()
            attn = nn.ModuleList()
            res_in = channels * in_channel_multiplier[i]
            res_out = channels * channel_multiplier[i]
            for j in range(self.num_res_blocks):
                res.append(
                    ResnetBlock(
                        in_channels=res_in,
                        out_channels=res_out,
                        temb_channels=self.temb_channels,
                        dropout=dropout
                    )
                )
                res_in = res_out
                if current_resolution in attn_resolutions:
                    attn.append(make_attn(res_in, attn_type=attn_type))   
            down = nn.Module()
            down.res = res
            down.attn = attn
            if i != self.num_resolutions - 1:
                down.downsample = Downsample(res_in, resample_with_conv)
                current_resolution = current_resolution // 2
            self.down.append(down)

        # Middle Layers
        self.middle = nn.Module()
        self.middle.res_1 = ResnetBlock(
            in_channels=res_in,
            out_channels=res_in,
            temb_channels=self.temb_channels,
            dropout=dropout
        )
        self.middle.attn = make_attn(res_in, attn_type=attn_type)
        self.middle.res_2 = ResnetBlock(
            in_channels=res_in,
            out_channels=res_in,
            temb_channels=self.temb_channels,
            dropout=dropout
        )

        # Upsampling Layers
        self.up = nn.ModuleList()
        for i in reversed(range(self.num_resolutions)):
            res = nn.ModuleList()
            attn = nn.ModuleList()
            res_out = channels * channel_multiplier[i]
            skip_in = channels * channel_multiplier[i]
            for j in range(self.num_res_blocks+1):
                if j == self.num_res_blocks:
                    skip_in = channels * in_channel_multiplier[i]
                res.append(
                    ResnetBlock(
                        in_channels=res_in+skip_in,
                        out_channels=res_out,
                        temb_channels=self.temb_channels,
                        dropout=dropout
                    )
                )
                res_in = res_out
                if current_resolution in attn_resolutions:
                    attn.append(make_attn(res_in, attn_type=attn_type))
            up = nn.Module()
            up.res = res
            up.attn = attn
            if i != 0:
                up.upsample = Upsample(res_in, resample_with_conv)
                current_resolution = current_resolution * 2
            self.up.insert(0, up)

        # end 
        self.norm_out = Normalize(res_in)
        self.conv_out = nn.Conv2d(res_in, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t=None, context=None):
        if context is not None:
            x = torch.cat((x, context), dim=1)
        if self.use_timestep:
            assert t is not None:
            temb = get_timestep_embedding(t, self.channels)
            temb = self.temb.dense[0](temb)
            temb = swish(temb)
            temb = self.temb.dense[1](temb)
        else: 
            temb = None

        # Downsampling
        hs = [self.conv_in(x)]
        for i in range(self.num_resolutions):
            for j in range(self.num_res_blocks):
                h = self.down[i].res[j](hs[-1], temb)
                if len(self.down[i].attn) > 0:
                    h = self.down[i].attn[j](h)
                hs.append(h)
            if i != self.num_resolutions - 1:
                hs.append(self.down[i].downsample(hs[-1]))    
        
        # Middle
        h = hs[-1]
        h = self.middle.res_1(h, temb)
        h = self.middle.attn(h)
        h = self.middle.res_2(h, temb)

        # Upsampling
        for i in reversed(range(self.num_resolutions)):
            for j in range(self.num_res_blocks + 1):
                h = self.up[i].block[j](torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i].attn) > 0:
                    h = self.up[i].attn[j](h)
            if i != 0:
                h = self.up[i].unsample(h)
        
        # End
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h

    def get_last_layer(self):
        return self.conv_out.weight


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        channels,
        in_channels,
        out_channels,
        channels_multiplier=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resample_with_conv=True,
        resolution,
        z_channels,
        double_z=True,
        use_linear_attn=False,
        attn_type='vanilla'
    ):
        super().__init__()

        if use_linear_attn:
            attn_type = "linear"
        self.channels = channels
        self.in_channels = in_channels
        self.temb_channels = 0
        self.num_resolutions = len(channels_multiplier)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        
        # Downsampling Layers
        self.conv_in = nn.Conv2d(in_channels, self.channels, kernel_size=3, stride=1, padding=1)

        current_resolution = resolution
        in_channel_multiplier = (1, ) + tuple(channels_multiplier)
        self.in_channel_multiplier = in_channel_multiplier
        self.down = nn.ModuleList()
        for i in range(self.num_resolutions):
            res = nn.ModuleList()
            attn = nn.ModuleList()
            res_in = channels * in_channel_multiplier[i]
            res_out = channels * channels_multiplier[i]
            for j in range(self.num_res_blocks):
                res.append(
                    ResnetBlock(
                        in_channels=res_in,
                        out_channels=res_out,
                        temb_channels=self.temb_channels,
                        dropout=dropout
                    )
                )
                res_in = res_out
                if current_resolution in attn_resolutions:
                    attn.append(make_attn(res_in, attn_type=attn_type))
            down = nn.Module()
            down.res = res
            down.attn = attn
            if i != self.num_resolutions:
                down.downsample = Downsample(res_in, resample_with_conv)
                current_resolution = current_resolution // 2
            self.down.append(down)

        # Middle Layers
        self.middle = nn.Module()
        self.middle.res_1 = ResnetBlock(
            in_channels=res_in,
            out_channels=res_in,
            temb_channels=self.temb_channels,
            dropout=dropout
        )
        self.middle_attn = make_attn(res_in, attn_type=attn_type)  
        self.middle.res_2 = ResnetBlock(
            in_channels=res_in,
            out_channels=res_in,
            temb_channels=self.temb_channels,
            dropout=dropout
        )

        # end
        self.norm_out = Normalize(res_in)
        self.conv_out = nn.Conv2d(
            res_in, 
            2 * z_channels if double_z else z_channels,
            kernel_size=3, 
            stride=1,
            padding=1
        )

    def forward(self, x):
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i in range(self.num_resolutions):
            for j in range(self.num_res_blocks):
                h = self.down[i].res[j](hs[-1], temb)
                if len(self.down[i].attn) > 0:
                    h = self.down[i].attn[j](h)
                hs.append(h)
            if i != self.num_resolutions - 1:
                hs.append(self.down[i].downsample(hs[-1]))
        
        # middle
        h = hs[-1]
        h = self.middle.res_1(h, temb)
        h = self.middle.attn(h)
        h = self.middle.res_2(h, temb)

        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h

    
class Decoder(nn.Module):
    def __init__(
        self,
        *,
        channels,
        in_channels,
        out_channels,
        channel_multiplier=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resample_with_conv=True,
        resolution,
        z_channels,
        give_pre_end=False,
        tanh_out=False,
        use_linear_attn=False,
        attn_type='vanilla',
        **ignorekwargs
    ):
        super().__init__()

        if use_linear_attn: 
            attn_type = 'linear'
        self.channels = channels
        self.in_channels = in_channels
        self.temb_chanenls = 0
        self.num_resolutions = len(channel_multiplier)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_channels_mutliplier, res_in, current_resolution
        in_channel_multiplier = (1, ) + tuple(channel_multiplier)
        res_in = channels * channel_multiplier[self.num_resolutions - 1]
        current_resolution = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, current_resolution)
        print(F"Working with z of shape {self.z_shape} = {np.prod(self.z_shape)} dimensions.")

        # z to res_in
        self.conv_in = nn.Conv2d(z_channels, res_in, kernel_size=3, stride=1, padding=1)

        # Middle Layers
        self.middle = nn.Module()
        self.middle.res_1 = ResnetBlock(
            in_channels=res_in, 
            out_channels=res_in, 
            temb_channels=self.temb_chanenls,
            dropout=dropout
        )
        self.middle.attn = make_attn(res_in, attn_type=attn_type)
        self.middle.res_2 = ResnetBlock(
            in_channels=res_in,
            out_channels=res_in,
            temb_channels=self.temb_chanenls,
            dropout=dropout
        )

        # Upsampling Layers
        self.up = nn.ModuleList()
        for i in reversed(range(self.num_resolutions)):
            res = nn.ModuleList()
            attn = nn.ModuleList()
            res_out = channels * channel_multiplier[i]
            for j in range(self.num_res_blocks + 1):
                res.append(
                    ResnetBlock(
                        in_channels=res_in,
                        out_channels=res_out,
                        temb_channels=self.temb_chanenls,
                        dropout=dropout
                    )
                )
                res_in = res_out
                if current_resolution in attn_resolutions:
                    attn.append(make_attn(res_in, attn_type=attn_type))
            up = nn.Module()
            up.res = res
            up.attn = attn
            if i != 0:
                up.upsample = Upsample(res_in, resample_with_conv)
                current_resolution = current_resolution * 2
            self.up.insert(0, up)

        # end
        self.norm_out = Normalize(res_in)
        self.conv_out = nn.Conv2d(res_in, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        self.last_z_sahpe = z.shape

        temb = None
        
        h = self.conv_in(z)

        # Middle
        h = self.middle.res_1(h, temb)
        h = self.middle.attn(h)
        h = self.middle.res_2(h, temb)

        # Upsampling
        for i in reversed(range(self.num_resolutions)):
            for j in range(self.num_res_blocks + 1):
                h = self.up[i].res[j](h, temb)
                if len(self.up[i].attn) > 0:
                    h = self.up[i].attn[j](h)
            if i != 0:
                h = self.up[i].upsample(h)
            
        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)

        if self.tanh_out:
            h = torch.tanh(h)
        return h


class SimpleDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__()

        self.model = nn.ModuleLIst([
            nn.Conv2d(in_channels, in_channels, 1),
            ResnetBlock(
                in_channels=in_channels,
                out_channels=2 * in_channels,
                temb_channels=0,
                dropout=0.0,
            ),
            ResnetBlock(
                in_channels=2 * in_channels,
                out_channels=4 * in_channels,
                temb_channels=0,
                dropout=0.0,
            ),
            ResnetBlock(
                in_channels=4* in_channels,
                out_channels=2 * in_channels,
                temb_channels=0,
                dropout=0.0,
            ),
            nn.Conv2d(2 * in_channels, in_channels, 1),
            Upsample(in_channels, with_conv=True)
        ])       

        # end
        self.norm_out = Normalize(in_channels)
        self.conv_out = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        for i, layer in enumerate(self.model):
            if i in [1, 2, 3]:
                z = layer(z, None)
            else:
                z = layer(z)

        h = self.norm_out(z)
        h = swish(h)
        x = self.conv_out(h)
        return x


class UpsampleDecoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        channels,
        num_res_blocks,
        resolution,
        channel_multiplier=(2, 2),
        dropout=0.0
    ):
        super().__init__()

        # Unsampling Layers
        self.temb_channels = 0
        self.num_resolutions = len(channel_multiplier)
        self.num_res_blocks = num_res_blocks
        res_in = in_channels
        current_resolution = resolution // 2 ** (self.num_resolution - 1)
        self.res = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for i in range(self.num_resolutions):
            res = []
            res_out = channels * channel_multiplier[i]
            for j in range(self.num_res_blocks + 1):
                res.append(
                    ResnetBlock(
                        in_channels=res_in,
                        out_channels=res_out,
                        temb_channels=self.temb_channels,
                        dropout=dropout
                    )
                )
                res_in = res_out
            self.res.append(nn.ModuleList(res))
            if i != self.num_resolutions - 1:
                self.upsample.append(Upsample(res_in, True))
                current_resolution = current_resolution * 2

        # end
        self.norm_out = Normalize(res_in)
        self.conv_out = nn.Conv2d(res_in, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        h = x
        for k, i in enumerate(range(self.num_resolutions)):
            for j in range(self.num_res_blocks + 1):
                h = self.res[i][j](h, None)
            if i != self.num_resolutions - 1:
                h = self.upsample[k](h)
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h 


class LatentRescaler(nn.Module):
    def __init__(self, factor, in_channels, mid_channels, out_channels, depth=2):
        super().__init__()

        self.factor = factor
        self.conv_in = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.res_1 = nn.ModuleList([
            ResnetBlock(
                in_channels=mid_channels,
                out_channels=mid_channels,
                temb_channels=0,
                dropout=0.0
            ) for _ in range(depth)
        ])
        self.attn = AttentionBlock(mid_channels)
        self.res_2 = nn.ModuleList([
            ResnetBlock(
                in_channels=mid_channels,
                out_channels=mid_channels,
                temb_channels=0,
                dropout=0.0
            ) for _ in range(depth)
        ])
        self.conv_out = nn.Conv2d(mid_channels, out_channelsm kernel_size=1)
    
    def forward(self, x):
        x = self.conv_in(x)
        for res in self.res_1:
            x = res(x, None)
        x = F.interpolate(x, size=(int(round(x.shape[2]*self.factor)), int(round(x.shape[3]*self.factor))))
        x = self.attn(x)
        for res in self.res_2:
            x = res(x, None)
        x = self.conv_out(x)
        return x


class MergedRescaleEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        channels,
        out_channels,
        resolution,
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resample_with_conv=True,
        channel_multiplier=(1, 2, 4, 8),
        rescale_factor=1.0,
        rescale_module_depth=1
    ):
        super().__init__()
        intermediate_channels = channels * channel_multiplier[-1]
        self.encoder = Encoder(
            channels=channels, in_channels=in_channels, out_channels=None,
            channels_multiplier=channel_multiplier, num_res_blocks=num_res_blocks, 
            z_channels=intermediate_channels, double_z=False,
            resolution=resolution, attn_resolutions=attn_resolutions, dropout=dropout, resample_with_conv=resample_with_conv
        )
        self.rescaler = LatentRescaler(
            factor=rescale_factor, in_channels=intermediate_channels,
            mid_channels=intermediate_channels, out_channels=out_channels, depth=rescale_module_depth
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.rescaler(x)
        return x


class MergedRescaleDecoder(nn.Module):
    def __init__(
        self, 
        channels,
        z_channels,
        out_channels,
        resolution,
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resample_with_conv=True,
        channel_multiplier=(1, 2, 4, 8),
        rescale_factor=1.0,
        rescale_module_depth=1,
    ):
        super().__init__()
        intermediate_channels = z_channels * channel_multiplier[-1]
        self.decoder = Decoder(
            channels=channels, in_channels=None, out_channels=out_channels, z_channels=intermediate_channels,
            channel_multiplier=channel_multiplier, num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions, dropout=dropout, resample_with_conv=resample_with_conv,
            resolution=resolution
        )
        self.rescaler = LatentRescaler(
            factor=rescale_factor, in_channels=z_channels
            mid_channels=intermediate_channels, out_channels=intermediate_channels,
            depth=rescale_module_depth
        )

    def forward(self, x):
        x = self.rescaler(x)
        x = self.decoder(x)
        return x


class Upsampler(nn.Module):
    def __init__(self, in_size, out_size, in_channels, out_channels, channel_multiplier=2):
        super().__init__()
        assert out_size >= in_size
        num_blocks = int(np.log2(out_size // in_size)) + 1
        factor_up = 1. + (out_size % in_size)
        print(f"Building {self.__class__.__name__} with in_size: {in_size} --> out_size {out_size} and factor {factor_up}")
        self.rescaler = LatentRescaler(
            factor=factor_up, in_channels=in_channels, mid_channels=2*in_channels,
            out_channels=in_channels
        )
        self.decoder = Decoder(
            out_channels=out_channels, resolution=out_size, z_channels=in_channels, num_res_blocks=2,
            attn_resolutions=[], in_channels=None, ch=in_channels,
            channel_multiplier=[channel_multiplier for _ in range(num_blocks)]
        )

    def forward(self, x):
        x = self.rescaler(x)
        x = self.decoder(x)
        return x


class Resize(nn.Module):
    def __init__(self, in_channels=None, learned=False, mode='bilinear'):
        super().__init__()

        self.with_conv = learned
        self.mode = mode
        if self.with_conv:
            print(f"Note: {self.__class__.__name__} uses learned downsampling and will ignore the fixed {mode} mode")
            raise NotImplementedError()
            assert in_channels is not None
            self.with_conv = nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x, scale_factor=1.0):
        if scale_factor == 1.0:
            return x
        else:
            x = F.interpolate(x, mode=self.mode, align_corners=False, scale_factor=scale_factor)
        return x
