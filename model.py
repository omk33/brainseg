import torch
import torch.nn as nn

# a residual 3D convolution block
class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False, dropout_probability=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
            nn.LeakyReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
            nn.LeakyReLU()
        )

        self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.dropout = nn.Dropout3d(dropout_probability) if dropout else nn.Identity()

    def forward(self, x):
        # the goal of this block is not to learn a mapping x -> y, but to learn a mapping F(x) that allows x + F(x) = y
        # in other words, how should I tweak x by parameterizing F(x) and adding it to x - in order to approximate y
        x_conv = self.block(x)
        x_res = self.residual(x)

        return self.dropout(x_conv + x_res)

# a 3D attention gate module
class AttentionGate3D(nn.Module):
    def __init__(self, in_channels_enc, in_channels_dec, latent_channels):
        super().__init__()
        self.W_enc = nn.Sequential(
            nn.Conv3d(in_channels_enc, latent_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=min(8, latent_channels), num_channels=latent_channels)
        )

        self.W_dec = nn.Sequential(
            nn.Conv3d(in_channels_dec, latent_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=min(8, latent_channels), num_channels=latent_channels)
        )

        self.attention = nn.Sequential(
            nn.Conv3d(latent_channels, 1, kernel_size=1, bias=True), # gives us spatial attention scores
            nn.Sigmoid()
        )

        self.leaky_relu = nn.LeakyReLU()

    def forward(self, enc, dec):
        enc_latent = self.W_enc(enc)
        dec_latent = self.W_dec(dec)
        att_scores = self.attention(self.leaky_relu(enc_latent + dec_latent))
        return enc * att_scores

# a 3D cross attention module
class CrossAttention3D(nn.Module):
    def __init__(self, in_channels_enc, in_channels_dec, latent_channels, num_heads, dropout, dropout_probability):
        super().__init__()
        if latent_channels % num_heads != 0:
            raise ValueError(f"latent_channels must be divisible by num_heads")

        self.lc = latent_channels
        self.n_h = num_heads
        self.n_lc_per_h = latent_channels // num_heads

        # projections to token latent space
        self.k_proj = nn.Conv3d(in_channels_enc, latent_channels, kernel_size=1, bias=False)
        self.q_proj = nn.Conv3d(in_channels_dec, latent_channels, kernel_size=1, bias=False)
        self.v_proj = nn.Conv3d(in_channels_enc, latent_channels, kernel_size=1, bias=False)

        self.norm_k = nn.LayerNorm(latent_channels)
        self.norm_q = nn.LayerNorm(latent_channels)
        self.norm_v = nn.LayerNorm(latent_channels)

        self.out_proj = nn.Conv3d(latent_channels, in_channels_dec, kernel_size=1, bias=False)

        self.proj_dropout = nn.Dropout3d(dropout_probability) if dropout else nn.Identity()
        self.att_dropout = nn.Dropout(dropout_probability) if dropout else nn.Identity()

    # flatten spatial dimensions, (b, c, d, h, w) to (b, d * h * w, c)
    def _to_tokens(self, x):
        b, c, d, h, w = x.shape
        tokens = x.view(b, c, d * h * w).transpose(1, 2)
        return tokens, (b, c, d, h, w)

    # reconstruct spatial dimensions, (b, d * h * w, lc) to (b, lc, d, h, w)
    def _from_tokens(self, x, shape):
        b, _, d, h, w = shape
        rec_spatial = x.transpose(1, 2).view(b, self.lc, d, h, w)
        return rec_spatial

    # split attention heads, (b, d * h * w, n_h, n_lc_per_h) to (b, n_h, d * h * w, n_lc_per_h)
    def _split_heads(self, x):
        b, dhw, _ = x.shape
        att_heads = x.view(b, dhw, self.n_h, self.n_lc_per_h).transpose(1, 2)
        return att_heads

    def forward(self, enc, dec):
        k = self.k_proj(enc)
        q = self.q_proj(dec)
        v = self.v_proj(enc)

        k_t, _ = self._to_tokens(k)
        q_t, _ = self._to_tokens(q)
        v_t, _ = self._to_tokens(v)

        k_t = self.norm_k(k_t)
        q_t = self.norm_q(q_t)
        v_t = self.norm_v(v_t)

        k_h = self._split_heads(k_t)
        q_h = self._split_heads(q_t)
        v_h = self._split_heads(v_t)

        # scaled dot-product attention
        att_raw = (q_h @ k_h.transpose(-2, -1)) / (self.n_lc_per_h ** 0.5)
        att_scores = self.att_dropout(att_raw.softmax(dim=-1))
        att_ctx = att_scores @ v_h # (b, n_h, d * h * w, n_lc_per_h)

        # merge attention heads
        att_ctx = att_ctx.transpose(1, 2).contiguous().view(q_t.shape[0], q_t.shape[1], self.lc) # (b, d * h * w, lc)

        # reconstruct spatial feature map
        att_ctx_map = self._from_tokens(att_ctx, (dec.shape[0], self.lc, dec.shape[2], dec.shape[3], dec.shape[4]))
        att_ctx_map = self.proj_dropout(self.out_proj(att_ctx_map)) # (b, in_channels_dec, d, h, w)

        return dec + att_ctx_map

# a 3D deconvolution and skip block with an attention gate
class UpBlock3D_AG(nn.Module):
    def __init__(self, in_channels, skip_channels, latent_channels, dropout, dropout_probability):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, skip_channels, kernel_size=2, stride=2)
        self.att_gate = AttentionGate3D(skip_channels, skip_channels, latent_channels)
        self.conv = ConvBlock3D(skip_channels * 2, skip_channels, dropout=dropout, dropout_probability=dropout_probability)

    def forward(self, x, skip=None):
        x = self.up(x)
        enc_ctx = self.att_gate(skip, x) # apply attention gating to encoder features
        x = torch.cat((enc_ctx, x), dim=1) # concatenate contextualized encoder and raw decoder features
        return self.conv(x) # learn a transformation that condenses them

# a 3D deconvolution and skip block with cross attention
class UpBlock3D_CA(nn.Module):
    def __init__(self, in_channels, skip_channels, latent_channels, num_heads, dropout, dropout_probability):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, skip_channels, kernel_size=2, stride=2)
        self.cross_att = CrossAttention3D(
            skip_channels, skip_channels, latent_channels,
            num_heads=num_heads, dropout=dropout, dropout_probability=dropout_probability
        )
        self.conv = ConvBlock3D(skip_channels * 2, skip_channels, dropout=dropout, dropout_probability=dropout_probability)

    def forward(self, x, skip):
        x = self.up(x)
        enc_ctx = self.cross_att(skip, x) # apply cross attention to decoder features
        x = torch.cat((enc_ctx, x), dim=1) # concatenate contextualized encoder and raw decoder features
        return self.conv(x)

# a 3D unet with multiple types of attention
class Att3DUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=4, num_filters=32, num_heads=2, dropout=False, dropout_probability=0.2):
        super().__init__()
        f = num_filters

        # encoder blocks
        self.enc0 = ConvBlock3D(in_channels, f, dropout=dropout, dropout_probability=dropout_probability)
        self.enc1 = ConvBlock3D(f, f * 2, dropout=dropout, dropout_probability=dropout_probability)
        self.enc2 = ConvBlock3D(f * 2, f * 4, dropout=dropout, dropout_probability=dropout_probability)
        self.enc3 = ConvBlock3D(f * 4, f * 8, dropout=dropout, dropout_probability=dropout_probability)

        self.bottleneck = ConvBlock3D(f * 8, f * 16, dropout=dropout, dropout_probability=dropout_probability) # deepest feature map

        self.pool = nn.MaxPool3d(2) # pooling layer (downsampling)

        # decoder blocks
        self.dec3 = UpBlock3D_CA(f * 16, f * 8, f * 16, num_heads=num_heads, dropout=dropout, dropout_probability=dropout_probability)
        self.dec2 = UpBlock3D_AG(f * 8, f * 4, f * 8, dropout=dropout, dropout_probability=dropout_probability)
        self.dec1 = UpBlock3D_AG(f * 4, f * 2, f * 4, dropout=dropout, dropout_probability=dropout_probability)
        self.dec0 = UpBlock3D_AG(f * 2, f, f * 2, dropout=dropout, dropout_probability=dropout_probability)

        self.final_conv = nn.Conv3d(f, out_channels, kernel_size=1)

    def forward(self, x):
        # encoder path
        e0 = self.enc0(x)
        e1 = self.enc1(self.pool(e0))
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # deepest feature map
        b = self.bottleneck(self.pool(e3))

        # decoder path
        d3 = self.dec3(b, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        d0 = self.dec0(d1, e0)

        return self.final_conv(d0)
