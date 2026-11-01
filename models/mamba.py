"""
Sleep Dynamics Modeling (SDM): a ShortMamba summarizes each 30-second epoch and a
bidirectional LongMamba models the whole night.

Mamba layers: https://github.com/state-spaces/mamba (Gu & Dao, 2023).
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import torch.nn as nn
from mamba_ssm import Mamba, Mamba2


class SleepMambaBlock(nn.Module):
    """
    Mamba block wrapper with optional bidirectional processing.

    Args:
        d_model: Model dimension
        mamba_version: 1 for Mamba, 2 for Mamba2
        d_state: SSM state dimension
        d_conv: Convolution width
        expand: Expansion factor
        bidirectional: Whether to use bidirectional processing
        dropout: Dropout rate
    """
    def __init__(
        self,
        d_model: int,
        mamba_version: int = 1,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        bidirectional: bool = False,
        dropout: float = 0.1
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        MambaClass = Mamba if mamba_version == 1 else Mamba2

        self.mamba_forward = MambaClass(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

        if bidirectional:
            self.mamba_backward = MambaClass(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        res = x

        if self.bidirectional:
            x = self.mamba_forward(x)
            x = torch.flip(x, [1])  # Reverse time dimension
            x = self.mamba_backward(x)
            x = torch.flip(x, [1])  # Restore order
        else:
            x = self.mamba_forward(x)

        x = self.dropout(x)
        return res + x


class DoubleMambaModel(nn.Module):
    """
    Sleep Dynamics Modeling (SDM) for sleep stage classification.

    Architecture:
    - Short path (ShortMamba): processes each epoch's temporal features independently
    - Long path (LongMamba): models temporal dependencies across epochs

    The short path uses a summary token to aggregate temporal information
    within each epoch, which is then passed to the long path for
    cross-epoch temporal modeling.

    Args:
        d_model: Long path model dimension
        d_state: Long path SSM state dimension
        n_layer: Number of long path Mamba blocks
        d_conv: Long path convolution width
        expand: Long path expansion factor
        bidirectional: Whether long path uses bidirectional processing
        d_model_short: Short path model dimension
        d_state_short: Short path SSM state dimension
        n_layer_short: Number of short path Mamba blocks
        d_conv_short: Short path convolution width
        d_expand_short: Short path expansion factor
        bidirectional_short: Whether short path uses bidirectional processing
        num_classes: Number of output classes
        Time_Max: Maximum sequence length (for time embedding)
        mamba_version: 1 for Mamba, 2 for Mamba2
        get_feature: Whether to return intermediate features
    """
    def __init__(
        self,
        d_model: int = 128,
        d_state: int = 32,
        n_layer: int = 4,
        d_conv: int = 4,
        expand: int = 2,
        bidirectional: bool = True,
        d_model_short: int = 256,
        d_state_short: int = 64,
        n_layer_short: int = 2,
        d_conv_short: int = 4,
        d_expand_short: int = 2,
        bidirectional_short: bool = False,
        num_classes: int = 5,
        Time_Max: int = 1500,
        mamba_version: int = 1,
        get_feature: bool = False
    ):
        super().__init__()

        # Input projection
        input_dim = 640  # MoViNet-A2 conv7 channels
        self.proj_in = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, d_model_short * 2),
            nn.GELU(),
            nn.Linear(d_model_short * 2, d_model_short),
            nn.GELU()
        )

        # Short path: process each epoch's temporal features
        self.summary_token = nn.Parameter(torch.zeros(1, 1, d_model_short))
        self.short_mamba_blocks = nn.ModuleList([
            SleepMambaBlock(
                d_model=d_model_short,
                mamba_version=mamba_version,
                d_state=d_state_short,
                d_conv=d_conv_short,
                expand=d_expand_short,
                bidirectional=bidirectional_short
            ) for _ in range(n_layer_short)
        ])

        # Projection from short to long path
        self.linear0 = nn.Linear(d_model_short, d_model)
        self.act = nn.GELU()

        # Time embedding for long path
        self.time_embedding = nn.Embedding(Time_Max, d_model)

        # Long path: model cross-epoch temporal dependencies
        self.mamba_base_blocks = nn.ModuleList([
            SleepMambaBlock(
                d_model=d_model,
                mamba_version=mamba_version,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                bidirectional=bidirectional
            ) for _ in range(n_layer)
        ])

        # Classification head
        self.classification = nn.Sequential(
            nn.Linear(d_model, d_model),
            self.act,
            nn.Linear(d_model, num_classes)
        )

        self.get_feature = get_feature

    def forward(self, x: torch.Tensor):
        """
        Forward pass.

        Args:
            x: Input tensor of shape [B, E, C, T, H, W]
               B: batch size
               E: number of epochs
               C: channels (640 for MoViNet features)
               T: temporal dimension (50)
               H, W: spatial dimensions (1, 1 for pooled features)

        Returns:
            If get_feature=False: logits [B, E, num_classes]
            If get_feature=True: (logits, long_features, short_features)
        """
        B, E, C, T, H, W = x.shape

        # Reshape: [B*E, T, C]
        x = x.squeeze(-1).squeeze(-1).reshape(B * E, C, T).transpose(1, 2)

        # Project input
        x = self.proj_in(x)  # [B*E, T, d_model_short]

        # Add summary token
        summary_token = self.summary_token.expand(B * E, 1, -1)
        x = torch.cat((x, summary_token), dim=1)  # [B*E, T+1, d_model_short]

        # Short path Mamba blocks
        for block in self.short_mamba_blocks:
            x = block(x)

        # Extract summary token output
        x = x[:, -1, :]  # [B*E, d_model_short]
        token_x = x

        # Project to long path dimension
        x = self.act(self.linear0(x))  # [B*E, d_model]
        x = x.view(B, E, -1)  # [B, E, d_model]

        # Add time embedding
        time_indices = torch.arange(E, device=x.device)
        time_embeds = self.time_embedding(time_indices)
        x = x + time_embeds.unsqueeze(0)  # [B, E, d_model]

        # Long path Mamba blocks
        for block in self.mamba_base_blocks:
            x = block(x)

        # Classification
        x_class = self.classification(x)  # [B, E, num_classes]

        if self.get_feature:
            return x_class, x, token_x
        return x_class


def build_mamba(**kwargs) -> nn.Module:
    """
    Build the SDM model (ShortMamba + LongMamba) for sleep stage classification.

    Args:
        **kwargs: Model configuration parameters
            - d_model: Long path model dimension
            - d_state: Long path SSM state dimension
            - n_layer: Number of long path Mamba blocks
            - d_conv: Long path convolution width
            - expand: Long path expansion factor
            - bidirectional: Whether long path uses bidirectional processing
            - d_model_short: Short path model dimension
            - d_state_short: Short path SSM state dimension
            - n_layer_short: Number of short path Mamba blocks
            - d_conv_short: Short path convolution width
            - d_expand_short: Short path expansion factor
            - bidirectional_short: Whether short path uses bidirectional processing
            - num_classes: Number of output classes
            - Time_Max: Maximum sequence length
            - mamba_version: 1 for Mamba, 2 for Mamba2
            - get_feature: Whether to return intermediate features

    Returns:
        Mamba model
    """
    print("Building SDM model (ShortMamba + LongMamba)")
    return DoubleMambaModel(**kwargs)
