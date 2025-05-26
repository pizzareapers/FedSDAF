import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

############################################################################

class DualAdapterInjection(nn.Module):
    def __init__(self, dim = 768):
        super().__init__()
        self.activation = nn.GELU()
        self.adapter_invariant_down = nn.Linear(dim, dim // 4)
        self.adapter_invariant_up = nn.Linear(dim // 4, dim)
        self.adapter_aware_down = nn.Linear(dim, dim // 4)
        self.adapter_aware_up = nn.Linear(dim // 4, dim)
        self.adapter_norm = nn.LayerNorm(dim)
        self.num_heads = 8
        self.num_heads = self.num_heads
        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.mhsa_query = nn.Linear(dim, dim)
        self.mhsa_key = nn.Linear(dim, dim)
        self.mhsa_value = nn.Linear(dim, dim)
        self.mhsa_output = nn.Linear(dim, dim)
        self.mhsa_norm = nn.LayerNorm(dim)

    def forward(self, x):
        invariant = self.adapter_invariant_down(x)
        invariant = self.activation(invariant)
        invariant = self.adapter_invariant_up(invariant) * 0.5
        aware = self.adapter_aware_down(x)
        aware = self.activation(aware)
        aware = self.adapter_aware_up(aware) * 0.5
        return self.adapter_norm(x + invariant + aware)


class DualAdapterViT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

        hidden_dim = 768

        self.adapter_injections = nn.ModuleList([
            DualAdapterInjection(hidden_dim)
            for _ in range(len(self.vit.encoder.layers))
        ])

        self.fc_class = nn.Linear(hidden_dim, num_classes)

        for param in self.parameters():
            param.requires_grad = False

        for layer in self.adapter_injections:
            for name, param in layer.named_parameters():
                if "adapter_aware_down" in name or "adapter_aware_up" in name or "adapter_norm" in name:
                    param.requires_grad = True

        for param in self.fc_class.parameters():
            param.requires_grad = True

    def forward(self, x):

        x = self.vit.conv_proj(x)
        batch_size = x.shape[0]

        x = x.flatten(2).transpose(1, 2)
        cls_token = self.vit.class_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], dim=1)

        x = x + self.vit.encoder.pos_embedding

        x = self.vit.encoder.dropout(x)

        for idx, block in enumerate(self.vit.encoder.layers):
            x = block(x)
            x = self.adapter_injections[idx](x)

        feature_out = x[:, 0]
        return self.fc_class(feature_out)


############################################################################################

class SingleAdapterInjection(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.activation = nn.GELU()
        self.adapter_invariant_down = nn.Linear(dim, dim // 4)
        self.adapter_invariant_up = nn.Linear(dim // 4, dim)
        self.adapter_norm = nn.LayerNorm(dim)
        self.num_heads = 8
        self.num_heads = self.num_heads
        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.mhsa_query = nn.Linear(dim, dim)
        self.mhsa_key = nn.Linear(dim, dim)
        self.mhsa_value = nn.Linear(dim, dim)
        self.mhsa_output = nn.Linear(dim, dim)
        self.mhsa_norm = nn.LayerNorm(dim)


    def forward(self, x):
        invariant = self.adapter_invariant_down(x)
        invariant = self.activation(invariant)
        invariant = self.adapter_invariant_up(invariant)
        return self.adapter_norm(x + invariant)


class SingleAdapterViT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

        hidden_dim = 768

        self.adapter_injections = nn.ModuleList([
            SingleAdapterInjection(hidden_dim)
            for _ in range(len(self.vit.encoder.layers))
        ])

        self.fc_class = nn.Linear(hidden_dim, num_classes)

        for param in self.parameters():
            param.requires_grad = False

        for layer in self.adapter_injections:
            for name, param in layer.named_parameters():
                if "adapter_invariant_down" in name or "adapter_invariant_up" in name or "adapter_norm" in name:
                    param.requires_grad = True
        for param in self.fc_class.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.vit.conv_proj(x)
        batch_size = x.shape[0]

        x = x.flatten(2).transpose(1, 2)
        cls_token = self.vit.class_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], dim=1)

        x = x + self.vit.encoder.pos_embedding

        x = self.vit.encoder.dropout(x)

        for idx, block in enumerate(self.vit.encoder.layers):
            x = block(x)
            x = self.adapter_injections[idx](x)

        feature_out = x[:, 0]
        return self.fc_class(feature_out)
