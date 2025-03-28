import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

class DualAdapterInjection(nn.Module):
    def __init__(self, dim, shared_adapter_down, shared_adapter_up):
        super().__init__()
        self.activation = nn.GELU()
        self.adapter_shared_down = shared_adapter_down
        self.adapter_shared_up = shared_adapter_up
        self.adapter_local_down = nn.Linear(dim, dim // 4)
        self.adapter_local_up = nn.Linear(dim // 4, dim)
        self.adapter_norm = nn.LayerNorm(dim)

    def forward(self, x):
        shared = self.adapter_shared_down(x)
        shared = self.activation(shared)
        shared = self.adapter_shared_up(shared) * 0.5
        local = self.adapter_local_down(x)
        local = self.activation(local)
        local = self.adapter_local_up(local) * 0.5
        return self.adapter_norm(x + shared + local)


class DualAdapterViT(nn.Module):
    def __init__(self, shared_adapter_down, shared_adapter_up, num_classes):
        super().__init__()
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

        hidden_dim = 768

        self.adapter_injections = nn.ModuleList([
            DualAdapterInjection(hidden_dim, shared_adapter_down, shared_adapter_up)
            for _ in range(len(self.vit.encoder.layers))
        ])
        self.fc_class = nn.Linear(hidden_dim, num_classes)

        for param in self.parameters():
            param.requires_grad = False

        for layer in self.adapter_injections:
            for name, param in layer.named_parameters():
                if "adapter_local_down" in name or "adapter_local_up" in name or "adapter_norm" in name:
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




class SingleAdapterInjection(nn.Module):
    def __init__(self, dim, shared_adapter_down, shared_adapter_up):
        super().__init__()
        self.activation = nn.GELU()
        self.adapter_shared_down = shared_adapter_down
        self.adapter_shared_up = shared_adapter_up
        self.adapter_norm = nn.LayerNorm(dim)

    def forward(self, x):
        shared = self.adapter_shared_down(x)
        shared = self.activation(shared)
        shared = self.adapter_shared_up(shared)
        return self.adapter_norm(x + shared)


class SingleAdapterViT(nn.Module):
    def __init__(self, shared_adapter_down, shared_adapter_up, num_classes):
        super().__init__()

        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

        hidden_dim = 768

        self.adapter_injections = nn.ModuleList([
            SingleAdapterInjection(hidden_dim, shared_adapter_down, shared_adapter_up)
            for _ in range(len(self.vit.encoder.layers))
        ])


        self.fc_class = nn.Linear(hidden_dim, num_classes)

        for param in self.parameters():
            param.requires_grad = False

        for layer in self.adapter_injections:
            for name, param in layer.named_parameters():
                if "adapter_shared_down" in name or "adapter_shared_up" in name or "adapter_norm" in name:
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