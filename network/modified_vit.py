import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


class DualAdapterViT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        hidden_dim = 768
        self.adapter_invariant_downs = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim // 4) for _ in range(len(self.vit.encoder.layers))])
        self.adapter_invariant_ups = nn.ModuleList(
            [nn.Linear(hidden_dim // 4, hidden_dim) for _ in range(len(self.vit.encoder.layers))])
        self.adapter_aware_downs = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim // 4) for _ in range(len(self.vit.encoder.layers))])
        self.adapter_aware_ups = nn.ModuleList(
            [nn.Linear(hidden_dim // 4, hidden_dim) for _ in range(len(self.vit.encoder.layers))])
        self.adapter_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(len(self.vit.encoder.layers))])
        self.num_heads = 8
        self.head_dim = hidden_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.mhsa_queries = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(len(self.vit.encoder.layers))])
        self.mhsa_keys = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(len(self.vit.encoder.layers))])
        self.mhsa_values = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(len(self.vit.encoder.layers))])
        self.mhsa_outputs = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(len(self.vit.encoder.layers))])
        self.mhsa_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(len(self.vit.encoder.layers))])
        self.activation = nn.GELU()
        self.fc_class = nn.Linear(hidden_dim, num_classes)
        for param in self.parameters():
            param.requires_grad = False
        for i in range(len(self.vit.encoder.layers)):
            for param in self.adapter_aware_downs[i].parameters():
                param.requires_grad = True
            for param in self.adapter_aware_ups[i].parameters():
                param.requires_grad = True
            for param in self.adapter_norms[i].parameters():
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
            # Integrated adapter injection logic
            invariant = self.adapter_invariant_downs[idx](x)
            invariant = self.activation(invariant)
            invariant = self.adapter_invariant_ups[idx](invariant) * 0.5
            aware = self.adapter_aware_downs[idx](x)
            aware = self.activation(aware)
            aware = self.adapter_aware_ups[idx](aware) * 0.5
            x = self.adapter_norms[idx](x + invariant + aware)
        feature_out = x[:, 0]
        return self.fc_class(feature_out)


class SingleAdapterViT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        hidden_dim = 768
        self.adapter_invariant_downs = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim // 4) for _ in range(len(self.vit.encoder.layers))])
        self.adapter_invariant_ups = nn.ModuleList(
            [nn.Linear(hidden_dim // 4, hidden_dim) for _ in range(len(self.vit.encoder.layers))])
        self.adapter_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(len(self.vit.encoder.layers))])
        self.num_heads = 8
        self.head_dim = hidden_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.mhsa_queries = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(len(self.vit.encoder.layers))])
        self.mhsa_keys = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(len(self.vit.encoder.layers))])
        self.mhsa_values = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(len(self.vit.encoder.layers))])
        self.mhsa_outputs = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(len(self.vit.encoder.layers))])
        self.mhsa_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(len(self.vit.encoder.layers))])
        self.activation = nn.GELU()
        self.fc_class = nn.Linear(hidden_dim, num_classes)
        for param in self.parameters():
            param.requires_grad = False
        for i in range(len(self.vit.encoder.layers)):
            for param in self.adapter_invariant_downs[i].parameters():
                param.requires_grad = True
            for param in self.adapter_invariant_ups[i].parameters():
                param.requires_grad = True
            for param in self.adapter_norms[i].parameters():
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
            invariant = self.adapter_invariant_downs[idx](x)
            invariant = self.activation(invariant)
            invariant = self.adapter_invariant_ups[idx](invariant)
            x = self.adapter_norms[idx](x + invariant)
        feature_out = x[:, 0]
        return self.fc_class(feature_out)
