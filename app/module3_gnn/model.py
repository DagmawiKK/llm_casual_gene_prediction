import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class FusionGeneGNN(nn.Module):
    def __init__(self, num_go_features, num_reactome_features, num_mesh_features,
                 common_embed_dim, num_attention_heads,
                 gnn_hidden_dim, gnn_out_dim):
        super().__init__()

        self.num_go_features = num_go_features
        self.num_reactome_features = num_reactome_features
        self.num_mesh_features = num_mesh_features

        self.go_proj = nn.Linear(self.num_go_features, common_embed_dim)
        self.reactome_proj = nn.Linear(self.num_reactome_features, common_embed_dim)
        self.mesh_proj = nn.Linear(self.num_mesh_features, common_embed_dim)
        self.gwas_proj = nn.Linear(1, common_embed_dim)

        self.attention = nn.MultiheadAttention(embed_dim=common_embed_dim,
                                               num_heads=num_attention_heads,
                                               batch_first=True)

        self.conv1 = GCNConv(common_embed_dim, gnn_hidden_dim)
        self.conv2 = GCNConv(gnn_hidden_dim, gnn_out_dim)

    def forward(self, data):
        x_original, edge_index = data.x, data.edge_index

        current_idx = 0
        go_feat = x_original[:, current_idx : current_idx + self.num_go_features]
        current_idx += self.num_go_features

        reactome_feat = x_original[:, current_idx : current_idx + self.num_reactome_features]
        current_idx += self.num_reactome_features

        mesh_feat = x_original[:, current_idx : current_idx + self.num_mesh_features]

        gwas_feat = x_original[:, -1].unsqueeze(1)

        proj_go = F.relu(self.go_proj(go_feat))
        proj_reactome = F.relu(self.reactome_proj(reactome_feat))
        proj_mesh = F.relu(self.mesh_proj(mesh_feat))
        proj_gwas = F.relu(self.gwas_proj(gwas_feat))

        modalities_stacked = torch.stack([proj_go, proj_reactome, proj_mesh, proj_gwas], dim=1)

        attn_output, attn_weights = self.attention(modalities_stacked, modalities_stacked, modalities_stacked)

        fused_x = attn_output.mean(dim=1)

        x_gnn = F.relu(self.conv1(fused_x, edge_index))
        x_gnn = F.dropout(x_gnn, p=0.3, training=self.training)
        final_gene_embeddings = self.conv2(x_gnn, edge_index)

        return final_gene_embeddings