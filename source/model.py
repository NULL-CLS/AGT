import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import os

import matplotlib.pyplot as plt


def sparsemax(input, dim=-1):
    input = input.contiguous()
    number_of_logits = input.size(dim)
    input = input - torch.max(input, dim=dim, keepdim=True)[0]
    input_sorted, _ = torch.sort(input, dim=dim, descending=True)

    range_values = torch.arange(number_of_logits).to(input.device).float() + 1
    shape = [1] * input.dim()
    shape[dim] = number_of_logits
    range_values = range_values.view(*shape)

    input_cumsum = input_sorted.cumsum(dim=dim) - 1
    input_cumsum = input_cumsum / range_values

    k_selector = range_values * input_sorted > input_cumsum
    k = k_selector.sum(dim=dim, keepdim=True)

    tau = input_cumsum.gather(dim, k.long() - 1)
    output = torch.clamp(input - tau, min=0)
    return output

class HASTANConfig:
    def __init__(self):
        # Default values
        pass


class KernelSparseAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, sparsity_alpha=0):
        super(KernelSparseAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.sparsity_alpha = sparsity_alpha
        self.theta = nn.Parameter(torch.zeros(1, num_heads, 1, 1))

    def forward(self, x):
        B, L, D = x.shape
        H = self.num_heads
        d_k = self.head_dim
        q = self.q_proj(x).view(B, L, H, d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, L, H, d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, d_k).transpose(1, 2)

        dist_matrix = torch.cdist(q, k, p=2)
        dist_sq = torch.pow(dist_matrix, 2) / d_k
        dist_max = dist_sq.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        dist_norm = dist_sq / (dist_max + 1e-6)

        dsexp = torch.exp(self.theta)
        attn_logits = torch.exp(-1 * dsexp * dist_norm)

        if self.sparsity_alpha == 2:
            attn_weights = sparsemax(attn_logits, dim=-1)
        else:
            attn_weights = F.softmax(attn_logits, dim=-1)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(B, L, D)
        output = self.out_proj(context)
        return output, attn_weights.mean(dim=1)

class SpatialPositionalEncoding(nn.Module):
    def __init__(self, input_dim=3, embed_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(input_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim))

    def forward(self, x): return self.mlp(x).unsqueeze(0)

class GlobalAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.self_attn = KernelSparseAttention(embed_dim, num_heads, sparsity_alpha=0)
        self.norm1 = nn.LayerNorm(embed_dim);
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout);
        self.dropout2 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(embed_dim, embed_dim * 4);
        self.linear2 = nn.Linear(embed_dim * 4, embed_dim)
        self.activation = nn.GELU()

    def forward(self, src):
        src2, attn_weights = self.self_attn(src)
        src = src + self.dropout1(src2);
        src = self.norm1(src)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2);
        src = self.norm2(src)
        return src, attn_weights


# 5. Panel A Components
class UnitGCN(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes):
        super().__init__()
        A = np.ones((num_nodes, num_nodes)) / num_nodes
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels);
        self.relu = nn.ReLU()
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out')

    def forward(self, x):
        N, C, T, V = x.size();
        A = self.PA.to(x.device)
        x_reshaped = x.permute(0, 1, 2, 3).contiguous().view(-1, V)
        x_agg = torch.matmul(x_reshaped, A).contiguous().view(N, C, T, V)
        return self.relu(self.bn(self.conv(x_agg)))


class AttentionPooling(nn.Module):
    def __init__(self, in_channels, hidden_dim=8):
        super().__init__()
        self.score_net = nn.Sequential(nn.Linear(in_channels, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        x_perm = x.permute(0, 2, 3, 1)
        attn_score = self.score_net(x_perm)
        attn_weights = F.softmax(attn_score, dim=2)
        return torch.sum(x_perm * attn_weights, dim=2).permute(0, 2, 1).unsqueeze(-1)

class IntraModuleAggregation(nn.Module):
    def __init__(self, config, roi_mapping):
        super().__init__()
        self.num_modules = config.num_modules
        self.module_gcns = nn.ModuleList();
        self.module_pools = nn.ModuleList();
        self.module_indices = []
        for m in range(self.num_modules):
            indices = (roi_mapping == m).nonzero(as_tuple=False).squeeze()
            if indices.dim() == 0: indices = indices.unsqueeze(0)
            self.module_indices.append(indices)
            self.module_gcns.append(UnitGCN(1, 1, len(indices)))
            self.module_pools.append(AttentionPooling(1))

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        outputs = []
        for m in range(self.num_modules):
            indices = self.module_indices[m].to(x.device)
            x_m = torch.index_select(x, 3, indices)
            x_m = self.module_gcns[m](x_m)
            outputs.append(self.module_pools[m](x_m))
        return torch.cat(outputs, dim=1)


def sliding_window(x, window_size, stride, projector):
    x = x.squeeze(-1).unfold(2, window_size, stride).contiguous()
    return projector(x)


class PanelA_StructureAwareInput(nn.Module):
    def __init__(self, config, roi_mapping, mni_coords):
        super().__init__()
        self.config = config
        self.spatial_encoder = SpatialPositionalEncoding(3, config.embed_dim)
        self.intra_module_agg = IntraModuleAggregation(config, roi_mapping)
        self.register_buffer('mni_coords', mni_coords / 100.0)
        self.register_buffer('module_centroids', self._compute_centroids(roi_mapping, mni_coords, config.num_modules))
        self.micro_projector = nn.Linear(config.window_size, config.embed_dim)
        self.macro_projector = nn.Linear(config.window_size, config.embed_dim)

    def _compute_centroids(self, roi_mapping, mni_coords, num_modules):
        centroids = []
        for m in range(num_modules):
            indices = (roi_mapping == m).nonzero(as_tuple=False).squeeze()
            if indices.dim() == 0: indices = indices.unsqueeze(0)
            coords = torch.index_select(mni_coords, 0, indices)
            centroids.append(torch.mean(coords, dim=0))
        return torch.stack(centroids)

    def forward(self, x):
        micro_tokens = sliding_window(x, self.config.window_size, self.config.stride, self.micro_projector)
        micro_tokens = micro_tokens + self.spatial_encoder(self.mni_coords).unsqueeze(2)
        macro_signal = self.intra_module_agg(x)
        macro_tokens = sliding_window(macro_signal, self.config.window_size, self.config.stride, self.macro_projector)
        macro_tokens = macro_tokens + self.spatial_encoder(self.module_centroids).unsqueeze(2)
        return micro_tokens, macro_tokens


class MicroTransformer(nn.Module):
    def __init__(self, config): super().__init__(); self.block = GlobalAttentionBlock(config.embed_dim,
                                                                                      config.num_heads)

    def forward(self, h): return self.block(h)


class MacroTransformer(nn.Module):
    def __init__(self, config): super().__init__(); self.block = GlobalAttentionBlock(config.embed_dim,
                                                                                      config.num_heads)

    def forward(self, h): return self.block(h)


class GraphormerSpatialEncoder(nn.Module):
    def __init__(self, num_heads, max_dist=10):
        super(GraphormerSpatialEncoder, self).__init__()
        self.num_heads = num_heads
        self.max_dist = max_dist
        self.bias_embedding = nn.Embedding(max_dist + 2, num_heads)

    def forward(self, dist_matrix):
        B, N, _ = dist_matrix.shape
        indices = dist_matrix + 1
        indices = indices.clamp(0, self.max_dist + 1)
        bias = self.bias_embedding(indices).permute(0, 3, 1, 2).reshape(B * self.num_heads, N, N)
        return bias

def get_shortest_path_matrix(adj_binary, max_dist=10):
    B, N, _ = adj_binary.shape
    device = adj_binary.device

    dist = torch.full_like(adj_binary, float('inf'))
    dist[adj_binary == 1] = 1

    ids = torch.arange(N, device=device)
    dist[:, ids, ids] = 0

    for k in range(N):
        dist = torch.minimum(dist, dist[:, :, k].unsqueeze(2) + dist[:, k, :].unsqueeze(1))

    dist[dist > max_dist] = -1
    return dist.long()

class SubGraphTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.2, top_k_ratio=0.2):
        super(SubGraphTransformer, self).__init__()
        self.top_k_ratio = top_k_ratio
        self.num_heads = num_heads

        self.spd_encoder = GraphormerSpatialEncoder(num_heads, max_dist=5)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                        dim_feedforward=embed_dim * 4, dropout=dropout,
                                                        batch_first=True)
        self.pooling_layer = nn.Sequential(nn.Linear(embed_dim, 1), nn.Tanh())

    def forward(self, x, adj):
        B_size, N, _ = adj.shape
        k = max(1, int(N * self.top_k_ratio))

        _, indices = torch.topk(adj, k, dim=-1)
        binary_adj = torch.zeros_like(adj)
        binary_adj.scatter_(dim=-1, index=indices, value=1.0)

        dist_matrix = get_shortest_path_matrix(binary_adj)
        spd_bias = self.spd_encoder(dist_matrix)  # (B*H, N, N)

        base_mask = torch.zeros_like(adj, dtype=torch.float)
        base_mask.masked_fill_(binary_adj == 0, float('-inf'))
        base_mask = base_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1).reshape(B_size * self.num_heads, N, N)

        combined_mask = base_mask + spd_bias

        x = self.encoder_layer(x, src_mask=combined_mask)

        weights = F.softmax(self.pooling_layer(x), dim=1)
        return torch.sum(x * weights, dim=1)

class RowSequenceTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.2):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, 20, embed_dim))
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                        dim_feedforward=embed_dim * 4, dropout=dropout,
                                                        batch_first=True)
        self.pooling_layer = nn.Sequential(nn.Linear(embed_dim, 1), nn.Tanh())

    def forward(self, x):
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.encoder_layer(x)
        weights = F.softmax(self.pooling_layer(x), dim=1)
        return torch.sum(x * weights, dim=1)


class SecondOrderStateModeler(nn.Module):
    def __init__(self, embed_dim, num_windows, num_rois, num_heads=4, dropout=0.2, top_k_ratio=0.2):
        super().__init__()
        self.num_windows, self.num_rois = num_windows, num_rois
        self.level1_model = SubGraphTransformer(embed_dim, num_heads, dropout, top_k_ratio)
        self.level2_model = RowSequenceTransformer(embed_dim, num_heads, dropout)

    def forward(self, h_flat, attn_matrix):
        B, L, D = h_flat.shape
        T, N = self.num_windows, self.num_rois

        h_windows = h_flat.view(B, T, N, D).contiguous()
        matrix_blocks = attn_matrix.view(B, T, N, T, N).permute(0, 1, 3, 2, 4).contiguous()

        final_rows = []
        for i in range(T):
            src_feat = h_windows[:, i, :, :]
            batch_feats = src_feat.unsqueeze(1).expand(-1, T, -1, -1).reshape(B * T, N, D)
            batch_adjs = matrix_blocks[:, i, :, :, :].reshape(B * T, N, N)

            g1_out = self.level1_model(batch_feats, batch_adjs)
            g1_seq = g1_out.view(B, T, D)
            row_final = self.level2_model(g1_seq)
            final_rows.append(row_final)

        all_rows = torch.stack(final_rows, dim=1)
        x_max = torch.max(all_rows, dim=1)[0]
        x_avg = torch.mean(all_rows, dim=1)
        return torch.cat([x_max, x_avg], dim=1)


class BrainAGT(nn.Module):
    def __init__(self, config, roi_mapping, mni_coords, num_classes=2):
        super().__init__()
        self.has_plotted = False

        self.config = config
        self.panel_a = PanelA_StructureAwareInput(config, roi_mapping, mni_coords)
        self.micro_transformer = MicroTransformer(config)
        self.macro_transformer = MacroTransformer(config)

        top_k = getattr(config, 'top_k_ratio', 0.2)
        self.micro_modeler = SecondOrderStateModeler(config.embed_dim, config.num_windows, config.num_rois,
                                                     config.num_heads, config.dropout, top_k)
        self.macro_modeler = SecondOrderStateModeler(config.embed_dim, config.num_windows, config.num_modules,
                                                     config.num_heads, config.dropout, top_k)

        self.classifier = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(config.dropout),
                                        nn.Linear(64, num_classes))

    def forward(self, x):
        h_micro, h_macro = self.panel_a(x)
        B = x.size(0)
        h_micro_flat = h_micro.permute(0, 2, 1, 3).contiguous().reshape(B, -1, self.config.embed_dim)
        h_macro_flat = h_macro.permute(0, 2, 1, 3).contiguous().reshape(B, -1, self.config.embed_dim)

        _, attn_micro = self.micro_transformer(h_micro_flat)

        _, attn_macro = self.macro_transformer(h_macro_flat)


        feat_micro = self.micro_modeler(h_micro_flat, attn_micro)
        feat_macro = self.macro_modeler(h_macro_flat, attn_macro)

        return self.classifier(torch.cat([feat_micro, feat_macro], dim=1))

def load_roi_mapping(pkl_path, num_rois=90):
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            m = pickle.load(f)
        if isinstance(m, dict): m = list(m.values())
        return torch.tensor(np.array(m, dtype=int), dtype=torch.long)
    return torch.randint(0, 8, (num_rois,))


def load_mni_coordinates(file_path, num_rois=90):
    if os.path.exists(file_path): return torch.tensor(np.loadtxt(file_path), dtype=torch.float32)
    return torch.randn(num_rois, 3)





