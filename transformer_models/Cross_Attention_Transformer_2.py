import torch
import torch.nn as nn
import timm
import math

class MMViTCrossAttentionTransformer_2(nn.Module):
    def __init__(self, num_tabular_features, num_transformer_layers, num_heads, hidden_dim, dropout):
        super(MMViTCrossAttentionTransformer_2, self).__init__()
        
        self.encoder = timm.create_model('vit_base_patch14_reg4_dinov2.lvd142m', pretrained=True)
        self.vit_embed_dim = self.encoder.embed_dim # Typically 768 for vit_base

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.hidden_dim = hidden_dim
        self.image_projection = nn.Linear(self.vit_embed_dim, self.hidden_dim)

        self.num_labs = num_tabular_features // 3
        self.tabular_embed = nn.Linear(3, self.hidden_dim)

        # Positional encoding for tabular data
        self.pos_encoding_tab = nn.Parameter(torch.zeros(1, self.num_labs, hidden_dim))
        nn.init.normal_(self.pos_encoding_tab, mean=0, std=0.02)
        
        # Cross attention layers for Tabular attending to Image
        self.tab_to_img_attention_layers = nn.ModuleList([
            CrossAttentionLayer(self.hidden_dim, num_heads, dropout)
            for _ in range(num_transformer_layers)
        ])

        # Cross attention layers for Image attending to Tabular
        self.img_to_tab_attention_layers = nn.ModuleList([
            CrossAttentionLayer(self.hidden_dim, num_heads, dropout)
            for _ in range(num_transformer_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        # Classifier input dimension is doubled due to concatenation
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim), # Adjusted input
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1),      # Adjusted input
            nn.Sigmoid()
        )

    def forward(self, images, tabular_data):
        batch_size = images.size(0)
        
        # Get image tokens
        # image_vectors = self.encoder(images) # Original call, might return CLS token or pooled output
        image_patch_embeddings = self.encoder.forward_features(images) # Shape: [bs, num_patches, vit_embed_dim]
        image_tokens = self.image_projection(image_patch_embeddings)   # Shape: [bs, num_patches, hidden_dim]
        image_tokens = self.dropout(image_tokens) # Apply dropout

        # Get lab tokens
        tabular_data = tabular_data.view(batch_size, self.num_labs, 3)  # reshape for each token [bs, 14, 3]
        tabular_tokens = self.tabular_embed(tabular_data) # [bs, 14, hidden_dim]
        tabular_tokens += self.pos_encoding_tab
        tabular_tokens = self.dropout(tabular_tokens)
        
        # Branch 1: Tabular features attend to Image features
        attended_tabular_tokens = tabular_tokens.clone() # Use clone if layers might modify in place or original needed
        for layer in self.tab_to_img_attention_layers:
            attended_tabular_tokens = layer(attended_tabular_tokens, image_tokens) # x=tabular, context=image
        
        # Branch 2: Image features attend to Tabular features
        attended_image_tokens = image_tokens.clone() # Use clone
        for layer in self.img_to_tab_attention_layers:
            attended_image_tokens = layer(attended_image_tokens, tabular_tokens) # x=image, context=tabular
        
        # Mean pooling over tokens for classification
        avg_tabular_features = torch.mean(attended_tabular_tokens, dim=1)
        avg_image_features = torch.mean(attended_image_tokens, dim=1)
        
        # Concatenate features from both branches
        fused_features = torch.cat((avg_tabular_features, avg_image_features), dim=1)
        
        output = self.classifier(fused_features)

        return output


class CrossAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super(CrossAttentionLayer, self).__init__()
        
        # Multi-head cross attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, context):
        # Cross attention: x (images) attends to context (tabular)
        # x shape: [batch_size, seq_len_q, hidden_dim]
        # context shape: [batch_size, seq_len_k, hidden_dim]
        
        # Layer normalization before attention
        norm_x = self.norm1(x)
        norm_context = self.norm2(context)
        
        # Cross attention
        attn_output, _ = self.cross_attention(
            query=norm_x,
            key=norm_context,
            value=norm_context
        )
        
        # Residual connection
        x = x + self.dropout(attn_output)
        
        # Feed-forward network
        norm_x = self.norm3(x)
        ff_output = self.feed_forward(norm_x)
        
        # Residual connection
        output = x + ff_output
        
        return output
    