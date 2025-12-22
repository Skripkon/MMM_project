import torch
from torch import nn


class TabularTransformer(nn.Module):
    """Single-modal transformer model for tabular data with multiple feature groups"""
    def __init__(self, num_classes: int = 11255, d_model: int = 128, nhead: int = 4, 
                 num_layers: int = 3, dim_feedforward: int = 256, dropout: float = 0.1):
        super().__init__()
        
        # Feature dimensions for each group
        self.feature_dims = {
            'climate': 19,
            'elevation': 1,
            'human_footprint': 22,
            'land_cover': 13,
            'soil_grids': 9
        }
        self.num_groups = len(self.feature_dims)
        
        # Project each feature group to d_model dimensions
        self.feature_projections = nn.ModuleDict({
            'climate': nn.Linear(19, d_model),
            'elevation': nn.Linear(1, d_model),
            'human_footprint': nn.Linear(22, d_model),
            'land_cover': nn.Linear(13, d_model),
            'soil_grids': nn.Linear(9, d_model)
        })
        
        # Learnable group embeddings (like positional encoding but for feature groups)
        self.group_embeddings = nn.Parameter(torch.randn(self.num_groups, d_model) * 0.02)
        
        # CLS token for aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer norm before classifier
        self.norm = nn.LayerNorm(d_model)
        
        # Final classifier
        self.head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_classes)
        )

    def forward(self, climate_features, elevation_features, human_footprint_features, 
                land_cover_features, soil_grids_features, **batch):
        """
        Model forward method.

        Args:
            climate_features (Tensor): climatic data. (B, 19)
            elevation_features (Tensor): elevation data. (B, 1)
            human_footprint_features (Tensor): human footprint data. (B, 22)
            land_cover_features (Tensor): land cover data. (B, 13)
            soil_grids_features (Tensor): soil grids data. (B, 9)
            **batch: other batch data.
        Returns:
            output (dict): output dict containing logits.
        """
        B = climate_features.size(0)
        
        # Project each feature group to d_model and stack as sequence
        projected = [
            self.feature_projections['climate'](climate_features),
            self.feature_projections['elevation'](elevation_features),
            self.feature_projections['human_footprint'](human_footprint_features),
            self.feature_projections['land_cover'](land_cover_features),
            self.feature_projections['soil_grids'](soil_grids_features)
        ] # (B, d_model)
        
        # Stack into sequence: (B, num_groups, d_model)
        x = torch.stack(projected, dim=1)
        
        # Add group embeddings
        x = x + self.group_embeddings.unsqueeze(0)
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_groups + 1, d_model)
        
        # Transformer encoding
        x = self.transformer_encoder(x)  # (B, num_groups + 1, d_model)
        
        # Take CLS token output
        cls_output = x[:, 0]  # (B, d_model)
        cls_output = self.norm(cls_output)
        
        # Classification
        logits = self.head(cls_output)
        
        return {"logits": logits}
