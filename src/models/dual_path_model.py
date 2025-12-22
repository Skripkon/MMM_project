import torch
from torch import nn

from src.models.backbones.parts import DualPath, MLP


class DualPathModel(nn.Module):
    """Time series fusion model with cross-modal attention"""
    def __init__(self, num_classes: int = 11255, number_of_layers: int = 4, hidden_size=512, bioclimatic_shape=(4, 19, 12), landsat_shape=(6, 4, 21)):
        super().__init__()

        # LSTM 1: bioclimatic time series
        bioclimatic_items_cnt = bioclimatic_shape[0] * bioclimatic_shape[1] * bioclimatic_shape[2]

        self.bioclimatic_lstms = nn.ModuleList()
        for _ in range(number_of_layers):
            self.bioclimatic_lstms.append(
                DualPath(feature_size=bioclimatic_shape[1], time_size=bioclimatic_shape[2], num_layers=2)
            )
        self.bioclimatic_fc = nn.Linear(bioclimatic_items_cnt, hidden_size)

        # LSTM 2: landsat time series
        landsat_items_cnt = landsat_shape[0] * landsat_shape[1] * landsat_shape[2]

        self.landsat_lstms = nn.ModuleList()
        for _ in range(number_of_layers):
            self.landsat_lstms.append(
                DualPath(feature_size=landsat_shape[1], time_size=landsat_shape[2], num_layers=2)
            )
        self.landsat_fc = nn.Linear(landsat_items_cnt, hidden_size)
        
        # Cross-modal attention
        self.cross_modal_attentions = nn.ModuleList()
        self.cross_modal_attentions_posts = nn.ModuleList()
        for _ in range(number_of_layers):
            self.cross_modal_attentions.append(
                nn.Sequential(
                    nn.LayerNorm(landsat_items_cnt + bioclimatic_items_cnt),
                    nn.MultiheadAttention(embed_dim=landsat_items_cnt + bioclimatic_items_cnt, num_heads=8, batch_first=True)
                )
            )
            self.cross_modal_attentions_posts.append(
                nn.Sequential(
                    nn.LayerNorm(landsat_items_cnt + bioclimatic_items_cnt),
                    nn.Linear(landsat_items_cnt + bioclimatic_items_cnt, hidden_size * 4),
                    nn.ReLU(),
                    nn.Linear(hidden_size * 4, landsat_items_cnt + bioclimatic_items_cnt)
                )
            )
        
        self.head = MLP(
            input_size=hidden_size * 2,
            hidden_size=[hidden_size],
            output_size=num_classes,
            activation=nn.Tanh,
            dropout=0.1
        )

    def forward(self, bioclimatic, landsat, **batch):
        """
        Model forward method.

        Args:
            bioclimatic (Tensor): bioclimatic data. (B, 4, 19, 12)
            landsat (Tensor): landsat data. (B, 6, 4, 21)
            **batch: other batch data.
        Returns:
            output (dict): output dict containing logits.
        """
        B = bioclimatic.size(0)

        # Encode each modality
        for i in range(4):
            bioclimatic = self.bioclimatic_lstms[i](bioclimatic)
            landsat = self.landsat_lstms[i](landsat)

            bioclimatic_shape = bioclimatic.size()
            landsat_shape = landsat.size()

            features = torch.cat([
                bioclimatic.view(B, -1),
                landsat.view(B, -1)
            ], dim=-1)  # (B, feature_dim)

            # Cross-modal attention layers
            features, _ = self.cross_modal_attentions[i](features, features, features)
            features = self.cross_modal_attentions_posts[i](features)

            bioclimatic_items = bioclimatic_shape[1]*bioclimatic_shape[2]*bioclimatic_shape[3]
            
            bioclimatic = features[:, :bioclimatic_items].view(bioclimatic_shape)
            landsat = features[:, bioclimatic_items:].view(landsat_shape)

        bioclim_feat = self.lstm1_fc(bioclimatic.reshape(B, -1))  # (B, 512)
        landsat_feat = self.lstm2_fc(landsat.reshape(B, -1))  # (B, 512)
        
        # Stack features for cross-modal attention
        features = torch.cat([landsat_feat, bioclim_feat], dim=1)  # (B, 2*512)
        
        # Classification
        logits = self.head(features)
        
        return { "logits": logits }
