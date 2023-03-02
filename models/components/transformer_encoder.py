import torch
from torch import nn as nn

from models.components import StoTrans
from models.bcluster.ocread import OCRead
from models.bcluster.learnable_assignment import LearnableAssignment


class TransPoolingEncoder(nn.Module):
    """
    Transformer encoder with Pooling mechanism.
    Input size: (batch_size, input_node_num, input_feature_size)
    Output size: (batch_size, output_node_num, input_feature_size)
    """

    def __init__(self, input_feature_size, input_node_num, hidden_size, output_node_num, pooling=True, orthogonal=True,
                 freeze_center=False, project_assignment=True, encoder=None, average_assignment=False, noise=0,
                 att_noise=False, clustering_type='ocread', no_encoder=False, mask_top_k=False, encoder_hidden_size=None):
        super().__init__()
        self.transformer = StoTrans(d_model=input_feature_size, nhead=4,
                                    dim_feedforward=hidden_size,
                                    batch_first=True, noise=att_noise)
        self.noise = noise
        self.pooling = pooling
        if encoder_hidden_size is None:
            encoder_hidden_size = hidden_size
        if pooling:
            if no_encoder:
                self.encoder = None
            elif encoder is not None:
                self.encoder = encoder
            else:
                self.encoder = nn.Sequential(
                    nn.Linear(input_feature_size * input_node_num, hidden_size),
                    nn.LeakyReLU(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.LeakyReLU(),
                    nn.Linear(hidden_size, input_feature_size * input_node_num),
                )  # Default encoder
            if clustering_type == 'ocread':
                self.clustering = OCRead(cluster_number=output_node_num, hidden_dimension=input_feature_size,
                                         encoder=self.encoder, orthogonal=orthogonal, freeze_center=freeze_center,
                                         project_assignment=project_assignment, average_assignment=average_assignment)
            elif clustering_type == 'learnable':
                self.clustering = LearnableAssignment(cluster_number=output_node_num, input_dim=input_node_num,
                                                      feature_size=input_feature_size, encoder=self.encoder,
                                                      orthogonal=orthogonal, mask_top_k=mask_top_k)

            self.norm = nn.LayerNorm(input_feature_size)

    def is_pooling_enabled(self):
        return self.pooling

    def forward(self, x):
        x = self.transformer(x)
        attention = self.transformer.attention_weights
        # if self.noise > 0:
        #     x = self.sampling(x, self.training)
        if self.pooling:
            x, assignment = self.clustering(x, attention)
            x = self.norm(x)
            return x, assignment
        return x, None

    def sampling(self, att_log_logit, training):
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, self.noise)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_log_logit = att_log_logit + random_noise
        # else:
        #     att_log_logit = att_log_logit.sigmoid()
        return att_log_logit

    def get_attention_weights(self):
        return self.transformer.get_attention_weights()

    def loss(self, assignment):
        return self.clustering.loss(assignment)
