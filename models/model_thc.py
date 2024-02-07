import torch
import torch.nn as nn

from .components.readout import DecReadout
from .components.transformer_encoder import TransPoolingEncoder


class THC(nn.Module):
    def __init__(self, model_config, node_num=360, pos_embed_dim=360, lr=1e-4):

        super().__init__()
        # self.layers = nn.Sequential([])
        self.attention_list = nn.ModuleList()
        self.readout_list = nn.ModuleList()
        forward_dim = node_num

        self.pos_encoding = model_config['pos_encoding']
        if self.pos_encoding == 'identity':
            self.node_identity = nn.Parameter(torch.zeros(node_num, pos_embed_dim), requires_grad=True)
            forward_dim = node_num + pos_embed_dim
            nn.init.kaiming_normal_(self.node_identity)

        sizes = list(model_config["sizes"])
        in_sizes = [node_num] + sizes[:-1]
        do_pooling = model_config["pooling"]
        self.hierarchical = model_config.get("hierarchical", False)
        self.global_pooling = model_config.get("global_pooling", False)
        self.readout = model_config.get("readout", "concat")
        self.noise = model_config.get("noise", 0)
        self.att_noise = model_config.get("att_noise", False)
        self.clustering_type = model_config.get("clustering_type", "ocread")
        self.no_encoder = model_config.get("no_encoder", False)
        self.do_pooling = do_pooling
        for index, size in enumerate(sizes):
            self.attention_list.append(
                TransPoolingEncoder(input_feature_size=forward_dim,
                                    input_node_num=in_sizes[index],
                                    hidden_size=model_config.get("transformer_hidden_size", 1024),
                                    output_node_num=size,
                                    pooling=do_pooling[index], orthogonal=model_config['orthogonal'],
                                    freeze_center=model_config.get('freeze_center', False),
                                    project_assignment=model_config['project_assignment'],
                                    average_assignment=model_config.get('average_assignment', False),
                                    noise=self.noise, att_noise=self.att_noise, clustering_type=self.clustering_type,
                                    no_encoder=self.no_encoder, mask_top_k=model_config.get('mask_top_k', False),
                                    encoder_hidden_size=model_config.get('encoder_hidden_size', 1024),))
            if self.hierarchical:
                self.readout_list.append(
                    DecReadout(feature_dim=forward_dim,
                               node_num=size,
                               hidden_dim=model_config.get("readout_hidden_size", 256),
                               feature_reduction_dim=model_config.get("feature_reduction_dim", 128)
                               )
                )

        if self.global_pooling:
            self.readout_list.append(DecReadout(feature_dim=forward_dim, node_num=sum(sizes)))
        elif self.readout == "concat":
            self.readout_list.append(nn.Sequential(
                nn.Linear(2 * len(sizes), 256),
                nn.LeakyReLU(),
                nn.Linear(256, 32),
                nn.LeakyReLU(),
                nn.Linear(32, 2)
            ))
        elif not self.hierarchical:
            self.readout_list.append(DecReadout(feature_dim=forward_dim, node_num=sizes[-1]))

    def forward(self, x):

        bz, _, _, = x.shape

        if self.pos_encoding == 'identity':
            pos_emb = self.node_identity.expand(bz, *self.node_identity.shape)
            x = torch.cat([x, pos_emb], dim=-1)

        assignments = []
        outputs = []
        for index, atten in enumerate(self.attention_list):
            x, assignment = atten(x)
            assignments.append(assignment)
            if self.hierarchical:
                out = self.readout_list[index](x)
                outputs.append(out)
            elif self.global_pooling:
                outputs.append(x)

        if self.hierarchical:
            if self.readout == "concat":
                x = torch.cat(outputs, dim=1)
                out = self.readout_list[-1](x)
            elif self.readout == "sum":
                out = torch.sum(torch.stack(outputs, dim=1), dim=1)
            elif self.readout == "mean":
                out = torch.mean(torch.stack(outputs, dim=1), dim=1)
            elif self.readout == "max":
                out = torch.max(torch.stack(outputs, dim=1), dim=1)[0]
            else:
                raise ValueError("Unknown readout method: {}".format(self.readout))
        elif self.global_pooling:
            x = torch.cat(outputs, dim=1)
            out = self.readout_list[-1](x)
        else:
            out = self.readout_list[-1](x)

        return out, assignments

    def get_attention_weights(self):
        return [atten.get_attention_weights() for atten in self.attention_list]

    def get_cluster_centers(self) -> torch.Tensor:
        """
        Get the cluster centers, as computed by the encoder.

        :return: [number of clusters, hidden dimension] Tensor of dtype float
        """
        return self.dec.get_cluster_centers()

    def loss(self, assignments):
        """
        Compute KL loss for the given assignments. Note that not all encoders contain a pooling layer.
        Inputs: assignments: [batch size, number of clusters]
        Output: KL loss
        """
        decs = list(filter(lambda x: x.is_pooling_enabled(), self.attention_list))
        assignments = list(filter(lambda x: x is not None, assignments))
        loss_all = None

        for index, assignment in enumerate(assignments):
            if loss_all is None:
                loss_all = decs[index].loss(assignment)
            else:
                loss_all += decs[index].loss(assignment)
        return loss_all
