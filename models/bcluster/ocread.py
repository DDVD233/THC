import torch
import torch.nn as nn

from .cluster import ClusterAssignment


class OCRead(nn.Module):
    def __init__(
            self,
            cluster_number: int,
            hidden_dimension: int,
            encoder: torch.nn.Module,
            alpha: float = 1.0,
            orthogonal=True,
            freeze_center=True, project_assignment=True,
            average_assignment=False
    ):
        """
        Initialize the OC-Read module.

        :param cluster_number: number of clusters
        :param hidden_dimension: hidden dimension, output of the encoder
        :param encoder: encoder to use
        :param alpha: parameter representing the degrees of freedom in the t-distribution, default 1.0
        """
        super(OCRead, self).__init__()
        self.encoder = encoder
        self.hidden_dimension = hidden_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        self.assignment = ClusterAssignment(
            cluster_number, self.hidden_dimension, alpha, orthogonal=orthogonal, freeze_center=freeze_center,
            project_assignment=project_assignment
        )
        self.average_assignment = average_assignment

        self.loss_fn = nn.KLDivLoss(size_average=False)

    def forward(self, batch: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Compute the cluster assignment using the ClusterAssignment after running the batch
        through the encoder part of the associated AutoEncoder module.
        :param batch: [batch size, embedding dimension] FloatTensor
        :return: [batch size, number of clusters] FloatTensor
        """
        node_num = batch.size(1)
        batch_size = batch.size(0)

        batch = batch.view(batch_size, -1)  # [batch size, embedding dimension]
        encoded = self.encoder(batch)
        encoded = encoded.view(batch_size * node_num, -1)  # [batch size * node_num, hidden dimension]
        assignment = self.assignment(encoded)  # [batch size * node_num, cluster_number]
        assignment = assignment.view(batch_size, node_num, -1)  # [batch size, node_num, cluster_number]
        if self.average_assignment:
            assignment = torch.mean(assignment, dim=0, keepdim=True).repeat([batch_size, 1, 1])
        encoded = encoded.view(batch_size, node_num, -1)  # [batch size, node_num, hidden dimension]
        # Multiply the encoded vectors by the cluster assignment to get the final node representations
        node_repr = torch.bmm(assignment.transpose(1, 2), encoded)  # [batch size, cluster_number, hidden dimension]
        return node_repr, assignment

    @staticmethod
    def target_distribution(batch: torch.Tensor) -> torch.Tensor:
        """
        Compute the target distribution p_ij, given the batch (q_ij), as in 3.1.3 Equation 3 of
        Xie/Girshick/Farhadi; this is used the KL-divergence loss function.

        :param batch: [batch size, number of clusters] Tensor of dtype float
        :return: [batch size, number of clusters] Tensor of dtype float
        """
        weight = (batch ** 2) / torch.sum(batch, 0)
        return (weight.t() / torch.sum(weight, 1)).t()

    def loss(self, assignment):
        flattened_assignment = assignment.view(-1, assignment.size(-1))
        target = self.target_distribution(flattened_assignment).detach()
        return self.loss_fn(flattened_assignment.log(), target) / flattened_assignment.size(0)

    def get_cluster_centers(self) -> torch.Tensor:
        """
        Get the cluster centers, as computed by the encoder.

        :return: [number of clusters, hidden dimension] Tensor of dtype float
        """
        return self.assignment.get_cluster_centers()
