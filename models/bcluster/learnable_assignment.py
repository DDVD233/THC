import torch
import torch.nn as nn
from torch import Tensor


class LearnableAssignment(nn.Module):
    EPS = 1e-15

    def __init__(
            self,
            cluster_number: int,
            input_dim: int,
            feature_size: int,
            encoder=None,
            orthogonal=False,
            mask_top_k=False,
    ) -> None:
        """
        Initialize the cluster assignment module.

        :param cluster_number: number of clusters
        :param input_dim: number of nodes

        assignment: [input_dim, cluster_number]
        """
        super(LearnableAssignment, self).__init__()
        self.input_dim = input_dim
        self.cluster_number = cluster_number
        self.encoder = encoder
        self.embedding_dim = feature_size
        self.mask_top_k = mask_top_k
        self.softmax = nn.Softmax(dim=1)
        assignment: Tensor = nn.init.xavier_uniform_(
            torch.zeros(input_dim, cluster_number, dtype=torch.float)
        )

        if orthogonal:
            assignment = torch.transpose(assignment, 0, 1)  # [cluster_number, input_dim]
            orthogonal_assignment = torch.zeros(
                cluster_number, input_dim, dtype=torch.float
            )
            orthogonal_assignment[0] = assignment[0]
            for i in range(1, cluster_number):
                project = 0
                for j in range(i):
                    project += self.project(assignment[j], assignment[i])
                assignment[i] = assignment[i] - project
                orthogonal_assignment[i] = assignment[i]/torch.norm(assignment[i], p=2)

            assignment = torch.transpose(orthogonal_assignment, 0, 1)  # [input_dim, cluster_number]
        self.assignment = nn.Parameter(assignment, requires_grad=True)

    @staticmethod
    def project(u, v):
        return (torch.dot(u, v) / torch.dot(u, u)) * u

    def forward(self, batch: torch.Tensor, attention: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Compute the soft assignment for a batch of feature vectors, returning a batch of assignments
        for each cluster.

        :param batch: FloatTensor of [batch size, input_dim, embedding_dim]
        :param attention: FloatTensor of [batch size, input_dim, input_dim]
        :return: FloatTensor [batch size, cluster_number, embedding_dim]
        """
        flattened = batch.view(-1, self.input_dim * self.embedding_dim)
        if self.encoder is not None:
            flattened = self.encoder(flattened)
        batch = flattened.view(-1, self.input_dim, self.embedding_dim)
        transposed = torch.transpose(batch, 1, 2)

        mean_attention = attention.mean(dim=0)  # [input_dim, input_dim]
        soft_assignment = self.softmax(mean_attention @ self.assignment)
        # Find top 10% indices for each row
        if self.mask_top_k:
            top_k = int(self.cluster_number * 0.1) if self.cluster_number > 50 else int(self.cluster_number * 0.5)
            top_10, top_10_indices = torch.topk(soft_assignment, top_k, dim=1)
            soft_assignment = torch.zeros_like(soft_assignment).scatter_(1, top_10_indices, top_10)
        return torch.matmul(transposed, soft_assignment).transpose(1, 2), soft_assignment

    def loss(self, assignment):
        m = self.assignment.sigmoid()
        entropy_loss = (- m * torch.log(m + self.EPS) - (1 - m) * torch.log(1 - m + self.EPS)) * 0.2
        # sparsity_loss = m.mean() * 0.2
        # std = - torch.std(self.assignment)
        return entropy_loss.mean()

    def get_assignment(self) -> torch.Tensor:
        """
        Get the soft assignment.

        :return: FloatTensor [number of clusters, embedding dimension]
        """
        return self.assignment
