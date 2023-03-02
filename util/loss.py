import torch
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def inner_loss(label, matrixs):

    loss = 0

    if torch.sum(label == 0) > 1:
        loss += torch.mean(torch.var(matrixs[label == 0], dim=0))

    if torch.sum(label == 1) > 1:
        loss += torch.mean(torch.var(matrixs[label == 1], dim=0))

    return loss


def intra_loss(label, matrixs):
    a, b = None, None

    if torch.sum(label == 0) > 0:
        a = torch.mean(matrixs[label == 0], dim=0)

    if torch.sum(label == 1) > 0:
        b = torch.mean(matrixs[label == 1], dim=0)
    if a is not None and b is not None:
        return 1 - torch.mean(torch.pow(a-b, 2))
    else:
        return 0


def mixup_cluster_loss(matrixs, y_a, y_b, lam, intra_weight=2):

    y_1 = lam * y_a.float() + (1 - lam) * y_b.float()

    y_0 = 1 - y_1

    bz, roi_num, _ = matrixs.shape
    matrixs = matrixs.reshape((bz, -1))
    sum_1 = torch.sum(y_1)
    sum_0 = torch.sum(y_0)
    loss = 0.0

    if sum_0 > 0:
        center_0 = torch.matmul(y_0, matrixs)/sum_0
        diff_0 = torch.norm(matrixs-center_0, p=1, dim=1)
        loss += torch.matmul(y_0, diff_0)/(sum_0*roi_num*roi_num)
    if sum_1 > 0:
        center_1 = torch.matmul(y_1, matrixs)/sum_1
        diff_1 = torch.norm(matrixs-center_1, p=1, dim=1)
        loss += torch.matmul(y_1, diff_1)/(sum_1*roi_num*roi_num)
    if sum_0 > 0 and sum_1 > 0:
        loss += intra_weight * \
            (1 - torch.norm(center_0-center_1, p=1)/(roi_num*roi_num))

    return loss


def dominate_loss(A, soft_max=False):

    sz = A.shape[-1]

    m = torch.ones((sz, sz)).to(device=device)

    m.fill_diagonal_(-1)

    m = -m

    A = torch.matmul(A, m)

    if soft_max:
        max_ele = torch.logsumexp(A, dim=-1)
    else:
        max_ele, _ = torch.max(A, dim=-1, keepdim=False)

    max_ele = -max_ele
    max_ele = F.relu(max_ele)
    max_ele = torch.pow(max_ele, 2)
    return torch.sum(max_ele)


def topk_dominate_loss(A, k=3):

    all_ele = torch.sum(A, dim=-1)

    max_ele, _ = torch.topk(A, k, dim=-1, sorted=False)

    max_ele = torch.sum(max_ele, dim=-1)

    residual = F.relu(all_ele-2*max_ele)
    residual = torch.pow(residual, 2)

    return torch.sum(residual)