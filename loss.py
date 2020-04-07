import torch
from torch import nn

class ContrastiveLoss(nn.Module):
    def __init__(self, tau=0.1):
        self.cuda_enabled = False
        super(ContrastiveLoss, self).__init__()
        self.tau = tau

    def forward(self, x):
        num_inputs = x.size(0)
        b = torch.mm(x, x.transpose(1, 0))
        n = torch.norm(x, p=2, dim=1).unsqueeze(1)
        n = torch.mm(n, n.transpose(1, 0))
        s = b / n
        m = torch.ones(num_inputs) - torch.eye(num_inputs)
        if self.cuda_enabled:
            m = m.cuda()
        d = torch.sum(torch.exp(s * m / self.tau), dim=1)
        v = torch.zeros(num_inputs).float()
        if self.cuda_enabled:
            v = v.cuda()
        for k in range(num_inputs):
            i, j = (k // 2) * 2, (k // 2) * 2 + 1
            v[k] = s[i][j]
        loss = torch.sum(-torch.log(torch.exp(v / self.tau) / d)) / num_inputs
        return loss

    def cuda(self):
        self.cuda_enabled = True
        return self
