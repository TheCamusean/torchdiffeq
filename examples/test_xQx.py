import numpy as np
import torch.nn as nn
import torch



Q_x = np.array([[20.,0.],[0.0,10.0]])

Q_x = torch.tensor(Q_x).type(torch.FloatTensor)

x = torch.tensor([[10., -10.]]).t()

x.requires_grad=True

class IntegerLoss(nn.Module):
    def __init__(self):
        super(IntegerLoss, self).__init__()


        self.Q = Q_x


    def forward(self, x):

        xQx = torch.matmul(x.t(),torch.matmul(self.Q, x))
        return xQx


integer_loss = IntegerLoss()

loss = integer_loss(x)
print(loss)



dLdy = torch.autograd.grad(loss,x,(torch.ones(1,1),))

print(2*torch.matmul(Q_x, x))
print(dLdy)