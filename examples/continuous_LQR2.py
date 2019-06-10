import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torchdiffeq._impl.adjoint_PMP import odeint_adjoint as odeint

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=500)
parser.add_argument('--batch_time', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=16000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')

parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--tol', type=float, default=1e-3)
args = parser.parse_args()


device = torch.device('cuda:' + str(1) if torch.cuda.is_available() else 'cpu')


# from optimal_control.LQR import LQR
# ###################### System Dynamics and cost ######################
A = np.array([[-2.,0.3],[0.8,-0.1]])

B = np.array([[10.0,-3.0],[0.0,3.0]])


Qx = np.array([[10.,0.],[0.0,10.0]])
Qu = np.array([[0,0],[0,0]])
#######################################################################


true_y0 = torch.tensor([[10., 10.]])



t = torch.linspace(0.01, 55., args.data_size)
true_y = torch.empty(1000,1,2).uniform_(-10,10)

def get_batch():
    s = torch.from_numpy(
        np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))

    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    return batch_y0, batch_t, batch_y

class IntegerLoss(nn.Module):
    def __init__(self):
        super(IntegerLoss, self).__init__()


        self.Q = torch.tensor(Qx).type(torch.FloatTensor)
        self.R =  torch.tensor(Qu).type(torch.FloatTensor)


    def forward(self, u, x):
        d=x.dim()


        x = x-5
        # uRu = u.mm(R.mm(u.T))
        uR = torch.matmul(u, R)
        ut = u.t()
        uRu = torch.matmul(uR, ut)

        xQ = torch.matmul(x, Q)
        xt = x.t()
        xQx = torch.matmul(xQ, xt)

        loss = xQx + uRu
        #print(loss)
        return loss


class Controller(nn.Module):
    def __init__(self, dim_t, dim_u):
        super(Controller, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(dim_t, 10),
            nn.Tanh(),
            nn.Linear(10, dim_u),
            #nn.Tanh()
            # nn.Linear(5, dim_u),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, x):
        return self.net(t)


class system_linear_dynamics(nn.Module):
    def __init__(self, A, B ):
        super(system_linear_dynamics, self).__init__()
        self.A = torch.tensor(A).type(torch.FloatTensor)
        self.B = torch.tensor(B).type(torch.FloatTensor)

    def forward(self, t, x, u):
        dynamics_x = torch.matmul(x,self.A)
        dynamics_u = torch.matmul(u,self.B)
        dynamics = torch.add(dynamics_x,dynamics_u)
        return dynamics


class closed_loop_dynamics(nn.Module):
    def __init__(self, linear_dyn, controller):
        super(closed_loop_dynamics, self).__init__()

        self.linear_dyn = linear_dyn
        self.controller = controller

    def forward(self, t, x):
        t_plus = t.unsqueeze(0)
        u = self.controller(t_plus,x)
        x_1 = self.linear_dyn(t,x,u)
        return x_1


class ODEBlock(nn.Module):

    def __init__(self, odefunc, integer_loss=None):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()

        self.integer_loss = integer_loss

    def forward(self, x,t):
        out = odeint(self.odefunc, x, t, rtol=args.tol, atol=args.tol, integer_loss= self.integer_loss)
        return out

    # @property
    # def nfe(self):
    #     return self.odefunc.nfe
    #
    # @nfe.setter
    # def nfe(self, value):
    #     self.odefunc.nfe = value



class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val



Q =  torch.tensor(Qx).type(torch.FloatTensor)
R =  torch.tensor(Qu).type(torch.FloatTensor)

def loss_LQR(x,batch_t):
    d = x.ndimension()

    x = x-5
    xt = torch.transpose(x, d-2, d-1)
    Qx = torch.matmul(Q,xt)
    xQx = torch.matmul(x, Qx)
    # loss = torch.add(uRu_int,xQx_int)
    # loss_clamp = torch.clamp(loss, min = -1000, max = 1000)

    return xQx



if __name__ == '__main__':

    #################### Setup Learning Model ##################
    ii = 0
    dim = 2
    dim_t = 1

    integer_loss = IntegerLoss()

    controller = Controller(dim_t,dim)
    lin_dyn = system_linear_dynamics(A,B)
    close_loop = closed_loop_dynamics(lin_dyn,controller)
    func = ODEBlock(close_loop,integer_loss)



    #optimizer = torch.optim.SGD(func.parameters(), lr=1e-6, momentum=0.9)
    #optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
    optimizer = torch.optim.Adam(func.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)


    end = time.time()


    # lr_fn = learning_rate_with_decay(
    #     args.batch_size, batch_denom=128, batches_per_epoch=batches_per_epoch, boundary_epochs=[60, 100, 140],
    #     decay_rates=[1, 0.1, 0.01, 0.001]
    # )

    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)

    ##############################################################

     ################## Train iteration #########################
    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch()

        pred_y = func(true_y0, batch_t)
        #pred_y2 = odeint(func2,batch_y0,batch_t)


        loss = loss_LQR(pred_y,batch_t)[-1]
        # print("############## TIME ###############")
        # print(batch_t)
        # print(batch_y0)




        print("############## LOSS ###############")
        #print(pred_y)
        print(loss)


        info = loss.backward()
        optimizer.step()
        #print(loss)


        #time_meter.update(time.time() - end)
        #loss_meter.update(loss.item())

        if itr % args.test_freq == 0:
            with torch.no_grad():
                pred_y = func(true_y0, batch_t)

                y_plot = pred_y.numpy()
                timer = batch_t.numpy()
                #plt.close()
                plt.clf()
                plt.plot(timer,y_plot[:,0,0])
                plt.plot(timer,y_plot[:,0,1])
                plt.draw()

                plt.pause(0.001)

                ii += 1

