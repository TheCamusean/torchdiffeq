import os
import argparse
import time

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchdiffeq._impl.adjoint_PMP import odeint_adjoint as odeint



parser = argparse.ArgumentParser('ODE pendulum')
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




class IntegerLoss(nn.Module):
    def __init__(self):
        super(IntegerLoss, self).__init__()


    def forward(self, u, x):
        x = x[0, 0] + np.pi
        loss = x*x
        return loss


class Controller(nn.Module):
    def __init__(self, dim_t, dim_u):
        super(Controller, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(dim_t, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, dim_u),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, x):
        return self.net(t)

class pendulum_dynamics(nn.Module):
    def __init__(self):
        super(pendulum_dynamics, self).__init__()
        self.b = 1.
        self.g = 9.8
        self.m = 0.5
        self.L = 1.

        self.b_m = -self.b/self.m
        self.g_L = -self.g/self.L
    def forward(self, t, x, u):
        sin_x1 = torch.sin(x[:,0])

        out_x1 = self.b_m*x[:,1] + self.g_L*sin_x1
        out_x2 = x[:,0] + u[:,0]

        out_x2 = out_x2.unsqueeze(1)
        out_x1 = out_x1.unsqueeze(1)
        out = torch.cat([out_x1,out_x2],dim=1)

        return out


class closed_loop_dynamics(nn.Module):
    def __init__(self, dynamics, controller):
        super(closed_loop_dynamics, self).__init__()

        self.dynamics = dynamics
        self.controller = controller

    def forward(self, t, x):
        t_plus = t.unsqueeze(0)
        t_plus = t_plus.unsqueeze(0)
        u = self.controller(t_plus,x)
        x_1 = self.dynamics(t,x,u)
        return x_1

class ODE_net(nn.Module):

    def __init__(self, dynamics, loss_integer):
        super(ODE_net, self).__init__()
        self.dynamics = dynamics
        self.loss_integer = loss_integer

    def forward(self, x,t):
        out = odeint(self.dynamics, x, t, rtol=args.tol, atol=args.tol, integer_loss=self.loss_integer)
        return out



def loss_LQR(x,batch_t):

    x1 = x[:,0,0] + np.pi

    y = x[:,0,1]
    loss = x1*x1

    return loss



x_0 = torch.tensor([[0.0,0.0]])

t = torch.linspace(0.05,20,100)


if __name__ == '__main__':

    #################### Setup Learning Model ##################
    ii = 0

    dim = 1
    loss_integer = IntegerLoss()


    controller = Controller(dim,dim)
    pendulum = pendulum_dynamics()
    close_dyn = closed_loop_dynamics(pendulum, controller)
    func = ODE_net(close_dyn, loss_integer)


    optimizer = torch.optim.Adam(func.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)


    end = time.time()

     ################## Train iteration #########################
    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()

        pred_y = func(x_0, t)
        loss = torch.mean(loss_LQR(pred_y,t))


        print("############## LOSS ###############")
        print(loss)

        info = loss.backward()
        optimizer.step()
        #print(loss)

        if itr % args.test_freq == 0:
            with torch.no_grad():
                pred_y = func(x_0, t)

                y_plot = pred_y.numpy()
                timer = t.numpy()
                #plt.close()
                plt.clf()
                plt.plot(timer,y_plot[:,0,0])
                plt.plot(timer,y_plot[:,0,1])
                plt.draw()

                plt.pause(0.001)

                ii += 1

