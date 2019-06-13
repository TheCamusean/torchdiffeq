import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


save_dir = "/home/julen/Documents/IAS/PMP_NODE/torchdiffeq/examples/pendulum_direct/"
file_name = "direct_output.npy"
model_name = "pendulum_init.pt"

out = np.load(save_dir+file_name, allow_pickle=True)

cuda_B = torch.cuda.is_available()


device = torch.device('cuda:' + str(1) if torch.cuda.is_available() else 'cpu')


class Controller(nn.Module):
    def __init__(self, dim_t, dim_u):
        super(Controller, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(dim_t, 50),
            nn.Tanh(),
            nn.Linear(50, 10),
            nn.Tanh(),
            nn.Linear(10, 10),
            nn.Tanh(),
            nn.Linear(10, dim_u),
        )


        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t):
        return self.net(t)




time = np.asarray(out[0][1:])

time = torch.from_numpy(time).type(torch.FloatTensor)

u = np.asarray(out[1])
u_t = torch.from_numpy(u).double().type(torch.FloatTensor)


def batch_data(batch_size):

    t = time.unsqueeze(1)
    u = u_t.unsqueeze(1)
    rand = np.random.randint(0,t.__len__()-1, size=batch_size)
    return u[rand],t[rand]



niters = 1000000
test_freq = 200
if __name__ == '__main__':

    #################### Setup Learning Model ##################
    ii = 0

    dim = 1
    controller = Controller(dim,dim)

    optimizer = torch.optim.Adam(controller.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    #optimizer = torch.optim.RMSprop(controller.parameters(), lr=1e-7)


     ################## Train iteration #########################
    for itr in range(1, niters + 1):
        optimizer.zero_grad()

        true_u,t = batch_data(170)


        pred_u = controller(t)
        loss = torch.mean(torch.abs(pred_u - true_u))




        info = loss.backward()
        optimizer.step()
        #print(loss)


        if itr % test_freq == 0:
            with torch.no_grad():
                print(loss)
                u_plot = pred_u.detach().numpy()
                timer = t.numpy()
                #plt.close()
                plt.clf()
                plt.plot(timer[:,0],u_plot[:,0],'*')
                plt.draw()

                plt.pause(0.001)

                torch.save(controller.state_dict(),save_dir+model_name)




