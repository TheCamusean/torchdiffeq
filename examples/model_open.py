

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


save_dir = "/home/julen/Documents/IAS/PMP_NODE/torchdiffeq/examples/pendulum_direct/"
model_name = "pendulum_init.pt"


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



model = Controller(1,1)

model.load_state_dict(torch.load(save_dir+model_name))

print(model.eval())


t = torch.linspace(0,20,100)
t = t.unsqueeze(1)
y = model(t)

y_np = y.detach().numpy()
plt.plot(y_np)
plt.show()