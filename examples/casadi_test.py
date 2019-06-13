from sys import path
path.append(r"/home/julen/Documents/py_libraries/casadi/casadi-linux-py36-v3.4.5-64bit")
from casadi import *
x = MX.sym("x")
print(jacobian(tanh(x),x))