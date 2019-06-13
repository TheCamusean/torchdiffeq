from sys import path
path.append(r"/home/julen/Documents/py_libraries/casadi/casadi-linux-py36-v3.4.5-64bit")


from casadi import *
import numpy as np

save_dir = "/home/julen/Documents/IAS/PMP_NODE/torchdiffeq/examples/pendulum_direct/"



T = 30. # Time horizon
N = 200 # number of control intervals

# Declare model variables
x1 = MX.sym('x1')
x2 = MX.sym('x2')
x = vertcat(x1, x2)
u = MX.sym('u')

#########################################################################
######################### Model Equations ###############################
################################ Pendulum #########################################

b = 1.
g = 9.8
m = 0.6
L = 1

b_m = -b / m
g_L = -g / L

x1_dot =  b_m*x2 + g_L*sin(x1)
x2_dot = x1 + u

xdot = vertcat(x1_dot, x2_dot)

# Objective term
L = (x1-np.pi)**2

# Formulate discrete time dynamics
if False:
   # CVODES from the SUNDIALS suite
   dae = {'x':x, 'p':u, 'ode':xdot, 'quad':L}
   opts = {'tf':T/N}
   F = integrator('F', 'cvodes', dae, opts)
else:
   # Fixed step Runge-Kutta 4 integrator
   M = 4 # RK4 steps per interval
   DT = T/N/M
   f = Function('f', [x, u], [xdot, L])
   X0 = MX.sym('X0', 2)
   U = MX.sym('U')
   X = X0
   Q = 0
   for j in range(M):
       k1, k1_q = f(X, U)
       k2, k2_q = f(X + DT/2 * k1, U)
       k3, k3_q = f(X + DT/2 * k2, U)
       k4, k4_q = f(X + DT * k3, U)
       X=X+DT/6*(k1 +2*k2 +2*k3 +k4)
       Q = Q + DT/6*(k1_q + 2*k2_q + 2*k3_q + k4_q)
   F = Function('F', [X0, U], [X, Q],['x0','p'],['xf','qf'])

# Evaluate at a test point
Fk = F(x0=[0.0,0.0],p=0.4)
print(Fk['xf'])
print(Fk['qf'])

# Start with an empty NLP
w=[]
w0 = []
lbw = []
ubw = []
J = 0
g=[]
lbg = []
ubg = []

# "Lift" initial conditions
Xk = MX.sym('X0', 2)
w += [Xk]
lbw += [0, 1]
ubw += [0, 1]
w0 += [0, 0]

# Formulate the NLP
for k in range(N):
    # New NLP variable for the control
    Uk = MX.sym('U_' + str(k))
    w   += [Uk]
    lbw += [-10]
    ubw += [10]
    w0  += [0]


    # Integrate till the end of the interval
    Fk = F(x0=Xk, p=Uk)
    Xk_end = Fk['xf']
    J=J+Fk['qf']

    # New NLP variable for state at end of interval
    Xk = MX.sym('X_' + str(k+1), 2)
    w   += [Xk]
    lbw += [-inf, -inf]
    ubw += [  inf,  inf]
    w0  += [0, 0]

    # Add equality constraint
    g   += [Xk_end-Xk]
    lbg += [0, 0]
    ubg += [0, 0]


    if k>0:
        g += [Uk-Uk_old]
        lbg += [-0.01]
        ubg += [0.01]

    Uk_old = Uk

# Create an NLP solver
prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}
solver = nlpsol('solver', 'ipopt', prob);

# Solve the NLP
sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
w_opt = sol['x'].full().flatten()

# Plot the solution
x1_opt = w_opt[0::3]
x2_opt = w_opt[1::3]
u_opt = w_opt[2::3]

tgrid = [T/N*k for k in range(N+1)]
import matplotlib.pyplot as plt
plt.figure(1)
plt.clf()
plt.plot(tgrid, x1_opt, '--')
plt.plot(tgrid, x2_opt, '-')
plt.step(tgrid, vertcat(DM.nan(1), u_opt), '-.')
plt.xlabel('t')
plt.legend(['x1','x2','u'])
plt.grid()
plt.show()

plt.clf()
plt.plot(u_opt,'b*')
plt.show()


x = np.array([x1_opt,x2_opt])
output = [tgrid,u_opt,x]

file_name = "direct_output.npy"

np.save(save_dir+file_name , output)
