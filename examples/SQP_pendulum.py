import numpy as np
from scipy.optimize import minimize

from scipy.optimize import Bounds

from scipy.optimize import LinearConstraint

from scipy.optimize import NonlinearConstraint

from scipy.sparse import csc_matrix ##FOR SPARSE MATRIX




def rosen(x):
     """The Rosenbrock function"""
     return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)



x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
bounds = Bounds([0, -0.5], [1.0, 2.0])






res = minimize(rosen, x0, method='nelder-mead',
               options={'xtol': 1e-8, 'disp': True})

print(res.x)


################# BFGS ################

def rosen_der(x):
     xm = x[1:-1]
     xm_m1 = x[:-2]
     xm_p1 = x[2:]
     der = np.zeros_like(x)
     der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
     der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
     der[-1] = 200*(x[-1]-x[-2]**2)
     return der

res = minimize(rosen, x0, method='BFGS', jac=rosen_der,
                options={'disp': True})


print("BFGS")
print(res.x)


######## Newton-Gradient ##############
def rosen_hess(x):
     x = np.asarray(x)
     H = np.diag(-400*x[:-1],1) - np.diag(400*x[:-1],-1)
     diagonal = np.zeros_like(x)
     diagonal[0] = 1200*x[0]**2-400*x[1]+2
     diagonal[-1] = 200
     diagonal[1:-1] = 202 + 1200*x[1:-1]**2 - 400*x[2:]
     H = H + np.diag(diagonal)
     return H

res = minimize(rosen, x0, method='Newton-CG',
                jac=rosen_der, hess=rosen_hess,
                options={'xtol': 1e-8, 'disp': True})


print(res.x)


######### Newton with Hessian Product ############
def rosen_hess_p(x, p):
     x = np.asarray(x)
     Hp = np.zeros_like(x)
     Hp[0] = (1200*x[0]**2 - 400*x[1] + 2)*p[0] - 400*x[0]*p[1]
     Hp[1:-1] = -400*x[:-2]*p[:-2]+(202+1200*x[1:-1]**2-400*x[2:])*p[1:-1] \
                -400*x[1:-1]*p[2:]
     Hp[-1] = -400*x[-2]*p[-2] + 200*p[-1]
     return Hp

res = minimize(rosen, x0, method='Newton-CG',
                jac=rosen_der, hessp=rosen_hess_p,
                options={'xtol': 1e-8, 'disp': True})

print(res.x)


######### Constraints ################
linear_constraint = LinearConstraint([[1, 2], [2, 1]], [-np.inf, 1], [1, 1])  # LinearCOntraint(A, min,max) : min < Ax <Max


###### Non-Linear Constraints ######

########NONLINEAR f(x) < 1

def cons_f(x):
     return [x[0]**2 + x[1], x[0]**2 - x[1]]

def cons_J(x):
     return [[2*x[0], 1], [2*x[0], -1]]
def cons_H(x, v):
     return v[0]*np.array([[2, 0], [0, 0]]) + v[1]*np.array([[2, 0], [0, 0]])

nonlinear_constraint = NonlinearConstraint(cons_f, -np.inf, 1, jac=cons_J, hess=cons_H)


############  
def cons_H_sparse(x, v):
     return v[0]*csc_matrix([[2, 0], [0, 0]]) + v[1]*csc_matrix([[2, 0], [0, 0]])

nonlinear_constraint = NonlinearConstraint(cons_f, -np.inf, 1,
                                         jac=cons_J, hess=cons_H_spars

############ SLSQP ######################

ineq_cons = {'type': 'ineq',
              'fun' : lambda x: np.array([1 - x[0] - 2*x[1],
                                          1 - x[0]**2 - x[1],
                                          1 - x[0]**2 + x[1]]),
              'jac' : lambda x: np.array([[-1.0, -2.0],
                                          [-2*x[0], -1.0],
                                          [-2*x[0], 1.0]])}

eq_cons = {'type': 'eq',
            'fun' : lambda x: np.array([2*x[0] + x[1] - 1]),
            'jac' : lambda x: np.array([2.0, 1.0])}


x0 = np.array([0.5, 0])


res = minimize(rosen, x0, method='SLSQP', jac=rosen_der,
                constraints=[eq_cons, ineq_cons], options={'ftol': 1e-9, 'disp': True},
                bounds=bounds)

