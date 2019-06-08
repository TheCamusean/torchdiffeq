import torch
import torch.nn as nn
from . import odeint
from .misc import _flatten, _flatten_convert_none_to_zeros


class OdeintAdjointMethod(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args):
        assert len(args) >= 9, 'Internal error: all arguments required.'
        y0, func, t, flat_params, rtol, atol, method, options, integer_loss = \
            args[:-8], args[-8], args[-7], args[-6], args[-5], args[-4], args[-3], args[-2], args[-1]

        ctx.func, ctx.rtol, ctx.atol, ctx.method, ctx.options, ctx.integer_loss = func, rtol, atol, method, options, integer_loss

        with torch.no_grad():
            ans = odeint(func, y0, t, rtol=rtol, atol=atol, method=method, options=options)
        ctx.save_for_backward(t, flat_params, *ans)
        return ans

    @staticmethod
    def backward(ctx, *grad_output):

        t, flat_params, *ans = ctx.saved_tensors
        ans = tuple(ans)
        func, rtol, atol, method, options, integer_loss = ctx.func, ctx.rtol, ctx.atol, ctx.method, ctx.options, ctx.integer_loss
        n_tensors = len(ans)
        f_params = tuple(func.parameters())

        # TODO: use a nn.Module and call odeint_adjoint to implement higher order derivatives.
        def augmented_dynamics(t, y_aug):
            # Dynamics of the original system augmented with
            # the adjoint wrt y, and an integrator wrt t and args.
            y, adj_y = y_aug[:n_tensors], y_aug[n_tensors:2 * n_tensors]  # Ignore adj_time and adj_params.

            with torch.set_grad_enabled(True):
                t = t.to(y[0].device).detach().requires_grad_(True)
                y = tuple(y_.detach().requires_grad_(True) for y_ in y)

                u = (func.base_func.controller(t,y[0]),)

                y_1 = (func.base_func.linear_dyn(t,y[0],u[0]),)

                loss_integer = integer_loss(u, y)

                #func_eval = func(t, y)


                dudy, *dudtheta = torch.autograd.grad(
                    u, y + f_params, [torch.ones([20, 1, 2]), ],
                    allow_unused=True, retain_graph=True
                )

                ############ We have to compute the gradient of the integer_loss with respect to u and with respect to x
                dLdy, dLdu, *dLdtheta = torch.autograd.grad(
                    loss_integer, y + u + f_params, [torch.ones([20,1,1]),],
                    allow_unused=True, retain_graph=True
                )

                ### This grad is actually pretty smart. Using the autograd, he knows he is going to have a chain rule, of dl/dout dout/dx, so he is smart and says
                ### as d lambda/dt = -lambda df()/dx , and lambda=grad_output dl/dout = lambda
                vjp_t, vjp_y, dfdu, *vjp_y_and_params = torch.autograd.grad(
                    y_1, (t,) + y + u + f_params,
                    tuple(-adj_y_ for adj_y_ in adj_y), allow_unused=True, retain_graph=True
                )
                ## From total derivative to partial derivative
                vjp_y = vjp_y - dfdu*dudy
                dLdy  =  dLdy - dLdu*dudy





            #vjp_y = vjp_y_and_params[:n_tensors]

            vjp_params = vjp_y_and_params
            #### Extra term from cost
            vjp_params = [a - b for a, b in zip(vjp_y_and_params, dLdtheta)]
            vjp_y = [a - b for a, b in zip(vjp_y, dLdy)]

            # autograd.grad returns None if no gradient, set to zero.
            vjp_t = torch.zeros_like(t) if vjp_t is None else vjp_t
            vjp_y = tuple(torch.zeros_like(y_) if vjp_y_ is None else vjp_y_ for vjp_y_, y_ in zip(vjp_y, y))
            vjp_params = _flatten_convert_none_to_zeros(vjp_params, f_params)

            if len(f_params) == 0:
                vjp_params = torch.tensor(0.).to(vjp_y[0])
            return (*y_1, *vjp_y, vjp_t, vjp_params)

        T = ans[0].shape[0]
        with torch.no_grad():
            adj_y = tuple(grad_output_[-1] for grad_output_ in grad_output)
            adj_params = torch.zeros_like(flat_params)
            adj_time = torch.tensor(0.).to(t)
            time_vjps = []
            for i in range(T - 1, 0, -1):

                ans_i = tuple(ans_[i] for ans_ in ans)
                grad_output_i = tuple(grad_output_[i] for grad_output_ in grad_output)
                func_i = func(t[i], ans_i)

                # Compute the effect of moving the current time measurement point.
                dLd_cur_t = sum(
                    torch.dot(func_i_.reshape(-1), grad_output_i_.reshape(-1)).reshape(1)
                    for func_i_, grad_output_i_ in zip(func_i, grad_output_i)
                )
                adj_time = adj_time - dLd_cur_t
                time_vjps.append(dLd_cur_t)

                # Run the augmented system backwards in time.
                if adj_params.numel() == 0:
                    adj_params = torch.tensor(0.).to(adj_y[0])
                aug_y0 = (*ans_i, *adj_y, adj_time, adj_params)
                aug_ans = odeint(
                    augmented_dynamics, aug_y0,
                    torch.tensor([t[i], t[i - 1]]), rtol=rtol, atol=atol, method=method, options=options
                )

                # Unpack aug_ans.
                adj_y = aug_ans[n_tensors:2 * n_tensors]
                adj_time = aug_ans[2 * n_tensors]
                adj_params = aug_ans[2 * n_tensors + 1]

                adj_y = tuple(adj_y_[1] if len(adj_y_) > 0 else adj_y_ for adj_y_ in adj_y)
                if len(adj_time) > 0: adj_time = adj_time[1]
                if len(adj_params) > 0: adj_params = adj_params[1]

                #adj_y = tuple(adj_y_ + grad_output_[i - 1] for adj_y_, grad_output_ in zip(adj_y, grad_output))

                del aug_y0, aug_ans

            time_vjps.append(adj_time)
            time_vjps = torch.cat(time_vjps[::-1])

            result = (*adj_y, None, time_vjps, adj_params, None, None, None, None, None)
            #print(result)

            return (*adj_y, None, time_vjps, adj_params, None, None, None, None, None)


def odeint_adjoint(func, y0, t, rtol=1e-6, atol=1e-12, method=None, options=None, integer_loss=None):

    # We need this in order to access the variables inside this module,
    # since we have no other way of getting variables along the execution path.
    if not isinstance(func, nn.Module):
        raise ValueError('func is required to be an instance of nn.Module.')

    if (integer_loss != None):
        if not isinstance(integer_loss, nn.Module):
            raise ValueError('integer_loss is required to be an instance of nn.Module.')





    tensor_input = False
    if torch.is_tensor(y0):

        class TupleFunc(nn.Module):

            def __init__(self, base_func):
                super(TupleFunc, self).__init__()
                self.base_func = base_func

            def forward(self, t, y):
                return (self.base_func(t, y[0]),)

        class TupleLoss(nn.Module):

            def __init__(self, int_loss):
                super(TupleLoss, self).__init__()
                self.int_loss = int_loss

            def forward(self, u, x):
                return (self.int_loss(u[0], x[0]),)

        tensor_input = True
        y0 = (y0,)
        func = TupleFunc(func)
        if (integer_loss != None):
            integer_loss = TupleLoss(integer_loss)

    flat_params = _flatten(func.parameters())
    ys = OdeintAdjointMethod.apply(*y0, func, t, flat_params, rtol, atol, method, options, integer_loss)

    if tensor_input:
        ys = ys[0]
    return ys
