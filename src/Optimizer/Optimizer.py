from torch.optim import Optimizer
import torch

class BBOptimizer(Optimizer):
    def __init__(self, params, alpha=0.05, lamda=0.5):

        alpha_bound=(0.005, 0.05)
        defaults = dict(alpha=alpha, alpha_bound=alpha_bound, lamda=lamda)
        super(BBOptimizer, self).__init__(params, defaults)

    def step(self, cluster_model, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            alpha_bound = group['alpha_bound']
            # print(alpha_bound)
            
            for p, cluster_weight in zip(group['params'],cluster_model.parameters()):
                if p.grad is None:
                    continue

                # grad = p.grad.data
                # state = self.state[p]

                # State initialization
                # if len(state) == 0:
                #    state['step'] = 0
                #    state['alpha'] = group['alpha']
                #    state['old_grad'] = torch.zeros_like(p.data)
                #    state['old_param'] = torch.clone(p.data).detach()

                # old_grad = state['old_grad']
                # old_param = state['old_param']
                # state['step'] += 1

                # BB step size calculation
                # s = p.data - old_param
                # y = grad - old_grad

                # Avoid division by zero
                # sty = torch.dot(s.view(-1), y.view(-1))
                # if sty <= 0:
                #     alpha = group['alpha']
                # else:
                #    alpha = sty / torch.dot(y.view(-1), y.view(-1))
                    # Bound the step size for stability
                    # print("alpha :",alpha.item())
                    # print("alpha_bound[1]:",alpha_bound[1])
                    # print("alpha_bound[0]:",alpha_bound[0])
                    
                #   alpha = max(min(alpha, alpha_bound[1]), alpha_bound[0])
                    # print("after maximization minimization alpha :",alpha)

                # Update parameters
                #p.data.add_(-alpha, grad)
                # print("alpha = ",alpha)
                p.data = p.data - 0.01*p.grad.data +  0.005*(p.data - cluster_weight.data)
                # Update state
                # state['old_grad'] = torch.clone(grad).detach()
                # state['old_param'] = torch.clone(p.data).detach()

        return loss

class PerMFL(Optimizer):
    def __init__(self, params, alpha=0.01, lamda=0.1):
        # self.local_weight_updated = local_weight # w_i,K
        if alpha < 0.0:
            raise ValueError("Invalid learning rate: {}".format(alpha))
        if lamda < 0.0:
            raise ValueError("Invalid learning rate: {}".format(alpha))

        defaults = dict(alpha=alpha, lamda=lamda)
        super(PerMFL, self).__init__(params, defaults)

    def step(self, cluster_updated, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        weight_update = cluster_updated.copy()
        for group in self.param_groups:
            for theta, clusterweight in zip(group['params'], weight_update):
                theta.data = theta.data - group['alpha'] * theta.grad.data + group['lamda'] * (theta.data - clusterweight.data)
        return group['params'], loss

    def update_param(self, local_weight_updated, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        weight_update = local_weight_updated.copy()
        for group in self.param_groups:
            for theta, localweight in zip(group['params'], weight_update):
                theta.data = localweight.data
        # return  p.data
        return group['params']
