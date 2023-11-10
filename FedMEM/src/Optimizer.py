from torch.optim import Optimizer

class PerMFL(Optimizer):
    def __init__(self, params, alpha=0.01, lamda=0.1):
        # self.local_weight_updated = local_weight # w_i,K
        if alpha < 0.0:
            raise ValueError("Invalid learning rate: {}".format(alpha))
        if lamda < 0.0:
            raise ValueError("Invalid learning rate: {}".format(alpha))

        defaults = dict(alpha=alpha, lamda=lamda)
        super(PerMFL, self).__init__(params, defaults)

    def step(self, local_weight_updated, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        weight_update = local_weight_updated.copy()
        for group in self.param_groups:
            for theta, localweight in zip(group['params'], weight_update):
                theta.data = theta.data - group['alpha'] * ( theta.grad.data + group['lamda'] * (theta.data - localweight.data))
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
