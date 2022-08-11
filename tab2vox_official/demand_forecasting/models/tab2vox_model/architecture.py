import torch

class Architecture(object):    
    def __init__(self, model, lr=3e-4, momentum=0.9, weight_decay=1e-3):
        self.model = model
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=weight_decay)

    def step(self, valid_inputs, valid_targets):
        self.optimizer.zero_grad()
        self.backward_step(valid_inputs, valid_targets) 
        self.optimizer.step()                           

    def backward_step(self, valid_inputs, valid_targets):
        loss = self.model.loss(valid_inputs, valid_targets) 
        loss.backward()