import copy
import numpy as np
import torch
from torch.autograd import Variable
import copy



class SaveOutput:
    def __init__(self):
        self.outputs = []
        self.inputs = []

    def __call__(self, module, module_in, module_out):
        self.inputs.append(module_in)
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []
        self.inputs = []


class feature_map_attack(object):
    def __init__(
        self, model, epsilon=16 / 255.0, step_size=0.004, device = 'cuda:0', steps=10, decay_factor=1, random_start = True
    ):
        self.saveout = SaveOutput()
        self.hook_handles = []
        self.step_size = step_size
        self.epsilon = epsilon
        self.steps = steps
        self.device = device
        self.model = copy.deepcopy(model).to(self.device)
        for p in self.model.parameters():
            p.requires_grad = False
        self.std_loss_fn = self._std_loss
        self.mse_loss = torch.nn.MSELoss().to(self.device)
        self.x_ent_loss_fn = torch.nn.CrossEntropyLoss().to(self.device)
        self.decay_factor = decay_factor
        self.random_start = random_start
        for layer in self.model.modules():
            if isinstance(layer, torch.nn.modules.conv.Conv2d):
                handle = layer.register_forward_hook(self.saveout)
                self.hook_handles.append(handle)

    def _random_start(self, x_nat_np):
        return x_nat_np + np.random.uniform(-self.epsilon, self.epsilon, x_nat_np.shape).astype('float32')
    
    def __call__(self, X_nat, attack_layers):
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.to(self.device).eval()
        X_nat_np = X_nat.detach().numpy()
        avg = X_nat_np.mean()
        if self.random_start:
            X = self._random_start(X_nat_np)
        else:
            X = np.copy(X_nat_np)
        ori_var = Variable(
                torch.from_numpy(X_nat_np).to(self.device), requires_grad=True, volatile=False
            )
        ori_scores = self.model(ori_var).detach()
        y = ori_scores.argmax().long().unsqueeze(0)
        original_features = []
        for original_feature in self.saveout.outputs:
            original_features.append(copy.copy(original_feature).detach())
        momentum = 0
        self.saveout.clear()
        out_list = []
        small_list = []
        large_list = []

        for i in range(self.steps):
            X_var = Variable(
                torch.from_numpy(X).to(self.device), requires_grad=True, volatile=False
            )

            logit = self.model(X_var)
            X_ent_loss = self.x_ent_loss_fn(logit,y)
            pred = logit.argmax().long().unsqueeze(0)
            mse_loss = 0
            for idx in attack_layers:
                mse_loss += self.mse_loss(self.saveout.outputs[idx],original_features[idx])
            loss = mse_loss - X_ent_loss
            self.model.zero_grad()
            loss.backward()
            grad = X_var.grad.data.cpu().numpy()
            X_var.grad.zero_()
            velocity = grad / np.sum(np.absolute(grad))
            momentum = self.decay_factor * momentum + velocity
            X += self.step_size * np.sign(momentum)
            dif = X - X_nat_np
            out_list.append((np.abs(dif) > 16).astype(np.uint8))
            small_list.append((dif < -16).astype(np.uint8))
            large_list.append((dif > 16).astype(np.uint8))
            X = np.clip(X, X_nat_np - self.epsilon, X_nat_np + self.epsilon)
            X = np.clip(X, 0, 1)
            self.saveout.clear()
        return torch.from_numpy(X), out_list, small_list, large_list

    def _std_loss(self, logit):
        return -1 * logit.view(logit.shape[0], -1).std(1)