import numpy as np
import copy
from torch.autograd import Variable
import torch

class Mifgsm():
    def __init__(self, model, epsilon, steps, step_size, device, decay_factor, random_start, surrogate, sigmoid_loss, weight_lambda, adaptive):
        self._decay_factor = decay_factor
        self._rand = random_start
        self._loss_fn = torch.nn.CrossEntropyLoss().to(device)
        self._model = copy.deepcopy(model)
        self._surrogate = copy.deepcopy(surrogate)
        self._epsilon = epsilon
        self._steps = steps
        self._step_size = step_size
        self._device = device
        self._sigmoid_loss = sigmoid_loss
        self._lambda = weight_lambda
        self._adaptive = adaptive

    def _random_start(self, x_nat_np):
        return x_nat_np + np.random.uniform(-self.epsilon, self.epsilon, x_nat_np.shape).astype('float32')
    
    def _update_lambda(self, weight_lambda, gt, pd):
        if gt == pd:
            weight_lambda = self._lambda/10
        return weight_lambda

    def __call__(self, X_nat):
        X_nat_np = X_nat.detach().numpy()
        for p in self._model.parameters():
            p.requires_grad = False
        self._model.eval()
        if self._rand:
            X = self._random_start(X_nat_np)
        else:
            X = np.copy(X_nat_np)
        momentum = 0
        ori_tensor = Variable(torch.from_numpy(X).to(self._device), requires_grad=True, volatile=False)
        ori_logits = self._model(ori_tensor)
        y = ori_logits.argmax().long().unsqueeze(0)
        y_for_lambda = ori_logits.argmax()
        surrogate_ori_logits = self._surrogate(ori_tensor)
        surrogate_y = surrogate_ori_logits.argmax().long().unsqueeze(0)
        weight_lambda = self._lambda
        for i in range(self._steps):
            X_var = Variable(torch.from_numpy(X).to(self._device), requires_grad=True, volatile=False)
            y_var = y.to(self._device)
            surrogate_y_var = surrogate_y.to(self._device)
            scores = self._model(X_var)
            pd = scores.argmax()
            surrogate_scores = self._surrogate(X_var)
            protect_loss = self._loss_fn(scores, y_var)
            attack_loss = self._loss_fn(surrogate_scores, surrogate_y_var)
            # print('protect:', protect_loss)
            # print('attack:',attack_loss)
            if self._adaptive:
                weight_lambda = self._update_lambda(weight_lambda, y_for_lambda, pd)
            loss = -weight_lambda * protect_loss + attack_loss
            self._model.zero_grad()
            self._surrogate.zero_grad()
            loss.backward()
            grad = X_var.grad.data.cpu().numpy()
            X_var.grad.zero_()
            velocity = grad / np.sum(np.absolute(grad))
            momentum = self._decay_factor * momentum + velocity
            X += self._step_size * np.sign(momentum)
            X = np.clip(X, X_nat_np - self._epsilon, X_nat_np + self._epsilon)
            X = np.clip(X, 0, 1)
        return torch.from_numpy(X)
