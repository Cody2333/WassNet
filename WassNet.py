
"""
using neural network to calculate wasserstein distance
between two data sets.
"""
from __future__ import print_function

import os
import argparse

import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from advertorch_examples.utils import get_mnist_train_loader
from advertorch_examples.utils import get_mnist_test_loader
from advertorch_examples.utils import TRAINED_MODEL_PATH
from advertorch.test_utils import LeNet5
from advertorch.attacks import LinfPGDAttack,L1BasicIterativeAttack
import torch.autograd as autograd

DIM = 64
one = torch.tensor(1.0)
mone = one * -1
use_cuda = torch.cuda.is_available()

def calc_gradient_penalty(netD, real_data, fake_data):
    #print real_data.size()
    batch_size = real_data.size()[0]
    alpha = torch.rand(batch_size, 1,1,1)
    alpha = alpha.expand_as(real_data)
    alpha = alpha.cuda(gpu) if use_cuda else alpha

    interpolates = alpha * real_data + (1 - alpha) * fake_data

    if use_cuda:
        interpolates = interpolates.cuda(gpu)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(gpu) if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(batch_size, -1)
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    gradient_penalty = ((gradients_norm - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        main = nn.Sequential(
            nn.Conv2d(1, DIM, 5, stride=2, padding=2),
            # nn.Linear(OUTPUT_DIM, 4*4*4*DIM),
            nn.LeakyReLU(True),
            nn.Conv2d(DIM, 2*DIM, 5, stride=2, padding=2),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            nn.LeakyReLU(True),
            nn.Conv2d(2*DIM, 4*DIM, 5, stride=2, padding=2),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            nn.LeakyReLU(True),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            # nn.LeakyReLU(True),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            # nn.LeakyReLU(True),
        )
        self.main = main
        self.output = nn.Linear(4*4*4*DIM, 1)

    def forward(self, input):
        input = input.view(-1, 1, 28, 28)
        out = self.main(input)
        out = out.view(-1, 4*4*4*DIM)
        out = self.output(out)
        return out.view(-1)

class Wasserstein():
    def __init__(self,Net, batch_size, LAMBDA, iters, train_loader, adversary,use_cuda):
        self.batch_size = batch_size
        self.LAMBDA = LAMBDA
        self.iters = iters
        self.model = Net()
        self.train_loader = train_loader
        self.adversary = adversary
        self.use_cuda = use_cuda
        self.losses = {'loss': [], 'w_dist': []}

    def train(self, lr = 1e-4, betas = (0.5,0.9)):
        model = self.model
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
        for epoch in range(self.iters):
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                real_data = data
                real_data_v = autograd.Variable(real_data)
                model.zero_grad()
                # train with real
                D_real = model(real_data_v)
                D_real = D_real.mean()
                fake_data = adversary.perturb(data, target)
                fake_data_v = autograd.Variable(fake_data)
                # train with fake
                D_fake = model(fake_data_v)
                D_fake = D_fake.mean()
                gradient_penalty = calc_gradient_penalty(model, real_data_v.data, fake_data_v.data)

                loss = D_fake - D_real + gradient_penalty
                loss.backward()
                Wasserstein_D = D_real - D_fake
                optimizer.step()
                self.losses['loss'].append(loss.item())
                self.losses['w_dist'].append(Wasserstein_D)

                if batch_idx % args.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tW_dist: {:.6f}'.format(
                        epoch, batch_idx *
                        len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item(), Wasserstein_D))
        self.save_model(epoch)
        self.save_loss(epoch)

    def distance(self,real_data=None, fake_data=None):
        if not torch.is_tensor(real_data):
            real_data = self.real_data
        if not torch.is_tensor(fake_data):
            fake_data = self.fake_data
        w_distance = (self.model(real_data)-self.model(fake_data)).mean()
        return w_distance
    def save_model(self,epoch):
        torch.save(
            self.model.state_dict(),
            os.path.join(TRAINED_MODEL_PATH,  'epoch_' + str(epoch) + '_' + model_filename))
    def save_loss(self, epoch):
        np.save('loss_'+ str(epoch)+'.npy',self.losses['loss'])
        np.save('w_dist_'+ str(epoch)+'.npy',self.losses['w_dist'])


if __name__ == '__main__':
    LAMBDA = 10
    ITERS = 10
    parser = argparse.ArgumentParser(description='Train MNIST')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--mode', default="cln")
    parser.add_argument('--train_batch_size', default=10, type=int)
    parser.add_argument('--test_batch_size', default=1000, type=int)
    parser.add_argument('--log_interval', default=10, type=int)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if args.mode == "cln":
        nb_epoch = 10
        model_filename = "mnist_wass_net.pt"
    else:
        raise

    train_loader = get_mnist_train_loader(
        batch_size=args.train_batch_size, shuffle=True)
    test_loader = get_mnist_test_loader(
        batch_size=args.test_batch_size, shuffle=False)

    filename = "mnist_lenet5_clntrained.pt"
    model = LeNet5()
    model.load_state_dict(
        torch.load(os.path.join(TRAINED_MODEL_PATH, filename), map_location='cpu'))
    model.to(device)
    model.eval()
    adversary = L1BasicIterativeAttack(
        model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.2,
        nb_iter=10, eps_iter=0.01, clip_min=0.0, clip_max=1.0,
        targeted=False)
    w = Wasserstein(Net,args.train_batch_size,LAMBDA,ITERS,train_loader, adversary,use_cuda)
    w.train()