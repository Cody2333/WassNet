# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import print_function

import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from WassNet import WassNet

from advertorch.context import ctx_noparamgrad_and_eval
from advertorch.test_utils import LeNet5
from advertorch_examples.utils import get_mnist_train_loader
from advertorch_examples.utils import get_mnist_test_loader
from advertorch_examples.utils import TRAINED_MODEL_PATH

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MNIST')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--mode', default="cln", help="cln | adv")
    parser.add_argument('--train_batch_size', default=50, type=int)
    parser.add_argument('--test_batch_size', default=1000, type=int)
    parser.add_argument('--log_interval', default=200, type=int)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    wass_model_filename = 'mnist_wass_net.pt'
    wass_model = WassNet()
    wass_model.load_state_dict(
        torch.load(os.path.join(TRAINED_MODEL_PATH, wass_model_filename),map_location='cpu'))
    wass_model.to(device)
    wass_model.eval()
    # define loss
    def myloss(model,yvar,xvar,xadv):
        outputs = model(xadv)
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(outputs, yvar)
        w_loss = torch.mean(wass_model(xadv))
        return loss,w_loss
    if args.mode == "cln":
        flag_advtrain = False
        nb_epoch = 10
        model_filename = "mnist_lenet5_clntrained.pt"
    elif args.mode == "adv":
        flag_advtrain = True
        nb_epoch = 90
        model_filename = "mnist_lenet5_wass_advtrained.pt"
    else:
        raise

    train_loader = get_mnist_train_loader(
        batch_size=args.train_batch_size, shuffle=True)
    test_loader = get_mnist_test_loader(
        batch_size=args.test_batch_size, shuffle=False)

    model = LeNet5()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    if flag_advtrain:
        from advertorch.attacks import LinfPGDAttack, L1BasicIterativeAttack
        adversary = L1BasicIterativeAttack(
            model, loss_fn=myloss, eps=0.3,
            nb_iter=10, eps_iter=0.02, clip_min=0.0,
            clip_max=1.0, targeted=False,beta=0.5, early_stop=False)

    for epoch in range(nb_epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            ori = data
            if flag_advtrain:
                # when performing attack, the model needs to be in eval mode
                # also the parameters should be accumulating gradients
                with ctx_noparamgrad_and_eval(model):
                    data,_ = adversary.perturb(data, target)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(
                output, target, reduction='elementwise_mean')
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx *
                    len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

        model.eval()
        test_clnloss = 0
        clncorrect = 0

        if flag_advtrain:
            test_advloss = 0
            advcorrect = 0

        for clndata, target in test_loader:
            clndata, target = clndata.to(device), target.to(device)
            with torch.no_grad():
                output = model(clndata)
            test_clnloss += F.cross_entropy(
                output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            clncorrect += pred.eq(target.view_as(pred)).sum().item()

            if flag_advtrain:
                advdata = adversary.perturb(clndata, target)
                with torch.no_grad():
                    output = model(advdata)
                test_advloss += F.cross_entropy(
                    output, target, reduction='sum').item()
                pred = output.max(1, keepdim=True)[1]
                advcorrect += pred.eq(target.view_as(pred)).sum().item()

        test_clnloss /= len(test_loader.dataset)
        print('\nTest set: avg cln loss: {:.4f},'
              ' cln acc: {}/{} ({:.0f}%)\n'.format(
                  test_clnloss, clncorrect, len(test_loader.dataset),
                  100. * clncorrect / len(test_loader.dataset)))
        if flag_advtrain:
            test_advloss /= len(test_loader.dataset)
            print('Test set: avg adv loss: {:.4f},'
                  ' adv acc: {}/{} ({:.0f}%)\n'.format(
                      test_advloss, advcorrect, len(test_loader.dataset),
                      100. * advcorrect / len(test_loader.dataset)))

    torch.save(
        model.state_dict(),
        os.path.join(TRAINED_MODEL_PATH, model_filename))
