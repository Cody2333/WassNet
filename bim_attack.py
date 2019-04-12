import matplotlib.pyplot as plt
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from WassNet import WassNet
from advertorch.utils import predict_from_logits
from advertorch_examples.utils import get_mnist_test_loader
from advertorch_examples.utils import _imshow

torch.manual_seed(20)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
from advertorch.test_utils import LeNet5
from advertorch_examples.utils import TRAINED_MODEL_PATH

filename = "mnist_lenet5_clntrained.pt"
# filename = "mnist_lenet5_advtrained.pt"

model = LeNet5()
model.load_state_dict(
    torch.load(os.path.join(TRAINED_MODEL_PATH, filename),map_location='cpu'))
model.to(device)
model.eval()

wass_model_filename = 'mnist_wass_net.pt'
wass_model = WassNet()
wass_model.load_state_dict(
    torch.load(os.path.join(TRAINED_MODEL_PATH, wass_model_filename),map_location='cpu'))
wass_model.to(device)
wass_model.eval()

batch_size = 1
loader = get_mnist_test_loader(batch_size=batch_size,shuffle=True)
for cln_data, true_label in loader:
    break
cln_data, true_label = cln_data.to(device), true_label.to(device)

from advertorch.attacks import LinfPGDAttack,LinfBasicIterativeAttack,L1BasicIterativeAttack,GradientAttack,GradientSignAttack

def myloss(model,yvar,xvar,xadv):
    outputs = model(xadv)
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(outputs, yvar)
    w_loss = torch.mean(wass_model(xadv))
    return loss,w_loss



adversary = L1BasicIterativeAttack(
    model, loss_fn=myloss, eps=0.2,
    nb_iter=20, eps_iter=0.01, clip_min=0.0, clip_max=1.0,
    targeted=False)

adv_untargeted = adversary.perturb(cln_data, true_label)
pred_cln = predict_from_logits(model(cln_data))
pred_untargeted_adv = predict_from_logits(model(adv_untargeted))

def eu_dist(x,y):
    A=x.reshape(28*28)
    B=y.reshape(28*28)
    return np.linalg.norm(A - B,ord=2)
def w_dist(x,y):
    return (wass_model(x) - wass_model(y)).detach().numpy()[0]

print('l2 distance:', eu_dist(cln_data[0].numpy(), adv_untargeted[0].numpy()))
print('w distance:', w_dist(cln_data, adv_untargeted))

for ii in range(batch_size):
    plt.subplot(3, batch_size, ii + 1)
    _imshow(cln_data[ii])
    plt.title("clean \n pred: {}".format(pred_cln[ii]))
    plt.subplot(3, batch_size, ii + 1 + batch_size)
    _imshow(adv_untargeted[ii])
    plt.title("untargeted \n adv \n pred: {}".format(
        pred_untargeted_adv[ii]))
plt.tight_layout()
# plt.show()
plt.savefig("examples.jpg")
